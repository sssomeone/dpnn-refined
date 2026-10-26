//
// dpnn_flat.h -- latency-oriented query layout for any LayerIndex (level k or trie).
//
// A DPNN query is a chain of dependent memory accesses: one per jump (~log n of
// them).  With the dpnn.cpp layout every jump costs two dependent misses (the
// child list, then the children's coordinates); comparisons themselves are
// almost free because they overlap with those misses.  This layout stores each
// state's candidate list as consecutive 32-byte entries {x, y, z, next} followed
// by a terminator that carries the answer, and `next` is directly the offset of
// the next list -> one dependent miss per jump.
//
// queryBatch() additionally interleaves K independent queries (one list per
// query per round, prefetching the next list), which turns the latency-bound
// loop into a throughput-bound one; this is where shorter refined lists pay off.
//
#pragma once
#include "dpnn_layered.h"

#include <sys/mman.h>
#include <cstring>

namespace dpl {

// Read-only array in anonymous memory, optionally backed by transparent huge
// pages (random access over hundreds of MB is otherwise dominated by TLB misses).
template <class T>
class PagedArray {
public:
	PagedArray() = default;
	PagedArray(const PagedArray&) = delete;
	PagedArray& operator=(const PagedArray&) = delete;
	~PagedArray() { release(); }

	// Maps room for n elements (zero pages until first written) for the caller to fill.
	T* allocate(size_t n, bool hugePages) {
		release();
		n_ = n;
		bytes_ = std::max<size_t>(((n * sizeof(T) + kHuge - 1) / kHuge) * kHuge, kHuge);
		void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
#ifdef MADV_HUGEPAGE
		if (hugePages) madvise(p, bytes_, MADV_HUGEPAGE);
#endif
		p_ = static_cast<T*>(p);
		return p_;
	}
	void assign(const T* src, size_t n, bool hugePages) { std::memcpy(allocate(n, hugePages), src, n * sizeof(T)); }
	const T* data() const { return p_; }
	size_t size() const { return n_; }
	size_t bytes() const { return n_ * sizeof(T); }

private:
	static constexpr size_t kHuge = size_t(2) << 20;
	void release() {
		if (p_) munmap(p_, bytes_);
		p_ = nullptr;
	}
	T* p_ = nullptr;
	size_t n_ = 0, bytes_ = 0;
};

struct alignas(32) FlatEntry {
	double x, y, z;
	u32 next;  // offset of the next list, or kEnd for the terminator
	u32 node;  // terminator: the DAG node to return
};
static_assert(sizeof(FlatEntry) == 32, "FlatEntry must be 32 bytes");

// same for the trie (node t > 0 was created by entry t - 1)
inline std::vector<u32> trieStateNodes(const LayerIndex& Tr) {
	std::vector<u32> sn(Tr.numStates());
	sn[0] = Tr.rootNode;
	for (size_t t = 1; t < sn.size(); ++t) sn[t] = Tr.ent[t - 1].node;
	return sn;
}

class FlatIndex {
public:
	static constexpr u32 kEnd = 0xFFFFFFFFu;

	FlatIndex(const LayerIndex& L, const std::vector<u32>& stateNode, bool hugePages = true) {
		const long long S = (long long)L.numStates();
		if (L.ent.size() + S >= kEnd) throw std::runtime_error("FlatIndex: too many entries for 32-bit offsets");
		std::vector<u32> start(S);
#pragma omp parallel for schedule(static)
		for (long long s = 0; s < S; ++s) start[s] = L.beg[s] + (u32)s;  // one terminator per preceding list
		// filled in place and in parallel (also spreads the first touch of the pages;
		// on one thread this took ~17 s for the 7.5 GB of Lucy's 2-level index)
		FlatEntry* v = ent_.allocate(L.ent.size() + S, hugePages);
#pragma omp parallel for schedule(static, 4096)
		for (long long s = 0; s < S; ++s) {
			u32 w = start[s];
			for (u32 p = L.beg[s]; p < L.beg[s + 1]; ++p) {
				const Entry& e = L.ent[p];
				v[w++] = FlatEntry{e.x, e.y, e.z, start[e.next], e.node};
			}
			v[w] = FlatEntry{0, 0, 0, kEnd, stateNode[s]};
		}
		rootStart_ = start[L.root];
		rootPt_ = L.rootPt;
	}

	size_t bytes() const { return ent_.bytes(); }

	u32 query(const Vec3& q) const {
		const FlatEntry* B = ent_.data();
		u32 s = rootStart_;
		double best = sqDist(q, rootPt_);
		for (;;) {
			const FlatEntry* e = B + s;
			for (;; ++e) {
				if (e->next == kEnd) return e->node;
				const double dx = e->x - q.x, dy = e->y - q.y, dz = e->z - q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (d < best) {
					best = d;
					s = e->next;
					break;
				}
			}
		}
	}

	u32 query(const Vec3& q, QueryStats& st) const {
		const FlatEntry* B = ent_.data();
		u32 s = rootStart_;
		double best = sqDist(q, rootPt_);
		for (;;) {
			const FlatEntry* e = B + s;
			for (;; ++e) {
				if (e->next == kEnd) return e->node;
				st.comparisons++;
				const double d = sqDist(q, Vec3{e->x, e->y, e->z});
				if (d < best) {
					best = d;
					s = e->next;
					st.jumps++;
					break;
				}
			}
		}
	}

	// Answers Q[0..m) into out[0..m), K queries in flight.
	template <int K = 16>
	void queryBatch(const Vec3* Q, size_t m, u32* out) const {
		const FlatEntry* B = ent_.data();
		struct Slot {
			Vec3 q;
			double best;
			u32 s;
			size_t idx;
		};
		Slot sl[K];
		int active = 0;
		size_t nq = 0;
		for (; active < K && nq < m; ++active, ++nq) sl[active] = Slot{Q[nq], sqDist(Q[nq], rootPt_), rootStart_, nq};
		while (active) {
			for (int k = 0; k < active;) {
				Slot& S = sl[k];
				const FlatEntry* e = B + S.s;
				bool jumped = false;
				for (;; ++e) {
					if (e->next == kEnd) break;
					const double dx = e->x - S.q.x, dy = e->y - S.q.y, dz = e->z - S.q.z;
					const double d = dx * dx + dy * dy + dz * dz;
					if (d < S.best) {
						S.best = d;
						S.s = e->next;
						jumped = true;
						break;
					}
				}
				if (jumped) {  // next round for this slot reads B + S.s
					__builtin_prefetch(B + S.s);
					__builtin_prefetch(B + S.s + 2);
					++k;
					continue;
				}
				out[S.idx] = e->node;
				if (nq < m) {
					S = Slot{Q[nq], sqDist(Q[nq], rootPt_), rootStart_, nq};
					++nq;
					++k;
				} else {
					sl[k] = sl[--active];
				}
			}
		}
	}

private:
	PagedArray<FlatEntry> ent_;
	u32 rootStart_ = 0;
	Vec3 rootPt_{0, 0, 0};
};


// ---------------------------------------------------------------------------
// FlatIndex16: same layout with float coordinates (16-byte entries, half the
// memory) and *the same answers* as the double version.  Coordinates are stored
// relative to the bounding-box centre c; every stored coordinate is within
// 2^-24 R of the exact one (R = max |p - c|), so a float-based distance is
// within eta = sqrt(3) 2^-23 R (+ tiny double rounding terms) of the exact one.
// A comparison is decided from floats only when the gap exceeds 2 eta;
// otherwise both distances are recomputed from the double coordinates exactly as
// the reference implementation does.
//
// List layout: [header {node, count}] [count entries {x, y, z, next}]
// ---------------------------------------------------------------------------
struct alignas(16) Flat16Entry {
	float x, y, z;
	u32 next;  // header: number of entries
};
static_assert(sizeof(Flat16Entry) == 16, "Flat16Entry must be 16 bytes");

class FlatIndex16 {
public:
	FlatIndex16(const LayerIndex& L, const std::vector<u32>& stateNode, const Dag& g, bool hugePages = true)
	    : pts_(g.pts), c_(g.center) {
		const long long S = (long long)L.numStates();
		if (L.ent.size() + S >= 0xFFFFFFFFull) throw std::runtime_error("FlatIndex16: too many entries");
		double R = 0;
		for (auto& p : g.pts) R = std::max({R, std::fabs(p.x - c_.x), std::fabs(p.y - c_.y), std::fabs(p.z - c_.z)});
		etaData_ = std::sqrt(3.0) * std::ldexp(R, -23);
		std::vector<u32> start(S);
#pragma omp parallel for schedule(static)
		for (long long s = 0; s < S; ++s) start[s] = L.beg[s] + (u32)s;  // one header per preceding list
		Flat16Entry* v = ent_.allocate(L.ent.size() + S, hugePages);  // filled in place, in parallel
#pragma omp parallel for schedule(static, 4096)
		for (long long s = 0; s < S; ++s) {
			u32 w = start[s];
			v[w++] = header(stateNode[s], L.beg[s + 1] - L.beg[s]);
			for (u32 p = L.beg[s]; p < L.beg[s + 1]; ++p) {
				const Entry& e = L.ent[p];
				v[w++] = Flat16Entry{float(e.x - c_.x), float(e.y - c_.y), float(e.z - c_.z), start[e.next]};
			}
		}
		rootStart_ = start[L.root];
		const Vec3& r = L.rootPt;
		rootF_ = Vec3{double(float(r.x - c_.x)), double(float(r.y - c_.y)), double(float(r.z - c_.z))};
	}

	size_t bytes() const { return ent_.bytes() + pts_.size() * sizeof(Vec3); }

	u32 query(const Vec3& q) const { return run<false>(q, nullptr); }
	u32 query(const Vec3& q, QueryStats& st) const { return run<true>(q, &st); }
	template <int K = 16>
	void queryBatch(const Vec3* Q, size_t m, u32* out) const {
		const Flat16Entry* B = ent_.data();
		struct Slot {
			Vec3 q, qAbs;  // q relative to c
			double bestF, lo2, hi2, eta;
			u32 s;
			size_t idx;
		};
		Slot sl[K];
		int active = 0;
		size_t nq = 0;
		auto start = [&](Slot& S) {
			S.qAbs = Q[nq];
			S.q = Vec3{Q[nq].x - c_.x, Q[nq].y - c_.y, Q[nq].z - c_.z};
			S.eta = eta(S.q);
			S.s = rootStart_;
			S.idx = nq++;
			setBest(S.bestF, S.lo2, S.hi2, sqDist(S.q, rootF_), S.eta);
		};
		for (; active < K && nq < m; ++active) start(sl[active]);
		while (active) {
			for (int k = 0; k < active;) {
				Slot& S = sl[k];
				const Flat16Entry* h = B + S.s;
				const u32 cnt = h->next;
				u32 i = 1;
				for (; i <= cnt; ++i) {
					const Flat16Entry& e = h[i];
					const double d = distF(S.q, e);
					if (d < S.lo2 || (d <= S.hi2 && exactCloserAbs(S.qAbs, e.next, nodeOf(h)))) {
						S.s = e.next;
						setBest(S.bestF, S.lo2, S.hi2, d, S.eta);
						break;
					}
				}
				if (i <= cnt) {
					__builtin_prefetch(B + S.s);
					__builtin_prefetch(B + S.s + 4);
					++k;
					continue;
				}
				out[S.idx] = nodeOf(h);
				if (nq < m) {
					start(S);
					++k;
				} else {
					sl[k] = sl[--active];
				}
			}
		}
	}

private:
	PagedArray<Flat16Entry> ent_;
	std::vector<Vec3> pts_;  // exact coordinates (node order) for the rare ambiguous comparisons
	Vec3 c_;
	Vec3 rootF_;
	double etaData_ = 0;
	u32 rootStart_ = 0;

	static Flat16Entry header(u32 node, u32 count) {
		Flat16Entry h{0, 0, 0, count};
		std::memcpy(&h.x, &node, sizeof(u32));
		return h;
	}
	static u32 nodeOf(const Flat16Entry* h) {
		u32 node;
		std::memcpy(&node, &h->x, sizeof(u32));
		return node;
	}
	static double distF(const Vec3& q, const Flat16Entry& e) {
		const double dx = double(e.x) - q.x, dy = double(e.y) - q.y, dz = double(e.z) - q.z;
		return dx * dx + dy * dy + dz * dz;
	}
	// error bound of a float-based distance for this query (q relative to c)
	double eta(const Vec3& q) const {
		const double Q = std::max({std::fabs(q.x), std::fabs(q.y), std::fabs(q.z)});
		return etaData_ + std::ldexp(etaData_ + Q, -46);
	}
	// a candidate at float-distance^2 d is surely closer if d < lo2, surely not if d > hi2
	static void setBest(double& bestF, double& lo2, double& hi2, double d, double eta) {
		bestF = d;
		const double b = std::sqrt(d);
		lo2 = b > 2 * eta ? (b - 2 * eta) * (b - 2 * eta) : -1.0;
		hi2 = (b + 2 * eta) * (b + 2 * eta);
	}

	template <bool kStats>
	u32 run(const Vec3& qAbs, QueryStats* st) const {
		const Flat16Entry* B = ent_.data();
		const Vec3 q{qAbs.x - c_.x, qAbs.y - c_.y, qAbs.z - c_.z};
		const double et = eta(q);
		double bestF, lo2, hi2;
		setBest(bestF, lo2, hi2, sqDist(q, rootF_), et);
		u32 s = rootStart_;
		for (;;) {
			const Flat16Entry* h = B + s;
			const u32 cnt = h->next;
			u32 i = 1;
			for (; i <= cnt; ++i) {
				const Flat16Entry& e = h[i];
				if (kStats) st->comparisons++;
				const double d = distF(q, e);
				if (kStats && d >= lo2 && d <= hi2) st->exactRechecks++;
				if (d < lo2 || (d <= hi2 && exactCloserAbs(qAbs, e.next, nodeOf(h)))) {
					s = e.next;
					setBest(bestF, lo2, hi2, d, et);
					if (kStats) st->jumps++;
					break;
				}
			}
			if (i > cnt) return nodeOf(h);
		}
	}
	// Exact test in the reference arithmetic: is the candidate whose list starts at
	// `next` strictly closer to q than node `cur`?
	bool exactCloserAbs(const Vec3& q, u32 next, u32 cur) const {
		const u32 cand = nodeOf(ent_.data() + next);
		return sqDist(pts_[cand], q) < sqDist(pts_[cur], q);
	}
};
}  // namespace dpl
