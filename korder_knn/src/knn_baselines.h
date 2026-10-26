//
// knn_baselines.h -- the k-NN methods the order-k index is compared with:
//
//   bruteKnn        exhaustive (verification only)
//   KdKnn           nanoflann kd-tree (leaf size 10)
//   DelaunayKnn     exact NN, then best-first expansion over the Delaunay graph
//                   (the (i+1)-th neighbour is a Delaunay neighbour of one of the
//                   first i; used e.g. by the VoR-tree)
//   SuccessorKnn    the successor-list scheme of "Manifold k-NN" (arXiv 2605.02224):
//                   candidates = the transition sites of the DPNN path plus the
//                   successor (child) lists of the neighbours confirmed so far.
//                   Our re-implementation from the paper's description, not the
//                   authors' code.
//
// All of them return the k nearest points sorted by distance.
//
#pragma once
#include "korder.h"

#include <nanoflann.hpp>

namespace kov {

struct KnnStats {
	u64 dist = 0;  // distance evaluations
};

// sorted (ascending) list of at most cap candidates
struct Cand {
	double d;
	u32 id;
};
inline void candInsert(Cand* c, int& size, int cap, double d, u32 id) {
	if (size == cap) {
		if (!(d < c[size - 1].d)) return;
		--size;
	}
	int i = size - 1;
	while (i >= 0 && c[i].d > d) {
		c[i + 1] = c[i];
		--i;
	}
	c[i + 1] = Cand{d, id};
	++size;
}

// open-addressing set of point ids, emptied after every query
class SmallSet {
public:
	explicit SmallSet(int log2cap = 12) { init(log2cap); }
	bool insert(u32 x) {
		if (2 * (used_.size() + 1) > tab_.size()) grow();
		u32 i = (x * 2654435761u) >> shift_;
		for (;;) {
			if (tab_[i] == kEmpty) {
				tab_[i] = x;
				used_.push_back(i);
				return true;
			}
			if (tab_[i] == x) return false;
			i = (i + 1) & mask_;
		}
	}
	void clear() {
		for (u32 i : used_) tab_[i] = kEmpty;
		used_.clear();
	}

private:
	static constexpr u32 kEmpty = 0xFFFFFFFFu;
	std::vector<u32> tab_, used_;
	u32 mask_ = 0;
	int shift_ = 0;
	void init(int log2cap) {
		tab_.assign(size_t(1) << log2cap, kEmpty);
		mask_ = (u32)tab_.size() - 1;
		shift_ = 32 - log2cap;
		used_.clear();
		used_.reserve(tab_.size() / 2);
	}
	void grow() {
		std::vector<u32> keys;
		for (u32 i : used_) keys.push_back(tab_[i]);
		init(32 - shift_ + 1);
		for (u32 x : keys) insert(x);
	}
};

inline void bruteKnn(const std::vector<Vec3>& P, const Vec3& q, int k, double* d2, u32* id) {
	std::vector<Cand> c(k + 1);
	int sz = 0;
	for (size_t i = 0; i < P.size(); ++i) {
		const double d = dpl::sqDist(P[i], q);
		if (sz < k || d < c[sz - 1].d) candInsert(c.data(), sz, k, d, (u32)i);
	}
	for (int i = 0; i < sz; ++i) { d2[i] = c[i].d; id[i] = c[i].id; }
}

// ---------------------------------------------------------------- kd-tree
class KdKnn {
	struct Cloud {
		const std::vector<Vec3>* P;
		size_t kdtree_get_point_count() const { return P->size(); }
		double kdtree_get_pt(size_t i, size_t d) const { return d == 0 ? (*P)[i].x : d == 1 ? (*P)[i].y : (*P)[i].z; }
		template <class B>
		bool kdtree_get_bbox(B&) const { return false; }
	};
	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Cloud, double, u32>, Cloud, 3, u32> KD;

public:
	explicit KdKnn(const std::vector<Vec3>& P, int leaf = 10) : cloud_{&P}, kd_(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf)) {
		kd_.buildIndex();
	}
	void query(const Vec3& q, int k, u32* id, double* d2) const { (void)kd_.knnSearch(&q.x, k, id, d2); }
	size_t bytes() const { return kd_.usedMemory(kd_); }

private:
	Cloud cloud_;
	KD kd_;
};

// ---------------------------------------------------------------- Delaunay best-first
class DelaunayKnn {
public:
	struct E {
		double x, y, z;
		u32 id, pad;
	};
	DelaunayKnn(const std::vector<Vec3>& P, const std::vector<std::pair<u32, u32>>& edges) : P_(P) {
		const size_t n = P.size();
		beg_.assign(n + 1, 0);
		for (auto& e : edges) { beg_[e.first + 1]++; beg_[e.second + 1]++; }
		for (size_t i = 0; i < n; ++i) beg_[i + 1] += beg_[i];
		adj_.resize(beg_[n]);
		std::vector<u32> fill(beg_.begin(), beg_.end() - 1);
		auto put = [&](u32 a, u32 b) { adj_[fill[a]++] = E{P[b].x, P[b].y, P[b].z, b, 0}; };
		for (auto& e : edges) { put(e.first, e.second); put(e.second, e.first); }
	}
	size_t bytes() const { return adj_.size() * sizeof(E) + beg_.size() * 4; }
	double avgDegree() const { return double(adj_.size()) / (beg_.size() - 1); }

	// nn: the nearest point (from any exact NN structure)
	void query(const Vec3& q, u32 nn, int k, u32* id, double* d2, SmallSet& vis, KnnStats* st = nullptr) const {
		Cand buf[256];
		Cand* c = buf;
		int sz = 0;
		id[0] = nn;
		d2[0] = dpl::sqDist(P_[nn], q);
		vis.insert(nn);
		u64 ev = 0;
		for (int r = 0; r + 1 < k; ++r) {
			const int cap = k - 1 - r;  // farther candidates can no longer be reported
			if (c + cap + 1 > buf + 256) {  // keep room for one insertion
				std::memmove(buf, c, sz * sizeof(Cand));
				c = buf;
			}
			const u32 u = id[r];
			for (u32 p = beg_[u]; p < beg_[u + 1]; ++p) {
				const E& e = adj_[p];
				if (!vis.insert(e.id)) continue;
				++ev;
				const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (sz < cap || d < c[sz - 1].d) candInsert(c, sz, cap, d, e.id);
			}
			id[r + 1] = c[0].id;
			d2[r + 1] = c[0].d;
			++c;
			--sz;
		}
		vis.clear();
		if (st) st->dist += ev + 1;
	}

private:
	const std::vector<Vec3>& P_;
	std::vector<u32> beg_;
	std::vector<E> adj_;
};

// ---------------------------------------------------------------- successor lists
class SuccessorKnn {
public:
	struct E {
		double x, y, z;
		u32 id, pad;
	};
	// g: DPNN DAG, P: its points (P[i] == g.pts[i], possibly translated)
	SuccessorKnn(const dpl::Dag& g, const std::vector<Vec3>& P) : P_(P), beg_(g.chBeg) {
		succ_.resize(g.ch.size());
		for (size_t i = 0; i < g.ch.size(); ++i) {
			const u32 v = g.ch[i];
			succ_[i] = E{P[v].x, P[v].y, P[v].z, v, 0};
		}
	}
	size_t bytes() const { return succ_.size() * sizeof(E) + beg_.size() * 4; }

	void query(const Vec3& q, int k, u32* id, double* d2, SmallSet& vis, KnnStats* st = nullptr) const {
		Cand c[257];
		int sz = 0;
		u64 ev = 1;
		// 1-NN by DPNN, keeping the transition sites (the nearest point of every prefix)
		u32 u = 0;
		double du = dpl::sqDist(P_[0], q);
		vis.insert(0);
		candInsert(c, sz, k, du, 0);
		for (;;) {
			bool jumped = false;
			for (u32 p = beg_[u]; p < beg_[u + 1]; ++p) {
				const E& e = succ_[p];
				++ev;
				const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (d < du) {
					u = e.id;
					du = d;
					jumped = true;
					break;
				}
			}
			if (!jumped) break;
			vis.insert(u);
			candInsert(c, sz, k, du, u);
		}
		// confirm the r-th neighbour, then add its successors
		for (int r = 0; r < k; ++r) {
			id[r] = c[r].id;
			d2[r] = c[r].d;
			if (r + 1 == k) break;
			const u32 x = c[r].id;
			for (u32 p = beg_[x]; p < beg_[x + 1]; ++p) {
				const E& e = succ_[p];
				if (!vis.insert(e.id)) continue;
				++ev;
				const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (sz < k || d < c[sz - 1].d) candInsert(c, sz, k, d, e.id);
			}
		}
		vis.clear();
		if (st) st->dist += ev;
	}

private:
	const std::vector<Vec3>& P_;
	std::vector<u32> beg_;
	std::vector<E> succ_;
};

// ---------------------------------------------------------------- NN + candidate list
// C(u) = union of the order-k cells that contain u.  If u is the nearest point of
// q, the k-set of q is an order-k cell containing u, so kNN(q) ⊆ C(u): one exact
// NN query, then one contiguous scan.  C(u) is also the set of p for which some
// sphere through u and p has at most k-2 points inside (order-(k-2) Delaunay edges).
class CandKnn {
public:
	struct E {
		double x, y, z;
		u32 id, pad;
	};
	// from explicit lists: lists[beg[u]..beg[u+1]) = the candidates of u (must contain u)
	CandKnn(const std::vector<Vec3>& P, int k, const std::vector<u32>& beg, const std::vector<u32>& lists) : k_(k), beg_(beg) {
		ent_.resize(lists.size());
#pragma omp parallel for schedule(static, 65536)
		for (long long i = 0; i < (long long)lists.size(); ++i) {
			const u32 p = lists[i];
			ent_[i] = E{P[p].x, P[p].y, P[p].z, p, 0};
		}
	}
	CandKnn(const std::vector<Vec3>& P, const Level& Lk) : k_(Lk.k) {
		const size_t n = P.size();
		const int k = Lk.k;
		std::vector<u64> pr;
		pr.reserve(Lk.M * (size_t)k * (k - 1) + n);
		for (size_t u = 0; u < n; ++u) pr.push_back((u64)u << 32 | u);
		for (size_t t = 0; t < Lk.M; ++t) {
			const u32* m = Lk.members(t);
			for (int a = 0; a < k; ++a)
				for (int b = 0; b < k; ++b)
					if (a != b) pr.push_back((u64)m[a] << 32 | m[b]);
		}
		__gnu_parallel::sort(pr.begin(), pr.end());
		pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
		beg_.assign(n + 1, 0);
		for (u64 x : pr) beg_[(x >> 32) + 1]++;
		for (size_t u = 0; u < n; ++u) beg_[u + 1] += beg_[u];
		ent_.resize(pr.size());
#pragma omp parallel for schedule(static, 65536)
		for (long long i = 0; i < (long long)pr.size(); ++i) {
			const u32 p = u32(pr[i] & 0xFFFFFFFFu);
			ent_[i] = E{P[p].x, P[p].y, P[p].z, p, 0};
		}
	}
	size_t bytes() const { return ent_.size() * sizeof(E) + beg_.size() * 4; }
	double avgLen() const { return double(ent_.size()) / (beg_.size() - 1); }
	int k() const { return k_; }

	// nn: the nearest point of q; k <= the order the lists were built for
	void query(const Vec3& q, u32 nn, int k, u32* id, double* d2, KnnStats* st = nullptr) const {
		Cand c[257];
		int sz = 0;
		const E* e = ent_.data() + beg_[nn];
		const E* end = ent_.data() + beg_[nn + 1];
		for (; e < end; ++e) {
			const double dx = e->x - q.x, dy = e->y - q.y, dz = e->z - q.z;
			const double d = dx * dx + dy * dy + dz * dz;
			if (sz < k || d < c[sz - 1].d) candInsert(c, sz, k, d, e->id);
		}
		for (int i = 0; i < k; ++i) { id[i] = c[i].id; d2[i] = c[i].d; }
		if (st) st->dist += beg_[nn + 1] - beg_[nn];
	}

private:
	int k_;
	std::vector<u32> beg_;
	std::vector<E> ent_;
};

}  // namespace kov
