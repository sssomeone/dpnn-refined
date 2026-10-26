//
// dpnn_layered.h -- DPNN with history-refined query lists, built from Delaunay
// adjacency only (no Voronoi cell construction, no polyhedral clipping).
//
// Notation (paper, Sec. III / VI.E of arXiv:2409.15023v2-v4):
//   p_1..p_n           points in insertion order (node i == p_{i+1} here)
//   V_m                Voronoi diagram of the first m points
//   Phi_m(q)           nearest neighbour of q among the first m points
//   Parents(v)         Delaunay neighbours of v right after v is inserted
//                      (== the nodes whose query list contains v)
//
// Every DAG edge u -> v (u in Parents(v)) is the event "q's nearest neighbour
// switches from u to v when v is inserted".  The set of q for which that event
// happens is
//
//   Q(u->v) = Cell(u; V_{t_v - 1}) ∩ Cell(v; V_{t_v})
//           = { |x-v| <= |x-u| } ∩ ⋂_{c in Parents(v), c != u} { |x-u| <= |x-c| }      (*)
//
// i.e. an intersection of bisector half-spaces whose index set is just the
// Delaunay neighbourhood of v at insertion time (facets of u's old cell that do
// not meet v's new cell are redundant inside {|x-v| <= |x-u|}).
//
// A query that went through the edges e_1, ..., e_r knows q ∈ Q(e_1) ∩ ... ∩ Q(e_r),
// and candidate w in the list of the current node v can only move the answer if
//
//   Q(e_1) ∩ ... ∩ Q(e_r) ∩ Q(v->w) != ∅.                                           (**)
//
// (Q(v->w) already lies inside Cell(v; V_{t_w - 1}), so all the "cell shrinking"
// that happened between t_v and t_w is accounted for automatically - nothing has
// to be cut or maintained.)  (**) is a 3D LP feasibility test with
// ~16 (r+1) bisector half-spaces (lp3.h).
//
//   level 0      : the original DPNN (state = node)
//   level k >= 1 : state = last k edges of the path (lists filtered with (**) on k+1 pieces)
//   trie         : state = whole path (all layers of history) == cells of the
//                  overlay of V_1, ..., V_n
//
// All variants return exactly what the original DPNN returns: a candidate is
// dropped only if (**) is infeasible even after relaxing every half-space by
// kRelax (normalized units), so no candidate that could win a floating-point
// distance comparison is ever skipped.  The LPs live in the box [-LP3::M, LP3::M]^3
// of normalized coordinates, i.e. this holds for queries within ~1000 bounding-box
// diagonals of the data; use level 0 for anything farther away.
//
#pragma once
#include "lp3.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/property_map.h>
#include <CGAL/spatial_sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dpl {

using u32 = uint32_t;
using u64 = uint64_t;

struct Vec3 {
	double x, y, z;
};

inline double sqDist(const Vec3& a, const Vec3& b) {
	const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

enum class InsertionOrder { Brio, Random, Fps };

inline const char* orderName(InsertionOrder o) {
	return o == InsertionOrder::Brio ? "brio" : o == InsertionOrder::Random ? "random" : "fps";
}

// ---------------------------------------------------------------------------
// The query DAG ("Query Table" of the paper).  Node i is the i-th inserted point.
// ---------------------------------------------------------------------------
struct Dag {
	std::vector<Vec3> pts;    // coordinates, insertion order
	std::vector<Vec3> npts;   // same, normalized (bbox centre -> 0, bbox diagonal -> 1); used by the LPs
	std::vector<u32> origId;  // pts[i] == input[origId[i]]
	std::vector<u32> chBeg, ch;  // query lists (CSR); ascending == insertion order; edge id == index into ch
	std::vector<u32> paBeg, pa;  // Parents(v) (CSR)
	std::vector<u32> edgeSrc;    // edge e goes edgeSrc[e] -> ch[e]
	std::vector<u32> dupOf;      // input index -> input index of the identical point that was kept
	Vec3 center{0, 0, 0};
	double invScale = 1;

	size_t n() const { return pts.size(); }
	size_t numEdges() const { return ch.size(); }
};

namespace detail {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<u32, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> DT;

// max-segment tree over the not-yet-inserted points (farthest point sampling)
struct MaxTree {
	int size = 1;
	std::vector<double> v;
	std::vector<u32> arg;
	explicit MaxTree(const std::vector<double>& init) {
		while (size < (int)init.size()) size <<= 1;
		v.assign(2 * size, -1.0);
		arg.assign(2 * size, 0);
		for (size_t i = 0; i < init.size(); ++i) { v[size + i] = init[i]; arg[size + i] = (u32)i; }
		for (int i = size - 1; i >= 1; --i) pull(i);
	}
	void pull(int i) {
		const int l = 2 * i, r = 2 * i + 1;
		if (v[l] >= v[r]) { v[i] = v[l]; arg[i] = arg[l]; }
		else { v[i] = v[r]; arg[i] = arg[r]; }
	}
	void set(u32 pos, double val) {
		int i = size + (int)pos;
		v[i] = val;
		for (i >>= 1; i >= 1; i >>= 1) pull(i);
	}
};

}  // namespace detail

// Builds the DAG by incremental Delaunay insertion (same procedure as dpnn.cpp,
// but with exact predicates and recording Parents(v)).
inline Dag buildDag(const std::vector<Vec3>& input, InsertionOrder order, unsigned seed = 1) {
	using namespace detail;
	Dag g;
	const size_t nIn = input.size();
	if (nIn < 5) throw std::runtime_error("need at least 5 points");

	// exact duplicates: keep the first occurrence
	std::vector<u32> byCoord(nIn);
	std::iota(byCoord.begin(), byCoord.end(), 0u);
	std::sort(byCoord.begin(), byCoord.end(), [&](u32 a, u32 b) {
		const Vec3 &p = input[a], &q = input[b];
		if (p.x != q.x) return p.x < q.x;
		if (p.y != q.y) return p.y < q.y;
		if (p.z != q.z) return p.z < q.z;
		return a < b;
	});
	g.dupOf.resize(nIn);
	std::vector<u32> uniq;
	for (size_t i = 0; i < nIn; ++i) {
		const u32 a = byCoord[i];
		if (i > 0) {
			const u32 b = byCoord[i - 1];
			if (input[a].x == input[b].x && input[a].y == input[b].y && input[a].z == input[b].z) {
				g.dupOf[a] = g.dupOf[b];
				continue;
			}
		}
		g.dupOf[a] = a;
		uniq.push_back(a);
	}
	std::sort(uniq.begin(), uniq.end());
	const size_t n = uniq.size();

	// insertion order (indices into input)
	std::vector<u32> perm;
	if (order == InsertionOrder::Brio) {
		// identical to dpnn.cpp: CGAL::spatial_sort (random shuffle + multiscale Hilbert sort)
		typedef std::pair<K::Point_3, u32> PI;
		typedef CGAL::Spatial_sort_traits_adapter_3<K, CGAL::First_of_pair_property_map<PI>> Traits;
		std::vector<PI> tmp;
		tmp.reserve(n);
		for (u32 id : uniq) tmp.emplace_back(K::Point_3(input[id].x, input[id].y, input[id].z), id);
		CGAL::spatial_sort(tmp.begin(), tmp.end(), Traits());
		for (auto& t : tmp) perm.push_back(t.second);
	} else if (order == InsertionOrder::Random) {
		perm = uniq;
		std::mt19937 gen(seed);
		std::shuffle(perm.begin(), perm.end(), gen);
	}
	// (Fps: the order is produced on the fly below)

	std::vector<std::vector<u32>> parents(n);
	g.pts.resize(n);
	g.origId.resize(n);
	DT dt;
	std::vector<DT::Vertex_handle> vh(n);
	std::vector<DT::Vertex_handle> nb;
	nb.reserve(64);

	// farthest point sampling state (only used for Fps)
	std::vector<u32> local;               // position in uniq -> current bucket owner (insertion index)
	std::vector<std::vector<u32>> bucket;  // insertion index -> positions (in uniq) of uninserted points it owns
	std::unique_ptr<MaxTree> tree;
	if (order == InsertionOrder::Fps) {
		local.assign(n, 0);
		bucket.resize(n);
		std::vector<double> d0(n);
		const Vec3 p0 = input[uniq[0]];
		for (size_t i = 0; i < n; ++i) d0[i] = sqDist(input[uniq[i]], p0);
		d0[0] = -1;
		tree.reset(new MaxTree(d0));
		bucket[0].resize(n - 1);
		std::iota(bucket[0].begin(), bucket[0].end(), 1u);
	}

	DT::Vertex_handle hint;
	for (size_t i = 0; i < n; ++i) {
		u32 id, pos = 0;
		if (order == InsertionOrder::Fps) {
			pos = (i == 0) ? 0 : tree->arg[1];
			id = uniq[pos];
			hint = (i == 0) ? DT::Vertex_handle() : vh[local[pos]];
		} else {
			id = perm[i];
		}
		const Vec3 p = input[id];
		g.pts[i] = p;
		g.origId[i] = id;
		const size_t before = dt.number_of_vertices();
		DT::Vertex_handle v = (hint == DT::Vertex_handle()) ? dt.insert(K::Point_3(p.x, p.y, p.z))
		                                                    : dt.insert(K::Point_3(p.x, p.y, p.z), hint);
		if (dt.number_of_vertices() != before + 1) throw std::runtime_error("duplicate point survived dedup");
		v->info() = (u32)i;
		vh[i] = v;
		if (order != InsertionOrder::Fps) hint = v;
		if (i < 4) {  // dpnn.cpp: the first four points form a complete graph
			for (u32 j = 0; j < i; ++j) parents[i].push_back(j);
		} else {
			nb.clear();
			dt.finite_adjacent_vertices(v, std::back_inserter(nb));
			for (auto& w : nb) parents[i].push_back(w->info());
			std::sort(parents[i].begin(), parents[i].end());
			parents[i].erase(std::unique(parents[i].begin(), parents[i].end()), parents[i].end());
		}
		if (order == InsertionOrder::Fps && i > 0) {
			// points whose nearest sample changes live in the cells of Parents(i)
			tree->set(pos, -1.0);
			for (u32 u : parents[i]) {
				std::vector<u32>& B = bucket[u];
				size_t keep = 0;
				const Vec3 pu = g.pts[u];
				for (size_t k = 0; k < B.size(); ++k) {
					const u32 w = B[k];
					if (w == pos) continue;
					const Vec3 pw = input[uniq[w]];
					const double dNew = sqDist(pw, p);
					if (dNew < sqDist(pw, pu)) {
						bucket[i].push_back(w);
						local[w] = (u32)i;
						tree->set(w, dNew);
					} else {
						B[keep++] = w;
					}
				}
				B.resize(keep);
				B.shrink_to_fit();
			}
		}
	}

	// CSR: parents, children (= transpose; ascending child == insertion order)
	g.paBeg.assign(n + 1, 0);
	g.chBeg.assign(n + 1, 0);
	for (size_t v = 0; v < n; ++v) {
		g.paBeg[v + 1] = g.paBeg[v] + (u32)parents[v].size();
		for (u32 u : parents[v]) g.chBeg[u + 1]++;
	}
	for (size_t v = 0; v < n; ++v) g.chBeg[v + 1] += g.chBeg[v];
	g.pa.resize(g.paBeg[n]);
	g.ch.resize(g.chBeg[n]);
	g.edgeSrc.resize(g.chBeg[n]);
	std::vector<u32> fill(g.chBeg.begin(), g.chBeg.end() - 1);
	for (size_t v = 0; v < n; ++v) {
		std::copy(parents[v].begin(), parents[v].end(), g.pa.begin() + g.paBeg[v]);
		for (u32 u : parents[v]) {
			g.ch[fill[u]] = (u32)v;
			g.edgeSrc[fill[u]] = u;
			fill[u]++;
		}
	}

	// normalization used by the LPs
	Vec3 lo = g.pts[0], hi = g.pts[0];
	for (auto& p : g.pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
	g.center = Vec3{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
	g.invScale = 1.0 / std::sqrt(sqDist(lo, hi));
	g.npts.resize(n);
	for (size_t i = 0; i < n; ++i)
		g.npts[i] = Vec3{(g.pts[i].x - g.center.x) * g.invScale, (g.pts[i].y - g.center.y) * g.invScale,
		                 (g.pts[i].z - g.center.z) * g.invScale};
	return g;
}

// ---------------------------------------------------------------------------
// Bisector pieces Q(u -> v), see (*) above.
// ---------------------------------------------------------------------------
// Every half-space is relaxed by this much (normalized units: bounding-box
// diagonal == 1).  1e-9 is enough for points in general position, but on exactly
// degenerate inputs (points on a sphere: all bisectors pass through its centre)
// the LPs are degenerate too and double rounding in Seidel's projections can eat
// a 1e-9 margin; 1e-6 removed every such case we found and costs < 0.5% extra
// list entries on ordinary data.
#ifndef DPL_RELAX
#define DPL_RELAX 1e-6
#endif
constexpr double kRelax = DPL_RELAX;

inline void appendBisector(const Dag& g, u32 nearer, u32 farther, std::vector<HalfSpace>& out) {
	// |x - nearer| <= |x - farther|  <=>  n . x <= n . (nearer + farther) / 2,  n = unit(farther - nearer)
	const Vec3 a = g.npts[nearer], b = g.npts[farther];
	double nx = b.x - a.x, ny = b.y - a.y, nz = b.z - a.z;
	const double l = std::sqrt(nx * nx + ny * ny + nz * nz);
	nx /= l; ny /= l; nz /= l;
	out.push_back(HalfSpace{{nx, ny, nz}, 0.5 * (nx * (a.x + b.x) + ny * (a.y + b.y) + nz * (a.z + b.z)) + kRelax});
}

inline void appendPiece(const Dag& g, u32 edge, std::vector<HalfSpace>& out) {
	const u32 u = g.edgeSrc[edge], v = g.ch[edge];
	appendBisector(g, v, u, out);
	for (u32 k = g.paBeg[v]; k < g.paBeg[v + 1]; ++k)
		if (g.pa[k] != u) appendBisector(g, u, g.pa[k], out);
}

// ---------------------------------------------------------------------------
// A query index: states with time-ordered candidate lists.  Entries store the
// candidate's coordinates inline so scanning a list is a sequential read.
// ---------------------------------------------------------------------------
// std::allocator, except that resize() default-initializes (leaves trivially
// constructible elements unwritten): the big index arrays are filled in parallel
// right after resize(), and zeroing them first on one thread cost ~5 s on Lucy.
template <class T>
struct NoInitAlloc : std::allocator<T> {
	template <class U>
	struct rebind {
		using other = NoInitAlloc<U>;
	};
	NoInitAlloc() = default;
	template <class U>
	NoInitAlloc(const NoInitAlloc<U>&) noexcept {}
	template <class U>
	void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
		::new ((void*)p) U;
	}
	template <class U, class... A>
	void construct(U* p, A&&... a) {
		::new ((void*)p) U(std::forward<A>(a)...);
	}
};
template <class T>
using RawVector = std::vector<T, NoInitAlloc<T>>;

struct alignas(32) Entry {
	double x, y, z;
	u32 next;  // state after jumping to this candidate
	u32 node;  // DAG node of the candidate
};
static_assert(sizeof(Entry) == 32, "Entry must be 32 bytes");

struct QueryStats {
	u64 comparisons = 0;  // distance evaluations (the root's own distance excluded)
	u64 jumps = 0;
	u64 exactRechecks = 0;  // FlatIndex16 only: float comparisons too close to call
};

struct LayerIndex {
	std::vector<u32> beg;  // CSR over states (size numStates + 1)
	RawVector<Entry> ent;  // candidate lists
	u32 root = 0;
	Vec3 rootPt{0, 0, 0};
	u32 rootNode = 0;
	int level = 0;  // number of history edges in a state; -1 == whole path (trie)
	// build-time metadata: for each entry, the state that owns it and the DAG edge it jumps along
	RawVector<u32> owner, edge;

	size_t numStates() const { return beg.empty() ? 0 : beg.size() - 1; }
	size_t queryBytes() const { return beg.size() * sizeof(u32) + ent.size() * sizeof(Entry); }

	u32 query(const Vec3& q) const {
		const Entry* E = ent.data();
		const u32* B = beg.data();
		u32 s = root, node = rootNode;
		double best = sqDist(q, rootPt);
		for (;;) {
			u32 i = B[s];
			const u32 end = B[s + 1];
			for (; i < end; ++i) {
				const Entry& e = E[i];
				const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (d < best) {
					best = d;
					s = e.next;
					node = e.node;
					break;
				}
			}
			if (i == end) return node;
		}
	}

	u32 query(const Vec3& q, QueryStats& st) const {
		u32 s = root, node = rootNode;
		double best = sqDist(q, rootPt);
		for (;;) {
			u32 i = beg[s];
			const u32 end = beg[s + 1];
			for (; i < end; ++i) {
				const Entry& e = ent[i];
				st.comparisons++;
				const double d = sqDist(q, Vec3{e.x, e.y, e.z});
				if (d < best) {
					best = d;
					s = e.next;
					node = e.node;
					st.jumps++;
					break;
				}
			}
			if (i == end) return node;
		}
	}
};

inline Entry makeEntry(const Dag& g, u32 node, u32 next) {
	const Vec3& p = g.pts[node];
	return Entry{p.x, p.y, p.z, next, node};
}

// level 0 == the original DPNN query table
inline LayerIndex buildLevel0(const Dag& g) {
	LayerIndex L;
	L.level = 0;
	L.beg = g.chBeg;
	L.ent.resize(g.ch.size());
	L.owner.assign(g.edgeSrc.begin(), g.edgeSrc.end());
	L.edge.resize(g.ch.size());
	for (size_t e = 0; e < g.ch.size(); ++e) {
		L.ent[e] = makeEntry(g, g.ch[e], g.ch[e]);
		L.edge[e] = (u32)e;
	}
	L.root = 0;
	L.rootNode = 0;
	L.rootPt = g.pts[0];
	return L;
}

struct BuildStats {
	u64 lpCalls = 0;
	u64 lpConstraints = 0;
	double seconds = 0;
};

namespace detail {

// History edges (oldest first) represented by state s of levels[lv].
inline void pathOf(const std::vector<LayerIndex>& levels, int lv, u32 s, std::vector<u32>& out) {
	out.clear();
	while (lv > 0) {
		const LayerIndex& prev = levels[lv - 1];  // states of level lv are entries of level lv-1
		if (s == prev.ent.size()) break;          // ... plus the root state
		out.push_back(prev.edge[s]);
		s = prev.owner[s];
		--lv;
	}
	std::reverse(out.begin(), out.end());
}

inline int numThreads() {
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}
// Per-thread scratch of the LP filter.  It is constructed inside the parallel
// region: kept in shared per-thread arrays, the LP's RNG state and the vector
// headers of neighbouring threads share cache lines and are written on every LP
// step, which made 48 threads only ~7x faster than one.
struct FilterScratch {
	LP3 lp;
	std::vector<HalfSpace> hs, wide;
	std::vector<u32> path;
	u64 lpCalls = 0, lpCons = 0;
};

// Called when the path's own half-spaces hs[0..base) are infeasible.  If they
// stay infeasible with every half-space relaxed 100x more, no query can follow
// the path and the state needs no candidates at all.  Such states do exist:
// dpnn.cpp links the first four points completely, and for degenerate (e.g.
// coplanar) inputs some of those edges have empty pieces Q(u -> v).  Keeping
// every candidate there (as for a numerical failure) makes the descendants'
// paths infeasible too, and the trie then grows exponentially.
inline bool pathEmpty(FilterScratch& sc, size_t base) {
	sc.wide.assign(sc.hs.begin(), sc.hs.begin() + base);
	for (HalfSpace& h : sc.wide) h.b += 99 * kRelax;  // unit normals: total relaxation 100 kRelax
	double x[3];
	return !sc.lp.prepare(sc.wide.data(), (int)base, x);
}

// (**) for one candidate: hs[0..base) are the path's half-spaces, already
// solved into x0 by prepare(); hs[base..m) is the candidate's piece, whose first
// half-space is its own bisector |x-w| <= |x-v| (see appendPiece).  That one is
// added first: x0 usually violates it, and projecting only the path's
// half-spaces onto its plane is cheaper (~10% faster than shuffling it in).
inline bool candidateFeasible(LP3& lp, HalfSpace* hs, int base, int m, const double x0[3]) {
	double x[3] = {x0[0], x0[1], x0[2]};
	return lp.extend(hs, base, base + 1, x) && lp.extend(hs, base + 1, m, x);
}

// Filters the candidate lists of states [s0, s1) in parallel.  numCand(s) bounds
// the number of candidates state s may keep; body(s, out, scratch) writes the
// kept ones (positions of the previous level / L1) to out[] and returns their
// number.  The kept lists are appended to keep (in state order) and the offset
// of each one in keep to beg.  Everything but two prefix sums runs in parallel.
template <class NumCand, class Body>
inline void filterStates(size_t s0, size_t s1, NumCand&& numCand, Body&& body, RawVector<u32>& keep,
                         std::vector<u32>& beg, BuildStats* bs) {
	const size_t chunk = 1 << 20;
	std::vector<size_t> slot;  // where state c0 + i may write its kept candidates in buf
	std::vector<u32> cnt;
	RawVector<u32> buf;
	for (size_t c0 = s0; c0 < s1; c0 += chunk) {
		const long long m = (long long)(std::min(s1, c0 + chunk) - c0);
		slot.resize(m + 1);
		cnt.resize(m);
		slot[0] = 0;
#pragma omp parallel for schedule(static)
		for (long long i = 0; i < m; ++i) slot[i + 1] = numCand(c0 + i);
		for (long long i = 0; i < m; ++i) slot[i + 1] += slot[i];
		buf.resize(slot[m]);
		u64 calls = 0, cons = 0;
#pragma omp parallel reduction(+ : calls, cons)
		{
			FilterScratch sc;
#pragma omp for schedule(dynamic, 256)
			for (long long i = 0; i < m; ++i) {
				sc.lp.seed(c0 + i);
				cnt[i] = body(c0 + i, buf.data() + slot[i], sc);
			}
			calls += sc.lpCalls;
			cons += sc.lpCons;
		}
		if (bs) { bs->lpCalls += calls; bs->lpConstraints += cons; }
		const size_t b0 = beg.size();
		beg.resize(b0 + m);
		size_t w = keep.size();
		for (long long i = 0; i < m; ++i) {
			beg[b0 + i] = (u32)w;
			w += cnt[i];
		}
		if (w >= 0xFFFFFFFFull) throw std::runtime_error("index too large for 32-bit offsets");
		keep.resize(w);
#pragma omp parallel for schedule(static)
		for (long long i = 0; i < m; ++i) std::copy_n(buf.data() + slot[i], cnt[i], keep.data() + beg[b0 + i]);
	}
}

}  // namespace detail

// Level k (k >= 1) from levels[0..k-1]: a state is an entry of level k-1
// (== the last <= k edges of the path), plus a root state.
//
// nodeLimit: states whose DAG node is >= nodeLimit get an empty list (they are
// never used by makeHybrid below, which answers that part with level 0).
inline LayerIndex buildNextLevel(const Dag& g, const std::vector<LayerIndex>& levels, BuildStats* bs = nullptr,
                                 u32 nodeLimit = 0xFFFFFFFFu) {
	const int k = (int)levels.size();
	const LayerIndex& P = levels[k - 1];
	LayerIndex L;
	L.level = k;
	const size_t S = P.ent.size() + 1;
	L.root = (u32)P.ent.size();
	L.rootNode = P.rootNode;
	L.rootPt = P.rootPt;

	auto t0 = std::chrono::steady_clock::now();
	auto skipped = [&](size_t s) { return s != L.root && P.ent[s].node >= nodeLimit; };
	// state after the jump (a state of level k-1); its list holds the candidates
	auto tauOf = [&](size_t s) { return s == L.root ? P.root : P.ent[s].next; };
	RawVector<u32> keep;  // kept positions (entries of P), in state order
	L.beg.reserve(S + 1);
	detail::filterStates(
	    0, S, [&](size_t s) -> size_t { return skipped(s) ? 0 : P.beg[tauOf(s) + 1] - P.beg[tauOf(s)]; },
	    [&](size_t s, u32* out, detail::FilterScratch& sc) -> u32 {
		    if (skipped(s)) return 0;
		    const u32 cb = P.beg[tauOf(s)], ce = P.beg[tauOf(s) + 1];
		    u32 n = 0;
		    if (s != L.root) detail::pathOf(levels, k, (u32)s, sc.path);
		    if (s == L.root || (int)sc.path.size() < k) {
			    // the whole history fits in level k-1: its list is already exact for us
			    for (u32 c = cb; c < ce; ++c) out[n++] = c;
			    return n;
		    }
		    std::vector<HalfSpace>& hs = sc.hs;
		    hs.clear();
		    for (u32 e : sc.path) appendPiece(g, e, hs);
		    const size_t base = hs.size();
		    // the path's half-spaces are shared by all candidates: solve them once,
		    // then only add each candidate's piece.  An empty path keeps nothing;
		    // a failure that is not confirmed as empty counts as numerical and
		    // keeps every candidate.
		    double x0[3];
		    const bool pathOk = sc.lp.prepare(hs.data(), (int)base, x0);
		    if (!pathOk && detail::pathEmpty(sc, base)) return 0;
		    for (u32 c = cb; c < ce; ++c) {
			    hs.resize(base);
			    appendPiece(g, P.edge[c], hs);
			    sc.lpCalls++;
			    sc.lpCons += hs.size();
			    if (!pathOk || detail::candidateFeasible(sc.lp, hs.data(), (int)base, (int)hs.size(), x0)) out[n++] = c;
		    }
		    return n;
	    },
	    keep, L.beg, bs);
	L.beg.push_back((u32)keep.size());

	L.ent.resize(keep.size());
	L.owner.resize(keep.size());
	L.edge.resize(keep.size());
#pragma omp parallel for schedule(static, 1 << 16)
	for (long long s = 0; s < (long long)S; ++s)
		for (u32 p = L.beg[s]; p < L.beg[s + 1]; ++p) {
			const u32 c = keep[p];
			L.ent[p] = makeEntry(g, P.ent[c].node, c);  // next state == entry c of level k-1
			L.owner[p] = (u32)s;
			L.edge[p] = P.edge[c];
		}
	if (bs) bs->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	return L;
}

// DAG node reached in each state of levels[k] (state == node for level 0,
// state == entry of level k-1 otherwise).
inline std::vector<u32> stateNodes(const std::vector<LayerIndex>& levels, int k) {
	const LayerIndex& L = levels[k];
	std::vector<u32> sn(L.numStates());
#pragma omp parallel for schedule(static)
	for (long long s = 0; s < (long long)sn.size(); ++s)
		sn[s] = (k == 0) ? (u32)s : (s == L.root ? L.rootNode : levels[k - 1].ent[s].node);
	return sn;
}

// Level k for the coarse part of the hierarchy (DAG nodes < T), level 0 for the
// rest.  Valid because a level-0 list is correct under any history: once a query
// reaches a node >= T it simply forgets its history (and never comes back,
// since nodes are visited in insertion order).  Comparisons are spread evenly
// over log(insertion time), so T = n/16 keeps ~ln(n/16)/ln(n) of the savings
// for a fraction of the memory.  stateNode receives the DAG node of each state.
inline LayerIndex makeHybrid(const std::vector<LayerIndex>& levels, int k, u32 T, std::vector<u32>& stateNode) {
	const LayerIndex& Lk = levels[k];
	const LayerIndex& L0 = levels[0];
	const std::vector<u32> nodeOfK = stateNodes(levels, k);
	const u32 n = (u32)L0.numStates();
	T = std::max<u32>(1, std::min(T, n));
	std::vector<u32> idK(Lk.numStates(), 0xFFFFFFFFu);
	u32 cnt = 0;
	for (u32 s = 0; s < Lk.numStates(); ++s)
		if (nodeOfK[s] < T) idK[s] = cnt++;
	const u32 base0 = cnt;  // node v >= T -> state base0 + (v - T)
	LayerIndex H;
	H.level = k;
	H.root = idK[Lk.root];
	H.rootNode = Lk.rootNode;
	H.rootPt = Lk.rootPt;
	stateNode.assign(cnt + (n - T), 0);
	H.beg.push_back(0);
	for (u32 s = 0; s < Lk.numStates(); ++s) {
		if (idK[s] == 0xFFFFFFFFu) continue;
		stateNode[idK[s]] = nodeOfK[s];
		for (u32 p = Lk.beg[s]; p < Lk.beg[s + 1]; ++p) {
			Entry e = Lk.ent[p];
			e.next = e.node < T ? idK[e.next] : base0 + (e.node - T);
			H.ent.push_back(e);
		}
		H.beg.push_back((u32)H.ent.size());
	}
	for (u32 v = T; v < n; ++v) {
		stateNode[base0 + (v - T)] = v;
		for (u32 p = L0.beg[v]; p < L0.beg[v + 1]; ++p) {
			Entry e = L0.ent[p];
			e.next = base0 + (e.node - T);
			H.ent.push_back(e);
		}
		H.beg.push_back((u32)H.ent.size());
	}
	return H;
}

// All layers: one state per realizable path prefix (a trie).  Candidates of a
// trie node are pre-filtered with the level-1 list of its last edge, then (**)
// is tested on the whole path.  Node ids are assigned in BFS order, so the
// child created by entry i has id i + 1.
inline LayerIndex buildTrie(const Dag& g, const LayerIndex& L1, BuildStats* bs = nullptr, size_t maxNodes = 0,
                            std::vector<u64>* nodesPerDepth = nullptr) {
	LayerIndex Tr;
	Tr.level = -1;
	Tr.root = 0;
	Tr.rootNode = L1.rootNode;
	Tr.rootPt = L1.rootPt;
	std::vector<u32> parent{0}, lastEdge{0}, l1state{L1.root};  // per trie node

	auto t0 = std::chrono::steady_clock::now();

	Tr.beg.assign(1, 0);
	size_t fb = 0, fe = 1;  // current BFS frontier [fb, fe)
	int depth = 0;
	RawVector<u32> keep;     // kept L1 positions of the frontier
	std::vector<u32> kbeg;  // ... and where each frontier node's list starts in keep
	while (fb < fe) {
		if (nodesPerDepth) nodesPerDepth->push_back(fe - fb);
		keep.clear();
		kbeg.clear();
		detail::filterStates(
		    fb, fe, [&](size_t ni) -> size_t { return L1.beg[l1state[ni] + 1] - L1.beg[l1state[ni]]; },
		    [&](size_t ni, u32* out, detail::FilterScratch& sc) -> u32 {
			    const u32 cb = L1.beg[l1state[ni]], ce = L1.beg[l1state[ni] + 1];
			    u32 n = 0;
			    if (depth <= 1) {  // level-1 lists are exact for histories of <= 1 edge
				    for (u32 c = cb; c < ce; ++c) out[n++] = c;
				    return n;
			    }
			    std::vector<HalfSpace>& hs = sc.hs;
			    hs.clear();
			    for (u32 t = (u32)ni; t != 0; t = parent[t]) appendPiece(g, lastEdge[t], hs);
			    const size_t base = hs.size();
			    double x0[3];
			    const bool pathOk = sc.lp.prepare(hs.data(), (int)base, x0);
			    if (!pathOk && detail::pathEmpty(sc, base)) return 0;  // see buildNextLevel
			    for (u32 c = cb; c < ce; ++c) {
				    hs.resize(base);
				    appendPiece(g, L1.edge[c], hs);
				    sc.lpCalls++;
				    sc.lpCons += hs.size();
				    if (!pathOk || detail::candidateFeasible(sc.lp, hs.data(), (int)base, (int)hs.size(), x0)) out[n++] = c;
			    }
			    return n;
		    },
		    keep, kbeg, bs);
		kbeg.push_back((u32)keep.size());
		for (size_t i = 0; i + 1 < kbeg.size(); ++i) {
			for (u32 p = kbeg[i]; p < kbeg[i + 1]; ++p) {
				const u32 c = keep[p];
				const u32 child = (u32)parent.size();
				parent.push_back((u32)(fb + i));
				lastEdge.push_back(L1.edge[c]);
				l1state.push_back(L1.ent[c].next);  // level-1 state == the edge just taken
				Tr.ent.push_back(makeEntry(g, L1.ent[c].node, child));
			}
			Tr.beg.push_back((u32)Tr.ent.size());
		}
		fb = fe;
		fe = parent.size();
		++depth;
		if (maxNodes && parent.size() > maxNodes) throw std::runtime_error("trie exceeds maxNodes");
		if (parent.size() >= 0xFFFFFFFFull) throw std::runtime_error("trie too large for 32-bit ids");
	}
	// per-entry metadata (entry i created trie node i + 1)
	Tr.edge.assign(lastEdge.begin() + 1, lastEdge.end());
	Tr.owner.assign(parent.begin() + 1, parent.end());
	if (bs) bs->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	return Tr;
}

}  // namespace dpl
