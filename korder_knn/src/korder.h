//
// korder.h -- order-k Voronoi diagrams of 3D points as power diagrams, built
// order by order with CGAL's regular triangulation, and a DPNN query DAG on
// them: a k-nearest-neighbour query becomes ONE power-nearest-site query.
//
// For a k-set T with centroid c_T and spread var_T = (1/k) Σ_{t∈T} |t - c_T|^2,
//
//     pow(x, T) := |x - c_T|^2 + var_T  =  (1/k) Σ_{t∈T} |x - t|^2 ,
//
// and the k-set minimizing the right-hand side is the set of the k nearest
// points of x.  So the order-k Voronoi diagram is the power diagram of the
// weighted points (c_T, -var_T) over the k-sets T that own a cell (Aurenhammer
// 1987/1990; the lifting used by Edelsbrunner & Osang for order-k Delaunay mosaics).
//
// Cells of order j+1 from cells of order j (V_j(S) = order-j cell of the j-set S):
//   (a) every order-(j+1) cell is S ∪ S' for two order-j cells sharing a facet
//       (S' = S - s + p).  Proof: take x in V_{j+1}(T), S = its j nearest points,
//       p = the (j+1)-th; walk from x towards p.  The balls B(y, |y-p|) shrink
//       inside B(x, |x-p|), so the points nearer than p stay a subset of S until
//       some s in S ties with p: there V_j(S) and V_j(S - s + p) share a facet.
//   (b) every such union is a cell: on the facet of V_j(S) ∩ V_j(S') the j-th and
//       (j+1)-th neighbours are s and p (tied), all other points are farther, so
//       nearby points have S ∪ S' as their j+1 nearest.
// The regular triangulation of the order-j sites contains every facet adjacency
// as an edge (in 3D the order-j Delaunay mosaic also has octahedra, triangulated
// with one diagonal whose ends differ in two points; those are skipped because
// their union has j+2 points).  Hence: order-(j+1) cells = unions of size j+1
// over the triangulation edges of order j.  No LP and no clipping is needed;
// the regular triangulation itself drops any candidate that is not a cell
// ("hidden" point), which can only happen through rounding.
//
// Query DAG: exactly DPNN, with power distance instead of distance.  Inserting a
// site v changes the power-nearest site only inside v's new cell, which is carved
// out of the cells of v's neighbours in the new triangulation, so the DAG edges
// u -> v (u adjacent to v right after v was inserted, in insertion order) are
// all a query needs.
//
#pragma once
#include "dpnn_flat.h"  // ../DPNN/src: Vec3, u32, PagedArray, QueryStats

#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>

#include <parallel/algorithm>

namespace kov {

using dpl::u32;
using dpl::u64;
using dpl::Vec3;

inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

inline u64 mix64(u64 x) {  // splitmix64 finalizer
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

// The cells of one order-k Voronoi diagram, i.e. the sites of its power diagram.
struct Level {
	int k = 0;
	size_t M = 0;
	std::vector<u32> mem;     // M * k point ids, each k-set sorted ascending
	std::vector<Vec3> c;      // centroid of the k-set
	std::vector<double> var;  // mean squared distance of the members to c
	std::vector<u64> h;       // set hash: sum of mix64(member)

	const u32* members(size_t t) const { return mem.data() + t * (size_t)k; }
	size_t bytes() const { return mem.size() * 4 + c.size() * sizeof(Vec3) + var.size() * 8 + h.size() * 8; }
};

inline Level levelOne(const std::vector<Vec3>& P) {
	Level L;
	L.k = 1;
	L.M = P.size();
	L.mem.resize(L.M);
	L.h.resize(L.M);
	for (size_t i = 0; i < L.M; ++i) { L.mem[i] = (u32)i; L.h[i] = mix64(i); }
	L.c = P;
	L.var.assign(L.M, 0.0);
	return L;
}

namespace rt {
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_vertex_base_3<K> Vb0;
typedef CGAL::Triangulation_vertex_base_with_info_3<u32, K, Vb0> Vb;
typedef CGAL::Regular_triangulation_cell_base_3<K, CGAL::Triangulation_cell_base_3<K>, CGAL::Discard_hidden_points> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Regular_triangulation_3<K, Tds> RT;
typedef K::Weighted_point_3 WP;
typedef K::Point_3 P3;
}  // namespace rt

// Regular triangulation of one level, built by incremental insertion in BRIO
// order (CGAL::spatial_sort, as in dpnn.cpp).  Node i of the DAG is the i-th
// inserted site; order[i] is its cell id.
struct PowerTri {
	std::vector<u32> order;
	std::vector<u32> paBeg, pa;               // Parents(i) as node ids (only if requested)
	std::vector<std::pair<u32, u32>> edges;   // final adjacency as cell ids (only if requested)
	size_t hiddenAtInsert = 0;                // sites that never owned a cell (rounding only)
	size_t removedLater = 0;                  // vertices hidden by a later insertion (rounding only)
	double seconds = 0;
};

inline PowerTri triangulate(const Level& L, bool wantDag, bool wantEdges) {
	using namespace rt;
	const double t0 = now();
	PowerTri T;
	const size_t M = L.M;
	if (M >= 0xFFFFFFFFull) throw std::runtime_error("too many sites for 32-bit ids");
	{
		typedef std::pair<P3, u32> PI;
		typedef CGAL::Spatial_sort_traits_adapter_3<K, CGAL::First_of_pair_property_map<PI>> Traits;
		std::vector<PI> tmp;
		tmp.reserve(M);
		for (size_t t = 0; t < M; ++t) tmp.emplace_back(P3(L.c[t].x, L.c[t].y, L.c[t].z), (u32)t);
		CGAL::spatial_sort(tmp.begin(), tmp.end(), Traits());
		T.order.resize(M);
		for (size_t i = 0; i < M; ++i) T.order[i] = tmp[i].second;
	}
	// weights are only defined up to a common constant: keep them >= 0
	double W0 = 0;
	for (double v : L.var) W0 = std::max(W0, v);

	RT tri;
	RT::Vertex_handle hint;
	std::vector<u32> alive;  // inserted, non-hidden nodes (for the first four)
	std::vector<RT::Vertex_handle> nb;
	std::vector<RT::Cell_handle> conflict;
	std::vector<RT::Facet> boundary;
	std::vector<u32> tmpPa;
	if (wantDag) {
		T.paBeg.reserve(M + 1);
		T.paBeg.push_back(0);
		T.pa.reserve(M * 16);
	}
	for (size_t i = 0; i < M; ++i) {
		const u32 t = T.order[i];
		const WP wp(P3(L.c[t].x, L.c[t].y, L.c[t].z), W0 - L.var[t]);
		const size_t before = tri.number_of_vertices();
		RT::Locate_type lt;
		int li, lj;
		const RT::Cell_handle loc = tri.locate(wp, lt, li, lj, hint == RT::Vertex_handle() ? RT::Cell_handle() : hint->cell());
		RT::Vertex_handle v;
		tmpPa.clear();
		if (lt == RT::VERTEX && loc->vertex(li)->point() == wp) {
			// the same weighted point is already there
		} else if (tri.dimension() == 3) {
			// Parents(v) = all vertices of the conflict zone: the boundary ones become v's
			// neighbours, the interior ones get hidden by v (their whole cell goes to v; with
			// exact sites that cannot happen, with rounded centroids it can)
			conflict.clear();
			boundary.clear();
			tri.find_conflicts(wp, loc, std::back_inserter(boundary), std::back_inserter(conflict));
			if (!conflict.empty()) {
				if (wantDag)
					for (const RT::Cell_handle& c : conflict)
						for (int x = 0; x < 4; ++x)
							if (!tri.is_infinite(c->vertex(x))) tmpPa.push_back(c->vertex(x)->info());
				v = tri.insert_in_hole(wp, conflict.begin(), conflict.end(), boundary.back().first, boundary.back().second);
			}
		} else {
			v = tri.insert(wp, lt, loc, li, lj);
			if (v != RT::Vertex_handle() && wantDag) {
				if (alive.size() < 4) {
					tmpPa = alive;  // dpnn.cpp: the first four points form a complete graph
				} else {
					nb.clear();
					tri.finite_adjacent_vertices(v, std::back_inserter(nb));
					for (auto& w : nb) tmpPa.push_back(w->info());
				}
			}
		}
		if (v == RT::Vertex_handle()) {
			// hidden: never a power-nearest site
			T.hiddenAtInsert++;
			if (wantDag) T.paBeg.push_back((u32)T.pa.size());
			continue;
		}
		const size_t after = tri.number_of_vertices();
		if (after < before + 1) T.removedLater += before + 1 - after;
		v->info() = (u32)i;
		hint = v;
		if (wantDag) {
			std::sort(tmpPa.begin(), tmpPa.end());
			tmpPa.erase(std::unique(tmpPa.begin(), tmpPa.end()), tmpPa.end());
			T.pa.insert(T.pa.end(), tmpPa.begin(), tmpPa.end());
			T.paBeg.push_back((u32)T.pa.size());
		}
		if (alive.size() < 4) alive.push_back((u32)i);
	}
	if (wantEdges) {
		// the six edges of every finite cell, deduplicated (CGAL's finite_edges iterator
		// walks around every edge and took as long as the insertion itself)
		std::vector<u64> pairs;
		if (tri.dimension() == 3) {
			pairs.reserve(tri.number_of_finite_cells() * 6);
			for (auto c = tri.finite_cells_begin(); c != tri.finite_cells_end(); ++c) {
				u32 id[4];
				for (int x = 0; x < 4; ++x) id[x] = c->vertex(x)->info();
				for (int x = 0; x < 4; ++x)
					for (int y = x + 1; y < 4; ++y)
						pairs.push_back(id[x] < id[y] ? (u64)id[x] << 32 | id[y] : (u64)id[y] << 32 | id[x]);
			}
		} else {
			for (auto e = tri.finite_edges_begin(); e != tri.finite_edges_end(); ++e) {
				const u32 a = e->first->vertex(e->second)->info(), b = e->first->vertex(e->third)->info();
				pairs.push_back(a < b ? (u64)a << 32 | b : (u64)b << 32 | a);
			}
		}
		__gnu_parallel::sort(pairs.begin(), pairs.end());
		pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
		T.edges.resize(pairs.size());
#pragma omp parallel for schedule(static, 65536)
		for (long long e = 0; e < (long long)pairs.size(); ++e)
			T.edges[e] = {T.order[pairs[e] >> 32], T.order[pairs[e] & 0xFFFFFFFFu]};
	}
	T.seconds = now() - t0;
	return T;
}

#ifdef CGAL_LINKED_WITH_TBB
// Edges only (no DAG): CGAL's parallel regular triangulation (TBB), ~10x faster
// with 48 threads.  Used for the orders that are only a stepping stone.
inline PowerTri triangulateParallel(const Level& L) {
	typedef CGAL::Triangulation_data_structure_3<rt::Vb, rt::Cb, CGAL::Parallel_tag> TdsP;
	typedef CGAL::Regular_triangulation_3<rt::K, TdsP, CGAL::Spatial_lock_grid_3<CGAL::Tag_priority_blocking>> RTP;
	const double t0 = now();
	PowerTri T;
	const size_t M = L.M;
	double W0 = 0;
	for (double v : L.var) W0 = std::max(W0, v);
	std::vector<std::pair<rt::WP, u32>> wp(M);
	Vec3 lo = L.c[0], hi = L.c[0];
	for (size_t t = 0; t < M; ++t) {
		const Vec3& c = L.c[t];
		lo.x = std::min(lo.x, c.x); lo.y = std::min(lo.y, c.y); lo.z = std::min(lo.z, c.z);
		hi.x = std::max(hi.x, c.x); hi.y = std::max(hi.y, c.y); hi.z = std::max(hi.z, c.z);
		wp[t] = {rt::WP(rt::P3(c.x, c.y, c.z), W0 - L.var[t]), (u32)t};
	}
	const double pad = 1e-3 * std::sqrt(dpl::sqDist(lo, hi)) + 1e-12;
	typename RTP::Lock_data_structure lock(CGAL::Bbox_3(lo.x - pad, lo.y - pad, lo.z - pad, hi.x + pad, hi.y + pad, hi.z + pad), 50);
	RTP tri(&lock);
	tri.insert(wp.begin(), wp.end());
	std::vector<std::pair<rt::WP, u32>>().swap(wp);
	T.hiddenAtInsert = M - tri.number_of_vertices();
	std::vector<u64> pairs;
	pairs.reserve(tri.number_of_finite_cells() * 6);
	for (auto c = tri.finite_cells_begin(); c != tri.finite_cells_end(); ++c) {
		u32 id[4];
		for (int x = 0; x < 4; ++x) id[x] = c->vertex(x)->info();
		for (int x = 0; x < 4; ++x)
			for (int y = x + 1; y < 4; ++y)
				pairs.push_back(id[x] < id[y] ? (u64)id[x] << 32 | id[y] : (u64)id[y] << 32 | id[x]);
	}
	__gnu_parallel::sort(pairs.begin(), pairs.end());
	pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
	T.edges.resize(pairs.size());
#pragma omp parallel for schedule(static, 65536)
	for (long long e = 0; e < (long long)pairs.size(); ++e) T.edges[e] = {u32(pairs[e] >> 32), u32(pairs[e] & 0xFFFFFFFFu)};
	T.seconds = now() - t0;
	return T;
}
#endif

// Order k+1 from order k: unions of size k+1 over the triangulation edges, deduplicated.
struct NextStats {
	size_t candidates = 0;  // edges whose union has k+1 points (before deduplication)
	double seconds = 0;
};

inline Level nextLevel(const Level& L, const std::vector<std::pair<u32, u32>>& edges, NextStats* st = nullptr) {
	const double t0 = now();
	const int k = L.k;
	struct Cand {
		u64 h;
		u32 a, p;  // the set is members(a) ∪ {p}
	};
	const int T = dpl::detail::numThreads();
	std::vector<std::vector<Cand>> per(T);
#pragma omp parallel num_threads(T)
	{
#ifdef _OPENMP
		std::vector<Cand>& out = per[omp_get_thread_num()];
#else
		std::vector<Cand>& out = per[0];
#endif
#pragma omp for schedule(static, 1 << 14)
		for (long long e = 0; e < (long long)edges.size(); ++e) {
			const u32 a = edges[e].first, b = edges[e].second;
			const u32 *A = L.members(a), *B = L.members(b);
			int i = 0, j = 0, onlyA = 0, onlyB = 0;
			u32 p = 0;
			while (i < k && j < k && onlyA <= 1 && onlyB <= 1) {
				if (A[i] == B[j]) { ++i; ++j; }
				else if (A[i] < B[j]) { ++onlyA; ++i; }
				else { ++onlyB; p = B[j]; ++j; }
			}
			if (onlyA > 1 || onlyB > 1) continue;
			onlyA += k - i;
			if (j < k) { onlyB += k - j; p = B[j]; }
			if (onlyA == 1 && onlyB == 1) out.push_back(Cand{L.h[a] + mix64(p), a, p});
		}
	}
	size_t total = 0;
	for (auto& v : per) total += v.size();
	std::vector<Cand> all;
	all.reserve(total);
	for (auto& v : per) {
		all.insert(all.end(), v.begin(), v.end());
		std::vector<Cand>().swap(v);
	}
	__gnu_parallel::sort(all.begin(), all.end(), [](const Cand& x, const Cand& y) {
		if (x.h != y.h) return x.h < y.h;
		if (x.a != y.a) return x.a < y.a;
		return x.p < y.p;
	});
	// the sorted member list of a candidate
	auto setOf = [&](const Cand& c, u32* out) {
		const u32* A = L.members(c.a);
		int i = 0, w = 0;
		while (i < k && A[i] < c.p) out[w++] = A[i++];
		out[w++] = c.p;
		while (i < k) out[w++] = A[i++];
	};
	// keep the first candidate of every distinct set (equal hash == equal set unless a
	// 64-bit collision, which is checked)
	std::vector<u32> keep;
	keep.reserve(all.size() / 2 + 1);
	{
		std::vector<u32> s1(k + 1), s2(k + 1);
		size_t r = 0;
		while (r < all.size()) {
			size_t e = r + 1;
			while (e < all.size() && all[e].h == all[r].h) ++e;
			const size_t first = keep.size();
			for (size_t x = r; x < e; ++x) {
				bool dup = false;
				if (x > r) {
					setOf(all[x], s1.data());
					for (size_t y = first; y < keep.size() && !dup; ++y) {
						setOf(all[keep[y]], s2.data());
						dup = s1 == s2;
					}
				}
				if (!dup) keep.push_back((u32)x);
			}
			r = e;
		}
	}
	Level N;
	N.k = k + 1;
	N.M = keep.size();
	N.mem.resize(N.M * (size_t)N.k);
	N.c.resize(N.M);
	N.var.resize(N.M);
	N.h.resize(N.M);
#pragma omp parallel for schedule(static, 4096) num_threads(T)
	for (long long t = 0; t < (long long)N.M; ++t) {
		const Cand& cd = all[keep[t]];
		setOf(cd, N.mem.data() + t * (size_t)N.k);
		N.h[t] = cd.h;
	}
	if (st) {
		st->candidates = all.size();
		st->seconds = now() - t0;
	}
	return N;  // sites (c, var) still to be filled by computeSites()
}

// Fills centroids and spreads of a level from the point coordinates (two-pass, so
// var does not suffer from cancellation).
inline void computeSites(Level& L, const std::vector<Vec3>& P) {
	const int k = L.k;
#pragma omp parallel for schedule(static, 4096)
	for (long long t = 0; t < (long long)L.M; ++t) {
		const u32* m = L.members(t);
		double sx = 0, sy = 0, sz = 0;
		for (int i = 0; i < k; ++i) { sx += P[m[i]].x; sy += P[m[i]].y; sz += P[m[i]].z; }
		const Vec3 c{sx / k, sy / k, sz / k};
		double v = 0;
		for (int i = 0; i < k; ++i) v += dpl::sqDist(P[m[i]], c);
		L.c[t] = c;
		L.var[t] = v / k;
	}
}

// ---------------------------------------------------------------------------
// Exact candidate lists: Cx(u) = union of the order-k cells T that meet the
// (order-1) Voronoi cell of u, i.e. the k-sets of all queries whose nearest point
// is u.  Needs the facets of every cell: V_k(T) = ∩ { |x-t| <= |x-p| : t in T,
// p in N(T) }, N(T) = the points that enter T across a facet, read off the
// triangulation edges (T, T - s + p).  One 3D LP per (T, member u):
//   V_k(T) ∩ { |x-u| <= |x-t| : t in T }  non-empty?
// (half-spaces relaxed by 1e-9 of the bounding-box diagonal, so rounding can only
// add candidates, never drop one).
// ---------------------------------------------------------------------------
struct CandLists {
	std::vector<u32> beg, list;  // CSR, every list sorted and containing u itself
	u64 lps = 0;
	double seconds = 0;
};

inline CandLists exactCandidates(const std::vector<Vec3>& P, const Level& L, const std::vector<std::pair<u32, u32>>& edges) {
	const double t0 = now();
	const int k = L.k;
	const size_t M = L.M, n = P.size();
	// outer facet neighbours N(T)
	std::vector<u64> nb;  // (cell << 32) | p
	nb.reserve(edges.size() * 2);
	for (const auto& e : edges) {
		const u32 *A = L.members(e.first), *B = L.members(e.second);
		int i = 0, j = 0, onlyA = 0, onlyB = 0;
		u32 pa = 0, pb = 0;
		while (i < k && j < k && onlyA <= 1 && onlyB <= 1) {
			if (A[i] == B[j]) { ++i; ++j; }
			else if (A[i] < B[j]) { ++onlyA; pa = A[i]; ++i; }
			else { ++onlyB; pb = B[j]; ++j; }
		}
		if (onlyA > 1 || onlyB > 1) continue;
		if (i < k) { onlyA += k - i; pa = A[i]; }
		if (j < k) { onlyB += k - j; pb = B[j]; }
		if (onlyA != 1 || onlyB != 1) continue;
		nb.push_back((u64)e.first << 32 | pb);
		nb.push_back((u64)e.second << 32 | pa);
	}
	__gnu_parallel::sort(nb.begin(), nb.end());
	nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
	std::vector<u64> nbBeg(M + 1, 0);
	for (u64 x : nb) nbBeg[(x >> 32) + 1]++;
	for (size_t t = 0; t < M; ++t) nbBeg[t + 1] += nbBeg[t];
	// LPs in normalized coordinates (bounding-box diagonal 1)
	Vec3 lo = P[0], hi = P[0];
	for (auto& p : P) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
	const Vec3 ctr{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
	const double inv = 1.0 / std::sqrt(dpl::sqDist(lo, hi));
	std::vector<Vec3> N(n);
	for (size_t i = 0; i < n; ++i) N[i] = Vec3{(P[i].x - ctr.x) * inv, (P[i].y - ctr.y) * inv, (P[i].z - ctr.z) * inv};
	auto bis = [&](u32 a, u32 b) {  // |x - a| <= |x - b|, unit normal, relaxed
		const Vec3 &A = N[a], &B = N[b];
		double nx = B.x - A.x, ny = B.y - A.y, nz = B.z - A.z;
		const double l = std::sqrt(nx * nx + ny * ny + nz * nz);
		nx /= l; ny /= l; nz /= l;
		return dpl::HalfSpace{{nx, ny, nz}, 0.5 * (nx * (A.x + B.x) + ny * (A.y + B.y) + nz * (A.z + B.z)) + 1e-9};
	};
	// owner pairs (u, cell): the cell meets Vor(u)
	const int T = dpl::detail::numThreads();
	std::vector<std::vector<u64>> per(T);
	u64 lps = 0;
#pragma omp parallel num_threads(T) reduction(+ : lps)
	{
#ifdef _OPENMP
		std::vector<u64>& out = per[omp_get_thread_num()];
#else
		std::vector<u64>& out = per[0];
#endif
		dpl::LP3 lp;
		std::vector<dpl::HalfSpace> hs;
#pragma omp for schedule(dynamic, 1024)
		for (long long t = 0; t < (long long)M; ++t) {
			const u32* m = L.members(t);
			hs.clear();
			for (u64 x = nbBeg[t]; x < nbBeg[t + 1]; ++x)
				for (int i = 0; i < k; ++i) hs.push_back(bis(m[i], u32(nb[x] & 0xFFFFFFFFu)));
			const size_t base = hs.size();
			for (int a = 0; a < k; ++a) {
				hs.resize(base);
				for (int i = 0; i < k; ++i)
					if (i != a) hs.push_back(bis(m[a], m[i]));
				lp.seed((u64)t * 64 + a);
				++lps;
				if (lp.feasible(hs.data(), (int)hs.size())) out.push_back((u64)m[a] << 32 | (u64)t);
			}
		}
	}
	std::vector<u64> own;
	for (auto& v : per) {
		own.insert(own.end(), v.begin(), v.end());
		std::vector<u64>().swap(v);
	}
	// expand to (u, p) pairs
	std::vector<u64> pr;
	pr.reserve(own.size() * k + n);
	for (size_t u = 0; u < n; ++u) pr.push_back((u64)u << 32 | u);
	for (u64 x : own) {
		const u64 u = x >> 32;
		const u32* m = L.members(x & 0xFFFFFFFFu);
		for (int i = 0; i < k; ++i) pr.push_back(u << 32 | m[i]);
	}
	std::vector<u64>().swap(own);
	__gnu_parallel::sort(pr.begin(), pr.end());
	pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
	CandLists C;
	C.beg.assign(n + 1, 0);
	for (u64 x : pr) C.beg[(x >> 32) + 1]++;
	for (size_t u = 0; u < n; ++u) C.beg[u + 1] += C.beg[u];
	C.list.resize(pr.size());
	for (size_t i = 0; i < pr.size(); ++i) C.list[i] = u32(pr[i] & 0xFFFFFFFFu);
	C.lps = lps;
	C.seconds = now() - t0;
	return C;
}

// ---------------------------------------------------------------------------
// The query structure: DPNN's flat layout with power distance.  Each node's
// children (in insertion order) are consecutive 40-byte entries holding the
// child's weighted site and the offset of the child's own list; a terminator
// carries the cell id to return.
// ---------------------------------------------------------------------------
struct PowerEntry {
	double x, y, z, v;  // centroid and spread of the child's k-set
	u32 next;           // offset of the child's list, kEnd for the terminator
	u32 cell;           // terminator: the cell (k-set) to return
};
static_assert(sizeof(PowerEntry) == 40, "PowerEntry must be 40 bytes");

class PowerDpnn {
public:
	static constexpr u32 kEnd = 0xFFFFFFFFu;

	PowerDpnn(const Level& L, const PowerTri& T, bool hugePages = true) : k_(L.k) {
		const size_t M = T.order.size();
		if (T.paBeg.size() != M + 1) throw std::runtime_error("PowerDpnn: triangulation built without DAG");
		// children = transpose of Parents (ascending == insertion order)
		std::vector<u32> chBeg(M + 1, 0);
		for (size_t v = 0; v < M; ++v)
			for (u32 p = T.paBeg[v]; p < T.paBeg[v + 1]; ++p) chBeg[T.pa[p] + 1]++;
		for (size_t u = 0; u < M; ++u) chBeg[u + 1] += chBeg[u];
		std::vector<u32> ch(chBeg[M]);
		{
			std::vector<u32> fill(chBeg.begin(), chBeg.end() - 1);
			for (size_t v = 0; v < M; ++v)
				for (u32 p = T.paBeg[v]; p < T.paBeg[v + 1]; ++p) ch[fill[T.pa[p]]++] = (u32)v;
		}
		edges_ = ch.size();
		if (ch.size() + M >= kEnd) throw std::runtime_error("PowerDpnn: too many entries for 32-bit offsets");
		std::vector<u32> start(M);
		for (size_t u = 0; u < M; ++u) start[u] = chBeg[u] + (u32)u;
		PowerEntry* e = ent_.allocate(ch.size() + M, hugePages);
#pragma omp parallel for schedule(static, 4096)
		for (long long u = 0; u < (long long)M; ++u) {
			u32 w = start[u];
			for (u32 p = chBeg[u]; p < chBeg[u + 1]; ++p) {
				const u32 v = ch[p], cell = T.order[v];
				e[w++] = PowerEntry{L.c[cell].x, L.c[cell].y, L.c[cell].z, L.var[cell], start[v], cell};
			}
			e[w] = PowerEntry{0, 0, 0, 0, kEnd, T.order[u]};
		}
		const u32 r = T.order[0];
		root_ = PowerEntry{L.c[r].x, L.c[r].y, L.c[r].z, L.var[r], start[0], r};
	}

	size_t bytes() const { return ent_.bytes(); }
	size_t numEdges() const { return edges_; }
	int k() const { return k_; }

	static double pw(const Vec3& q, const PowerEntry& e) {
		const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
		return dx * dx + dy * dy + dz * dz + e.v;
	}

	// cell id of the power-nearest site == the k-set of the k nearest points of q
	u32 query(const Vec3& q) const {
		const PowerEntry* B = ent_.data();
		u32 s = root_.next;
		double best = pw(q, root_);
		for (;;) {
			const PowerEntry* e = B + s;
			for (;; ++e) {
				if (e->next == kEnd) return e->cell;
				const double d = pw(q, *e);
				if (d < best) {
					best = d;
					s = e->next;
					break;
				}
			}
		}
	}

	u32 query(const Vec3& q, dpl::QueryStats& st) const {
		const PowerEntry* B = ent_.data();
		u32 s = root_.next;
		double best = pw(q, root_);
		for (;;) {
			const PowerEntry* e = B + s;
			for (;; ++e) {
				if (e->next == kEnd) return e->cell;
				st.comparisons++;
				const double d = pw(q, *e);
				if (d < best) {
					best = d;
					s = e->next;
					st.jumps++;
					break;
				}
			}
		}
	}

	// Q[0..m) -> out[0..m) (cell ids), Kq queries interleaved (see dpnn_flat.h)
	template <int Kq = 16>
	void queryBatch(const Vec3* Q, size_t m, u32* out) const {
		const PowerEntry* B = ent_.data();
		struct Slot {
			Vec3 q;
			double best;
			u32 s;
			size_t idx;
		};
		Slot sl[Kq];
		int active = 0;
		size_t nq = 0;
		for (; active < Kq && nq < m; ++active, ++nq) sl[active] = Slot{Q[nq], pw(Q[nq], root_), root_.next, nq};
		while (active) {
			for (int a = 0; a < active;) {
				Slot& S = sl[a];
				const PowerEntry* e = B + S.s;
				bool jumped = false;
				for (;; ++e) {
					if (e->next == kEnd) break;
					const double d = pw(S.q, *e);
					if (d < S.best) {
						S.best = d;
						S.s = e->next;
						jumped = true;
						break;
					}
				}
				if (jumped) {
					__builtin_prefetch(B + S.s);
					__builtin_prefetch(B + S.s + 2);
					++a;
					continue;
				}
				out[S.idx] = e->cell;
				if (nq < m) {
					S = Slot{Q[nq], pw(Q[nq], root_), root_.next, nq};
					++nq;
					++a;
				} else {
					sl[a] = sl[--active];
				}
			}
		}
	}

private:
	int k_;
	dpl::PagedArray<PowerEntry> ent_;
	PowerEntry root_{};
	size_t edges_ = 0;
};

}  // namespace kov
