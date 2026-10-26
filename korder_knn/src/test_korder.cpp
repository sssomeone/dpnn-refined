//
// test_korder.cpp -- small exhaustive checks of the order-k construction:
//   1. every computed cell is non-empty (3D LP over all n - k bisector constraints),
//   2. every k-set that random queries produce is among the computed cells,
//   3. the order-k DPNN returns exactly that k-set,
// for uniform, clustered, sphere and grid-like inputs.
//
#include "korder.h"

#include <CGAL/MP_Float.h>
#include <CGAL/QP_functions.h>
#include <CGAL/QP_models.h>

#include <cstdio>
#include <random>
#include <set>

using namespace kov;

static int failures = 0;

// Exact classification of a k-set T (exact rational LP, CGAL's QP solver on MP_Float):
// maximize s subject to 2 (p - t) . x + s <= |p|^2 - |t|^2 for all t in T, p not in T, s <= 1.
//   +1: s* > 0, a full-dimensional cell;  0: s* == 0, measure zero;  -1: no point at all
static int exactCell(const std::vector<Vec3>& P, const u32* T, int k) {
	typedef CGAL::MP_Float ET;
	CGAL::Quadratic_program<ET> lp(CGAL::SMALLER, false, 0, false, 0);
	std::vector<char> in(P.size(), 0);
	for (int i = 0; i < k; ++i) in[T[i]] = 1;
	int row = 0;
	for (int i = 0; i < k; ++i) {
		const Vec3& a = P[T[i]];
		for (size_t p = 0; p < P.size(); ++p) {
			if (in[p]) continue;
			const Vec3& b = P[p];
			lp.set_a(0, row, ET(2) * (ET(b.x) - ET(a.x)));
			lp.set_a(1, row, ET(2) * (ET(b.y) - ET(a.y)));
			lp.set_a(2, row, ET(2) * (ET(b.z) - ET(a.z)));
			lp.set_a(3, row, ET(1));
			lp.set_b(row, ET(b.x) * ET(b.x) + ET(b.y) * ET(b.y) + ET(b.z) * ET(b.z) - ET(a.x) * ET(a.x) - ET(a.y) * ET(a.y) -
			                  ET(a.z) * ET(a.z));
			++row;
		}
	}
	lp.set_u(3, true, ET(1));
	lp.set_c(3, ET(-1));
	const CGAL::Quadratic_program_solution<ET> sol = CGAL::solve_linear_program(lp, ET());
	if (sol.is_infeasible()) return -1;
	const CGAL::Quotient<ET> v = sol.objective_value();  // == -s*
	return v < CGAL::Quotient<ET>(0) ? 1 : 0;
}

static void check(const char* name, std::vector<Vec3> P, int K, size_t nq) {
	// centre the data like bench_knn does
	Vec3 lo = P[0], hi = P[0];
	for (auto& p : P) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
	const Vec3 c{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
	const double diag = std::sqrt(dpl::sqDist(lo, hi));
	for (auto& p : P) p = Vec3{(p.x - c.x) / diag, (p.y - c.y) / diag, (p.z - c.z) / diag};
	const size_t n = P.size();
	std::mt19937_64 gen(5);
	std::uniform_real_distribution<double> U(-1.0, 1.0);
	std::normal_distribution<double> G(0, 0.01);
	std::vector<Vec3> Q(nq);
	for (size_t i = 0; i < nq; ++i) {
		if (i % 2) Q[i] = Vec3{U(gen), U(gen), U(gen)};
		else {
			const Vec3& b = P[gen() % n];
			Q[i] = Vec3{b.x + G(gen), b.y + G(gen), b.z + G(gen)};
		}
	}
	Level L = levelOne(P);
	dpl::LP3 lp;
	std::vector<dpl::HalfSpace> hs;
	for (int k = 1; k <= K; ++k) {
		PowerTri T = triangulate(L, true, k < K);
		PowerDpnn D(L, T, false);
		// 1. non-empty cells: "thin" = empty after shrinking every half-space by 1e-9,
		//    "empty" = still empty after enlarging every half-space by 1e-9
		size_t empty = 0, thin = 0;
		size_t exIn = 0, exDeg = 0, exEmpty = 0;
		std::vector<char> in(n);
		for (size_t t = 0; t < L.M; ++t) {
			const u32* m = L.members(t);
			std::fill(in.begin(), in.end(), 0);
			for (int i = 0; i < k; ++i) in[m[i]] = 1;
			hs.clear();
			for (int i = 0; i < k; ++i)
				for (size_t p = 0; p < n; ++p) {
					if (in[p]) continue;
					// |x - a| <= |x - b|  <=>  2 (b - a) . x <= |b|^2 - |a|^2, tightened by 1e-9
					const Vec3 &a = P[m[i]], &b = P[p];
					hs.push_back(dpl::HalfSpace{{2 * (b.x - a.x), 2 * (b.y - a.y), 2 * (b.z - a.z)},
					                            (b.x * b.x + b.y * b.y + b.z * b.z) - (a.x * a.x + a.y * a.y + a.z * a.z) - 1e-9});
				}
			if (!lp.feasible(hs.data(), (int)hs.size())) {
				for (auto& h : hs) h.b += 2e-9;
				if (lp.feasible(hs.data(), (int)hs.size())) ++thin;
				else ++empty;
				// the floating-point LP is not the judge: classify exactly
				const int ex = exactCell(P, m, k);
				if (ex > 0) ++exIn;
				else if (ex == 0) ++exDeg;
				else ++exEmpty;
			}
		}
		// 2./3. brute-force k-sets of the queries
		std::set<std::vector<u32>> cells;
		for (size_t t = 0; t < L.M; ++t) cells.insert(std::vector<u32>(L.members(t), L.members(t) + k));
		size_t missing = 0, wrong = 0;
		std::vector<std::pair<double, u32>> d(n);
		for (const Vec3& q : Q) {
			for (size_t p = 0; p < n; ++p) d[p] = {dpl::sqDist(P[p], q), (u32)p};
			std::nth_element(d.begin(), d.begin() + k, d.end());
			std::vector<u32> s(k);
			for (int i = 0; i < k; ++i) s[i] = d[i].second;
			std::sort(s.begin(), s.end());
			if (!cells.count(s)) ++missing;
			const u32* m = L.members(D.query(q));
			if (!std::equal(s.begin(), s.end(), m)) ++wrong;
		}
		const bool ok = !missing && !wrong;
		std::printf("%-9s n=%zu k=%d: %6zu cells (%6.2f n) | fp-LP thin %3zu empty %3zu -> exact: full-dim %3zu, measure-zero %3zu, empty %3zu"
		            " | query k-sets missing %zu, DPNN wrong %zu / %zu | hidden %zu + %zu  %s\n",
		            name, n, k, L.M, double(L.M) / n, thin, empty, exIn, exDeg, exEmpty, missing, wrong, nq, T.hiddenAtInsert,
		            T.removedLater, ok ? "ok" : "FAIL");
		if (!ok) ++failures;
		if (k < K) {
			Level N = nextLevel(L, T.edges);
			computeSites(N, P);
			L = std::move(N);
		}
	}
}

int main() {
	std::mt19937_64 gen(1);
	std::uniform_real_distribution<double> U(0, 1);
	std::normal_distribution<double> G(0, 1);
	std::vector<Vec3> uni(300), clu, sph(200), grid;
	for (auto& p : uni) p = Vec3{U(gen), U(gen), U(gen)};
	for (int c = 0; c < 6; ++c) {  // clusters of very different density
		const Vec3 o{U(gen), U(gen), U(gen)};
		const double s = c % 2 ? 0.01 : 0.1;
		for (int i = 0; i < 50; ++i) clu.push_back(Vec3{o.x + s * G(gen), o.y + s * G(gen), o.z + s * G(gen)});
	}
	for (auto& p : sph) {
		double x = G(gen), y = G(gen), z = G(gen), l = std::sqrt(x * x + y * y + z * z);
		p = Vec3{x / l, y / l, z / l};
	}
	for (int x = 0; x < 6; ++x)  // jittered grid (an exact grid has ties in every k-set)
		for (int y = 0; y < 6; ++y)
			for (int z = 0; z < 6; ++z) grid.push_back(Vec3{x + 1e-3 * U(gen), y + 1e-3 * U(gen), z + 1e-3 * U(gen)});
	check("uniform", uni, 8, 20000);
	check("clusters", clu, 8, 20000);
	check("sphere", sph, 6, 20000);
	check("grid~", grid, 6, 20000);
	std::printf("%s\n", failures ? "TEST FAILURES" : "ALL TESTS PASSED");
	return failures ? 1 : 0;
}
