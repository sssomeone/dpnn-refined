//
// lp3.h -- tiny 3D linear feasibility solver (Seidel's randomized incremental LP).
//
// Used by dpnn_layered.h to decide whether an intersection of bisector
// half-spaces {x : a_i . x <= b_i} is non-empty.  No polytope is ever built or
// clipped: the solver only keeps one witness point and recurses onto a plane
// (2D) / line (1D) when that point violates a new constraint.
//
// Expected time O(m) for m constraints (d = 3).
//
// Conservativeness: every tolerance in here can only turn "infeasible" into
// "feasible", never the other way round.  Callers additionally relax every
// constraint by a small epsilon, so a non-empty region is never reported empty
// because of rounding.  (A false "feasible" only costs one extra distance
// comparison at query time; a false "infeasible" would break exactness.)
//
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace dpl {

struct HalfSpace {
	double a[3];
	double b;  // a . x <= b
};

class LP3 {
public:
	// Every variable is also boxed into [-M, M].  The box only keeps the LP
	// bounded; dpnn_layered.h works in normalized coordinates (data bounding-box
	// diagonal == 1), so M = 1e3 covers queries up to ~1000 diagonals away.
	double M = 1e3;
	double tol = 1e-12;

	// Returns true iff {x : h[i].a . x <= h[i].b, i < m} ∩ [-M,M]^3 is non-empty
	// (up to the tolerances above).  If xout != nullptr it receives a witness.
	bool feasible(const HalfSpace* h, int m, double* xout = nullptr) {
		if ((int)work.size() < m) work.resize(m);
		for (int i = 0; i < m; ++i) work[i] = h[i];
		shuffle(work.data(), m);
		double x[3];
		start(x);
		const bool ok = run(work.data(), 0, m, x);
		if (ok && xout) { xout[0] = x[0]; xout[1] = x[1]; xout[2] = x[2]; }
		return ok;
	}

	// Incremental use, for many LPs that share a prefix of constraints:
	//   prepare(h, m0, x)        shuffles h[0..m0) in place and sets x to their optimum;
	//   extend(h, m0, m, x)      shuffles h[m0..m) in place and continues Seidel's
	//                            algorithm from x (which must be the value prepare() gave).
	// extend() only pays for the new constraints (plus the recursions they trigger).
	bool prepare(HalfSpace* h, int m0, double x[3]) {
		shuffle(h, m0);
		start(x);
		return run(h, 0, m0, x);
	}
	bool extend(HalfSpace* h, int m0, int m, double x[3]) {
		shuffle(h + m0, m - m0);
		return run(h, m0, m, x);
	}
	// Makes the shuffles (and so the result up to the tolerances) a function of s only.
	void seed(uint64_t s) { rng = (s + 1) * 0x9E3779B97F4A7C15ull; }

private:
	struct H2 { double a[2]; double b; };
	static constexpr double c[3] = {0.5377, 0.8338, 0.1252};  // generic objective
	std::vector<HalfSpace> work;
	std::vector<H2> buf2;
	uint64_t rng = 0x9E3779B97F4A7C15ull;

	uint64_t next() {  // xorshift64*
		rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
		return rng * 2685821657736338717ull;
	}
	void shuffle(HalfSpace* h, int m) {  // Fisher-Yates, multiply-shift range reduction
		for (int i = m - 1; i > 0; --i) {
			const int j = (int)(((next() >> 32) * (uint64_t)(i + 1)) >> 32);
			const HalfSpace t = h[i]; h[i] = h[j]; h[j] = t;
		}
	}
	void start(double x[3]) const {
		for (int k = 0; k < 3; ++k) x[k] = c[k] > 0 ? -M : M;
	}

	// Seidel's main loop over h[from..m); x is the optimum of h[0..from).
	bool run(const HalfSpace* h, int from, int m, double x[3]) {
		if ((int)buf2.size() < m + 2) buf2.resize(m + 2);
		for (int idx = from; idx < m; ++idx) {
			const HalfSpace& hi = h[idx];
			if (hi.a[0] * x[0] + hi.a[1] * x[1] + hi.a[2] * x[2] <= hi.b + tol) continue;
			// The optimum over the first idx+1 constraints lies on the plane of hi.
			int k = 0;
			if (std::fabs(hi.a[1]) > std::fabs(hi.a[k])) k = 1;
			if (std::fabs(hi.a[2]) > std::fabs(hi.a[k])) k = 2;
			if (std::fabs(hi.a[k]) < 1e-300) {  // degenerate 0 . x <= b
				if (hi.b < -tol) return false;
				continue;
			}
			const int l0 = (k + 1) % 3, l1 = (k + 2) % 3;
			const double inv = 1.0 / hi.a[k];
			const double beta = hi.b * inv, g0 = hi.a[l0] * inv, g1 = hi.a[l1] * inv;
			// on the plane: x_k = beta - g0 * x_l0 - g1 * x_l1
			int cnt = 0;
			buf2[cnt++] = H2{{-g0, -g1}, M - beta};  // x_k <=  M
			buf2[cnt++] = H2{{g0, g1}, M + beta};    // x_k >= -M
			for (int j = 0; j < idx; ++j) {
				const HalfSpace& hj = h[j];
				buf2[cnt++] = H2{{hj.a[l0] - hj.a[k] * g0, hj.a[l1] - hj.a[k] * g1}, hj.b - hj.a[k] * beta};
			}
			const double c2[2] = {c[l0] - c[k] * g0, c[l1] - c[k] * g1};
			double y[2];
			if (!lp2(cnt, c2, y)) return false;
			x[l0] = y[0];
			x[l1] = y[1];
			x[k] = beta - g0 * y[0] - g1 * y[1];
		}
		return true;
	}

	// 2D LP over buf2[0..m) (both variables boxed in [-M, M]).
	bool lp2(int m, const double c2[2], double y[2]) {
		y[0] = c2[0] > 0 ? -M : M;
		y[1] = c2[1] > 0 ? -M : M;
		for (int i = 0; i < m; ++i) {
			const H2& hi = buf2[i];
			if (hi.a[0] * y[0] + hi.a[1] * y[1] <= hi.b + tol) continue;
			const int k = std::fabs(hi.a[1]) > std::fabs(hi.a[0]) ? 1 : 0;
			const int l = 1 - k;
			if (std::fabs(hi.a[k]) < 1e-300) {
				if (hi.b < -tol) return false;
				continue;
			}
			const double inv = 1.0 / hi.a[k];
			const double beta = hi.b * inv, g = hi.a[l] * inv;  // y_k = beta - g * y_l
			double lo = -M, up = M;
			bool ok = true;
			auto add = [&](double a, double b) {  // a * y_l <= b
				if (a > 1e-14) { double t = b / a; if (t < up) up = t; }
				else if (a < -1e-14) { double t = b / a; if (t > lo) lo = t; }
				else if (b < -tol) ok = false;
			};
			add(-g, M - beta);  // y_k <=  M
			add(g, M + beta);   // y_k >= -M
			for (int j = 0; j < i && ok; ++j) {
				const H2& hj = buf2[j];
				add(hj.a[l] - hj.a[k] * g, hj.b - hj.a[k] * beta);
			}
			if (!ok) return false;
			if (lo > up + 1e-12 * (1.0 + std::fabs(lo) + std::fabs(up))) return false;
			const double cl = c2[l] - c2[k] * g;
			const double yl = lo > up ? 0.5 * (lo + up) : (cl > 0 ? lo : up);
			y[l] = yl;
			y[k] = beta - g * yl;
		}
		return true;
	}
};

}  // namespace dpl
