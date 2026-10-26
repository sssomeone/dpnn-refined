// Brute-force cross-check of lp3.h against the Chebyshev margin computed by
// enumerating all vertices of the 4D (x, s) constraint polytope.
#include "lp3.h"
#include <cstdio>
#include <random>
#include <vector>

using namespace dpl;

// Chebyshev margin: max s s.t. a_i.x + |a_i| s <= b_i (box included), by brute
// force over all 4-subsets of the 4D constraints.  s* > 0 <=> full-dimensional,
// s* < 0 <=> empty.
static bool solve4(double A[4][5], double x[4]) {
	for (int c = 0; c < 4; ++c) {
		int p = c;
		for (int r = c + 1; r < 4; ++r)
			if (std::fabs(A[r][c]) > std::fabs(A[p][c])) p = r;
		if (std::fabs(A[p][c]) < 1e-12) return false;
		for (int k = 0; k < 5; ++k) std::swap(A[c][k], A[p][k]);
		for (int r = 0; r < 4; ++r)
			if (r != c) {
				double f = A[r][c] / A[c][c];
				for (int k = c; k < 5; ++k) A[r][k] -= f * A[c][k];
			}
	}
	for (int c = 0; c < 4; ++c) x[c] = A[c][4] / A[c][c];
	return true;
}
static double bruteMaxMinSlack(std::vector<HalfSpace> h, double M) {
	for (int k = 0; k < 3; ++k) {
		HalfSpace u{{0, 0, 0}, M}, l{{0, 0, 0}, M};
		u.a[k] = 1; l.a[k] = -1;
		h.push_back(u); h.push_back(l);
	}
	const int m = (int)h.size();
	std::vector<double> nrm(m);
	for (int i = 0; i < m; ++i) nrm[i] = std::sqrt(h[i].a[0] * h[i].a[0] + h[i].a[1] * h[i].a[1] + h[i].a[2] * h[i].a[2]);
	double best = -1e300;
	int idx[4];
	for (idx[0] = 0; idx[0] < m; ++idx[0])
		for (idx[1] = idx[0] + 1; idx[1] < m; ++idx[1])
			for (idx[2] = idx[1] + 1; idx[2] < m; ++idx[2])
				for (idx[3] = idx[2] + 1; idx[3] < m; ++idx[3]) {
					double A[4][5], x[4];
					for (int r = 0; r < 4; ++r) {
						const HalfSpace& q = h[idx[r]];
						A[r][0] = q.a[0]; A[r][1] = q.a[1]; A[r][2] = q.a[2]; A[r][3] = nrm[idx[r]]; A[r][4] = q.b;
					}
					if (!solve4(A, x)) continue;
					bool ok = true;
					for (int i = 0; i < m && ok; ++i)
						if (h[i].a[0] * x[0] + h[i].a[1] * x[1] + h[i].a[2] * x[2] + nrm[i] * x[3] > h[i].b + 1e-9) ok = false;
					if (ok && x[3] > best) best = x[3];
				}
	return best;
}

int main() {
	std::mt19937_64 gen(12345);
	std::normal_distribution<double> N(0, 1);
	std::uniform_real_distribution<double> U(-1, 1);
	LP3 lp;
	lp.M = 10;
	int agree = 0, disagree = 0, skipped = 0, feas = 0;
	for (int it = 0; it < 20000; ++it) {
		int m = 1 + (int)(gen() % 14);
		std::vector<HalfSpace> h(m);
		// random tangent planes of a random ball -> mixes feasible/infeasible
		double cx = U(gen) * 2, cy = U(gen) * 2, cz = U(gen) * 2, R = 0.05 + std::fabs(U(gen));
		for (auto& q : h) {
			double nx = N(gen), ny = N(gen), nz = N(gen);
			double l = std::sqrt(nx * nx + ny * ny + nz * nz);
			nx /= l; ny /= l; nz /= l;
			q.a[0] = nx; q.a[1] = ny; q.a[2] = nz;
			q.b = nx * cx + ny * cy + nz * cz + R * U(gen) * 1.5;  // some planes cut through the ball
			if (gen() % 5 == 0) { q.a[0] *= -1; q.a[1] *= -1; q.a[2] *= -1; q.b = -q.b - R * std::fabs(U(gen)); }
		}
		double s = bruteMaxMinSlack(h, lp.M);
		bool f = lp.feasible(h.data(), m);
		if (std::fabs(s) < 1e-7) { ++skipped; continue; }
		if ((s > 0) == f) ++agree; else { ++disagree; std::printf("disagree: m=%d slack=%g lp=%d\n", m, s, (int)f); }
		feas += s > 0;
	}
	std::printf("agree %d  disagree %d  skipped(near-degenerate) %d  feasible-fraction %.3f\n", agree, disagree, skipped,
	            feas / double(agree + disagree));
	// witness check
	int badWitness = 0;
	for (int it = 0; it < 5000; ++it) {
		int m = 1 + (int)(gen() % 30);
		std::vector<HalfSpace> h(m);
		for (auto& q : h) {
			double nx = N(gen), ny = N(gen), nz = N(gen);
			q.a[0] = nx; q.a[1] = ny; q.a[2] = nz;
			q.b = std::fabs(N(gen));  // origin always feasible
		}
		double x[3];
		if (!lp.feasible(h.data(), m, x)) { ++badWitness; continue; }
		for (auto& q : h)
			if (q.a[0] * x[0] + q.a[1] * x[1] + q.a[2] * x[2] > q.b + 1e-9) { ++badWitness; break; }
	}
	std::printf("witness failures: %d / 5000\n", badWitness);
	// prepare(prefix) + extend(rest) must agree with feasible(all)
	int badExtend = 0;
	for (int it = 0; it < 20000; ++it) {
		int m = 2 + (int)(gen() % 40);
		std::vector<HalfSpace> h(m);
		double cx = U(gen), cy = U(gen), cz = U(gen), R = 0.05 + std::fabs(U(gen));
		for (auto& q : h) {
			double nx = N(gen), ny = N(gen), nz = N(gen), l = std::sqrt(nx * nx + ny * ny + nz * nz);
			q.a[0] = nx / l; q.a[1] = ny / l; q.a[2] = nz / l;
			q.b = q.a[0] * cx + q.a[1] * cy + q.a[2] * cz + R * U(gen) * 1.5;
		}
		const int m0 = 1 + (int)(gen() % (m - 1));
		const bool all = lp.feasible(h.data(), m);
		std::vector<HalfSpace> w = h;
		double x[3];
		const bool pre = lp.prepare(w.data(), m0, x);
		const bool ext = pre && lp.extend(w.data(), m0, m, x);
		if (all != ext) {
			// only acceptable for near-degenerate instances
			if (std::fabs(bruteMaxMinSlack(h, lp.M)) > 1e-7) ++badExtend;
		}
	}
	std::printf("prepare+extend disagreements: %d / 20000\n", badExtend);
	return disagree || badWitness || badExtend;
}
