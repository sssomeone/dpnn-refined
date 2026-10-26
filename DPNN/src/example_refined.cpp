// Minimal usage of dpnn_refined.h: build, query, and check against brute force.
//   example_refined data/bunny.xyz
#include "dpnn_refined.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>

int main(int argc, char** argv) {
	const char* path = argc > 1 ? argv[1] : "../data/bunny.xyz";
	std::vector<dpl::Vec3> pts;
	std::ifstream f(path);
	double x, y, z;
	while (f >> x >> y >> z) pts.push_back(dpl::Vec3{x, y, z});
	if (pts.size() < 5) { std::fprintf(stderr, "cannot read %s\n", path); return 1; }

	auto t0 = std::chrono::steady_clock::now();
	dpl::RefinedOptions opt;
	opt.levels = 2;
	dpl::RefinedDPNN nn(pts, opt);
	const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

	std::mt19937_64 gen(1);
	std::uniform_real_distribution<double> U(-1, 1);
	std::vector<dpl::Vec3> Q(200000);
	for (auto& q : Q) q = dpl::Vec3{U(gen), U(gen), U(gen)};
	std::vector<uint32_t> out(Q.size());
	t0 = std::chrono::steady_clock::now();
	nn.nearestBatch(Q.data(), Q.size(), out.data());
	const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / Q.size();

	size_t bad = 0;
	for (size_t i = 0; i < 2000; ++i) {
		double best = 1e300;
		for (auto& p : pts) best = std::min(best, dpl::sqDist(p, Q[i]));
		bad += dpl::sqDist(pts[out[i]], Q[i]) != best;
		bad += nn.nearest(Q[i]) != out[i];
	}
	std::printf("%zu points, build %.2f s, %.1f MB, %.3f us/query (batch), %zu mismatches in 2000 checks\n", pts.size(),
	            build, nn.bytes() / 1048576.0, us, bad);
	return bad != 0;
}
