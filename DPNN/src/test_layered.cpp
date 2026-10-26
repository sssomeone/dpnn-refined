// Edge-case tests for dpnn_layered.h / dpnn_flat.h / dpnn_refined.h against brute force:
// tiny inputs, exact duplicates, integer grids (many cospherical points),
// coplanar and collinear sets, all insertion orders, levels 0..3, trie,
// float coordinates and coarse-only refinement.
#include "dpnn_compact.h"
#include "dpnn_refined.h"

#include <cstdio>
#include <random>
#include <string>

using namespace dpl;

static int failures = 0;

static void check(const std::string& name, const std::vector<Vec3>& pts, std::mt19937_64& gen) {
	// queries: near the data, inside the box, and well outside it
	Vec3 lo = pts[0], hi = pts[0];
	for (auto& p : pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
	const double diag = std::max(1e-9, std::sqrt(sqDist(lo, hi)));
	std::uniform_real_distribution<double> U(-1, 1);
	std::normal_distribution<double> N(0, 1);
	std::vector<Vec3> Q;
	for (int i = 0; i < 3000; ++i) {
		const double s = (i % 3 == 0) ? 0.01 : (i % 3 == 1) ? 1.0 : 20.0;
		const Vec3& b = pts[gen() % pts.size()];
		Q.push_back(Vec3{b.x + s * diag * U(gen), b.y + s * diag * U(gen), b.z + s * diag * U(gen)});
	}
	for (int i = 0; i < 300; ++i) Q.push_back(pts[gen() % pts.size()]);  // exactly on data points
	std::vector<double> best(Q.size(), 1e300);
	for (size_t i = 0; i < Q.size(); ++i)
		for (auto& p : pts) best[i] = std::min(best[i], sqDist(p, Q[i]));

	for (InsertionOrder order : {InsertionOrder::Brio, InsertionOrder::Random, InsertionOrder::Fps}) {
		Dag g = buildDag(pts, order);
		std::vector<LayerIndex> L{buildLevel0(g)};
		for (int k = 1; k <= 3; ++k) L.push_back(buildNextLevel(g, L));
		// the all-layer trie explodes on (near-)degenerate inputs such as grids;
		// that is a property of the data (see LAYERED_README.md), so only cap it here
		LayerIndex Tr;
		bool haveTrie = true;
		try {
			Tr = buildTrie(g, L[1], nullptr, 2000 * g.n());
		} catch (const std::runtime_error&) {
			haveTrie = false;
			std::printf("note %-12s %-7s all-layer trie > 2000 states/point, skipped\n", name.c_str(), orderName(order));
		}
		auto run = [&](const std::string& what, auto&& nearest) {
			size_t bad = 0;
			for (size_t i = 0; i < Q.size(); ++i) bad += sqDist(pts[nearest(Q[i])], Q[i]) != best[i];
			if (bad) {
				++failures;
				std::printf("FAIL %-12s %-7s %-14s %zu / %zu\n", name.c_str(), orderName(order), what.c_str(), bad, Q.size());
			}
		};
		for (int k = 0; k <= 3; ++k) {
			const std::vector<u32> sn = stateNodes(L, k);
			FlatIndex f(L[k], sn, false);
			FlatIndex16 h(L[k], sn, g, false);
			run("flat L" + std::to_string(k), [&](const Vec3& q) { return g.origId[f.query(q)]; });
			run("f32 L" + std::to_string(k), [&](const Vec3& q) { return g.origId[h.query(q)]; });
		}
		if (haveTrie) {
			FlatIndex ft(Tr, trieStateNodes(Tr), false);
			run("flat trie", [&](const Vec3& q) { return g.origId[ft.query(q)]; });
			CompactTrie ct(g, Tr);
			run("bitmask trie", [&](const Vec3& q) { return g.origId[ct.query(q)]; });
		}
		{
			CompactL0 c0(g);
			CompactL1 c1(g, L[1]);
			CompactL2 c2(g, L[1], L[2]);
			run("bitmask L0", [&](const Vec3& q) { return g.origId[c0.query(q)]; });
			run("bitmask L1", [&](const Vec3& q) { return g.origId[c1.query(q)]; });
			run("bitmask L2", [&](const Vec3& q) { return g.origId[c2.query(q)]; });
		}
		// batch path
		{
			std::vector<u32> out(Q.size());
			FlatIndex f(L[2], stateNodes(L, 2), false);
			f.queryBatch<8>(Q.data(), Q.size(), out.data());
			size_t i = 0;
			run("batch L2", [&](const Vec3&) { return g.origId[out[i++]]; });
		}
		// coarse-only refinement
		{
			const u32 T = (u32)std::max<size_t>(1, g.n() / 4);
			std::vector<LayerIndex> C{buildLevel0(g)};
			for (int k = 1; k <= 2; ++k) C.push_back(buildNextLevel(g, C, nullptr, T));
			std::vector<u32> sn;
			LayerIndex H = makeHybrid(C, 2, T, sn);
			FlatIndex f(H, sn, false);
			run("coarse L2", [&](const Vec3& q) { return g.origId[f.query(q)]; });
		}
	}
	// wrapper
	RefinedOptions opt;
	opt.levels = 2;
	opt.floatCoords = true;
	RefinedDPNN nn(pts, opt);
	size_t bad = 0;
	for (size_t i = 0; i < Q.size(); ++i) bad += sqDist(pts[nn.nearest(Q[i])], Q[i]) != best[i];
	if (bad) { ++failures; std::printf("FAIL %-12s RefinedDPNN %zu\n", name.c_str(), bad); }
	std::printf("ok   %-12s n=%zu\n", name.c_str(), pts.size());
}

int main() {
	std::mt19937_64 gen(2024);
	std::uniform_real_distribution<double> U(0, 1);
	for (int n : {5, 6, 9, 17, 50}) {
		std::vector<Vec3> p(n);
		for (auto& v : p) v = Vec3{U(gen), U(gen), U(gen)};
		check("random" + std::to_string(n), p, gen);
	}
	{  // duplicates
		std::vector<Vec3> p(800);
		for (auto& v : p) v = Vec3{U(gen), U(gen), U(gen)};
		for (int i = 0; i < 200; ++i) p.push_back(p[gen() % 800]);
		std::shuffle(p.begin(), p.end(), gen);
		check("duplicates", p, gen);
	}
	{  // integer grid: massively cospherical
		std::vector<Vec3> p;
		for (int x = 0; x < 9; ++x)
			for (int y = 0; y < 9; ++y)
				for (int z = 0; z < 9; ++z) p.push_back(Vec3{double(x), double(y), double(z)});
		check("grid9^3", p, gen);
	}
	{  // coplanar
		std::vector<Vec3> p(600);
		for (auto& v : p) v = Vec3{U(gen), U(gen), 0.5};
		check("coplanar", p, gen);
	}
	{  // collinear
		std::vector<Vec3> p(200);
		for (auto& v : p) { double t = U(gen); v = Vec3{t, 2 * t, -t}; }
		check("collinear", p, gen);
	}
	{  // sphere surface
		std::normal_distribution<double> N(0, 1);
		std::vector<Vec3> p(3000);
		for (auto& v : p) {
			double x = N(gen), y = N(gen), z = N(gen), l = std::sqrt(x * x + y * y + z * z);
			v = Vec3{x / l, y / l, z / l};
		}
		check("sphere", p, gen);
	}
	std::printf("%s\n", failures ? "FAILURES" : "ALL EDGE-CASE TESTS PASSED");
	return failures != 0;
}
