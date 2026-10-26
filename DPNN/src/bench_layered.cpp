//
// bench_layered.cpp -- builds level 0..K (and optionally the all-layer trie),
// reports comparisons / query time / memory, and verifies exactness.
//
//   bench_layered <data.xyz | uniform:N | sphere:N> [options]
//     --order brio|random|fps   insertion order (default brio == dpnn.cpp)
//     --levels K                build levels 0..K (default 2)
//     --trie                    also build the all-layer trie
//     --sub N                   random subsample of N points
//     --queries N               queries per query set (default 1000000)
//     --check N                 brute-force check of the first N queries (default 20000)
//     --threads T               OpenMP threads for building / brute force
//     --no-huge                 do not request transparent huge pages
//     --coarse D                refine only nodes < n/D (level 0 below), see makeHybrid
//     --original                also time dpnn.cpp itself (needs -DWITH_ORIGINAL)
//
#ifdef WITH_ORIGINAL
#include <Eigen/Dense>
typedef Eigen::Vector3d Point;
#include "dpnn.cpp"
#endif
#include "dpnn_compact.h"
#include "dpnn_flat.h"
#include "dpnn_layered.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using dpl::u32;
using dpl::u64;
using dpl::Vec3;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<Vec3> readXYZ(const std::string& path) {
	std::vector<Vec3> pts;
	std::ifstream f(path);
	if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
	double x, y, z;
	while (f >> x >> y >> z) pts.push_back(Vec3{x, y, z});
	return pts;
}

static std::vector<Vec3> synth(const std::string& kind, size_t n, unsigned seed) {
	std::mt19937_64 gen(seed);
	std::uniform_real_distribution<double> U(0, 1);
	std::normal_distribution<double> N(0, 1);
	std::vector<Vec3> pts(n);
	for (auto& p : pts) {
		if (kind == "uniform") {
			p = Vec3{U(gen), U(gen), U(gen)};
		} else {  // sphere surface
			double x = N(gen), y = N(gen), z = N(gen), l = std::sqrt(x * x + y * y + z * z);
			p = Vec3{x / l, y / l, z / l};
		}
	}
	return pts;
}

static void bbox(const std::vector<Vec3>& pts, Vec3& lo, Vec3& hi) {
	lo = hi = pts[0];
	for (auto& p : pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
}

// Same distribution as generateRandomPoints() in main.cpp: uniform in the bounding
// box enlarged by scale * extent on every side (scale 0.5 -> 8x volume).
static std::vector<Vec3> boxQueries(const std::vector<Vec3>& pts, size_t m, double scale, unsigned seed) {
	Vec3 lo, hi;
	bbox(pts, lo, hi);
	std::mt19937_64 gen(seed);
	std::uniform_real_distribution<double> X(lo.x - scale * (hi.x - lo.x), hi.x + scale * (hi.x - lo.x));
	std::uniform_real_distribution<double> Y(lo.y - scale * (hi.y - lo.y), hi.y + scale * (hi.y - lo.y));
	std::uniform_real_distribution<double> Z(lo.z - scale * (hi.z - lo.z), hi.z + scale * (hi.z - lo.z));
	std::vector<Vec3> q(m);
	for (auto& p : q) p = Vec3{X(gen), Y(gen), Z(gen)};
	return q;
}

// Data point + isotropic Gaussian noise (sigma relative to the bbox diagonal):
// the typical ICP / point-to-surface situation.
static std::vector<Vec3> nearQueries(const std::vector<Vec3>& pts, size_t m, double sigmaRel, unsigned seed) {
	Vec3 lo, hi;
	bbox(pts, lo, hi);
	const double sigma = sigmaRel * std::sqrt(dpl::sqDist(lo, hi));
	std::mt19937_64 gen(seed);
	std::normal_distribution<double> N(0, sigma);
	std::vector<Vec3> q(m);
	for (auto& p : q) {
		const Vec3& b = pts[gen() % pts.size()];
		p = Vec3{b.x + N(gen), b.y + N(gen), b.z + N(gen)};
	}
	return q;
}

// One timed method: run() answers every query of the set once.
struct Method {
	std::string name, cmp, jumps, agree, note;
	std::function<void()> run;
	double us = 1e300;
};

// Every method runs once per round, rounds interleaved, minimum kept (µs per
// query): on a shared machine this compares all methods under the same load.
static void timeInterleaved(std::vector<Method>& ms, size_t m, int rounds) {
	for (int r = 0; r < rounds; ++r)
		for (auto& M : ms) {
			const double t0 = now();
			M.run();
			M.us = std::min(M.us, (now() - t0) * 1e6 / m);
		}
}

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <data.xyz|uniform:N|sphere:N> [options]\n", argv[0]); return 1; }
	std::string data = argv[1], orderS = "brio";
	int levels = 2;
	bool trie = false, original = false, huge = true;
	size_t sub = 0, nq = 1000000, check = 20000, coarse = 0;
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		auto val = [&]() { return std::string(argv[++i]); };
		if (a == "--order") orderS = val();
		else if (a == "--levels") levels = std::stoi(val());
		else if (a == "--trie") trie = true;
		else if (a == "--sub") sub = std::stoull(val());
		else if (a == "--queries") nq = std::stoull(val());
		else if (a == "--check") check = std::stoull(val());
		else if (a == "--original") original = true;
		else if (a == "--no-huge") huge = false;
		else if (a == "--coarse") coarse = std::stoull(val());
#ifdef _OPENMP
		else if (a == "--threads") omp_set_num_threads(std::stoi(val()));
#endif
		else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
	}
	const dpl::InsertionOrder order = orderS == "random" ? dpl::InsertionOrder::Random
	                                  : orderS == "fps"  ? dpl::InsertionOrder::Fps
	                                                     : dpl::InsertionOrder::Brio;

	std::vector<Vec3> pts;
	if (data.rfind("uniform:", 0) == 0) pts = synth("uniform", std::stoull(data.substr(8)), 7);
	else if (data.rfind("sphere:", 0) == 0) pts = synth("sphere", std::stoull(data.substr(7)), 7);
	else pts = readXYZ(data);
	if (sub && sub < pts.size()) {
		std::mt19937_64 gen(99);
		std::shuffle(pts.begin(), pts.end(), gen);
		pts.resize(sub);
	}
	std::printf("data %s  n=%zu  order=%s  threads=%d\n", data.c_str(), pts.size(), dpl::orderName(order),
	            dpl::detail::numThreads());

	// ---------------------------------------------------------------- build
	double t0 = now();
	dpl::Dag g = dpl::buildDag(pts, order);
	std::printf("DAG: %zu nodes, %zu edges (avg |Parents| %.2f), %.2fs\n", g.n(), g.numEdges(),
	            double(g.numEdges()) / g.n(), now() - t0);

	if (coarse && trie) { std::fprintf(stderr, "--coarse and --trie cannot be combined\n"); return 1; }
	const u32 T = coarse ? (u32)std::max<size_t>(1, g.n() / coarse) : (u32)g.n();
	if (coarse) std::printf("refining only nodes < %u (n/%zu); level 0 below\n", T, coarse);
	std::vector<dpl::LayerIndex> L;
	L.push_back(dpl::buildLevel0(g));
	for (int k = 1; k <= levels; ++k) {
		dpl::BuildStats bs;
		L.push_back(dpl::buildNextLevel(g, L, &bs, T));
		std::printf("level %d: %zu states, %zu entries (%.2f / state), %llu LPs (avg %.1f half-spaces), %.2fs\n", k,
		            L[k].numStates(), L[k].ent.size(), double(L[k].ent.size()) / L[k].numStates(),
		            (unsigned long long)bs.lpCalls, bs.lpCalls ? double(bs.lpConstraints) / bs.lpCalls : 0.0, bs.seconds);
		std::fflush(stdout);
	}
	dpl::LayerIndex Tr;
	if (trie) {
		dpl::BuildStats bs;
		std::vector<u64> perDepth;
		Tr = dpl::buildTrie(g, L[1], &bs, 0, &perDepth);
		std::printf("trie: %zu nodes (%.2f / point), max depth %zu, %llu LPs (avg %.1f half-spaces), %.2fs\n",
		            Tr.numStates(), double(Tr.numStates()) / g.n(), perDepth.size() - 1,
		            (unsigned long long)bs.lpCalls, bs.lpCalls ? double(bs.lpConstraints) / bs.lpCalls : 0.0, bs.seconds);
		std::fflush(stdout);
	}

	// query structures
	dpl::CompactL0 c0(g);  // == dpnn.cpp layout
	std::unique_ptr<dpl::CompactL1> c1;
	if (levels >= 1 && !coarse) c1.reset(new dpl::CompactL1(g, L[1]));
	struct Named {
		std::string name;
		std::unique_ptr<dpl::FlatIndex> idx;
	};
	std::vector<Named> flats;
	std::vector<std::unique_ptr<dpl::FlatIndex16>> flats16;
	for (int k = 0; k <= levels; ++k) {
		std::vector<u32> sn;
		dpl::LayerIndex H;
		const bool hybrid = coarse && k > 0;
		if (hybrid) H = dpl::makeHybrid(L, k, T, sn);
		else sn = dpl::stateNodes(L, k);
		const dpl::LayerIndex& src = hybrid ? H : L[k];
		flats.push_back({"L" + std::to_string(k), std::unique_ptr<dpl::FlatIndex>(new dpl::FlatIndex(src, sn, huge))});
		flats16.emplace_back(new dpl::FlatIndex16(src, sn, g, huge));
	}
	if (trie) {
		const std::vector<u32> sn = dpl::trieStateNodes(Tr);
		flats.push_back({"trie", std::unique_ptr<dpl::FlatIndex>(new dpl::FlatIndex(Tr, sn, huge))});
		flats16.emplace_back(new dpl::FlatIndex16(Tr, sn, g, huge));
	}
	std::printf("memory: dpnn.cpp layout %.1f MB", c0.bytes() / 1048576.0);
	if (c1) std::printf(", +level-1 bitmasks %.1f MB", c1->bytes() / 1048576.0);
	for (size_t i = 0; i < flats.size(); ++i)
		std::printf(", flat %s %.1f MB (f32: %.1f MB)", flats[i].name.c_str(), flats[i].idx->bytes() / 1048576.0,
		            flats16[i]->bytes() / 1048576.0);
	std::printf("\n");
	// the LayerIndex copies are no longer needed
	L.clear();
	Tr = dpl::LayerIndex();

#ifdef WITH_ORIGINAL
	std::unique_ptr<dpnn> orig;
	if (original) {
		std::vector<Point> P;
		for (auto& p : pts) P.emplace_back(p.x, p.y, p.z);
		double s = now();
		orig.reset(new dpnn(P, false));
		std::printf("original dpnn.cpp build: %.2fs\n", now() - s);
	}
#endif

	// ---------------------------------------------------------------- queries
	struct QSet {
		const char* name;
		std::vector<Vec3> q;
	};
	std::vector<QSet> sets;
	sets.push_back({"box2x", boxQueries(pts, nq, 0.5, 1)});
	sets.push_back({"near", nearQueries(pts, nq, 0.002, 2)});

	bool allOk = true;
	for (auto& S : sets) {
		const size_t m = S.q.size();
		std::vector<u32> ref(m), ans(m), scratch(m);
		u64 sink = 0;
		auto agreeWithRef = [&]() {
			size_t a = 0;
			for (size_t i = 0; i < m; ++i) a += ans[i] == ref[i];
			if (a != m) allOk = false;
			return std::to_string(a);
		};
		auto num = [](double v) {
			char b[32];
			std::snprintf(b, sizeof b, "%.2f", v);
			return std::string(b);
		};
		std::vector<Method> ms;
		{  // reference: dpnn.cpp layout
			dpl::QueryStats st;
			for (size_t i = 0; i < m; ++i) ref[i] = c0.query(S.q[i], st);
			ms.push_back({"dpnn.cpp layout", num(double(st.comparisons) / m), num(double(st.jumps) / m), "ref", "",
			              [&] { for (auto& q : S.q) sink += c0.query(q); }});
		}
		if (c1) {
			dpl::QueryStats st;
			for (size_t i = 0; i < m; ++i) ans[i] = c1->query(S.q[i], st);
			ms.push_back({"L1 bitmask (+4B/edge)", num(double(st.comparisons) / m), num(double(st.jumps) / m), agreeWithRef(), "",
			              [&] { for (auto& q : S.q) sink += c1->query(q); }});
		}
		for (size_t fi = 0; fi < flats.size(); ++fi) {
			const dpl::FlatIndex& f = *flats[fi].idx;
			const dpl::FlatIndex16& h = *flats16[fi];
			const std::string& nm = flats[fi].name;
			dpl::QueryStats st, st16;
			for (size_t i = 0; i < m; ++i) ans[i] = f.query(S.q[i], st);
			ms.push_back({"flat " + nm + " single", num(double(st.comparisons) / m), num(double(st.jumps) / m), agreeWithRef(), "",
			              [&] { for (auto& q : S.q) sink += f.query(q); }});
			f.queryBatch<16>(S.q.data(), m, ans.data());
			ms.push_back({"flat " + nm + " batch16", "", "", agreeWithRef(), "",
			              [&] { f.queryBatch<16>(S.q.data(), m, scratch.data()); }});
			for (size_t i = 0; i < m; ++i) ans[i] = h.query(S.q[i], st16);
			ms.push_back({"f32 " + nm + " single", "", "", agreeWithRef(),
			              "   (exact rechecks " + std::to_string(double(st16.exactRechecks) / m).substr(0, 6) + " / query)",
			              [&] { for (auto& q : S.q) sink += h.query(q); }});
			h.queryBatch<16>(S.q.data(), m, ans.data());
			ms.push_back({"f32 " + nm + " batch16", "", "", agreeWithRef(), "",
			              [&] { h.queryBatch<16>(S.q.data(), m, scratch.data()); }});
		}
#ifdef WITH_ORIGINAL
		std::vector<Point> P;
		if (orig) {
			for (auto& q : S.q) P.emplace_back(q.x, q.y, q.z);
			size_t same = 0;  // the original returns input indices: compare distances
			for (size_t i = 0; i < m; ++i)
				same += dpl::sqDist(pts[orig->getNearestNeighbor(P[i])], S.q[i]) == dpl::sqDist(g.pts[ref[i]], S.q[i]);
			if (same != m) allOk = false;
			ms.push_back({"dpnn.cpp (original)", "", "", std::to_string(same), "",
			              [&] { for (auto& p : P) sink += orig->getNearestNeighbor(p); }});
		}
#endif
		timeInterleaved(ms, m, 5);
		std::printf("\n== queries: %s (%zu)\n", S.name, m);
		std::printf("%-22s %10s %8s %12s %12s\n", "index", "cmp/query", "jumps", "us/query", "agree");
		for (auto& M : ms)
			std::printf("%-22s %10s %8s %12.4f %12s%s\n", M.name.c_str(), M.cmp.c_str(), M.jumps.c_str(), M.us, M.agree.c_str(),
			            M.note.c_str());
		const size_t mc = std::min(check, m);
		size_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
		for (long long i = 0; i < (long long)mc; ++i) {
			double best = 1e300;
			for (auto& p : g.pts) best = std::min(best, dpl::sqDist(p, S.q[i]));
			if (dpl::sqDist(g.pts[ref[i]], S.q[i]) != best) ++bad;
		}
		std::printf("brute force: %zu / %zu reference answers are not at minimum distance\n", bad, mc);
		std::fflush(stdout);
		if (bad) allOk = false;
		if (sink == 42) std::printf(" ");  // keep the timed loops observable
	}
	std::printf("\n%s\n", allOk ? "ALL CHECKS PASSED" : "CHECK FAILURES");
	return allOk ? 0 : 2;
}
