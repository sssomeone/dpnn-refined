//
// bench_dtree.cpp -- decision trees inside the states of level 0..K (and of the
// trie) versus the sequential scan: comparisons, query time, exactness.
//
//   bench_dtree <data.xyz | uniform:N> [options]
//     --order brio|fps|random   insertion order (default brio)
//     --levels K                levels 0..K (default 2)
//     --trie                    also the all-layer trie
//     --train N                 training queries per distribution (box + near, other
//                               seeds than the measured sets; default 1000000)
//     --train-on box|near|both  which distributions train the trees (default both)
//     --queries N               measured queries per set (default 1000000)
//     --check N                 brute-force check of the first N answers per set
//     --budget B                internal nodes per state <= B * list length (default 4)
//     --crit 0..3               test choice: 0 information gain; LP look-ahead scored by 1 outcomes left,
//                               2 log2(outcomes left) (default), 3 two-step look-ahead
//     --threads T
//     --no-timing               comparisons and exactness only
//     --no-huge
//
#include "dpnn_dtree.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>

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
static void bbox(const std::vector<Vec3>& pts, Vec3& lo, Vec3& hi) {
	lo = hi = pts[0];
	for (auto& p : pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
}
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

struct Method {
	std::string name, cmp, jumps, agree;
	std::function<void()> run;
	double us = 1e300;
};
static void timeInterleaved(std::vector<Method>& ms, size_t m, int rounds) {
	for (int r = 0; r < rounds; ++r)
		for (auto& M : ms) {
			const double t0 = now();
			M.run();
			M.us = std::min(M.us, (now() - t0) * 1e6 / m);
		}
}
static std::string num(double v) {
	char b[32];
	std::snprintf(b, sizeof b, "%.2f", v);
	return b;
}

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <data.xyz|uniform:N> [options]\n", argv[0]); return 1; }
	std::string data = argv[1], orderS = "brio", trainOn = "both";
	int levels = 2, budget = 4, crit = 2;
	bool trie = false, huge = true, timing = true;
	size_t nTrain = 1000000, nq = 1000000, check = 20000;
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		auto val = [&]() { return std::string(argv[++i]); };
		if (a == "--order") orderS = val();
		else if (a == "--levels") levels = std::stoi(val());
		else if (a == "--trie") trie = true;
		else if (a == "--train") nTrain = std::stoull(val());
		else if (a == "--train-on") trainOn = val();
		else if (a == "--queries") nq = std::stoull(val());
		else if (a == "--check") check = std::stoull(val());
		else if (a == "--budget") budget = std::stoi(val());
		else if (a == "--crit") crit = std::stoi(val());
		else if (a == "--no-huge") huge = false;
		else if (a == "--no-timing") timing = false;
#ifdef _OPENMP
		else if (a == "--threads") omp_set_num_threads(std::stoi(val()));
#endif
		else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
	}
	const dpl::InsertionOrder order = orderS == "random" ? dpl::InsertionOrder::Random
	                                  : orderS == "fps"  ? dpl::InsertionOrder::Fps
	                                                     : dpl::InsertionOrder::Brio;
	std::vector<Vec3> pts;
	if (data.rfind("uniform:", 0) == 0) {
		std::mt19937_64 gen(7);
		std::uniform_real_distribution<double> U(0, 1);
		pts.resize(std::stoull(data.substr(8)));
		for (auto& p : pts) p = Vec3{U(gen), U(gen), U(gen)};
	} else {
		pts = readXYZ(data);
	}
	std::printf("data %s  n=%zu  order=%s  threads=%d  train=%zu x %s  budget=%d  crit=%d\n", data.c_str(), pts.size(),
	            dpl::orderName(order), dpl::detail::numThreads(), nTrain, trainOn.c_str(), budget, crit);

	double t0 = now();
	const dpl::Dag g = dpl::buildDag(pts, order);
	std::vector<dpl::LayerIndex> L;
	L.push_back(dpl::buildLevel0(g));
	for (int k = 1; k <= levels; ++k) L.push_back(dpl::buildNextLevel(g, L));
	dpl::LayerIndex Tr;
	if (trie) Tr = dpl::buildTrie(g, L[1]);
	std::printf("DAG + levels%s: %.1fs\n", trie ? " + trie" : "", now() - t0);

	std::vector<Vec3> train;
	if (trainOn != "near") {
		auto t = boxQueries(pts, nTrain, 0.5, 101);
		train.insert(train.end(), t.begin(), t.end());
	}
	if (trainOn != "box") {
		auto t = nearQueries(pts, nTrain, 0.002, 102);
		train.insert(train.end(), t.begin(), t.end());
	}

	struct Built {
		std::string name;
		std::unique_ptr<dpl::FlatIndex> seq;
		std::unique_ptr<dpl::DtreeIndex> dt;
	};
	std::vector<Built> B;
	const int nStruct = levels + 1 + (trie ? 1 : 0);
	std::printf("\n%-6s %10s %10s %9s %12s %10s %10s %9s %8s\n", "index", "states", "seq MB", "dt MB", "LP calls", "tests/st",
	            "forced/st", "chains", "build s");
	for (int k = 0; k < nStruct; ++k) {
		const bool isTrie = k > levels;
		const dpl::LayerIndex& X = isTrie ? Tr : L[k];
		const std::vector<u32> sn = isTrie ? dpl::trieStateNodes(Tr) : dpl::stateNodes(L, k);
		dpl::DtreeStats st;
		std::unique_ptr<dpl::DtreeIndex> D;
		if (isTrie) D.reset(new dpl::DtreeIndex(g, X, sn, dpl::TrieRegion{g, Tr}, train, budget, &st, huge, crit));
		else D.reset(new dpl::DtreeIndex(g, X, sn, dpl::LevelRegion{g, L, k}, train, budget, &st, huge, crit));
		std::unique_ptr<dpl::FlatIndex> F(new dpl::FlatIndex(X, sn, huge));
		const std::string name = isTrie ? "trie" : "L" + std::to_string(k);
		std::printf("%-6s %10zu %10.1f %9.1f %12llu %10.2f %10.3f %9llu %8.1f\n", name.c_str(), X.numStates(), F->bytes() / 1048576.0,
		            D->bytes() / 1048576.0, (unsigned long long)st.lpCalls, double(st.internal) / st.states,
		            double(st.forced) / st.states, (unsigned long long)st.chainStates, st.seconds);
		std::fflush(stdout);
		B.push_back({name, std::move(F), std::move(D)});
	}

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
		std::vector<Method> ms;
		u64 sink = 0;
		for (auto& b : B) {
			const dpl::FlatIndex& F = *b.seq;
			const dpl::DtreeIndex& D = *b.dt;
			dpl::QueryStats s1, s2;
			for (size_t i = 0; i < m; ++i) ref[i] = F.query(S.q[i], s1);
			for (size_t i = 0; i < m; ++i) ans[i] = D.query(S.q[i], s2);
			size_t same = 0;
			for (size_t i = 0; i < m; ++i) same += ans[i] == ref[i];
			if (same != m) allOk = false;
			D.queryBatch<16>(S.q.data(), m, scratch.data());
			size_t sameB = 0;
			for (size_t i = 0; i < m; ++i) sameB += scratch[i] == ref[i];
			if (sameB != m) allOk = false;
			// brute force on the first answers
			const size_t mc = std::min(check, m);
			size_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
			for (long long i = 0; i < (long long)mc; ++i) {
				double best = 1e300;
				for (auto& p : g.pts) best = std::min(best, dpl::sqDist(p, S.q[i]));
				if (dpl::sqDist(g.pts[ans[i]], S.q[i]) != best) ++bad;
			}
			if (bad) allOk = false;
			ms.push_back({b.name + " seq", num(double(s1.comparisons) / m), num(double(s1.jumps) / m), "ref",
			              [&] { for (auto& q : S.q) sink += F.query(q); }});
			ms.push_back({b.name + " seq batch16", "", "", "", [&] { F.queryBatch<16>(S.q.data(), m, scratch.data()); }});
			ms.push_back({b.name + " dtree", num(double(s2.comparisons) / m), num(double(s2.jumps) / m),
			              std::to_string(same) + (bad ? "  BRUTE-FORCE ERRORS " + std::to_string(bad) : ""),
			              [&] { for (auto& q : S.q) sink += D.query(q); }});
			ms.push_back({b.name + " dtree batch16", "", "", std::to_string(sameB),
			              [&] { D.queryBatch<16>(S.q.data(), m, scratch.data()); }});
		}
		if (timing) timeInterleaved(ms, m, 5);
		std::printf("\n== queries: %s (%zu)\n%-20s %10s %8s %10s %12s\n", S.name, m, "index", "cmp/query", "jumps", "us/query",
		            "same as seq");
		for (auto& M : ms) {
			if (!timing && M.cmp.empty()) continue;  // batch rows only carry a time
			char t[32] = "-";
			if (timing) std::snprintf(t, sizeof t, "%.4f", M.us);
			std::printf("%-20s %10s %8s %10s %12s\n", M.name.c_str(), M.cmp.c_str(), M.jumps.c_str(), t, M.agree.c_str());
		}
		std::fflush(stdout);
		if (sink == 42) std::printf(" ");
	}
	std::printf("\n%s\n", allOk ? "ALL CHECKS PASSED" : "CHECK FAILURES");
	return allOk ? 0 : 2;
}
