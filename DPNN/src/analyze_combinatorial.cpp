//
// analyze_combinatorial.cpp -- can a purely combinatorial (graph-only) rule
// replace the LP test (**) of dpnn_layered.h?
//
// For every 2-path a -> b -> c of the DAG we compare
//   exact : Q(a->b) ∩ Q(b->c) != ∅          (LP over bisector half-spaces)
//   comb  : a ∈ Parents(c)                  (c also stole from a's cell)
// and then answer queries with level-1 lists filtered by `comb` instead of the
// LP to see how many answers become wrong.
//
//   analyze_combinatorial <data.xyz | uniform:N> [--order brio|random|fps] [--queries N]
//
#include "dpnn_layered.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <string>

using namespace dpl;

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <data.xyz|uniform:N> [--order o] [--queries N]\n", argv[0]); return 1; }
	std::string data = argv[1], orderS = "brio";
	size_t nq = 200000;
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--order") orderS = argv[++i];
		else if (a == "--queries") nq = std::stoull(argv[++i]);
	}
	const InsertionOrder order = orderS == "random" ? InsertionOrder::Random
	                             : orderS == "fps"  ? InsertionOrder::Fps
	                                                : InsertionOrder::Brio;
	std::vector<Vec3> pts;
	if (data.rfind("uniform:", 0) == 0) {
		std::mt19937_64 gen(7);
		std::uniform_real_distribution<double> U(0, 1);
		pts.resize(std::stoull(data.substr(8)));
		for (auto& p : pts) p = Vec3{U(gen), U(gen), U(gen)};
	} else {
		std::ifstream f(data);
		double x, y, z;
		while (f >> x >> y >> z) pts.push_back(Vec3{x, y, z});
	}
	Dag g = buildDag(pts, order);
	std::vector<LayerIndex> L;
	L.push_back(buildLevel0(g));
	L.push_back(buildNextLevel(g, L));
	const LayerIndex& L1 = L[1];

	// confusion matrix over all 2-paths (e, f), f in children(child(e))
	u64 n11 = 0, n10 = 0, n01 = 0, n00 = 0;  // [exact][comb]
	auto isParent = [&](u32 a, u32 c) {
		return std::binary_search(g.pa.begin() + g.paBeg[c], g.pa.begin() + g.paBeg[c + 1], a);
	};
	std::vector<u32> combBeg(L1.numStates() + 1, 0), combEdge;
	for (u32 e = 0; e < g.numEdges(); ++e) {
		const u32 a = g.edgeSrc[e], b = g.ch[e];
		u32 p = L1.beg[e];
		for (u32 f = g.chBeg[b]; f < g.chBeg[b + 1]; ++f) {
			const bool exact = (p < L1.beg[e + 1] && L1.edge[p] == f);
			if (exact) ++p;
			const bool comb = isParent(a, g.ch[f]);
			if (exact && comb) ++n11;
			else if (exact) ++n10;
			else if (comb) ++n01;
			else ++n00;
			if (comb) combEdge.push_back(f);
		}
		combBeg[e + 1] = (u32)combEdge.size();
	}
	// root keeps everything
	for (u32 f = g.chBeg[0]; f < g.chBeg[1]; ++f) combEdge.push_back(f);
	combBeg[L1.root + 1] = (u32)combEdge.size();
	const double tot = double(n11 + n10 + n01 + n00);
	std::printf("n=%zu order=%s  2-paths: %.0f\n", g.n(), orderName(order), tot);
	std::printf("  exact-realizable %.1f%%  |  a in Parents(c) %.1f%%\n", 100.0 * (n11 + n10) / tot, 100.0 * (n11 + n01) / tot);
	std::printf("  realizable but a not in Parents(c): %.1f%% of realizable  (a comb-only filter would DROP these)\n",
	            100.0 * n10 / double(n11 + n10));
	std::printf("  a in Parents(c) but not realizable: %.1f%% of comb-kept  (wasted comparisons)\n",
	            100.0 * n01 / double(n11 + n01));

	// queries with the combinatorial filter (approximate!)
	Vec3 lo = g.pts[0], hi = g.pts[0];
	for (auto& p : g.pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
	std::mt19937_64 gen(1);
	std::uniform_real_distribution<double> X(lo.x - .5 * (hi.x - lo.x), hi.x + .5 * (hi.x - lo.x));
	std::uniform_real_distribution<double> Y(lo.y - .5 * (hi.y - lo.y), hi.y + .5 * (hi.y - lo.y));
	std::uniform_real_distribution<double> Z(lo.z - .5 * (hi.z - lo.z), hi.z + .5 * (hi.z - lo.z));
	u64 wrong = 0, cmpExact = 0, cmpComb = 0;
	double relErr = 0;
	for (size_t i = 0; i < nq; ++i) {
		const Vec3 q{X(gen), Y(gen), Z(gen)};
		QueryStats st;
		const u32 exact = L1.query(q, st);
		cmpExact += st.comparisons;
		// level-1 query over the combinatorial lists
		u32 s = L1.root, v = 0;
		double best = sqDist(q, g.pts[0]);
		for (;;) {
			bool jumped = false;
			for (u32 k = combBeg[s]; k < combBeg[s + 1]; ++k) {
				const u32 f = combEdge[k], w = g.ch[f];
				++cmpComb;
				const double d = sqDist(q, g.pts[w]);
				if (d < best) { best = d; v = w; s = f; jumped = true; break; }
			}
			if (!jumped) break;
		}
		if (v != exact) {
			++wrong;
			relErr += std::sqrt(sqDist(q, g.pts[v]) / sqDist(q, g.pts[exact])) - 1.0;
		}
	}
	std::printf("queries (box2x, %zu): exact level-1 %.2f cmp/query; comb-filtered %.2f cmp/query, wrong answers %.3f%% "
	            "(mean distance excess of wrong ones %.2f%%)\n",
	            nq, double(cmpExact) / nq, double(cmpComb) / nq, 100.0 * wrong / nq, wrong ? 100.0 * relErr / wrong : 0.0);
	return 0;
}
