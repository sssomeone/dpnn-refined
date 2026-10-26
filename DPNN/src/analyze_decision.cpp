//
// analyze_decision.cpp -- where the comparisons of a (refined) DPNN query go, and how
// many a smarter order of the tests inside one state could save.
//
//   analyze_decision <data.xyz | uniform:N> [--order brio|fps|random] [--levels K]
//                    [--trie] [--queries N] [--threads T]
//
// Inside a state with candidate list c_1..c_m (insertion order) and current point u the
// query takes the FIRST c_i with |q-c_i| < |q-u| (outcome i), or stays (outcome "none").
// The sequential scan pays i tests (m for none).  By pure logic that is optimal: outcome
// i can only be certified by also showing that c_1..c_{i-1} are not closer.  A different
// test order can only win through geometry: inside the region already known, a positive
// (or negative) answer for c_j may rule out other candidates without testing them.
//
// For sampled queries we record, at every visited state, the full bit vector
// x_k = [|q-c_k| < |q-u|] and report per structure:
//   cmp      sequential comparisons per query (== bench_layered)
//   = jumps + failed tests before a jump + final scan
//   H        Σ over visited states of H(outcome | state): a lower bound for any strategy
//            that makes binary tests at these states
//   greedy   expected tests of one decision tree per state, built ID3-style on half of the
//            samples (leaf as soon as all training samples agree on the outcome) and
//            evaluated on the other half; a test sample whose leaf disagrees pays the
//            sequential tests it still needs.  An estimate of what LP-certified decision
//            trees could reach.
//
#include "dpnn_dtree.h"

#include <parallel/algorithm>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using dpl::u32;
using dpl::u64;
using u16 = uint16_t;
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
// same query distributions and seeds as bench_layered.cpp
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

// one visited state of one query
struct Rec {
	u64 mask;      // bit k: candidate k is closer than the current point (k < 64)
	u32 state;
	u16 m;         // list length
	u16 outcome;   // first set bit, m == none
};

static double entropy(const std::vector<double>& c) {
	double t = 0, h = 0;
	for (double x : c) t += x;
	for (double x : c)
		if (x > 0) h -= x * std::log2(x / t);
	return t > 0 ? h / t : 0;
}

// ID3-style decision tree over sampled bit vectors of one state
struct Sample {
	u64 mask;
	u16 outcome;
	u32 count;
};
struct TreeNode {
	int test = -1;          // candidate tested, -1 == leaf
	int child[2] = {-1, -1};  // [bit == 0, bit == 1]
	int outcome = -1;       // leaf
};

static int build(std::vector<TreeNode>& T, std::vector<Sample> S, u64 tested, int m) {
	const int me = (int)T.size();
	T.emplace_back();
	bool same = true;
	for (auto& s : S) same &= s.outcome == S[0].outcome;
	if (S.empty() || same) {
		T[me].outcome = S.empty() ? -1 : S[0].outcome;
		return me;
	}
	auto H = [&](const std::vector<Sample>& v) {
		std::map<int, double> c;
		for (auto& s : v) c[s.outcome] += s.count;
		std::vector<double> x;
		for (auto& kv : c) x.push_back(kv.second);
		double t = 0;
		for (double y : x) t += y;
		return std::make_pair(entropy(x), t);
	};
	const int lim = std::min(m, 64);
	int bestJ = -1;
	double bestH = 1e300;
	for (int j = 0; j < lim; ++j) {
		if (tested >> j & 1) continue;
		std::vector<Sample> a, b;
		for (auto& s : S) (s.mask >> j & 1 ? b : a).push_back(s);
		if (a.empty() || b.empty()) continue;
		const auto ha = H(a), hb = H(b);
		const double h = ha.first * ha.second + hb.first * hb.second;
		if (h < bestH - 1e-12) { bestH = h; bestJ = j; }
	}
	if (bestJ < 0) {  // no bit splits the samples although outcomes differ: cannot happen
		T[me].outcome = S[0].outcome;
		return me;
	}
	std::vector<Sample> a, b;
	for (auto& s : S) (s.mask >> bestJ & 1 ? b : a).push_back(s);
	T[me].test = bestJ;
	const int c0 = build(T, std::move(a), tested | (1ull << bestJ), m);
	const int c1 = build(T, std::move(b), tested | (1ull << bestJ), m);
	T[me].child[0] = c0;
	T[me].child[1] = c1;
	return me;
}

// tests a sample pays on tree T; a wrong leaf falls back to testing, in insertion order,
// every untested candidate up to the sample's outcome
static int evalCost(const std::vector<TreeNode>& T, u64 mask, int outcome, int m) {
	int node = 0, cost = 0;
	u64 tested = 0;
	while (T[node].test >= 0) {
		const int j = T[node].test;
		++cost;
		tested |= 1ull << j;
		node = T[node].child[mask >> j & 1];
	}
	if (T[node].outcome == outcome) return cost;
	const int upto = outcome < m ? outcome : m - 1;
	for (int k = 0; k <= upto && k < 64; ++k)
		if (!(tested >> k & 1)) ++cost;
	if (upto >= 64) cost += upto - 63;
	return cost;
}

struct Report {
	double cmp = 0, jumps = 0, failed = 0, finalScan = 0, H = 0, greedy = 0, hAnswer = 0, hFinalState = 0;
	double statesVisited = 0;
	double orchard = 0;  // sequential tests left if candidates with |u-c| >= 2|q-u| are skipped
};

static Report analyze(const dpl::LayerIndex& L, const std::vector<Vec3>& Q, const std::vector<Vec3>& pts) {
	const size_t nq = Q.size();
	const int T = dpl::detail::numThreads();
	std::vector<std::vector<Rec>> per(T);
	std::vector<u32> finalNode(nq), finalState(nq);
	double orchard = 0;
#pragma omp parallel num_threads(T) reduction(+ : orchard)
	{
#ifdef _OPENMP
		std::vector<Rec>& out = per[omp_get_thread_num()];
#else
		std::vector<Rec>& out = per[0];
#endif
#pragma omp for schedule(static, 4096)
		for (long long qi = 0; qi < (long long)nq; ++qi) {
			const Vec3& q = Q[qi];
			u32 s = L.root, node = L.rootNode;
			double best = dpl::sqDist(q, L.rootPt);
			for (;;) {
				const u32 b = L.beg[s], e = L.beg[s + 1], m = e - b;
				u64 mask = 0;
				u32 outc = m;
				double dOut = 0;
				const Vec3 up = pts[node];
				for (u32 i = 0; i < m; ++i) {
					const dpl::Entry& en = L.ent[b + i];
					const double d = dpl::sqDist(q, Vec3{en.x, en.y, en.z});
					if (outc == m && dpl::sqDist(up, Vec3{en.x, en.y, en.z}) < 4 * best) orchard += 1;
					if (d < best) {
						if (i < 64) mask |= 1ull << i;
						if (outc == m) { outc = i; dOut = d; }
					}
				}
				// bit 31 of 'state' carries the query's parity (train/test fold); states < 2^31
				out.push_back(Rec{mask, s | (qi & 1 ? 0x80000000u : 0u), (u16)std::min<u32>(m, 65535), (u16)std::min<u32>(outc, 65535)});
				if (outc == m) break;
				best = dOut;
				node = L.ent[b + outc].node;
				s = L.ent[b + outc].next;
			}
			finalNode[qi] = node;
			finalState[qi] = s;
		}
	}
	std::vector<Rec> R;
	{
		size_t tot = 0;
		for (auto& v : per) tot += v.size();
		R.reserve(tot);
		for (auto& v : per) {
			R.insert(R.end(), v.begin(), v.end());
			std::vector<Rec>().swap(v);
		}
	}
	Report rep;
	rep.orchard = orchard / nq;
	for (const Rec& r : R) {
		const bool none = r.outcome == r.m;
		rep.cmp += none ? r.m : r.outcome + 1;
		if (none) rep.finalScan += r.m;
		else { rep.jumps += 1; rep.failed += r.outcome; }
	}
	rep.statesVisited = double(R.size()) / nq;
	// group by state (ignoring the parity bit), then by mask
	__gnu_parallel::sort(R.begin(), R.end(), [](const Rec& a, const Rec& b) {
		const u32 sa = a.state & 0x7FFFFFFFu, sb = b.state & 0x7FFFFFFFu;
		if (sa != sb) return sa < sb;
		if (a.mask != b.mask) return a.mask < b.mask;
		return a.state < b.state;
	});
	std::vector<size_t> runs{0};
	for (size_t i = 1; i <= R.size(); ++i)
		if (i == R.size() || (R[i].state & 0x7FFFFFFFu) != (R[i - 1].state & 0x7FFFFFFFu)) runs.push_back(i);
	double H = 0, greedy = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : H, greedy) num_threads(T)
	for (long long r = 0; r < (long long)runs.size() - 1; ++r) {
		const size_t b = runs[r], e = runs[r + 1];
		const int m = R[b].m;
		// H(outcome | state), all samples
		std::map<int, double> oc;
		for (size_t i = b; i < e; ++i) oc[R[i].outcome] += 1;
		std::vector<double> c;
		for (auto& kv : oc) c.push_back(kv.second);
		H += entropy(c) * double(e - b);
		// greedy tree on the even queries, evaluated on the odd ones (and vice versa, averaged)
		for (int fold = 0; fold < 2; ++fold) {
			std::vector<Sample> train;
			for (size_t i = b; i < e;) {
				size_t j = i;
				u32 cnt = 0;
				while (j < e && R[j].mask == R[i].mask) {
					if ((R[j].state >> 31) == (u32)fold) ++cnt;
					++j;
				}
				if (cnt) train.push_back(Sample{R[i].mask, R[i].outcome, cnt});
				i = j;
			}
			if (m > 64 || train.empty()) {
				for (size_t i = b; i < e; ++i)
					if ((R[i].state >> 31) != (u32)fold) greedy += R[i].outcome == R[i].m ? R[i].m : R[i].outcome + 1;
				continue;
			}
			std::vector<TreeNode> Tr;
			build(Tr, std::move(train), 0, m);
			for (size_t i = b; i < e; ++i)
				if ((R[i].state >> 31) != (u32)fold) greedy += evalCost(Tr, R[i].mask, R[i].outcome, m);
		}
	}
	rep.cmp /= nq;
	rep.jumps /= nq;
	rep.failed /= nq;
	rep.finalScan /= nq;
	rep.H = H / nq;
	rep.greedy = greedy / nq;
	// H(answer) and H(final state), plug-in + Miller-Madow
	auto ent = [&](std::vector<u32> v) {
		std::sort(v.begin(), v.end());
		std::vector<double> c;
		for (size_t i = 0; i < v.size();) {
			size_t j = i;
			while (j < v.size() && v[j] == v[i]) ++j;
			c.push_back(double(j - i));
			i = j;
		}
		return entropy(c) + (c.size() - 1) / (2.0 * v.size() * std::log(2.0));
	};
	rep.hAnswer = ent(finalNode);
	rep.hFinalState = ent(finalState);
	return rep;
}

// Oracle certification cost: for a query q in state s with outcome i, the fewest
// tests (greedy set cover, each test answered as q answers it) after which the LP says
// that only outcome i is possible; +1 if c_i itself was not among them (its distance is
// needed to continue).  A lower-bound-like reference for any certified test order.
template <class RegionFn>
static double oracleCost(const dpl::Dag& g, const dpl::LayerIndex& X, const std::vector<u32>& sn, RegionFn&& region,
                         const std::vector<Vec3>& Q, size_t nOracle) {
	using dpl::HalfSpace;
	const size_t nq = std::min(nOracle, Q.size());
	double total = 0;
#pragma omp parallel reduction(+ : total)
	{
		dpl::LP3 lp;
		std::vector<HalfSpace> reg, hs;
		std::vector<std::vector<HalfSpace>> piece;
		std::vector<int> lits;
#pragma omp for schedule(dynamic, 16)
		for (long long qi = 0; qi < (long long)nq; ++qi) {
			const Vec3& q = Q[qi];
			lp.seed(qi);
			u32 s = X.root;
			double best = dpl::sqDist(q, X.rootPt);
			for (;;) {
				const u32 b = X.beg[s], e = X.beg[s + 1];
				const int m = int(e - b);
				const u32 u = sn[s];
				std::vector<char> x(m);
				int outc = m;
				double dOut = 0;
				for (int i = 0; i < m; ++i) {
					const dpl::Entry& en = X.ent[b + i];
					const double d = dpl::sqDist(q, Vec3{en.x, en.y, en.z});
					x[i] = d < best;
					if (x[i] && outc == m) { outc = i; dOut = d; }
				}
				if (m > 0) {
					reg.clear();
					region(s, reg);
					piece.assign(m, {});
					for (int i = 0; i < m; ++i) dpl::appendPiece(g, X.edge[b + i], piece[i]);
					lits.clear();
					// feasible outcomes under region + lits; -1 == none
					auto feasible = [&](const std::vector<int>& L, std::vector<int>& F) {
						F.clear();
						hs.assign(reg.begin(), reg.end());
						int posLast = -1;
						std::vector<signed char> st(m, 0);
						for (int l : L) {
							const int j = std::abs(l) - 1;
							st[j] = l > 0 ? 1 : -1;
							if (l > 0) { dpl::appendBisector(g, X.ent[b + j].node, u, hs); posLast = posLast < 0 ? j : std::min(posLast, j); }
							else dpl::appendBisector(g, u, X.ent[b + j].node, hs);
						}
						const int base = (int)hs.size();
						for (int i = 0; i < m; ++i) {
							if (st[i] < 0 || (posLast >= 0 && i > posLast)) continue;
							std::vector<HalfSpace> h(hs.begin(), hs.end());
							h.insert(h.end(), piece[i].begin(), piece[i].end());
							double y[3];
							if (lp.feasible(h.data(), (int)h.size(), y)) F.push_back(i);
						}
						if (posLast < 0) {
							std::vector<HalfSpace> h(hs.begin(), hs.end());
							for (int i = 0; i < m; ++i)
								if (st[i] == 0) dpl::appendBisector(g, u, X.ent[b + i].node, h);
							double y[3];
							if ((int)h.size() == 0 || lp.feasible(h.data(), (int)h.size(), y)) F.push_back(-1);
						}
						(void)base;
					};
					std::vector<int> F, Fj;
					feasible(lits, F);
					const int target = outc == m ? -1 : outc;
					std::vector<char> used(m, 0);
					int tests = 0;
					while (!(F.size() == 1 && F[0] == target) && !F.empty()) {
						int bestJ = -1;
						size_t bestN = 1u << 30;
						for (int j = 0; j < m; ++j) {
							if (used[j]) continue;
							lits.push_back(x[j] ? j + 1 : -(j + 1));
							feasible(lits, Fj);
							lits.pop_back();
							if (Fj.size() < bestN) { bestN = Fj.size(); bestJ = j; }
						}
						if (bestJ < 0) break;
						used[bestJ] = 1;
						lits.push_back(x[bestJ] ? bestJ + 1 : -(bestJ + 1));
						feasible(lits, F);
						++tests;
					}
					if (target >= 0 && !used[target]) ++tests;
					total += tests;
				}
				if (outc == m) break;
				best = dOut;
				s = X.ent[b + outc].next;
			}
		}
	}
	return total / nq;
}

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <data.xyz|uniform:N> [options]\n", argv[0]); return 1; }
	std::string data = argv[1], orderS = "brio";
	int levels = 3;
	bool trie = false;
	size_t nq = 2000000, nOracle = 0;
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		auto val = [&]() { return std::string(argv[++i]); };
		if (a == "--order") orderS = val();
		else if (a == "--levels") levels = std::stoi(val());
		else if (a == "--trie") trie = true;
		else if (a == "--queries") nq = std::stoull(val());
		else if (a == "--oracle") nOracle = std::stoull(val());
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
	std::printf("data %s  n=%zu  order=%s  queries=%zu per set\n", data.c_str(), pts.size(), dpl::orderName(order), nq);
	double t0 = now();
	const dpl::Dag g = dpl::buildDag(pts, order);
	std::vector<dpl::LayerIndex> L;
	L.push_back(dpl::buildLevel0(g));
	for (int k = 1; k <= levels; ++k) L.push_back(dpl::buildNextLevel(g, L));
	dpl::LayerIndex Tr;
	if (trie) Tr = dpl::buildTrie(g, L[1]);
	std::printf("built in %.1fs; log2 n = %.2f\n", now() - t0, std::log2((double)g.n()));

	struct QSet {
		const char* name;
		std::vector<Vec3> q;
	};
	std::vector<QSet> sets;
	sets.push_back({"box2x", boxQueries(pts, nq, 0.5, 1)});
	sets.push_back({"near", nearQueries(pts, nq, 0.002, 2)});
	for (auto& S : sets) {
		std::printf("\n== %s\n%-8s %8s %7s %8s %8s %8s %9s %8s %8s %8s | %8s %8s\n", S.name, "index", "cmp", "jumps", "failed", "final",
		            "states", "H(o|s)", "greedy", "oracle", "orchard", "H(ans)", "H(end)");
		for (int k = 0; k <= levels + (trie ? 1 : 0); ++k) {
			const dpl::LayerIndex& X = k <= levels ? L[k] : Tr;
			const Report r = analyze(X, S.q, g.pts);
			double orc = -1;
			if (nOracle) {
				const bool isTrie = k > levels;
				const std::vector<u32> sn = isTrie ? dpl::trieStateNodes(Tr) : dpl::stateNodes(L, k);
				orc = isTrie ? oracleCost(g, X, sn, dpl::TrieRegion{g, Tr}, S.q, nOracle)
				             : oracleCost(g, X, sn, dpl::LevelRegion{g, L, k}, S.q, nOracle);
			}
			std::printf("%-8s %8.2f %7.2f %8.2f %8.2f %8.2f %9.2f %8.2f %8.2f %8.2f | %8.2f %8.2f\n",
			            k <= levels ? ("L" + std::to_string(k)).c_str() : "trie", r.cmp, r.jumps, r.failed, r.finalScan,
			            r.statesVisited, r.H, r.greedy, orc, r.orchard, r.hAnswer, r.hFinalState);
			std::fflush(stdout);
		}
	}
	return 0;
}
