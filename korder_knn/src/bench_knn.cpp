//
// bench_knn.cpp -- k-NN through order-k Voronoi diagrams (power diagram + DPNN)
// versus kd-tree, Delaunay best-first expansion and the successor-list scheme.
//
//   bench_knn <data.xyz | uniform:N | sphere:N> [options]
//     --kmax K        build the order-1..K diagrams (default 8)
//     --ks a,b,..     orders that are benchmarked (default 1,2,4,8; capped at K)
//     --dag-ks a,b,.. orders that get an order-k DPNN (default: same as --ks; order 1 always)
//     --queries N     queries per query set (default 1000000)
//     --check N       brute-force check of the first N queries (default 20000)
//     --threads T     OpenMP threads (building, brute force)
//     --sub N         random subsample of N points
//     --sizes-only    build the diagrams and report their sizes, no queries
//     --no-huge       no transparent huge pages
//     --sequential    sequential CGAL insertion for every order (default: TBB for
//                     the orders without a query DAG)
//     --cand-stats    (with --sizes-only) size of C(u) = union of the order-k cells
//                     that contain u, for every order
//     --exact-cand    also build Cx(u) = union of the order-k cells that meet Vor(u)
//                     (one LP per cell member) and benchmark NN + Cx(u)
//     --nn refined    exact NN for the NN + list methods from RefinedDPNN (BRIO, 2
//                     history layers, ../DPNN/src/dpnn_refined.h) instead of order-1 DPNN
//
#include "dpnn_refined.h"
#include "knn_baselines.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>

using namespace kov;

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
		} else {
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

// the query distributions of DPNN/src/bench_layered.cpp (same seeds)
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
	std::string name, dist, jumps, check, note;
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
static std::string num(double v, int prec = 2) {
	char b[32];
	std::snprintf(b, sizeof b, "%.*f", prec, v);
	return b;
}
static double mb(size_t b) { return b / 1048576.0; }

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <data.xyz|uniform:N|sphere:N> [options]\n", argv[0]); return 1; }
	std::string data = argv[1], ksArg, dagArg = "same", nnKind = "dpnn";
	int K = 8;
	size_t nq = 1000000, check = 20000, sub = 0;
	bool sizesOnly = false, huge = true, sequential = false, candStats = false, exactCand = false;
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		auto val = [&]() { return std::string(argv[++i]); };
		if (a == "--kmax") K = std::stoi(val());
		else if (a == "--ks") ksArg = val();
		else if (a == "--dag-ks") dagArg = val();
		else if (a == "--nn") nnKind = val();
		else if (a == "--queries") nq = std::stoull(val());
		else if (a == "--check") check = std::stoull(val());
		else if (a == "--sub") sub = std::stoull(val());
		else if (a == "--sizes-only") sizesOnly = true;
		else if (a == "--no-huge") huge = false;
		else if (a == "--sequential") sequential = true;
		else if (a == "--cand-stats") candStats = true;
		else if (a == "--exact-cand") exactCand = true;
#ifdef _OPENMP
		else if (a == "--threads") omp_set_num_threads(std::stoi(val()));
#endif
		else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
	}
	std::set<int> ks;
	if (ksArg.empty()) ks = {1, 2, 4, 8};
	else {
		std::stringstream ss(ksArg);
		std::string t;
		while (std::getline(ss, t, ',')) ks.insert(std::stoi(t));
	}
	for (auto it = ks.begin(); it != ks.end();) it = *it > K ? ks.erase(it) : std::next(it);
	ks.insert(1);  // order 1 == DPNN: the exact NN for the Delaunay expansion
	std::set<int> dagKs = ks;
	if (dagArg != "same") {
		dagKs.clear();
		std::stringstream ss(dagArg);
		std::string t;
		while (std::getline(ss, t, ',')) dagKs.insert(std::stoi(t));
		dagKs.insert(1);
	}

	std::vector<Vec3> raw;
	if (data.rfind("uniform:", 0) == 0) raw = synth("uniform", std::stoull(data.substr(8)), 7);
	else if (data.rfind("sphere:", 0) == 0) raw = synth("sphere", std::stoull(data.substr(7)), 7);
	else raw = readXYZ(data);
	if (sub && sub < raw.size()) {
		std::mt19937_64 gen(99);
		std::shuffle(raw.begin(), raw.end(), gen);
		raw.resize(sub);
	}
	std::printf("data %s  n=%zu  kmax=%d  threads=%d\n", data.c_str(), raw.size(), K, dpl::detail::numThreads());

	// ------------------------------------------------------------ points
	double t0 = now();
	const dpl::Dag g = dpl::buildDag(raw, dpl::InsertionOrder::Brio);  // dedup + DPNN DAG (successor lists)
	const double tDag = now() - t0;
	// every method works on the same translated copy (bbox centre -> origin), so that
	// the centroids of the k-sets are computed with small coordinates
	Vec3 lo, hi;
	bbox(g.pts, lo, hi);
	const Vec3 ctr{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
	std::vector<Vec3> P(g.n());
	for (size_t i = 0; i < g.n(); ++i) P[i] = Vec3{g.pts[i].x - ctr.x, g.pts[i].y - ctr.y, g.pts[i].z - ctr.z};
	const size_t n = P.size();
	std::printf("unique points %zu, DPNN DAG %.2fs\n", n, tDag);

	// ------------------------------------------------------------ order-k diagrams
	std::printf("\n%5s %12s %8s %12s %8s %8s %9s %9s %9s %11s %11s\n", "order", "cells", "/n", "tri edges", "hidden",
	            "removed", "tri s", "next s", "DAG pa", "DAG MB", "sets MB");
	std::map<int, std::unique_ptr<PowerDpnn>> dpnn;
	std::map<int, Level> levels;  // member lists of the benchmarked orders
	std::unique_ptr<DelaunayKnn> del;
	std::map<int, std::unique_ptr<CandKnn>> candX;  // exact Cx(u) lists
	Level L = levelOne(P);
	double tBuild = 0;
	for (int j = 1; j <= K; ++j) {
		const bool wantDag = dagKs.count(j) && !sizesOnly;
		PowerTri T;
#ifdef CGAL_LINKED_WITH_TBB
		if (!wantDag && j > 1 && !sequential) T = triangulateParallel(L);
		else
#endif
			T = triangulate(L, wantDag || (sizesOnly && j == 1), j < K || j == 1 || (exactCand && ks.count(j)));
		double tq = 0;
		if (wantDag || (sizesOnly && j == 1)) {
			const double s = now();
			if (wantDag) dpnn[j].reset(new PowerDpnn(L, T, huge));
			tq = now() - s;
		}
		if (j == 1 && !sizesOnly) del.reset(new DelaunayKnn(P, T.edges));
		if (exactCand && ks.count(j) && j >= 2) {
			CandLists C = exactCandidates(P, L, T.edges);
			std::vector<u32> len(n);
			for (size_t u = 0; u < n; ++u) len[u] = C.beg[u + 1] - C.beg[u];
			std::sort(len.begin(), len.end());
			std::printf("      Cx(u) for k=%d: mean %.1f (%.2f k), median %u, p90 %u, p99 %u, max %u; %llu LPs, %.1fs\n", j,
			            double(C.list.size()) / n, double(C.list.size()) / n / j, len[n / 2], len[n * 9 / 10], len[n * 99 / 100],
			            len.back(), (unsigned long long)C.lps, C.seconds);
			if (!sizesOnly) candX[j].reset(new CandKnn(P, j, C.beg, C.list));
		}
		NextStats ns;
		Level N;
		if (j < K) {
			N = nextLevel(L, T.edges, &ns);
			const double s = now();
			computeSites(N, P);
			ns.seconds += now() - s;
		}
		const double paAvg = T.paBeg.empty() ? 0.0 : double(T.pa.size()) / L.M;
		std::printf("%5d %12zu %8.2f %12zu %8zu %8zu %9.2f %9.2f %9s %11s %11.1f\n", j, L.M, double(L.M) / n, T.edges.size(),
		            T.hiddenAtInsert, T.removedLater, T.seconds, ns.seconds, paAvg > 0 ? num(paAvg).c_str() : "-",
		            dpnn.count(j) ? num(mb(dpnn[j]->bytes()), 1).c_str() : "-", mb(L.mem.size() * 4));
		std::fflush(stdout);
		tBuild += T.seconds + ns.seconds + tq;
		if (candStats && j >= 2) {
			// C(u) = union of the members of all order-j cells that contain u: the exact
			// candidate set of every query whose nearest point is u
			const double s0 = now();
			std::vector<u64> pr;
			pr.reserve(L.M * (size_t)j * (j - 1));
			for (size_t t = 0; t < L.M; ++t) {
				const u32* m = L.members(t);
				for (int a = 0; a < j; ++a)
					for (int b = 0; b < j; ++b)
						if (a != b) pr.push_back((u64)m[a] << 32 | m[b]);
			}
			__gnu_parallel::sort(pr.begin(), pr.end());
			pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
			std::vector<u32> cnt(n, 0);
			for (u64 x : pr) cnt[x >> 32]++;
			std::vector<u32> srt(cnt);
			std::sort(srt.begin(), srt.end());
			std::printf("      C(u) for k=%d: mean %.1f (%.2f k), median %u, p90 %u, p99 %u, max %u; total %.1f MB as 32-byte entries (%.1fs)\n", j,
			            double(pr.size()) / n + 1, (double(pr.size()) / n + 1) / j, srt[n / 2] + 1, srt[n * 9 / 10] + 1, srt[n * 99 / 100] + 1,
			            srt.back() + 1, mb((pr.size() + n) * 32), now() - s0);
			std::fflush(stdout);
		}
		if (ks.count(j) && !sizesOnly) {
			Level keep;
			keep.k = L.k;
			keep.M = L.M;
			keep.mem = std::move(L.mem);
			levels[j] = std::move(keep);
		}
		L = std::move(N);
	}
	std::printf("order-1..%d build: %.1fs (orders with a DAG: sequential BRIO insertion; others: %s)\n", K, tBuild,
#ifdef CGAL_LINKED_WITH_TBB
	            sequential ? "sequential" : "CGAL parallel insertion (TBB)"
#else
	            "sequential"
#endif
	);
	if (sizesOnly) return 0;

	std::map<int, std::unique_ptr<CandKnn>> cand;
	for (int k : ks) {
		if (k < 2) continue;
		const double s = now();
		cand[k].reset(new CandKnn(P, levels[k]));
		std::printf("candidate lists C(u) for k=%d: avg %.1f entries, %.1f MB, %.2fs\n", k, cand[k]->avgLen(), mb(cand[k]->bytes()),
		            now() - s);
	}

	t0 = now();
	KdKnn kd(P);
	const double tKd = now() - t0;
	SuccessorKnn succ(g, P);
	std::printf("kd-tree (nanoflann, leaf 10): %.2fs, %.1f MB | Delaunay adjacency: %.1f MB (avg degree %.2f) | successor lists: %.1f MB\n",
	            tKd, mb(kd.bytes()), mb(del->bytes()), del->avgDegree(), mb(succ.bytes()));

	// ------------------------------------------------------------ queries
	struct QSet {
		const char* name;
		std::vector<Vec3> q;
	};
	std::vector<QSet> sets;
	sets.push_back({"box2x", boxQueries(g.pts, nq, 0.5, 1)});
	sets.push_back({"near", nearQueries(g.pts, nq, 0.002, 2)});
	for (auto& S : sets)
		for (auto& q : S.q) q = Vec3{q.x - ctr.x, q.y - ctr.y, q.z - ctr.z};

	bool allOk = true;
	const PowerDpnn& nn1 = *dpnn[1];
	const std::vector<u32>& mem1 = levels[1].mem;
	std::unique_ptr<dpl::RefinedDPNN> refined;
	if (nnKind == "refined") {
		const double s = now();
		refined.reset(new dpl::RefinedDPNN(P));
		std::printf("NN for the list methods: RefinedDPNN (BRIO, 2 layers) %.1fs, %.1f MB\n", now() - s, mb(refined->bytes()));
	} else {
		std::printf("NN for the list methods: order-1 DPNN (BRIO, no history layers), %.1f MB\n", mb(nn1.bytes()));
	}
	// exact nearest point (index into P), single and batch
	auto nnOne = [&](const Vec3& q) -> u32 { return refined ? refined->nearest(q) : mem1[nn1.query(q)]; };
	auto nnBatch = [&](const Vec3* Q, size_t m, u32* out) {
		if (refined) refined->nearestBatch(Q, m, out);
		else {
			nn1.queryBatch<16>(Q, m, out);
			for (size_t i = 0; i < m; ++i) out[i] = mem1[out[i]];
		}
	};
	for (auto& S : sets) {
		const size_t m = S.q.size();
		const size_t mc = std::min(check, m);
		// nearest neighbour of every query (input of the Delaunay expansion)
		std::vector<u32> nn(m);
		for (size_t i = 0; i < m; ++i) nn[i] = nnOne(S.q[i]);
		for (int k : ks) {
			// brute-force reference (squared distances, ascending)
			std::vector<double> ref(mc * k);
			{
				std::vector<u32> tmp(mc * k);
#pragma omp parallel for schedule(dynamic, 16)
				for (long long i = 0; i < (long long)mc; ++i) bruteKnn(P, S.q[i], k, &ref[i * k], &tmp[i * k]);
			}
			std::vector<u32> outId(m * (size_t)k);
			std::vector<double> outD(m * (size_t)k);
			std::vector<u32> cell(m);
			SmallSet vis;
			u64 sink = 0;
			// compares the first mc answers (as sorted squared distances) with the reference
			auto verify = [&](bool fromCells) {
				size_t exact = 0, close = 0;
				std::vector<double> d(k);
				for (size_t i = 0; i < mc; ++i) {
					if (fromCells) {
						const u32* M = levels[k].members(cell[i]);
						for (int t = 0; t < k; ++t) d[t] = dpl::sqDist(P[M[t]], S.q[i]);
						std::sort(d.begin(), d.end());
					} else {
						for (int t = 0; t < k; ++t) d[t] = outD[i * k + t];
					}
					bool eq = true, cl = true;
					for (int t = 0; t < k; ++t) {
						const double r = ref[i * k + t];
						if (d[t] != r) eq = false;
						if (std::fabs(d[t] - r) > 1e-12 * std::max(r, 1e-300) + 1e-300) cl = false;
					}
					exact += eq;
					close += cl;
				}
				if (close != mc) allOk = false;
				return std::to_string(exact) + (close != exact ? " (+" + std::to_string(close - exact) + " within 1e-12)" : "");
			};
			std::vector<Method> ms;
			{
				for (size_t i = 0; i < m; ++i) kd.query(S.q[i], k, &outId[i * k], &outD[i * k]);
				ms.push_back({"kd-tree (nanoflann)", "", "", verify(false), "",
				              [&] { for (size_t i = 0; i < m; ++i) kd.query(S.q[i], k, &outId[i * k], &outD[i * k]); }});
			}
			{
				KnnStats st;
				for (size_t i = 0; i < m; ++i) del->query(S.q[i], nn[i], k, &outId[i * k], &outD[i * k], vis, &st);
				ms.push_back({"Delaunay best-first", num(double(st.dist) / m), "", verify(false), "  (+ the NN query)",
				              [&] {
					              for (size_t i = 0; i < m; ++i) del->query(S.q[i], nnOne(S.q[i]), k, &outId[i * k], &outD[i * k], vis);
				              }});
			}
			{
				KnnStats st;
				for (size_t i = 0; i < m; ++i) succ.query(S.q[i], k, &outId[i * k], &outD[i * k], vis, &st);
				ms.push_back({"successor lists", num(double(st.dist) / m), "", verify(false), "",
				              [&] { for (size_t i = 0; i < m; ++i) succ.query(S.q[i], k, &outId[i * k], &outD[i * k], vis); }});
			}
			if (cand.count(k)) {
				const CandKnn& C = *cand[k];
				KnnStats st;
				for (size_t i = 0; i < m; ++i) C.query(S.q[i], nn[i], k, &outId[i * k], &outD[i * k], &st);
				ms.push_back({"NN + C(u) scan", num(double(st.dist) / m), "", verify(false), "  (+ the NN query)",
				              [&] {
					              for (size_t i = 0; i < m; ++i) C.query(S.q[i], nnOne(S.q[i]), k, &outId[i * k], &outD[i * k]);
				              }});
				ms.push_back({"NN batch16 + C(u) scan", "", "", "", "",
				              [&] {
					              nnBatch(S.q.data(), m, cell.data());
					              for (size_t i = 0; i < m; ++i) C.query(S.q[i], cell[i], k, &outId[i * k], &outD[i * k]);
				              }});
			}
			if (candX.count(k)) {
				const CandKnn& C = *candX[k];
				KnnStats st;
				for (size_t i = 0; i < m; ++i) C.query(S.q[i], nn[i], k, &outId[i * k], &outD[i * k], &st);
				ms.push_back({"NN + Cx(u) scan", num(double(st.dist) / m), "", verify(false), "  (+ the NN query)",
				              [&] {
					              for (size_t i = 0; i < m; ++i) C.query(S.q[i], nnOne(S.q[i]), k, &outId[i * k], &outD[i * k]);
				              }});
				ms.push_back({"NN batch16 + Cx(u) scan", "", "", "", "",
				              [&] {
					              nnBatch(S.q.data(), m, cell.data());
					              for (size_t i = 0; i < m; ++i) C.query(S.q[i], cell[i], k, &outId[i * k], &outD[i * k]);
				              }});
				{
					std::vector<u32> nk(m);
					std::vector<double> dk(m);
					for (size_t i = 0; i < m; ++i) {
						kd.query(S.q[i], 1, &nk[i], &dk[i]);
						C.query(S.q[i], nk[i], k, &outId[i * k], &outD[i * k]);
					}
					ms.push_back({"kd-tree NN + Cx(u) scan", "", "", verify(false), "",
					              [&] {
						              for (size_t i = 0; i < m; ++i) {
							              u32 a;
							              double d;
							              kd.query(S.q[i], 1, &a, &d);
							              C.query(S.q[i], a, k, &outId[i * k], &outD[i * k]);
						              }
					              }});
				}
			}
			if (dpnn.count(k)) {
				const PowerDpnn& D = *dpnn[k];
				const Level& Lk = levels[k];
				dpl::QueryStats st;
				for (size_t i = 0; i < m; ++i) cell[i] = D.query(S.q[i], st);
				ms.push_back({"order-k DPNN, set", num(double(st.comparisons) / m), num(double(st.jumps) / m, 1), verify(true), "",
				              [&] {
					              for (size_t i = 0; i < m; ++i) {
						              const u32* M = Lk.members(D.query(S.q[i]));
						              std::copy(M, M + k, &outId[i * k]);
					              }
				              }});
				ms.push_back({"order-k DPNN, sorted", "", "", "", "",
				              [&] {
					              for (size_t i = 0; i < m; ++i) {
						              const u32* M = Lk.members(D.query(S.q[i]));
						              Cand c[256];
						              int sz = 0;
						              for (int t = 0; t < k; ++t) candInsert(c, sz, k, dpl::sqDist(P[M[t]], S.q[i]), M[t]);
						              for (int t = 0; t < k; ++t) { outId[i * k + t] = c[t].id; outD[i * k + t] = c[t].d; }
					              }
				              }});
				D.queryBatch<16>(S.q.data(), m, cell.data());
				ms.push_back({"order-k DPNN batch16, set", "", "", verify(true), "",
				              [&] {
					              D.queryBatch<16>(S.q.data(), m, cell.data());
					              for (size_t i = 0; i < m; ++i) {
						              const u32* M = Lk.members(cell[i]);
						              std::copy(M, M + k, &outId[i * k]);
					              }
				              }});
			}
			timeInterleaved(ms, m, 5);
			for (size_t i = 0; i < m; i += 997) sink += outId[i];
			std::printf("\n== %s (%zu queries), k = %d   [order-%d diagram: %zu cells, DAG %s MB + sets %.1f MB; C(u) %s MB]\n", S.name,
			            m, k, k, levels[k].M, dpnn.count(k) ? num(mb(dpnn[k]->bytes()), 1).c_str() : "-", mb(levels[k].mem.size() * 4),
			            cand.count(k) ? num(mb(cand[k]->bytes()), 1).c_str() : "-");
			std::printf("%-28s %10s %8s %10s   %s\n", "method", "dist/query", "jumps", "us/query", ("exact vs brute force (of " + std::to_string(mc) + ")").c_str());
			for (auto& M : ms)
				std::printf("%-28s %10s %8s %10.4f   %s%s\n", M.name.c_str(), M.dist.c_str(), M.jumps.c_str(), M.us, M.check.c_str(),
				            M.note.c_str());
			if (sink == 42) std::printf(" ");
			std::fflush(stdout);
		}
	}
	std::printf("\n%s\n", allOk ? "ALL CHECKS PASSED" : "CHECK FAILURES");
	return allOk ? 0 : 2;
}
