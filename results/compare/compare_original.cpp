// For one dataset: end-to-end build times (original dpnn.cpp with BRIO / FPS order,
// RefinedDPNN configurations, the all-layer trie), the original's query time with
// FPS order, and the entropy of the answers of the bench's two query distributions.
#include <Eigen/Dense>
typedef Eigen::Vector3d Point;
#include "dpnn.cpp"
#include "dpnn_refined.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
using namespace dpl;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void bbox(const std::vector<Vec3>& pts, Vec3& lo, Vec3& hi) {
	lo = hi = pts[0];
	for (auto& p : pts) {
		lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
		hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
	}
}
// identical to bench_layered.cpp
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

int main(int argc, char** argv) {
	const std::string path = argv[1];
	const bool big = argc > 2 && std::string(argv[2]) == "big";  // Lucy: skip full 3 layers and the trie
	std::vector<Vec3> pts;
	{ std::ifstream f(path); double x, y, z; while (f >> x >> y >> z) pts.push_back(Vec3{x, y, z}); }
	const size_t n = pts.size();
	std::printf("data %s n=%zu log2(n)=%.3f H_n=%.3f threads=%d\n", path.c_str(), n, std::log2(double(n)),
	            std::log(double(n)) + 0.5772156649, dpl::detail::numThreads());
	std::fflush(stdout);

	// ---- original dpnn.cpp: build (BRIO, FPS) and query time (FPS; BRIO is in the bench)
	std::vector<Point> P;
	for (auto& p : pts) P.emplace_back(p.x, p.y, p.z);
	const std::vector<Vec3> qb = boxQueries(pts, 1000000, 0.5, 1), qn = nearQueries(pts, 1000000, 0.002, 2);
	RefinedOptions exactOpt;
	exactOpt.levels = 1;
	RefinedDPNN exact(pts, exactOpt);  // reference answers
	for (bool fps : {false, true}) {
		double t = now();
		dpnn* o = new dpnn(P, fps);
		const double b = now() - t;
		std::printf("RESULT original %s build %.2f s\n", fps ? "FPS " : "BRIO", b);
		for (int set = 0; set < 2; ++set) {
			const std::vector<Vec3>& Q = set ? qn : qb;
			std::vector<Point> QP;
			for (auto& q : Q) QP.emplace_back(q.x, q.y, q.z);
			double best = 1e30;
			u64 sink = 0;
			for (int r = 0; r < 5; ++r) {
				t = now();
				for (auto& q : QP) sink += o->getNearestNeighbor(q);
				best = std::min(best, (now() - t) / Q.size() * 1e6);
			}
			size_t bad = 0;
			for (size_t i = 0; i < Q.size(); ++i)
				bad += sqDist(pts[o->getNearestNeighbor(QP[i])], Q[i]) != sqDist(pts[exact.nearest(Q[i])], Q[i]);
			std::printf("RESULT original %s query %s %.4f us (min of 5), wrong %zu  [%llu]\n", fps ? "FPS " : "BRIO",
			            set ? "near" : "box ", best, bad, (unsigned long long)(sink & 1));
		}
		delete o;
		std::fflush(stdout);
	}

	// ---- ours, end to end (RefinedDPNN: DAG + levels + flat index)
	struct Cfg { const char* name; InsertionOrder o; int levels; u32 coarse; bool f32; bool bigOk; };
	const Cfg cfgs[] = {
	    {"BRIO 0 layers", InsertionOrder::Brio, 0, 0, false, true},
	    {"BRIO 1 layer", InsertionOrder::Brio, 1, 0, false, true},
	    {"BRIO 2 layers", InsertionOrder::Brio, 2, 0, false, true},
	    {"BRIO 3 layers", InsertionOrder::Brio, 3, 0, false, false},
	    {"BRIO 3 layers coarse16", InsertionOrder::Brio, 3, 16, false, true},
	    {"FPS 1 layer", InsertionOrder::Fps, 1, 0, false, true},
	    {"FPS 2 layers", InsertionOrder::Fps, 2, 0, false, true},
	    {"FPS 2 layers float", InsertionOrder::Fps, 2, 0, true, true},
	    {"FPS 3 layers", InsertionOrder::Fps, 3, 0, false, false},
	};
	for (const Cfg& c : cfgs) {
		if (big && !c.bigOk) continue;
		RefinedOptions opt;
		opt.order = c.o;
		opt.levels = c.levels;
		opt.coarseDivisor = c.coarse;
		opt.floatCoords = c.f32;
		const double t = now();
		RefinedDPNN nn(pts, opt);
		std::printf("RESULT ours %-24s build %.2f s, %.1f MB\n", c.name, now() - t, nn.bytes() / 1048576.0);
		std::fflush(stdout);
	}
	if (!big) {
		const double t = now();
		Dag g = buildDag(pts, InsertionOrder::Brio);
		std::vector<LayerIndex> L{buildLevel0(g)};
		L.push_back(buildNextLevel(g, L));
		LayerIndex Tr = buildTrie(g, L[1]);
		L.clear();
		FlatIndex f(Tr, trieStateNodes(Tr), true);
		std::printf("RESULT ours %-24s build %.2f s, %.1f MB\n", "BRIO all layers (trie)", now() - t, f.bytes() / 1048576.0);
		std::fflush(stdout);
	}

	// ---- entropy of the answer distribution (10M queries per distribution)
	for (int set = 0; set < 2; ++set) {
		const size_t m = 10000000;
		const std::vector<Vec3> Q = set ? nearQueries(pts, m, 0.002, 12) : boxQueries(pts, m, 0.5, 11);
		std::vector<u32> ans(m);
		const size_t chunk = 1 << 16;
#pragma omp parallel for schedule(dynamic, 1)
		for (long long c = 0; c < (long long)((m + chunk - 1) / chunk); ++c) {
			const size_t b = c * chunk, e = std::min(m, b + chunk);
			exact.nearestBatch(Q.data() + b, e - b, ans.data() + b);
		}
		std::vector<u32> cnt(n, 0);
		for (u32 a : ans) cnt[a]++;
		double H = 0;
		size_t K = 0;
		for (u32 c : cnt)
			if (c) { const double p = double(c) / m; H -= p * std::log2(p); ++K; }
		const double Hmm = H + (K - 1) / (2.0 * m * std::log(2.0));  // Miller-Madow bias correction
		std::printf("RESULT entropy %s: H = %.3f bits (Miller-Madow %.3f), %zu distinct answers of %zu points\n",
		            set ? "near" : "box ", H, Hmm, K, n);
		std::fflush(stdout);
	}
}
