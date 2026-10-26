//
// dpnn_dtree.h -- a decision tree instead of a sequential scan inside every state.
//
// In a state (of any level k, or of the trie) with current point u and candidate
// list c_1..c_m (insertion order), DPNN moves to the FIRST c_i closer than u, or
// stays.  With K the state's history region (the pieces Q(e) of its edges), the
// outcome regions are
//
//     R_i    = K ∩ Q(u -> c_i)                 (Q(u -> c_i) ⊆ Cell(u; V_{t_i - 1}): every
//                                                earlier candidate is farther than u)
//     R_none = K ∩ ⋂_i { |x-u| <= |x-c_i| }
//
// and they partition K.  The sequential scan pays i tests for outcome i (m for none),
// which is optimal by logic alone: outcome i needs c_1..c_{i-1} certified negative.
// Geometry certifies more: inside the region known after a few answers, often only
// one outcome is still possible.  So every state gets a binary tree over the same
// tests "|q-c_j| < |q-u|" in any order; a node becomes a leaf as soon as the LP says
// that only one outcome region meets the known region (K plus the answers so far,
// every half-space relaxed by kRelax, so no outcome is ever wrongly ruled out and the
// query follows exactly the DPNN path).  Which test to make next is chosen by the
// information gain on the outcomes of training queries that reach the node; nodes no
// training query reaches fall back to the remaining candidates in insertion order.
// Trees are capped at budget * m internal nodes per state, then continue as such
// sequential chains (always valid: a positive c_j after negatives for all earlier
// candidates is outcome j).
//
// Query: distances are compared with the current point's distance only.  The distance
// of the last positive test is kept ("pending"); it is the distance of the outcome's
// candidate whenever that candidate was tested (a later positive test can only be an
// earlier candidate, which changes the outcome), and becomes the current distance when
// the tree hands over to the next state.  If the outcome was certified without testing
// its candidate, a FORCE node computes that one distance.
//
#pragma once
#include "dpnn_flat.h"

#include <parallel/algorithm>

namespace dpl {

struct alignas(32) DtEntry {
	double x, y, z;
	u32 a, b;
};
static_assert(sizeof(DtEntry) == 32, "DtEntry must be 32 bytes");

struct DtreeStats {
	u64 lpCalls = 0, states = 0, internal = 0, forced = 0, chainStates = 0, trainSamples = 0;
	double seconds = 0;
};

class DtreeIndex {
public:
	// a: bits 31..30 type, bit 29 TRANS (current := pending when this edge is taken),
	//    bits 28..0 target of the positive edge (TEST) / of the next state (FORCE)
	// b: TEST: TRANS bit + target of the negative edge; END: the DAG node to return
	static constexpr u32 kTest = 0, kForce = 1, kEnd = 2;
	static constexpr u32 kTrans = 1u << 29, kOff = (1u << 29) - 1;

	size_t bytes() const { return ent_.bytes(); }
	size_t numEntries() const { return ent_.size(); }

	// The answers of informative tests are ~50/50, so the loop selects instead of
	// branching (only END / FORCE, which are rare, take a branch).
	u32 query(const Vec3& q) const {
		const DtEntry* E = ent_.data();
		u32 idx = root_;
		double best = sqDist(q, rootPt_), pending = best;
		for (;;) {
			const DtEntry& e = E[idx];
			const u32 a = e.a;
			const double dx = e.x - q.x, dy = e.y - q.y, dz = e.z - q.z;
			const double d = dx * dx + dy * dy + dz * dz;
			if (__builtin_expect((a >> 30) != kTest, 0)) {
				if ((a >> 30) == kEnd) return e.b;
				best = d;
				idx = a & kOff;
				continue;
			}
			const bool pos = d < best;
			pending = pos ? d : pending;
			const u32 nx = pos ? a : e.b;
			best = (nx & kTrans) ? pending : best;
			idx = nx & kOff;
		}
	}

	u32 query(const Vec3& q, QueryStats& st) const {
		const DtEntry* E = ent_.data();
		u32 idx = root_;
		double best = sqDist(q, rootPt_), pending = best;
		for (;;) {
			const DtEntry& e = E[idx];
			const u32 type = e.a >> 30;
			if (type == kEnd) return e.b;
			st.comparisons++;
			const double d = sqDist(q, Vec3{e.x, e.y, e.z});
			if (type == kForce) {
				st.jumps++;
				best = d;
				idx = e.a & kOff;
				continue;
			}
			u32 nx;
			if (d < best) {
				pending = d;
				nx = e.a;
			} else {
				nx = e.b;
			}
			if (nx & kTrans) {
				best = pending;
				st.jumps++;
			}
			idx = nx & kOff;
		}
	}

	// Kq queries interleaved one tree node at a time: the next node of a query depends
	// on its last comparison, so a lone query is latency-bound; Kq independent chains
	// keep the core busy.  A query's next state is prefetched when it moves there.
	template <int Kq = 16>
	void queryBatch(const Vec3* Q, size_t m, u32* out) const {
		const DtEntry* E = ent_.data();
		struct Slot {
			Vec3 q;
			double best, pending;
			u32 idx;
			size_t i;
		};
		Slot sl[Kq];
		int active = 0;
		size_t nq = 0;
		auto start = [&](Slot& S, size_t i) {
			const double d0 = sqDist(Q[i], rootPt_);
			S = Slot{Q[i], d0, d0, root_, i};
		};
		for (; active < Kq && nq < m; ++active, ++nq) start(sl[active], nq);
		while (active) {
			for (int k = 0; k < active;) {
				Slot& S = sl[k];
				const DtEntry& e = E[S.idx];
				const u32 a = e.a;
				const double dx = e.x - S.q.x, dy = e.y - S.q.y, dz = e.z - S.q.z;
				const double d = dx * dx + dy * dy + dz * dz;
				if (__builtin_expect((a >> 30) != kTest, 0)) {
					if ((a >> 30) == kEnd) {
						out[S.i] = e.b;
						if (nq < m) {
							start(S, nq++);
							++k;
						} else {
							sl[k] = sl[--active];
						}
						continue;
					}
					S.best = d;
					S.idx = a & kOff;
					__builtin_prefetch(E + S.idx);
					++k;
					continue;
				}
				const bool pos = d < S.best;
				S.pending = pos ? d : S.pending;
				const u32 nx = pos ? a : e.b;
				S.best = (nx & kTrans) ? S.pending : S.best;
				S.idx = nx & kOff;
				if (nx & kTrans) __builtin_prefetch(E + S.idx);
				++k;
			}
		}
	}

	// ------------------------------------------------------------------ build
	// X: levels[k] or the trie (with its build-time edge[] metadata); stateNode[s]:
	// the current point of state s; region(s, hs): appends the state's history
	// half-spaces; train: queries whose paths choose the tests.
	template <class RegionFn>
	DtreeIndex(const Dag& g, const LayerIndex& X, const std::vector<u32>& stateNode, RegionFn&& region,
	           const std::vector<Vec3>& train, int budget = 4, DtreeStats* st = nullptr, bool hugePages = true, int crit = 2) {
		const auto t0 = std::chrono::steady_clock::now();
		const size_t S = X.numStates();
		// 1. training: per state, the bit vectors x_k = [|q-c_k| < |q-u|] of the queries reaching it
		std::vector<u64> recState;  // (state << 32) | index into recMask
		std::vector<u64> rec;       // (state, mask) pairs, sorted
		{
			const int T = detail::numThreads();
			std::vector<std::vector<std::pair<u32, u64>>> per(T);
#pragma omp parallel num_threads(T)
			{
#ifdef _OPENMP
				auto& out = per[omp_get_thread_num()];
#else
				auto& out = per[0];
#endif
#pragma omp for schedule(static, 4096)
				for (long long qi = 0; qi < (long long)train.size(); ++qi) {
					const Vec3& q = train[qi];
					u32 s = X.root;
					double best = sqDist(q, X.rootPt);
					for (;;) {
						const u32 b = X.beg[s], e = X.beg[s + 1];
						u64 mask = 0;
						u32 outc = e - b;
						double dOut = 0;
						for (u32 i = b; i < e; ++i) {
							const Entry& en = X.ent[i];
							const double d = sqDist(q, Vec3{en.x, en.y, en.z});
							if (d < best) {
								if (i - b < 64) mask |= 1ull << (i - b);
								if (outc == e - b) { outc = i - b; dOut = d; }
							}
						}
						out.emplace_back(s, mask);
						if (outc == e - b) break;
						best = dOut;
						s = X.ent[b + outc].next;
					}
				}
			}
			std::vector<std::pair<u32, u64>> all;
			for (auto& v : per) {
				all.insert(all.end(), v.begin(), v.end());
				std::vector<std::pair<u32, u64>>().swap(v);
			}
			__gnu_parallel::sort(all.begin(), all.end());
			if (st) st->trainSamples = all.size();
			// compress to (mask, count) per state
			sampBeg_.assign(S + 1, 0);
			for (size_t i = 0; i < all.size();) {
				size_t j = i;
				while (j < all.size() && all[j] == all[i]) ++j;
				samp_.push_back(Samp{all[i].second, u32(j - i)});
				sampBeg_[all[i].first + 1]++;
				i = j;
			}
			for (size_t s = 0; s < S; ++s) sampBeg_[s + 1] += sampBeg_[s];
		}

		// 2. one tree per state (serialized preorder), in chunks
		std::vector<u64> treeBeg(S + 1, 0);
		std::vector<BNode> nodes;
		u64 lpCalls = 0, internal = 0, chainStates = 0;
		const size_t chunk = 1 << 18;
		std::vector<std::vector<BNode>> tmp;
		for (size_t c0 = 0; c0 < S; c0 += chunk) {
			const long long cm = (long long)(std::min(S, c0 + chunk) - c0);
			tmp.assign(cm, {});
#pragma omp parallel reduction(+ : lpCalls, internal, chainStates)
			{
				Builder B(g, X, stateNode, budget);
				B.crit = crit;
#pragma omp for schedule(dynamic, 64)
				for (long long i = 0; i < cm; ++i) {
					const u32 s = (u32)(c0 + i);
					B.lp.seed(s);
					B.region.clear();
					region(s, B.region);
					B.build(s, samp_.data() + sampBeg_[s], samp_.data() + sampBeg_[s + 1], tmp[i]);
				}
				lpCalls += B.lpCalls;
				internal += B.internal;
				chainStates += B.chainStates;
			}
			for (long long i = 0; i < cm; ++i) {
				treeBeg[c0 + i + 1] = treeBeg[c0 + i] + tmp[i].size();
				nodes.insert(nodes.end(), tmp[i].begin(), tmp[i].end());
			}
		}
		std::vector<std::vector<BNode>>().swap(tmp);
		std::vector<Samp>().swap(samp_);
		std::vector<u64>().swap(sampBeg_);

		// 3. layout: per state [TEST entries in preorder][FORCE entries][END]
		std::vector<u64> base(S + 1, 0), rootOff(S);
		u64 forcedTotal = 0;
#pragma omp parallel for schedule(static, 4096) reduction(+ : forcedTotal)
		for (long long s = 0; s < (long long)S; ++s) {
			const BNode* T = nodes.data() + treeBeg[s];
			const int n = int(treeBeg[s + 1] - treeBeg[s]);
			int nInt = 0, nForce = 0;
			bool end = false;
			u64 forcedSeen[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // candidates < 512 with a FORCE entry
			for (int i = 0; i < n; ++i) {
				if (T[i].test >= 0) ++nInt;
				else if (T[i].leaf < 0) end = true;
				else if (!T[i].tested) {
					const int c = T[i].leaf;
					if (!(forcedSeen[c >> 6] >> (c & 63) & 1)) {
						forcedSeen[c >> 6] |= 1ull << (c & 63);
						++nForce;
					}
				}
			}
			base[s + 1] = nInt + nForce + (end ? 1 : 0);
			forcedTotal += nForce;
		}
		for (size_t s = 0; s < S; ++s) base[s + 1] += base[s];
		if (base[S] >= kOff) throw std::runtime_error("DtreeIndex: too many entries for 29-bit offsets");
		// root offsets need every state's layout first
		auto layoutOf = [&](u32 s, std::vector<int>& slotOf, std::vector<int>& forceSlot, int& endSlot) {
			const BNode* T = nodes.data() + treeBeg[s];
			const int n = int(treeBeg[s + 1] - treeBeg[s]);
			slotOf.assign(n, -1);
			int w = 0;
			for (int i = 0; i < n; ++i)
				if (T[i].test >= 0) slotOf[i] = w++;
			forceSlot.assign(X.beg[s + 1] - X.beg[s], -1);
			for (int i = 0; i < n; ++i)
				if (T[i].test < 0 && T[i].leaf >= 0 && !T[i].tested && forceSlot[T[i].leaf] < 0) forceSlot[T[i].leaf] = w++;
			endSlot = -1;
			for (int i = 0; i < n; ++i)
				if (T[i].test < 0 && T[i].leaf < 0) { endSlot = w; break; }
		};
#pragma omp parallel
		{
			std::vector<int> slotOf, forceSlot;
			int endSlot;
#pragma omp for schedule(static, 4096)
			for (long long s = 0; s < (long long)S; ++s) {
				layoutOf((u32)s, slotOf, forceSlot, endSlot);
				const BNode& r = nodes[treeBeg[s]];
				int slot;
				if (r.test >= 0) slot = slotOf[0];
				else if (r.leaf < 0) slot = endSlot;
				else slot = forceSlot[r.leaf];  // a root leaf with a candidate is never 'tested'
				rootOff[s] = base[s] + (u64)slot;
			}
		}
		DtEntry* E = ent_.allocate(base[S], hugePages);
#pragma omp parallel
		{
			std::vector<int> slotOf, forceSlot;
			int endSlot;
#pragma omp for schedule(static, 4096)
			for (long long s = 0; s < (long long)S; ++s) {
				layoutOf((u32)s, slotOf, forceSlot, endSlot);
				const BNode* T = nodes.data() + treeBeg[s];
				const int n = int(treeBeg[s + 1] - treeBeg[s]);
				const u32 lb = X.beg[s];
				auto target = [&](int child) -> u32 {
					const BNode& c = T[child];
					if (c.test >= 0) return u32(base[s] + slotOf[child]);
					if (c.leaf < 0) return u32(base[s] + endSlot);
					if (c.tested) return u32(rootOff[X.ent[lb + c.leaf].next]) | kTrans;
					return u32(base[s] + forceSlot[c.leaf]);
				};
				for (int i = 0; i < n; ++i) {
					if (T[i].test < 0) continue;
					const Entry& en = X.ent[lb + T[i].test];
					E[base[s] + slotOf[i]] = DtEntry{en.x, en.y, en.z, target(T[i].child[1]), target(T[i].child[0])};
				}
				for (u32 c = 0; c < forceSlot.size(); ++c)
					if (forceSlot[c] >= 0) {
						const Entry& en = X.ent[lb + c];
						E[base[s] + forceSlot[c]] = DtEntry{en.x, en.y, en.z, (kForce << 30) | u32(rootOff[en.next]), 0};
					}
				if (endSlot >= 0) E[base[s] + endSlot] = DtEntry{0, 0, 0, kEnd << 30, stateNode[s]};
			}
		}
		root_ = u32(rootOff[X.root]);
		rootPt_ = X.rootPt;
		if (st) {
			st->lpCalls += lpCalls;
			st->states += S;
			st->internal += internal;
			st->forced += forcedTotal;
			st->chainStates += chainStates;
			st->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		}
	}

private:
	struct Samp {
		u64 mask;
		u32 count;
	};
	// serialized tree node (preorder); leaf: test == -1
	struct BNode {
		int test;      // candidate index tested, -1 == leaf
		int child[2];  // node index within the state's tree: [negative, positive]
		int leaf;      // leaf: -1 == stay (END), i >= 0 == move to candidate i
		bool tested;   // leaf i: c_i was tested positive on the way (its distance is pending)
	};

	// per-thread tree builder
	struct Builder {
		const Dag& g;
		const LayerIndex& X;
		const std::vector<u32>& stateNode;
		const int budget;
		LP3 lp;
		std::vector<HalfSpace> region, hs, wide;
		u64 lpCalls = 0, internal = 0, chainStates = 0;
		// current state
		u32 s = 0, u = 0, lb = 0;
		int m = 0, nodeBudget = 0;
		std::vector<std::vector<HalfSpace>> piece;  // Q(u -> c_i)
		std::vector<int> lit;                       // test literals: +(j+1) positive, -(j+1) negative
		std::vector<signed char> state;             // per candidate: 0 untested, 1 positive, -1 negative
		std::vector<BNode>* out = nullptr;

		int crit = 2;  // 0: information gain on the training outcomes; 1/2: LP look-ahead (outcomes left / log2); 3: two steps

		Builder(const Dag& g_, const LayerIndex& X_, const std::vector<u32>& sn, int b) : g(g_), X(X_), stateNode(sn), budget(b) {}

		void build(u32 s_, const Samp* sb, const Samp* se, std::vector<BNode>& out_) {
			s = s_;
			u = stateNode[s];
			lb = X.beg[s];
			m = int(X.beg[s + 1] - lb);
			out = &out_;
			out->clear();
			if (m == 0) {
				out->push_back(BNode{-1, {-1, -1}, -1, false});
				return;
			}
			piece.resize(m);
			for (int i = 0; i < m; ++i) {
				piece[i].clear();
				appendPiece(g, X.edge[lb + i], piece[i]);
			}
			lit.clear();
			state.assign(m, 0);
			nodeBudget = budget * m;
			std::vector<Samp> S(sb, se);
			if (m > 64) {  // masks cover 64 candidates: plain sequential chain
				++chainStates;
				chain(-1, std::vector<char>(m, 1));
				return;
			}
			node(S, -1);
		}

		// LP over region + literals (+ extra); returns feasibility of the base in 'baseOk'
		bool prepareBase(double x0[3], bool& empty) {
			hs.assign(region.begin(), region.end());
			for (int l : lit) {
				const int j = std::abs(l) - 1;
				const u32 c = X.ent[lb + j].node;
				if (l > 0) appendBisector(g, c, u, hs);  // c_j closer than u
				else appendBisector(g, u, c, hs);
			}
			const int base = (int)hs.size();
			if (base == 0) {
				x0[0] = x0[1] = x0[2] = 0;
				empty = false;
				return true;
			}
			++lpCalls;
			if (lp.prepare(hs.data(), base, x0)) {
				empty = false;
				return true;
			}
			wide.assign(hs.begin(), hs.begin() + base);
			for (HalfSpace& h : wide) h.b += 99 * kRelax;
			double y[3];
			empty = !lp.prepare(wide.data(), base, y);
			return false;
		}

		// feasible outcomes at the current literals; ok == false: numerical trouble.
		// sup/supNone: a known superset (feasibility only shrinks as literals are added)
		void outcomes(int posLast, std::vector<char>& feas, bool& noneFeas, bool& ok, bool& empty,
		              const std::vector<char>* sup = nullptr, bool supNone = true) {
			double x0[3];
			feas.assign(m, 0);
			noneFeas = false;
			const bool baseOk = prepareBase(x0, empty);
			ok = baseOk;
			if (!baseOk) return;
			const int base = (int)hs.size();
			const int lim = posLast >= 0 ? posLast : m - 1;
			for (int i = 0; i <= lim; ++i) {
				if (state[i] < 0 || (sup && !(*sup)[i])) continue;
				hs.resize(base);
				hs.insert(hs.end(), piece[i].begin(), piece[i].end());
				++lpCalls;
				double x[3] = {x0[0], x0[1], x0[2]};
				feas[i] = base == 0 ? lp.prepare(hs.data(), (int)hs.size(), x)
				                    : detail::candidateFeasible(lp, hs.data(), base, (int)hs.size(), x0);
			}
			if (posLast < 0 && supNone) {
				hs.resize(base);
				for (int i = 0; i < m; ++i)
					if (state[i] == 0) appendBisector(g, u, X.ent[lb + i].node, hs);
				++lpCalls;
				double x[3] = {x0[0], x0[1], x0[2]};
				noneFeas = (int)hs.size() == base ? true
				           : base == 0            ? lp.prepare(hs.data(), (int)hs.size(), x)
				                                  : lp.extend(hs.data(), base, (int)hs.size(), x);
			}
		}

		int emit(const BNode& b) {
			out->push_back(b);
			return (int)out->size() - 1;
		}

		// sequential chain over the untested candidates < limit whose outcome is still
		// possible ('maybe'); a positive c_j after negatives for all earlier ones is outcome j
		int chain(int posLast, const std::vector<char>& maybe) {
			const int lim = posLast >= 0 ? posLast : m;
			std::vector<int> js;
			for (int j = 0; j < lim; ++j)
				if (state[j] == 0 && maybe[j]) js.push_back(j);
			const int me = (int)out->size();
			// build front to back: node k tests js[k]; negative -> node k+1 or the final leaf
			for (size_t k = 0; k < js.size(); ++k) {
				out->push_back(BNode{js[k], {-1, -1}, -1, false});
				++internal;
			}
			const int finalLeaf = emit(BNode{-1, {-1, -1}, posLast, posLast >= 0});
			for (size_t k = 0; k < js.size(); ++k) {
				const int pos = emit(BNode{-1, {-1, -1}, js[k], true});
				(*out)[me + k].child[1] = pos;
				(*out)[me + k].child[0] = k + 1 < js.size() ? me + int(k) + 1 : finalLeaf;
			}
			return js.empty() ? finalLeaf : me;
		}

		static double ent(const std::vector<std::pair<int, double>>& c) {
			double t = 0, h = 0;
			for (auto& x : c) t += x.second;
			for (auto& x : c)
				if (x.second > 0) h -= x.second * std::log2(x.second / t);
			return t > 0 ? h : 0;  // total, not normalized
		}
		static int outcomeOf(u64 mask, int m) { return mask ? __builtin_ctzll(mask) : m; }

		// expected tests left after the best next test, at the current literals
		double secondStep(int posLast, const std::vector<char>& F, bool N, const std::vector<Samp>& S) {
			const int lim = posLast >= 0 ? posLast : m;
			int maxFeas = -1;
			for (int i = 0; i < m; ++i)
				if (F[i]) maxFeas = i;
			const int testLim = N ? lim : std::min(lim, maxFeas + 1);
			double total = 0;
			for (auto& x : S) total += x.count;
			double best = 1e300;
			std::vector<char> G;
			bool NG;
			for (int k = 0; k < testLim; ++k) {
				if (state[k] != 0) continue;
				double v[2];
				bool dead = false;
				for (int sk = 0; sk < 2; ++sk) {
					state[k] = sk ? 1 : -1;
					lit.push_back(sk ? k + 1 : -(k + 1));
					bool okb, emptyb;
					outcomes(sk ? k : posLast, G, NG, okb, emptyb, &F, N);
					lit.pop_back();
					state[k] = 0;
					if (!okb) { dead = true; break; }
					int c = NG ? 1 : 0, last = -1;
					for (int i = 0; i < m; ++i)
						if (G[i]) { ++c; last = i; }
					if (c == 0) { dead = true; break; }
					const bool force = c == 1 && !NG && !(sk == 1 && last == k) && state[last] <= 0;
					v[sk] = std::log2((double)c) + (force ? 1 : 0);
				}
				if (dead) continue;
				double w1 = 0;
				for (auto& x : S)
					if (x.mask >> k & 1) w1 += x.count;
				const double sc = 1 + (total > 0 ? (w1 * v[1] + (total - w1) * v[0]) / total : 0.5 * (v[0] + v[1]));
				best = std::min(best, sc);
			}
			return best;
		}

		int node(std::vector<Samp>& S, int posLast, const std::vector<char>* known = nullptr, bool knownNone = false) {
			std::vector<char> feas;
			bool noneFeas, ok, empty;
			if (known) {  // computed by the parent's look-ahead
				feas = *known;
				noneFeas = knownNone;
				ok = true;
				empty = false;
				bool any = noneFeas;
				for (char f : feas) any |= f != 0;
				if (!any) outcomes(posLast, feas, noneFeas, ok, empty);  // classify an empty branch
			} else {
				outcomes(posLast, feas, noneFeas, ok, empty);
			}
			if (!ok && empty) return emit(BNode{-1, {-1, -1}, -1, false});  // unreachable
			if (!ok) return chain(posLast, std::vector<char>(m, 1));         // numerical: no pruning
			int nFeas = noneFeas ? 1 : 0, only = -1;
			for (int i = 0; i < m; ++i)
				if (feas[i]) { ++nFeas; only = i; }
			if (nFeas == 0) return chain(posLast, std::vector<char>(m, 1));
			if (nFeas == 1) {
				if (noneFeas) return emit(BNode{-1, {-1, -1}, -1, false});
				return emit(BNode{-1, {-1, -1}, only, state[only] > 0});
			}
			// candidates worth testing: untested, before the last positive
			const int lim = posLast >= 0 ? posLast : m;
			int maxFeas = -1;
			for (int i = 0; i < m; ++i)
				if (feas[i]) maxFeas = i;
			const int testLim = noneFeas ? lim : std::min(lim, maxFeas + 1);
			if (nodeBudget <= 0) return chain(posLast, feas);
			int bestJ = -1;
			std::vector<char> bestF[2];
			bool bestNone[2] = {false, false};
			if (crit >= 1) {
				// LP look-ahead: the test after which the fewest outcomes stay possible, weighted
				// by how the training queries at this node answer it (equal weights without any)
				double total = 0;
				for (auto& x : S) total += x.count;
				double bestScore = 1e300;
				std::vector<char> F[2];
				bool N[2];
				for (int j = 0; j < testLim; ++j) {
					if (state[j] != 0) continue;
					double nOut[2];
					bool dead = false;
					for (int sg = 0; sg < 2; ++sg) {
						state[j] = sg ? 1 : -1;
						lit.push_back(sg ? j + 1 : -(j + 1));
						bool okb, emptyb;
						outcomes(sg ? j : posLast, F[sg], N[sg], okb, emptyb, &feas, noneFeas);
						lit.pop_back();
						state[j] = 0;
						if (!okb) { dead = true; break; }  // unreachable or numerical: no information
						int c = N[sg] ? 1 : 0, last = -1;
						for (int i = 0; i < m; ++i)
							if (F[sg][i]) { ++c; last = i; }
						if (c == 0) { dead = true; break; }
						// tests still needed, roughly: one per other outcome (crit 1) or log2 of
						// the outcomes (crit 2); +1 if a single candidate outcome remains whose
						// distance was never computed (FORCE)
						const bool force = c == 1 && !N[sg] && !(sg == 1 && last == j) && state[last] <= 0;
						nOut[sg] = (crit >= 2 ? std::log2((double)c) : c - 1) + (force ? 1 : 0);
						if (crit == 3 && c >= 2 && m <= 16) {
							// two-step look-ahead: 1 + the best expected value after a second test
							state[j] = sg ? 1 : -1;
							lit.push_back(sg ? j + 1 : -(j + 1));
							std::vector<Samp> SB;
							for (auto& x : S)
								if ((int)(x.mask >> j & 1) == sg) SB.push_back(x);
							nOut[sg] = std::min(nOut[sg], secondStep(sg ? j : posLast, F[sg], N[sg], SB));
							lit.pop_back();
							state[j] = 0;
						}
					}
					if (dead) continue;
					double score;
					if (total > 0) {
						double w1 = 0;
						for (auto& x : S)
							if (x.mask >> j & 1) w1 += x.count;
						score = (w1 * nOut[1] + (total - w1) * nOut[0]) / total;
					} else {
						score = 0.5 * (nOut[0] + nOut[1]);
					}
					if (score < bestScore - 1e-12) {
						bestScore = score;
						bestJ = j;
						bestF[0] = F[0];
						bestF[1] = F[1];
						bestNone[0] = N[0];
						bestNone[1] = N[1];
					}
				}
			}
			// information gain on the training outcomes
			double bestH = 1e300, h0 = 0;
			if (crit == 0 && !S.empty()) {
				std::vector<std::pair<int, double>> c;
				auto hist = [&](const std::vector<Samp>& v, std::vector<std::pair<int, double>>& out) {
					out.clear();
					for (auto& x : v) {
						const int o = outcomeOf(x.mask, m);
						bool f = false;
						for (auto& y : out)
							if (y.first == o) { y.second += x.count; f = true; break; }
						if (!f) out.emplace_back(o, x.count);
					}
				};
				hist(S, c);
				h0 = ent(c);
				for (int j = 0; j < testLim; ++j) {
					if (state[j] != 0) continue;
					std::vector<Samp> a, b;
					for (auto& x : S) (x.mask >> j & 1 ? b : a).push_back(x);
					if (a.empty() || b.empty()) continue;
					std::vector<std::pair<int, double>> ca, cb;
					hist(a, ca);
					hist(b, cb);
					const double h = ent(ca) + ent(cb);
					if (h < bestH - 1e-9) { bestH = h; bestJ = j; }
				}
				if (bestJ >= 0 && bestH >= h0 - 1e-9) bestJ = -1;  // no gain
			}
			if (bestJ < 0) {
				// no training signal: if the samples agree on a feasible candidate, test it;
				// otherwise the first untested candidate
				if (!S.empty()) {
					const int o = outcomeOf(S[0].mask, m);
					bool agree = true;
					for (auto& x : S) agree &= outcomeOf(x.mask, m) == o;
					if (agree && o < testLim && state[o] == 0) bestJ = o;
				}
				if (bestJ < 0)
					for (int j = 0; j < testLim; ++j)
						if (state[j] == 0) { bestJ = j; break; }
				if (bestJ < 0) return chain(posLast, feas);
			}
			--nodeBudget;
			++internal;
			const int me = emit(BNode{bestJ, {-1, -1}, -1, false});
			std::vector<Samp> a, b;
			for (auto& x : S) (x.mask >> bestJ & 1 ? b : a).push_back(x);
			std::vector<Samp>().swap(S);
			const bool haveF = !bestF[0].empty();
			std::vector<char> F0 = std::move(bestF[0]), F1 = std::move(bestF[1]);
			// negative branch
			state[bestJ] = -1;
			lit.push_back(-(bestJ + 1));
			const int cn = node(a, posLast, haveF ? &F0 : nullptr, bestNone[0]);
			lit.pop_back();
			// positive branch: outcome <= bestJ from now on
			state[bestJ] = 1;
			lit.push_back(bestJ + 1);
			const int cp = node(b, bestJ, haveF ? &F1 : nullptr, bestNone[1]);
			lit.pop_back();
			state[bestJ] = 0;
			(*out)[me].child[0] = cn;
			(*out)[me].child[1] = cp;
			return me;
		}
	};

	PagedArray<DtEntry> ent_;
	std::vector<Samp> samp_;
	std::vector<u64> sampBeg_;
	u32 root_ = 0;
	Vec3 rootPt_{0, 0, 0};
};

// history region of a state, for levels[k] (k >= 0) and for the trie
struct LevelRegion {
	const Dag& g;
	const std::vector<LayerIndex>& levels;
	int k;
	void operator()(u32 s, std::vector<HalfSpace>& hs) const {
		if (k == 0) {  // Cell(u; V_{t_u}): u is the nearest point when it gets inserted
			for (u32 p = g.paBeg[s]; p < g.paBeg[s + 1]; ++p) appendBisector(g, s, g.pa[p], hs);
			return;
		}
		if (s == levels[k].root) return;
		std::vector<u32> path;
		detail::pathOf(levels, k, s, path);
		for (u32 e : path) appendPiece(g, e, hs);
	}
};
struct TrieRegion {
	const Dag& g;
	const LayerIndex& Tr;
	void operator()(u32 s, std::vector<HalfSpace>& hs) const {
		for (u32 t = s; t != 0; t = Tr.owner[t - 1]) appendPiece(g, Tr.edge[t - 1], hs);
	}
};

}  // namespace dpl
