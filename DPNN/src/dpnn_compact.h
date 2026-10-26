//
// dpnn_compact.h -- memory-lean encodings of the refined query lists.
//
// LayerIndex (dpnn_layered.h) copies every candidate with its coordinates, so
// level k costs 32 bytes per list entry.  Here every refined list is stored as a
// *bitmask over the original query list* of the current node: the node array and
// the child array stay exactly as in dpnn.cpp, and refinement costs a few bits
// per state.  These are the smallest exact refined indexes, but not the fastest:
// iterating set bits costs about as much as the comparisons it skips, and every
// jump still pays two dependent cache misses.  For speed use dpnn_flat.h.
//
//   CompactL0   : dpnn.cpp layout (Node{x,y,z,first,end} + child ids)
//   CompactL1   : + 32 bits per DAG edge   (state = last edge)
//   CompactL2   : + 32 bits per realizable 2-edge path
//   CompactTrie : 8 bytes per trie node    (state = whole path)
//
// Masks of lists with more than 32 candidates live in `wide` (64-bit words);
// the 32-bit slot then holds the word offset.
//
#pragma once
#include "dpnn_layered.h"

namespace dpl {

struct alignas(32) CNode {
	double x, y, z;
	u32 first, end;  // child range of this node
};

namespace detail {

inline u32 ctz64(u64 m) { return (u32)__builtin_ctzll(m); }

inline std::vector<CNode> makeNodes(const Dag& g) {
	std::vector<CNode> N(g.n());
	for (size_t i = 0; i < g.n(); ++i) N[i] = CNode{g.pts[i].x, g.pts[i].y, g.pts[i].z, g.chBeg[i], g.chBeg[i + 1]};
	return N;
}

// Encodes a subset (sorted bit positions < len) as a 32-bit mask or a `wide` offset.
inline u32 encodeMask(const std::vector<u32>& bits, u32 len, std::vector<u64>& wide) {
	if (len <= 32) {
		u32 m = 0;
		for (u32 b : bits) m |= 1u << b;
		return m;
	}
	const size_t off = wide.size();
	if (off > 0xFFFFFFFFull) throw std::runtime_error("wide mask offset overflow");
	wide.resize(off + (len + 63) / 64, 0);
	for (u32 b : bits) wide[off + b / 64] |= 1ull << (b % 64);
	return (u32)off;
}

// Calls f(bit, rank) for every set bit of the mask, in increasing order, until f returns true.
template <class F>
inline bool forEachBit(u32 aux, u32 len, const u64* wide, F&& f) {
	if (len <= 32) {
		u32 m = aux, r = 0;
		while (m) {
			const u32 j = (u32)__builtin_ctz(m);
			m &= m - 1;
			if (f(j, r++)) return true;
		}
		return false;
	}
	const u64* W = wide + aux;
	u32 r = 0;
	for (u32 w = 0; w * 64 < len; ++w) {
		u64 m = W[w];
		while (m) {
			const u32 j = w * 64 + ctz64(m);
			m &= m - 1;
			if (f(j, r++)) return true;
		}
	}
	return false;
}

// number of set bits below position j
inline u32 rankBelow(u32 aux, u32 len, const u64* wide, u32 j) {
	if (len <= 32) return (u32)__builtin_popcount(aux & ((1u << j) - 1u));
	const u64* W = wide + aux;
	u32 r = 0;
	for (u32 w = 0; w < j / 64; ++w) r += (u32)__builtin_popcountll(W[w]);
	if (j % 64) r += (u32)__builtin_popcountll(W[j / 64] & ((1ull << (j % 64)) - 1ull));
	return r;
}

inline double dist2(const Vec3& q, const CNode& p) {
	const double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
	return dx * dx + dy * dy + dz * dz;
}

}  // namespace detail

// ---------------------------------------------------------------------------
struct CompactL0 {
	std::vector<CNode> nodes;
	std::vector<u32> child;

	explicit CompactL0(const Dag& g) : nodes(detail::makeNodes(g)), child(g.ch) {}
	size_t bytes() const { return nodes.size() * sizeof(CNode) + child.size() * sizeof(u32); }

	template <bool kStats>
	u32 run(const Vec3& q, QueryStats* st) const {
		const CNode* N = nodes.data();
		const u32* C = child.data();
		u32 v = 0;
		double best = detail::dist2(q, N[0]);
		for (;;) {
			u32 i = N[v].first;
			const u32 end = N[v].end;
			for (; i < end; ++i) {
				if (kStats) st->comparisons++;
				const double d = detail::dist2(q, N[C[i]]);
				if (d < best) {
					best = d;
					v = C[i];
					if (kStats) st->jumps++;
					break;
				}
			}
			if (i == end) return v;
		}
	}
	u32 query(const Vec3& q) const { return run<false>(q, nullptr); }
	u32 query(const Vec3& q, QueryStats& st) const { return run<true>(q, &st); }
};

// ---------------------------------------------------------------------------
struct CompactL1 {
	struct Edge {
		u32 node, aux;  // aux: level-1 list of the state "just took this edge", over children(node)
	};
	std::vector<CNode> nodes;
	std::vector<Edge> edges;
	std::vector<u64> wide;
	u32 rootAux = 0;

	CompactL1(const Dag& g, const LayerIndex& L1) : nodes(detail::makeNodes(g)), edges(g.ch.size()) {
		std::vector<u32> bits;
		auto listOf = [&](u32 state, u32 w) {
			bits.clear();
			for (u32 p = L1.beg[state]; p < L1.beg[state + 1]; ++p) bits.push_back(L1.edge[p] - g.chBeg[w]);
			return detail::encodeMask(bits, g.chBeg[w + 1] - g.chBeg[w], wide);
		};
		for (u32 f = 0; f < g.ch.size(); ++f) edges[f] = Edge{g.ch[f], listOf(f, g.ch[f])};
		rootAux = listOf(L1.root, 0);
	}
	size_t bytes() const { return nodes.size() * sizeof(CNode) + edges.size() * sizeof(Edge) + wide.size() * 8; }

	template <bool kStats>
	u32 run(const Vec3& q, QueryStats* st) const {
		const CNode* N = nodes.data();
		const Edge* E = edges.data();
		const u64* W = wide.data();
		u32 v = 0, aux = rootAux;
		double best = detail::dist2(q, N[0]);
		for (;;) {
			const u32 first = N[v].first, len = N[v].end - first;
			u32 hit = 0;
			const bool found = detail::forEachBit(aux, len, W, [&](u32 j, u32) {
				if (kStats) st->comparisons++;
				const double d = detail::dist2(q, N[E[first + j].node]);
				if (d < best) { best = d; hit = first + j; return true; }
				return false;
			});
			if (!found) return v;
			if (kStats) st->jumps++;
			v = E[hit].node;
			aux = E[hit].aux;
		}
	}
	u32 query(const Vec3& q) const { return run<false>(q, nullptr); }
	u32 query(const Vec3& q, QueryStats& st) const { return run<true>(q, &st); }
};

// ---------------------------------------------------------------------------
struct CompactL2 {
	struct Edge {
		u32 node;
		u32 aux1;  // level-1 list of the state (.., this edge) over children(node)
		u32 off2;  // level-2 lists of the states (this edge, g), g in aux1, start at aux2[off2]
		u32 pad;
	};
	std::vector<CNode> nodes;
	std::vector<Edge> edges;
	std::vector<u32> aux2;
	std::vector<u64> wide;
	u32 rootAux = 0;

	// L1 and L2 are levels[1], levels[2] of dpnn_layered.h.  Level-2 states are
	// entries of L1: entry p of L1 (owned by edge e, edge g = L1.edge[p]) is the
	// state (e, g), and its list (over children(child(g))) is L2's list of p.
	CompactL2(const Dag& g, const LayerIndex& L1, const LayerIndex& L2)
	    : nodes(detail::makeNodes(g)), edges(g.ch.size()) {
		std::vector<u32> bits;
		auto encode = [&](const LayerIndex& L, u32 state, u32 w) {
			bits.clear();
			for (u32 p = L.beg[state]; p < L.beg[state + 1]; ++p) bits.push_back(L.edge[p] - g.chBeg[w]);
			return detail::encodeMask(bits, g.chBeg[w + 1] - g.chBeg[w], wide);
		};
		aux2.resize(L1.ent.size());
		for (u32 f = 0; f < g.ch.size(); ++f) {
			edges[f] = Edge{g.ch[f], encode(L1, f, g.ch[f]), L1.beg[f], 0};
			for (u32 p = L1.beg[f]; p < L1.beg[f + 1]; ++p) aux2[p] = encode(L2, p, L1.ent[p].node);
		}
		rootAux = encode(L1, L1.root, 0);
	}
	size_t bytes() const {
		return nodes.size() * sizeof(CNode) + edges.size() * sizeof(Edge) + aux2.size() * 4 + wide.size() * 8;
	}

	template <bool kStats>
	u32 run(const Vec3& q, QueryStats* st) const {
		const CNode* N = nodes.data();
		const Edge* E = edges.data();
		const u64* W = wide.data();
		u32 v = 0, cur = rootAux;        // current list (mask over children(v))
		u32 m1 = 0, off2 = 0, len = 0;  // level-1 list / level-2 offset of the edge that led to v
		bool atRoot = true;
		double best = detail::dist2(q, N[0]);
		for (;;) {
			const u32 first = N[v].first;
			len = N[v].end - first;
			u32 hit = 0, hitBit = 0;
			const bool found = detail::forEachBit(cur, len, W, [&](u32 j, u32) {
				if (kStats) st->comparisons++;
				const double d = detail::dist2(q, N[E[first + j].node]);
				if (d < best) { best = d; hit = first + j; hitBit = j; return true; }
				return false;
			});
			if (!found) return v;
			if (kStats) st->jumps++;
			const Edge& e = E[hit];
			// new state (previous edge, hit): its list is the level-2 list stored with the previous edge
			cur = atRoot ? e.aux1 : aux2[off2 + detail::rankBelow(m1, len, W, hitBit)];
			atRoot = false;
			m1 = e.aux1;
			off2 = e.off2;
			v = e.node;
		}
	}
	u32 query(const Vec3& q) const { return run<false>(q, nullptr); }
	u32 query(const Vec3& q, QueryStats& st) const { return run<true>(q, &st); }
};

// ---------------------------------------------------------------------------
struct CompactTrie {
	struct TNode {
		u32 aux;        // candidate list over children(DAG node of this trie node)
		u32 childBase;  // trie id of the child created by the first candidate
	};
	std::vector<CNode> nodes;
	std::vector<u32> child;
	std::vector<TNode> trie;
	std::vector<u64> wide;

	CompactTrie(const Dag& g, const LayerIndex& Tr) : nodes(detail::makeNodes(g)), child(g.ch), trie(Tr.numStates()) {
		std::vector<u32> bits;
		std::vector<u32> dagNode(Tr.numStates(), 0);
		for (size_t t = 1; t < Tr.numStates(); ++t) dagNode[t] = Tr.ent[t - 1].node;
		for (u32 t = 0; t < Tr.numStates(); ++t) {
			const u32 w = dagNode[t];
			bits.clear();
			for (u32 p = Tr.beg[t]; p < Tr.beg[t + 1]; ++p) bits.push_back(Tr.edge[p] - g.chBeg[w]);
			trie[t] = TNode{detail::encodeMask(bits, g.chBeg[w + 1] - g.chBeg[w], wide), Tr.beg[t] + 1};
		}
	}
	size_t bytes() const {
		return nodes.size() * sizeof(CNode) + child.size() * 4 + trie.size() * sizeof(TNode) + wide.size() * 8;
	}

	template <bool kStats>
	u32 run(const Vec3& q, QueryStats* st) const {
		const CNode* N = nodes.data();
		const u32* C = child.data();
		const u64* W = wide.data();
		u32 v = 0, t = 0;
		double best = detail::dist2(q, N[0]);
		for (;;) {
			const u32 first = N[v].first, len = N[v].end - first;
			u32 hitNode = 0, hitRank = 0;
			const bool found = detail::forEachBit(trie[t].aux, len, W, [&](u32 j, u32 r) {
				if (kStats) st->comparisons++;
				const u32 w = C[first + j];
				const double d = detail::dist2(q, N[w]);
				if (d < best) { best = d; hitNode = w; hitRank = r; return true; }
				return false;
			});
			if (!found) return v;
			if (kStats) st->jumps++;
			t = trie[t].childBase + hitRank;
			v = hitNode;
		}
	}
	u32 query(const Vec3& q) const { return run<false>(q, nullptr); }
	u32 query(const Vec3& q, QueryStats& st) const { return run<true>(q, &st); }
};

}  // namespace dpl
