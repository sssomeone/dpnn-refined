//
// dpnn_refined.h -- drop-in wrapper: build once, query the exact nearest
// neighbour with history-refined lists (see LAYERED_README.md).
//
//   dpl::RefinedDPNN nn(points);                       // BRIO order, 2 layers
//   uint32_t i = nn.nearest(q);                        // index into `points`
//   nn.nearestBatch(Q.data(), Q.size(), out.data());   // faster for many queries
//
#pragma once
#include "dpnn_flat.h"
#include "dpnn_layered.h"

#include <memory>

namespace dpl {

struct RefinedOptions {
	InsertionOrder order = InsertionOrder::Brio;
	int levels = 2;          // history layers (0 == original DPNN)
	u32 coarseDivisor = 0;   // > 0: refine only the first n / coarseDivisor nodes (less memory)
	bool floatCoords = false;  // 16-byte entries (half the memory, same answers)
	bool hugePages = true;
};

class RefinedDPNN {
public:
	explicit RefinedDPNN(const std::vector<Vec3>& points, const RefinedOptions& opt = RefinedOptions())
	    : g_(buildDag(points, opt.order)) {
		std::vector<LayerIndex> L;
		L.push_back(buildLevel0(g_));
		const u32 T = opt.coarseDivisor ? (u32)std::max<size_t>(1, g_.n() / opt.coarseDivisor) : (u32)g_.n();
		for (int k = 1; k <= opt.levels; ++k) L.push_back(buildNextLevel(g_, L, nullptr, T));
		const int k = opt.levels;
		std::vector<u32> sn;
		LayerIndex H;
		if (opt.coarseDivisor && k > 0) {
			H = makeHybrid(L, k, T, sn);
		} else {
			sn = stateNodes(L, k);
			H = std::move(L[k]);
		}
		// the flat index needs H only: free the other levels and the build-time
		// metadata first (halves the peak memory)
		L.clear();
		RawVector<u32>().swap(H.owner);
		RawVector<u32>().swap(H.edge);
		if (opt.floatCoords) f32_.reset(new FlatIndex16(H, sn, g_, opt.hugePages));
		else f64_.reset(new FlatIndex(H, sn, opt.hugePages));
	}

	// index into the input point array (for exact duplicates: the first copy)
	u32 nearest(const Vec3& q) const { return g_.origId[f64_ ? f64_->query(q) : f32_->query(q)]; }

	void nearestBatch(const Vec3* Q, size_t m, u32* out) const {
		if (f64_) f64_->queryBatch<16>(Q, m, out);
		else f32_->queryBatch<16>(Q, m, out);
		for (size_t i = 0; i < m; ++i) out[i] = g_.origId[out[i]];
	}

	size_t bytes() const { return f64_ ? f64_->bytes() : f32_->bytes(); }
	const Dag& dag() const { return g_; }

private:
	Dag g_;
	std::unique_ptr<FlatIndex> f64_;
	std::unique_ptr<FlatIndex16> f32_;
};

}  // namespace dpl
