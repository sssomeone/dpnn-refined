# k-NN through order-k Voronoi diagrams.  Needs the headers of ../DPNN/src,
# CGAL (header-only, >= 5.1), Boost and nanoflann (../deps/nanoflann).
#   make            -> build/bench_knn build/test_korder
DPNN_INC  ?= ../DPNN/src
CGAL_INC  ?= ../deps/CGAL-5.6.1/include
NANO_INC  ?= ../deps/nanoflann/include
BUILD     ?= build
CXX       ?= g++
# -ffp-contract=off: the same squared distance must round identically in every
# method and in the brute-force check (see ../results/compare/NOTES.md)
CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -ffp-contract=off
# TBB: CGAL's parallel regular triangulation for the orders that need no query DAG
TBB       ?= -DCGAL_LINKED_WITH_TBB
TBB_LIBS  ?= -ltbb -ltbbmalloc

INC = -I src -I $(DPNN_INC) -I $(CGAL_INC) -I $(NANO_INC)
HDR = src/korder.h src/knn_baselines.h $(DPNN_INC)/dpnn_layered.h $(DPNN_INC)/dpnn_flat.h $(DPNN_INC)/dpnn_refined.h $(DPNN_INC)/lp3.h

all: $(BUILD)/bench_knn $(BUILD)/test_korder

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/bench_knn: src/bench_knn.cpp $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(TBB) $(INC) $< -o $@ $(TBB_LIBS)

$(BUILD)/test_korder: src/test_korder.cpp $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
