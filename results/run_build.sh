#!/bin/bash
# Build-time and full-verification runs of LAYERED_README.md (§7 "构建时间", §3 "精确性").
# The query tables come from run_all.sh (results/final/); these runs re-measure the
# build after the parallel-build fixes and brute-force *every* query (--check 1000000).
set -u
cd "$(dirname "$0")/../DPNN"
B=./build_layered/bench_layered
R=../results/build
T=48
mkdir -p $R
./build_layered/test_layered > $R/test_layered.txt 2>&1
$B data/lucy.xyz --levels 2 --queries 1000000 --check 1000000 --threads $T --original > $R/lucy_brio_verify.txt 2>&1
$B data/bunny.xyz --levels 3 --trie --queries 1000000 --check 1000000 --threads $T --original > $R/bunny_brio_verify.txt 2>&1
$B data/lucy.xyz --levels 3 --coarse 16 --queries 100000 --check 2000 --threads $T > $R/lucy_brio_coarse16.txt 2>&1
$B data/lucy.xyz --order fps --levels 2 --queries 100000 --check 2000 --threads $T > $R/lucy_fps.txt 2>&1
# parallel scaling of the LP filter: same 100K-point subsample, 1 vs $T threads
$B data/lucy.xyz --sub 100000 --levels 2 --queries 1000 --check 100 --threads 1 > $R/lucy100k_threads1.txt 2>&1
$B data/lucy.xyz --sub 100000 --levels 2 --queries 1000 --check 100 --threads $T > $R/lucy100k_threads$T.txt 2>&1
# end to end: RefinedDPNN (2 levels, flat layout) on Lucy, all hardware threads
( /usr/bin/time -f "wall %e s, peak RSS %M KB" ./build_layered/example_refined data/lucy.xyz ) > $R/example_refined_lucy.txt 2>&1
echo ALL_RUNS_DONE
