#!/bin/bash
# Decision trees inside the states (DPNN/LAYERED_README.md §8): where the comparisons
# go, entropy and oracle bounds, comparisons + exactness of the trees, query time.
# Results in results/decision/.  Needs `make -f Makefile.layered` first.
set -u
cd "$(dirname "$0")/../DPNN"
B=./build_layered
R=../results/decision
T=48
mkdir -p $R
# 1. where the comparisons go (sequential scan), entropy, sample-based estimate
$B/analyze_decision data/bunny.xyz --levels 3 --trie --queries 2000000 --threads $T > $R/bunny_brio.txt 2>&1
# 2. oracle: fewest LP-certified tests per query (3000 queries per set)
$B/analyze_decision data/bunny.xyz --levels 3 --queries 400000 --oracle 3000 --threads $T > $R/bunny_brio_oracle.txt 2>&1
$B/analyze_decision data/bunny.xyz --order fps --levels 3 --queries 400000 --oracle 3000 --threads $T > $R/bunny_fps_oracle.txt 2>&1
# 3. decision trees: comparisons + exactness (every query compared with the sequential scan);
#    bunny BRIO levels 0..3: see the timing run in 5.
for c in 0 1 2 3; do  # test choice: information gain / LP look-ahead variants (§8.3)
  $B/bench_dtree uniform:20000 --levels 2 --train 200000 --queries 200000 --check 2000 --crit $c --no-timing --threads $T > $R/uniform20k_crit$c.txt 2>&1
done
$B/bench_dtree data/bunny.xyz --order fps --levels 3 --no-timing --threads $T > $R/bunny_fps_dtree.txt 2>&1
$B/bench_dtree data/bunny.xyz --levels 1 --trie --no-timing --threads $T > $R/bunny_brio_trie_dtree.txt 2>&1
$B/bench_dtree data/lucy.xyz --levels 2 --check 2000 --no-timing --threads $T > $R/lucy_brio_dtree.txt 2>&1
$B/bench_dtree data/lucy.xyz --order fps --levels 2 --check 2000 --no-timing --threads $T > $R/lucy_fps_dtree.txt 2>&1
# 4. Orchard pruning |u-c| >= 2|q-u| (§8.7; a column of analyze_decision)
$B/analyze_decision data/bunny.xyz --levels 3 --queries 400000 --threads $T > $R/bunny_brio_orchard.txt 2>&1
$B/analyze_decision data/bunny.xyz --order fps --levels 3 --queries 400000 --threads $T > $R/bunny_fps_orchard.txt 2>&1
# 5. query time (5 interleaved rounds per method) -- also the comparisons of bunny BRIO levels 0..3
$B/bench_dtree data/bunny.xyz --levels 3 --threads $T > $R/bunny_brio_dtree_timing.txt 2>&1
$B/bench_dtree data/bunny.xyz --order fps --levels 3 --threads $T > $R/bunny_fps_dtree_timing.txt 2>&1
echo ALL_RUNS_DONE
