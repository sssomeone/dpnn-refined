#!/bin/bash
# Re-runs every experiment of LAYERED_README.md with the current build.
set -u
cd "$(dirname "$0")/../DPNN"
B=./build_layered/bench_layered
R=../results/final
T=48
$B uniform:20000 --levels 8 --queries 200000 --check 2000 --threads $T > $R/uniform20k_levels8.txt 2>&1
for n in 2500 5000 10000 20000 40000; do
  $B uniform:$n --levels 1 --trie --queries 100000 --check 1000 --threads $T > $R/uniform_trie_$n.txt 2>&1
done
$B data/bunny.xyz --levels 3 --trie --queries 1000000 --check 20000 --threads $T --original > $R/bunny_brio_full.txt 2>&1
$B data/bunny.xyz --levels 3 --coarse 16 --queries 1000000 --check 5000 --threads $T > $R/bunny_brio_coarse16.txt 2>&1
$B data/bunny.xyz --order random --levels 3 --queries 1000000 --check 5000 --threads $T > $R/bunny_random.txt 2>&1
$B data/bunny.xyz --order fps --levels 3 --queries 1000000 --check 5000 --threads $T > $R/bunny_fps.txt 2>&1
for s in 5000 10000 20000 40000; do
  $B data/bunny.xyz --sub $s --levels 1 --trie --queries 100000 --check 1000 --threads $T > $R/bunny_trie_$s.txt 2>&1
done
$B data/lucy.xyz --levels 2 --queries 1000000 --check 5000 --threads $T --original > $R/lucy_brio.txt 2>&1
$B data/lucy.xyz --levels 3 --coarse 16 --queries 1000000 --check 2000 --threads $T > $R/lucy_brio_coarse16.txt 2>&1
$B data/lucy.xyz --order fps --levels 2 --queries 1000000 --check 2000 --threads $T > $R/lucy_fps.txt 2>&1
./build_layered/analyze_combinatorial uniform:20000 > $R/combinatorial_uniform20k.txt 2>&1
./build_layered/analyze_combinatorial data/bunny.xyz > $R/combinatorial_bunny.txt 2>&1
echo ALL_RUNS_DONE
