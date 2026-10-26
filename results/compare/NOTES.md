# 与原版对比的补充测量

`compare_original.cpp`：端到端构建时间（原版 dpnn.cpp 的 BRIO / FPS 顺序；`RefinedDPNN` 各配置；全部层 trie），
原版用 FPS 顺序时的查询时间（每组 100 万查询，5 次取最小），以及 bench 两种查询分布下答案的熵（各 1000 万查询）。
比较次数与本方法的查询时间取自 bench 的输出（`results/build/*_verify.txt`、`results/final/*.txt`）。

    cd DPNN
    g++ -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -I ../deps/stubs -I src -I ../deps/CGAL-5.6.1/include \
        -I /usr/include/eigen3 ../results/compare/compare_original.cpp -o compare_original
    OMP_NUM_THREADS=48 ./compare_original data/bunny.xyz
    OMP_NUM_THREADS=48 ./compare_original data/lucy.xyz big

注意：输出中原版查询那一行的 `wrong` 计数是假象，不是原版出错。GCC 在 C++ 下默认 `-ffp-contract=fast`，
`-march=native` 时会把距离公式融合成 FMA，同一个点的距离在两处内联位置可能差最后一位，被 `!=` 判成不同。
用 `-ffp-contract=off` 编译同样的核对，错误数为 0。bench 的核对用的是点编号比较和同一段代码算出的距离，不受影响。
