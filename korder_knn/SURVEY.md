# order-k Voronoi 做 kNN 查询：文献调研

（2026-10 整理。只列出核对过出处的文献；"本目录"指 `korder_knn/`。）

## 1. 基本事实

**order-k Voronoi 图**：对每个 k 元子集 T，`V_k(T) = { x : T 恰好是 x 的 k 个最近点 }`，非空的 `V_k(T)` 把空间剖分成凸多面体。
kNN 查询（只要集合、不排序）就是在这个剖分里做一次点定位，答案是一张预存的 k 元表。

- Shamos & Hoey (FOCS 1975, *Closest-point problems*) 最早把它当作 "k closest point" 查询的数据结构提出，查询 O(log n + k)。
  这是 Edelsbrunner–Osang 2021 引言中的说法。
- **它是一个 power diagram**。对 k 元集 T，记质心 `c_T`、离散度 `var_T = (1/k)Σ|t−c_T|²`，则
  `(1/k) Σ_{t∈T} |x−t|² = |x−c_T|² + var_T`。
  使左边最小的 k 元集正是 x 的 k 近邻集，所以 order-k 图 = 加权点 `(c_T, −var_T)` 的 power diagram。
  - Aurenhammer (SIAM J. Comput. 16(1), 1987) §3 定义了 order-k power diagram，并证明它们是提升后超平面排列 k-level 的投影（Thm 6）。
  - Aurenhammer (DCG 5, 1990) 给出质心提升的写法；Edelsbrunner & Osang 2021 在算法里直接用这个形式（质心 + 权重 `|c|² − (1/k)Σ|a|²`）。
- **提升到抛物面**：order-k 图是 n 个切平面排列的 k-level 在底面上的投影（Edelsbrunner & Seidel, DCG 1986）。

## 2. 规模

| 场景 | order-k 图的规模 | 出处 |
|---|---|---|
| 平面 | Θ(k(n−k)) 个单元/边/顶点 | Lee, IEEE TC 31(6), 1982 |
| d 维最坏情况（前 k 阶合计） | O(k^⌈(d+1)/2⌉ · n^⌊(d+1)/2⌋)，3D 即 O(k²n²) | Clarkson & Shor, DCG 1989；Mulmuley, DCG 1991（Edelsbrunner–Osang 2021 中的综述） |
| Poisson 点（任意维） | 单位体积内各维面个数的期望有精确公式 | Edelsbrunner & Nikitenko, DCG 62(4), 2019 |
| 3D 小规模实验（n = 200） | 球内均匀点、凸多面体上的点：随 k 近似线性；环面：略超线性；矩曲线：近二次 | Edelsbrunner & Osang 2021 |

我们没有找到 3D 曲面采样点（bunny、Lucy 这类）在 n ≥ 10⁴ 时的 order-k 规模数据。本目录的实测：bunny 约为 (3.4k² − 2.9k)·n，10 万个均匀点几乎相同，Lucy（k ≤ 4）也一样。

## 3. 构造算法

**平面**
- Lee 1982：从 order-(k−1) 逐阶构造到 order-k，O(k²n log n)。
- Chazelle & Edelsbrunner, IEEE TC 1987：改进的构造算法。
- Aurenhammer & Schwarzkopf, SoCG 1991 / IJCGA 1992：在线随机增量算法，O(nk² log n + nk log³ n) 时间，O(k(n−k)) 空间。
  它同时是一个支持 kNN 查询、插入、删除的动态结构。
- Agarwal, de Berg, Matoušek, Schwarzkopf, SIAM J. Comput. 27(3), 1998：构造 ≤k-level 与 1..k 阶 Voronoi 图，适用于任意维。
- Ramos, SoCG 1999；Chan, SIAM J. Comput. 2000：3D 平面排列的 (≤k)-level 可在 O(n log n + nk²) 期望时间内构造。
- Chan, Cheng, Zheng, SODA 2024：平面 order-k 图 O(n log n + nk) 期望时间，最优。

**任意维**
- Boissonnat, Devillers, Teillaud, Algorithmica 9, 1993：**k-Delaunay tree**，即 Delaunay tree（DPNN 所用的历史结构）的 k 阶推广。
  - 半动态（只插入），随机增量构造。
  - 平面：O(n log n + k³n) 期望时间，O(k²n) 期望空间。
  - 也给出了 d 维的界。
- Mulmuley, STOC 1990 / DCG 1991：1..k 阶 Voronoi 图的输出敏感构造。
- Edelsbrunner & Osang, arXiv 2011.03617（2020，Algorithmica 2021）：
  - 用加权一阶 Delaunay 作黑盒，逐阶得到 order-k Delaunay 镶嵌，每阶的顶点（k 元集）来自低阶的 rhomboid（菱形体）。
  - 3D 中 order-k 镶嵌天然含八面体（非单纯形）单元。
  - 开源：github.com/geoo89/rhomboidtiling（CGAL，2D/3D）；github.com/geoo89/orderkdelaunay（python）。

本目录的构造与 Edelsbrunner–Osang 同一思路，但取顶点的规则更直接：order-(j+1) 的单元 = order-j 三角剖分中相邻两单元的并（并集大小为 j+1 者）。证明见 `src/korder.h` 开头。

## 4. 用于 kNN 查询：理论上的最好结果

**平面**
- order-k 图 + 点定位：O(log n + k) 查询，O(k(n−k)) 空间（Shamos–Hoey；Lee 1982）。
- 不依赖 k 的线性空间结构：平面 kNN 提升后是 3D 半空间报告。
  - Chan 2000：O(log n + k) 期望查询，O(n log log n) 空间。
  - Afshani & Chan, SODA 2009：线性空间，O(log n + k) 查询，O(n log n) 期望预处理。平面问题至此在理论上已最优。

**3D 点（我们的情形）**
- 提升后是 4 维半空间报告。
  Agarwal–Eppstein–Matoušek 1992 的空间/时间权衡：空间 m（n ≤ m ≤ n²）时，查询约 O(n/√m · log n + k)。
- 所以近线性空间只能做到 ~√n 的查询；要对数查询，最坏情况需要 ~n² 空间，与 order-≤k 图的 O(k²n²) 最坏规模一致。
- 没有平面那样"线性空间 + 对数查询"的结果。

**近似**
- Har-Peled & Kumar, FOCS 2012 / SIAM J. Comput. 2014（*Down the rabbit hole*）：(1+ε, k)-ANN 查询 O(log n)，空间 Õ(n/k)，k 越大越省。只适用于近似。

## 5. 数据库 / 应用方向

**用 order-k 单元作"安全区"（移动查询，基本都在 2D）**
- Zhang, Zhu, Papadias, Tao, Lee, SIGMOD 2003（*Location-based spatial queries*）：kNN 结果的有效区域就是 order-k 单元。
- Nutanong, Zhang, Tanin, Kulik, PVLDB 1(1), 2008（*V\*-Diagram*）：按查询现场构造的安全区。
- Li, Gu, Qi, Yu, Zhang, Yi, PVLDB 8(2), 2015（*Influential Neighbor Sets*）：
  - 明确指出计算完整 order-k 单元代价太高。
  - 改为维护一小组"守护点"：只要当前 k 个近邻都比守护点近，结果就不变。
  - 作者证明这样得到的区域就是最大的安全区（即 order-k 单元本身），但不必构造它。

**用一阶 Voronoi / Delaunay 做 kNN 扩展**
- 关键引理：第 i+1 近邻一定是前 i 个近邻之一的 Voronoi 邻居。
- Sharifzadeh & Shahabi, PVLDB 3(1), 2010（VoR-tree）：R-tree 定位，再沿 Voronoi 邻居扩展。
- Li et al., arXiv 1911.02788（2019，多层 Voronoi，MVD）：多层一阶 Voronoi。

**与我们最相关**
- Wang, Guo, Zhao, Xin, Chen, Tu, Wang，*Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds*，arXiv 2605.02224（2026-05）。
  - 把 DP-NNS 推广到 kNN：候选 = 1-NN 路径上的 transition sites + 已确认近邻的后继表（successor list）。
  - 复杂度 O(k log n)，支持前缀子集查询与点的增删。
  - 论文 Table 1：k=20、Lucy 上 2.264 µs，kd-tree 为 10.13 µs（论文机器上测得）。
  - 没有用到高阶 Voronoi 图。

## 6. 调研小结

1. **平面问题在理论上已解决**：order-k 图 + 点定位给出 O(log n + k)，Afshani–Chan 去掉了对 k 的空间依赖。数据库界用 order-k 单元，但都在 2D，而且尽量避免显式构造（INS）。
2. **3D 的 order-k 图很少被真正建出来**：理论最坏 O(k²n²)。唯一能直接构造 3D order-k 镶嵌的开源实现（Edelsbrunner–Osang）只做过 n = 200 的实验。
3. **"kNN = power diagram 上的一次最近点查询"这一点很适合 DPNN**：DPNN 的正确性只用到"新插入的站点只从邻居手里夺取区域"，对 power distance 同样成立。k-Delaunay tree（BDT 1993）就是这条思路的历史结构版本。
4. **因此本目录的尝试**：
   - 用加权 Delaunay 逐阶构造 3D order-k 图；
   - 在 order-k power diagram 上建 DPNN，一次查询直接得到 k 近邻集合；
   - 与 kd-tree、Delaunay 逐点扩展、Manifold k-NN 的后继表方案对比。
   - 实验结果见 `README.md`。

## 链接

- Lee 1982: IEEE Trans. Computers 31(6):478–487（无开放链接）
- Aurenhammer 1987: https://epubs.siam.org/doi/10.1137/0216006
- Aurenhammer & Schwarzkopf: https://tugraz.elsevierpure.com/en/publications/a-simple-on-line-randomized-incremental-algorithm-for-computing-h-3/
- Boissonnat, Devillers, Teillaud 1993: https://link.springer.com/article/10.1007/BF01228508
- Agarwal, de Berg, Matoušek, Schwarzkopf 1998: https://epubs.siam.org/doi/10.1137/S0097539795281840
- Chan 2000: https://epubs.siam.org/doi/10.1137/S0097539798349188
- Afshani & Chan 2009: https://tmc.web.engr.illinois.edu/pub_nn.html
- Chan, Cheng, Zheng 2024: https://arxiv.org/abs/2310.15363
- Edelsbrunner & Osang: https://arxiv.org/abs/2011.03617 ，代码 https://github.com/geoo89/rhomboidtiling
- Edelsbrunner & Nikitenko 2019: https://arxiv.org/abs/1709.09380
- Har-Peled & Kumar: https://arxiv.org/abs/1111.2942
- Zhang et al. SIGMOD 2003: https://doi.org/10.1145/872757.872812
- Nutanong et al. PVLDB 2008: https://people.eng.unimelb.edu.au/etanin/vldb08.pdf
- Li et al. PVLDB 2015 (INS): https://arxiv.org/abs/1602.00363
- VoR-tree PVLDB 2010: https://infolab.usc.edu/assets/pdf/courses/csci587/slides/session8-vor-tree.pdf
- MVD: https://arxiv.org/abs/1911.02788
- Manifold k-NN: https://arxiv.org/abs/2605.02224
