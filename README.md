# dpnn-refined

在 DPNN（*Efficient Nearest Neighbor Search Using Dynamic Programming*，arXiv:2409.15023）基础上做的两条研究线：

1. **主线：只用 Delaunay 信息减少 DPNN 的比较次数**（`DPNN/`）。
   - 利用查询已经走过的跳转，删掉不可能赢的候选。判定只需要 DAG 父节点给出的平分半空间和一个 3D 小线性规划，不做 Voronoi 胞腔的几何切割。
   - 提供 1 层 / k 层 / 全部层三种精炼表，以及状态内的 LP 认证决策树。
   - 配套面向访存延迟的查询布局：扁平表加批量交错查询。
   - 所有版本对每个查询返回的点都与原 DPNN 相同。
2. **支线：用 order-k Voronoi 图做 kNN**（`korder_knn/`）。
   - 包括文献调研、order-k 图的构造与查询，以及精确候选表 Cx(u)。
   - 与 kd-tree、Delaunay 扩展、后继表做了对比。

完整文档：
- 主线：[`DPNN/LAYERED_README.md`](DPNN/LAYERED_README.md)（方法、证明、复杂度、全部实验表格）；
- 支线：[`korder_knn/README.md`](korder_knn/README.md)，文献调研见 [`korder_knn/SURVEY.md`](korder_knn/SURVEY.md)。

---

## 1. 结论速览

### 比较次数

box 查询（2 倍包围盒内均匀分布，与论文设置相同），每组 100 万个，括号内是 log₂n 的倍数：

| 数据 | 原 DPNN | 1 层 | 多层 | 多层 + 决策树 | 神谕下界 |
|---|---|---|---|---|---|
| bunny 73K，BRIO | 184.1（11.4） | 112.9（7.0） | 3 层 70.7（4.4） | 53.7（3.3） | 32.1（2.0） |
| bunny 73K，FPS | 135.6（8.4） | 73.1（4.5） | 3 层 51.3（3.2） | **40.3（2.5）** | 24.0（1.5） |
| Lucy 1.02M，BRIO | 332.9（16.7） | 162.1（8.1） | 2 层 111.4（5.6） | 84.8（4.2） | – |
| Lucy 1.02M，FPS | 236.3（11.8） | 105.4（5.3） | 2 层 79.3（4.0） | **64.3（3.2）** | – |

- **与论文一致**：1 层就是论文 Sec. VI.E 的简化版，FPS 下 73.1 / 105.4，论文 Table V 为 74.1 / 106.3。区别是这里不做几何切割。
- **全部层是用尽历史信息的极限**：bunny BRIO 全部层为 57.2，加决策树 46.8。跳转次数 ≈ H_n 不变，所以查询仍是 Θ(log n)，常数只能降约 3 倍。
- **神谕下界**：神谕 = 已知查询点时，证明 DPNN 路径最少需要的测试次数。决策树拿到了"顺序扫描 → 神谕"之间约 40–50% 的空间。
- **精确性**：所有版本都与原 DPNN 逐点相同；顺序扫描版在 bunny、Lucy 的全部查询上做了暴力验证，0 错误。

### 查询时间

单线程，µs/查询，box / near；原版采用相同的插入顺序。near 查询为数据点加 0.2% 对角线的高斯噪声。

| 数据 | 原 DPNN | 本实现（批量查询） | 加速 | 内存 |
|---|---|---|---|---|
| bunny，BRIO | 0.77 / 0.83 | 3 层 0.50 / 0.58 | 1.5 / 1.4 倍 | 6.9 MB → 1.0 GB |
| Lucy，BRIO | 1.64 / 2.19 | 2 层 float 0.78 / 1.10 | 2.1 / 2.0 倍 | 97 MB → 3.7 GB |
| Lucy，FPS | 1.49 / 1.71 | 2 层 float 0.49 / 0.60 | 3.0 / 2.9 倍 | 87 MB → 2.9 GB |

- **加速主要来自批量查询**：单查询提升小得多，bunny 基本持平，Lucy 约 1.1–2 倍。
- **比较次数少不等于快**：查询时间主要由每次跳转的依赖访存决定。只换精炼列表、不换布局，时间几乎不变。
- **决策树**：比同结构的顺序扫描慢 1.3–2 倍。它是比较次数这一指标上的改进，不是提速手段。

### 构建时间

48 线程，端到端：

| 数据 | 原 DPNN | 1 层 | 2 层 | 3 层 |
|---|---|---|---|---|
| bunny 73K | 0.53 s | 1.4 s | 3.0 s | 6.2 s |
| Lucy 1.02M | 8.5 s | 21 s | 45 s（峰值 15 GB） | – |

- **只精炼粗层**：只精炼前 n/16 个点（`--coarse 16`）时，Lucy 3 层共 26 s，2.0 GB（float 版 1.0 GB），速度接近全量 2 层。
- **决策树**：线性规划次数多一个数量级（Lucy 2 层约 109 亿次）。

## 2. 方法要点

- **定理 1**：查询"从 u 跳到 v"的区域
  Q(u→v) = Cell(u; V_{t_v−1}) ∩ Cell(v; V_{t_v})，
  恰好等于 v 插入时的 Delaunay 邻居 Parents(v) 给出的约 16 个平分半空间之交。所以只需要原 DAG 加点坐标，不构造 Voronoi 胞腔，也不维护随时间收缩的区域。
- **定理 2**：查询沿边 e₁…e_r 到达 v 后，候选 w 需要比较，当且仅当
  Q(e₁)∩…∩Q(e_r)∩Q(v→w) ≠ ∅。
  这是一个 3D 线性可行性问题，用 Seidel LP 判定，期望 O(m)。记住最近 1 次 / k 次 / 全部跳转，就分别得到 1 层 / k 层 / 全部层（trie）的表。
- **定理 3**：同一状态内，各候选对应的结果区域两两不交，并覆盖整个状态区域。所以测试可以按任意顺序做，LP 判定只剩一个结果即可停止。决策树就建立在这一点上。
- **精确性**：所有半空间放宽 1e-6，只会多留候选、不会误删，查询路径与原 DPNN 完全相同。
- **纯组合判据不可行**：只看 Delaunay 邻接、不看坐标的规则，会漏掉 15–16% 的必要候选，使 59–83% 的查询答错。所以坐标谓词省不掉，这里把它压缩到了最小形式。
- **查询布局**：
  - 每个状态的候选连续存为 `{x, y, z, next}`，每次跳转只有一次依赖访存；
  - 16 个查询交错推进，并开启透明大页；
  - 可选 float 坐标：带严格误差界，必要时回退到 double，结果不变、内存减半。

## 3. 快速使用

```cpp
#include "dpnn_refined.h"           // DPNN/src

std::vector<dpl::Vec3> points = ...;
dpl::RefinedOptions opt;                 // 默认：BRIO 顺序、2 层、double 坐标
opt.order = dpl::InsertionOrder::Fps;    // 推荐：FPS 顺序跳转次数约减半
opt.levels = 2;                          // 0 = 原 DPNN
opt.floatCoords = true;                  // 可选：内存减半，结果不变
// opt.coarseDivisor = 16;               // 可选：只精炼前 n/16 个点，省内存
dpl::RefinedDPNN nn(points, opt);

uint32_t i = nn.nearest(q);                          // 返回输入数组中的下标
nn.nearestBatch(Q.data(), Q.size(), out.data());     // 批量查询，更快
```

**推荐配置**：FPS 顺序 + 1–2 层 + 批量查询；内存紧张时用 float 版，或 `coarseDivisor = 16`。

## 4. 目录结构

```
DPNN/                      原 DPNN（src/dpnn.cpp、src/main.cpp 未改动，提交历史保留）+ 新增实现
  LAYERED_README.md        主线完整文档
  Makefile.layered         新增程序的编译脚本（不依赖 CMake）
  data/                    bunny.xyz（72,911 点）、lucy.xyz（1,018,219 点）
  src/
    lp3.h                  3D 线性可行性求解（Seidel，支持共享前缀增量求解）
    dpnn_layered.h         DAG、0/k 层、全部层（trie）、只精炼粗层
    dpnn_flat.h            扁平布局、批量查询、大页、float 版
    dpnn_refined.h         RefinedDPNN 封装
    dpnn_compact.h         在原布局上的位掩码编码
    dpnn_dtree.h           状态内 LP 认证决策树
    bench_layered.cpp      构建、比较次数、计时、逐点比对与暴力验证
    bench_dtree.cpp        决策树与顺序扫描的对比
    analyze_decision.cpp   比较次数拆分、熵、神谕下界
    analyze_combinatorial.cpp  纯组合判据的反例实验
    test_lp3.cpp           LP 求解器交叉验证
    test_layered.cpp       退化输入（重复点、整数网格、共面、共线、球面点）的暴力比对
    example_refined.cpp    RefinedDPNN 最小示例
korder_knn/                order-k Voronoi 做 kNN（README.md、SURVEY.md、src/、results/）
results/                   主线实验日志与复现脚本
  final/  build/  decision/  compare/  prelim/
  run_all.sh  run_build.sh  run_dtree.sh  make_tables.py
deps/stubs/                编译原版 dpnn.cpp 用的空头文件
```

## 5. 依赖与编译

| 依赖 | 用途 |
|---|---|
| C++17 编译器（测试用 GCC）、OpenMP、Boost | 必需 |
| CGAL 5.6.1（header-only） | 必需 |
| Eigen3 | 只在与原版 `dpnn.cpp` 对比时需要 |
| TBB、nanoflann | 只有 `korder_knn` 需要 |

第三方源码不在仓库里，请放到 `deps/` 下（Makefile 的默认路径）：

```bash
# CGAL 5.6.1：从 https://github.com/CGAL/cgal/releases/tag/v5.6.1 下载，解压为 deps/CGAL-5.6.1
git clone https://github.com/jlblancoc/nanoflann deps/nanoflann
# 测试所用的 nanoflann 版本：18f4edf753e813644a7a12711a86139777a8ef00

cd DPNN && make -f Makefile.layered            # 产物在 DPNN/build_layered/
                                                # 也可指定 CGAL_INC=/path/to/CGAL/include
cd ../korder_knn && make                        # 产物在 korder_knn/build/
```

编译使用 `-ffp-contract=off`，避免 FMA 让同一距离在不同代码位置舍入不同，原因见 `results/compare/NOTES.md`。

## 6. 复现实验

```bash
cd DPNN
./build_layered/test_lp3
./build_layered/test_layered                                       # 退化输入暴力比对
./build_layered/bench_layered data/bunny.xyz --levels 3 --trie --original
./build_layered/bench_layered data/lucy.xyz --order fps --levels 2 --check 5000
./build_layered/bench_dtree data/bunny.xyz --order fps --levels 3

../results/run_all.sh      # 查询表格（约 1.5 小时）     → results/final/
../results/run_build.sh    # 构建计时 + 全量暴力验证（约 30 分钟） → results/build/
../results/run_dtree.sh    # 决策树、神谕下界、计时        → results/decision/
```

- **测试环境**：Xeon Platinum 8163（2×24 核），查询单线程，构建 48 线程。
- **计时方法**：机器与他人共享，所以采用 5 轮交错计时取最小值：每轮把所有方法各跑一遍。10–15% 以内的差异视为噪声。

## 7. 支线：order-k Voronoi 做 kNN

- **原理**：order-k Voronoi 图就是 k 元集质心的 power diagram（权重为负方差）。在上面跑 DPNN，一次点定位就得到 k 个近邻，结果精确。
- **规模是硬伤**：单元数 ≈ 3.4k²n。bunny 在 k=8 时查询图已达 8.7 GB，所以纯 order-k DPNN 不实用。
- **有用的副产品是精确候选表 Cx(u)**：每点约 3.2k+10 项，先找最近点再扫描这张表，是 box 查询中最快的精确方法：

  | 数据 | NN + Cx(u) | Delaunay 扩展 | 后继表 | kd-tree |
  |---|---|---|---|---|
  | bunny，k=8 | **1.41 µs**（批量 1.11） | 2.67 | 2.77 | 7.03 |
  | Lucy，k=4 | **1.46 µs**（批量 1.17） | 2.38 | 2.60 | 12.0 |

- **near 查询**：贴近曲面时，单查询仍是 kd-tree 最快。
- **瓶颈在构建**：Lucy k=4 峰值内存 65 GB。下一步是不建整张 order-k 图、直接求 Cx(u)。

## 8. 局限

- **全部层（trie）规模大**：
  - 曲面数据约 120·n·ln n 个状态，bunny 已是 5.8 GB、构建 87 s；
  - 网格、球面这类高度退化的输入会指数膨胀。
  - 实用配置是 1–3 层，或只精炼粗层。
- **查询范围**：精炼表只适用于离数据中心 1000 倍对角线以内的查询，更远的查询请用 0 层表。
- **内存**：精炼表比原 DPNN 大一到两个数量级，可以用 float 版或 `--coarse 16` 缓解。
- **构建**：DAG 构建是串行的（每个点的父节点是它插入那一刻的 Delaunay 邻居），使用精确谓词，比原版略慢。
- **下界**：凡是"逐步证明 DPNN 路径"的精确方法，都不可能低于神谕下界。要再降比较次数，需要只证明最终答案的新结构。

## 来源与许可

`DPNN/` 下的原始代码来自 [sssomeone/DPNN](https://github.com/sssomeone/DPNN)（commit `1139444`，提交历史保留在本仓库中），MIT 许可，见 [`DPNN/LICENSE`](DPNN/LICENSE)。
