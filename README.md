# dpnn-refined

在 DPNN（*Efficient Nearest Neighbor Search Using Dynamic Programming*，arXiv:2409.15023）基础上的两条研究线：

1. **只用 Delaunay 信息减少 DPNN 的比较次数**（主线，`DPNN/`）：
   - 用查询的历史跳转删掉不可能赢的候选，判定只需要 DAG 父节点的平分半空间和一个 3D 小线性规划，不做几何切割；
   - 有 1 层 / k 层 / 全部层三种表，以及状态内的 LP 认证决策树；
   - 查询布局：扁平表 + 批量交错查询。
   - 所有版本对每个查询返回的点都与原 DPNN 相同。
2. **用 order-k Voronoi 图做 kNN**（支线，`korder_knn/`）：
   - 文献调研；
   - order-k 图的构造与查询；
   - 精确候选表 Cx(u)；
   - 与 kd-tree、Delaunay 扩展、后继表的对比。

## 目录

| 路径 | 内容 |
|---|---|
| `DPNN/` | 原 DPNN 代码（`src/dpnn.cpp`、`src/main.cpp`，未改动；提交历史保留）+ 新增实现 |
| `DPNN/LAYERED_README.md` | 主线文档：方法、证明、复杂度、全部实验表格 |
| `korder_knn/` | 支线代码；`README.md` 是结论与实验，`SURVEY.md` 是文献调研 |
| `results/` | 主线实验日志与复现脚本（`run_all.sh`、`run_build.sh`、`run_dtree.sh`） |
| `deps/stubs/` | 编译原版 `dpnn.cpp` 用的空头文件 |

## 主要结果（比较次数，box 查询，括号内为 log₂n 的倍数）

| 数据 | 原 DPNN | 1 层 | 多层 | 多层 + 决策树 |
|---|---|---|---|---|
| bunny 73K，FPS | 135.6（8.4） | 73.1（4.5） | 3 层 51.3（3.2） | 40.3（2.5） |
| Lucy 1.02M，FPS | 236.3（11.8） | 105.4（5.3） | 2 层 79.3（4.0） | 64.3（3.2） |

- **查询时间**：批量查询比原版快 1.4–3 倍（Lucy FPS 2 层 float：1.49 → 0.49 µs）。
- **决策树**：只减少比较次数，查询时间反而比顺序扫描慢。

详见 `DPNN/LAYERED_README.md` §0。

## 依赖与编译

依赖：
- C++17 编译器、OpenMP、Boost；
- CGAL 5.6.1（header-only）；
- Eigen3：只在与原版对比时需要；
- TBB 与 nanoflann：只有 `korder_knn` 需要。

```bash
# 第三方源码不在仓库里，放到 deps/ 下（Makefile 的默认路径）
#   CGAL 5.6.1: https://github.com/CGAL/cgal/releases/tag/v5.6.1 ，解压为 deps/CGAL-5.6.1
#   nanoflann:  git clone https://github.com/jlblancoc/nanoflann deps/nanoflann
#               （测试用的版本：18f4edf753e813644a7a12711a86139777a8ef00）

cd DPNN && make -f Makefile.layered          # 产物在 DPNN/build_layered/
./build_layered/test_layered                  # 退化输入的暴力比对
../results/run_dtree.sh                       # 例：重跑决策树实验

cd ../korder_knn && make                      # 产物在 korder_knn/build/
```

数据 `DPNN/data/bunny.xyz`、`DPNN/data/lucy.xyz` 随原仓库一起提供。

## 来源与许可

`DPNN/` 下的原始代码来自 [sssomeone/DPNN](https://github.com/sssomeone/DPNN)（commit `1139444`），MIT 许可，见 `DPNN/LICENSE`。
