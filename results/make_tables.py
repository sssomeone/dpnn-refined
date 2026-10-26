#!/usr/bin/env python3
"""Builds the Markdown tables of LAYERED_README.md from results/final/*.txt."""
import math
import os
import re
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final")


def read(name):
    p = os.path.join(D, name)
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""


def rows(txt, qset):
    """{label: [numbers...]} for one query set block."""
    out = {}
    m = re.search(r"== queries: " + qset + r" \(\d+\)\n(.*?)\nbrute force", txt, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines()[1:]:
        label, rest = line[:22].strip(), line[22:].split()
        out[label] = rest
    return out


def cmp_(r, label):
    return float(r[label][0]) if label in r else None


def us_single(r, label):  # "flat Lk single": cmp jumps us agree
    return float(r[label][2]) if label in r else None


def us_plain(r, label):  # batch / f32 / original lines: us agree
    return float(r[label][0]) if label in r else None


def mem(txt):
    out = {}
    m = re.search(r"memory: (.*)", txt)
    if m:
        out["orig"] = float(re.search(r"dpnn.cpp layout ([\d.]+) MB", m.group(1)).group(1))
        for name, a, b in re.findall(r"flat (\S+) ([\d.]+) MB \(f32: ([\d.]+) MB\)", m.group(1)):
            out[name] = (float(a), float(b))
    return out


def fmt_mb(x):
    return f"{x / 1024:.1f} GB" if x >= 1000 else f"{x:.0f} MB" if x >= 10 else f"{x:.1f} MB"


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def timing_table(txt, title, names, with_f32=(), orig_label="dpnn.cpp (original)", orig_name="dpnn.cpp 原版", extra=()):
    rb, rn = rows(txt, "box2x"), rows(txt, "near")
    mm = mem(txt)
    if not mm:
        print(f"**{title}**: (no results yet)\n")
        return
    print(f"**{title}**\n")
    print("| 索引 | box 比较 | box 单查询 µs | box 批量 µs | near 比较 | near 单查询 µs | near 批量 µs | 内存 |")
    print("|---|---|---|---|---|---|---|---|")
    ob, on = us_plain(rb, orig_label), us_plain(rn, orig_label)
    if ob is None:  # no --original run: use the dpnn.cpp layout row
        ob, on = us_single(rb, "dpnn.cpp layout"), us_single(rn, "dpnn.cpp layout")
    print(f"| {orig_name} | {f(cmp_(rb, 'dpnn.cpp layout'), 1)} | {f(ob)} | – | {f(cmp_(rn, 'dpnn.cpp layout'), 1)} | {f(on)} | – | {fmt_mb(mm['orig'])} |")
    for key, label in names:
        print(f"| {label} | {f(cmp_(rb, f'flat {key} single'), 1)} | {f(us_single(rb, f'flat {key} single'))} | "
              f"{f(us_plain(rb, f'flat {key} batch16'))} | {f(cmp_(rn, f'flat {key} single'), 1)} | "
              f"{f(us_single(rn, f'flat {key} single'))} | {f(us_plain(rn, f'flat {key} batch16'))} | {fmt_mb(mm[key][0])} |")
        if key in with_f32:
            print(f"| {label}，float | {f(cmp_(rb, f'flat {key} single'), 1)} | {f(us_plain(rb, f'f32 {key} single'))} | "
                  f"{f(us_plain(rb, f'f32 {key} batch16'))} | {f(cmp_(rn, f'flat {key} single'), 1)} | "
                  f"{f(us_plain(rn, f'f32 {key} single'))} | {f(us_plain(rn, f'f32 {key} batch16'))} | {fmt_mb(mm[key][1])} |")
    for row in extra:
        print(row)
    print()


def coarse_rows(txt, key, label, with_f32=False):
    rb, rn = rows(txt, "box2x"), rows(txt, "near")
    mm = mem(txt)
    if key not in mm:
        return []
    out = [f"| {label} | {f(cmp_(rb, f'flat {key} single'), 1)} | {f(us_single(rb, f'flat {key} single'))} | "
           f"{f(us_plain(rb, f'flat {key} batch16'))} | {f(cmp_(rn, f'flat {key} single'), 1)} | "
           f"{f(us_single(rn, f'flat {key} single'))} | {f(us_plain(rn, f'flat {key} batch16'))} | {fmt_mb(mm[key][0])} |"]
    if with_f32:
        out.append(f"| {label}，float | {f(cmp_(rb, f'flat {key} single'), 1)} | {f(us_plain(rb, f'f32 {key} single'))} | "
                   f"{f(us_plain(rb, f'f32 {key} batch16'))} | {f(cmp_(rn, f'flat {key} single'), 1)} | "
                   f"{f(us_plain(rn, f'f32 {key} single'))} | {f(us_plain(rn, f'f32 {key} batch16'))} | {fmt_mb(mm[key][1])} |")
    return out


def main():
    # comparisons summary
    print("### comparisons (box2x)\n")
    for name, fn in [("bunny random", "bunny_random.txt"), ("bunny FPS", "bunny_fps.txt"),
                     ("bunny BRIO", "bunny_brio_full.txt"), ("uniform 20K BRIO", "uniform20k_levels8.txt"),
                     ("Lucy BRIO", "lucy_brio.txt"), ("Lucy FPS", "lucy_fps.txt")]:
        r = rows(read(fn), "box2x")
        vals = [cmp_(r, f"flat L{k} single") for k in range(4)] + [cmp_(r, "flat trie single")]
        print(f"| {name} | " + " | ".join(f(v, 1) for v in vals) + " |")
    print()
    # level convergence
    t = read("uniform20k_levels8.txt")
    r = rows(t, "box2x")
    ents = [int(x) for x in re.findall(r"level \d+: \d+ states, (\d+) entries", t)]
    per = [float(x) for x in re.findall(r"\((\d+\.\d+) / state\)", t)]
    print("### level convergence (uniform 20K)\n")
    print("entries/pt:", [round(e / 20000) for e in ents])
    print("per state:", per)
    print("cmp:", [cmp_(r, f"flat L{k} single") for k in range(9)])
    tri = read("uniform_trie_20000.txt")
    print("trie cmp:", cmp_(rows(tri, "box2x"), "flat trie single"))
    print()
    # trie scaling
    print("### trie scaling\n")
    for fn, n in [(f"uniform_trie_{n}.txt", n) for n in (2500, 5000, 10000, 20000, 40000)] + \
                 [(f"bunny_trie_{n}.txt", n) for n in (5000, 10000, 20000, 40000)] + [("bunny_brio_full.txt", 72911)]:
        m = re.search(r"trie: (\d+) nodes \(([\d.]+) / point\), max depth (\d+), (\d+) LPs \(avg ([\d.]+) half-spaces\), ([\d.]+)s",
                      read(fn))
        if m:
            per_pt = float(m.group(2))
            print(f"{fn}: {per_pt:.0f}/pt  /ln n = {per_pt / math.log(n):.0f}  depth {m.group(3)}  LPs {int(m.group(4)) / 1e6:.0f}M  "
                  f"avg {m.group(5)}  {m.group(6)}s")
        rb = rows(read(fn), "box2x")
        if "flat trie single" in rb:
            print(f"   cmp L0 {cmp_(rb, 'flat L0 single')}  trie {cmp_(rb, 'flat trie single')}  jumps {rb['flat L0 single'][1]}")
    print()
    # timing tables
    timing_table(read("bunny_brio_full.txt"), "bunny（72,911 点，BRIO）",
                 [("L0", "0 层 扁平"), ("L1", "1 层 扁平"), ("L2", "2 层 扁平"), ("L3", "3 层 扁平"), ("trie", "全部层 扁平")],
                 extra=coarse_rows(read("bunny_brio_coarse16.txt"), "L3", "3 层，只精炼前 n/16"))
    timing_table(read("lucy_brio.txt"), "Lucy（1,018,219 点，BRIO）",
                 [("L0", "0 层 扁平"), ("L1", "1 层 扁平"), ("L2", "2 层 扁平")], with_f32=("L1", "L2"),
                 extra=coarse_rows(read("lucy_brio_coarse16.txt"), "L3", "3 层，只精炼前 n/16", True))
    timing_table(read("lucy_fps.txt"), "Lucy（FPS 顺序）",
                 [("L0", "0 层 扁平"), ("L1", "1 层 扁平"), ("L2", "2 层 扁平")], with_f32=("L1", "L2"),
                 orig_label="__none__", orig_name="dpnn.cpp 布局 + FPS DAG")
    # build lines
    print("### build times\n")
    for fn in ["bunny_brio_full.txt", "bunny_brio_coarse16.txt", "bunny_random.txt", "bunny_fps.txt", "lucy_brio.txt",
               "lucy_brio_coarse16.txt", "lucy_fps.txt"]:
        t = read(fn)
        print(fn, re.findall(r"DAG: .*?([\d.]+)s\n", t), re.findall(r"level \d+: .*?, ([\d.]+)s", t),
              re.findall(r"trie: .*?([\d.]+)s", t), re.findall(r"original dpnn.cpp build: ([\d.]+)s", t))
    print()
    for fn in ["combinatorial_uniform20k.txt", "combinatorial_bunny.txt"]:
        print(fn)
        print(read(fn))
    # sanity
    bad = [fn for fn in sorted(os.listdir(D)) if fn.endswith(".txt") and "combinatorial" not in fn
           and "ALL CHECKS PASSED" not in read(fn)]
    print("files without ALL CHECKS PASSED:", bad)


if __name__ == "__main__":
    sys.exit(main())
