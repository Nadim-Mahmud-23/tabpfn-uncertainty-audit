#!/usr/bin/env python3
"""Authoritative 4-base-model pooled values for the RQ3 tables/prose.

Pools LAC rows over the four base models (drops tabpfn_temp) and prints every
value the manuscript quotes, so each fill is traceable. Run from repo root.
"""
import numpy as np
import pandas as pd

BASE = ["tabpfn", "xgboost", "lightgbm", "mlp"]
df = pd.read_csv("results/conformal_fairness.csv")
lac = df[(df.score == "lac") & (df.model.isin(BASE))].copy()


def pooled(axis, method, alpha, col):
    s = lac[(lac.axis == axis) & (lac.method == method) & np.isclose(lac.alpha, alpha)]
    # model-average within (dataset,seed), then mean over cells
    cell = s.groupby(["dataset", "seed"])[col].mean()
    return cell.mean()


AX = ["sex", "age", "race", "predclass"]
print("=== Table rq3 @90% (alpha=0.1): gap / WGC / mean size, marg -> mond ===")
for ax in AX:
    g = (pooled(ax, "marginal", 0.1, "coverage_gap"), pooled(ax, "mondrian", 0.1, "coverage_gap"))
    w = (pooled(ax, "marginal", 0.1, "worst_group_coverage"), pooled(ax, "mondrian", 0.1, "worst_group_coverage"))
    s = (pooled(ax, "marginal", 0.1, "mean_set_size"), pooled(ax, "mondrian", 0.1, "mean_set_size"))
    print(f"  {ax:10s} gap {g[0]:.4f}->{g[1]:.4f}  WGC {w[0]:.4f}->{w[1]:.4f}  size {s[0]:.4f}->{s[1]:.4f}")

print("\n=== Table tax: set_size_disparity @90%, marg -> mond ===")
for ax in AX:
    d = (pooled(ax, "marginal", 0.1, "set_size_disparity"), pooled(ax, "mondrian", 0.1, "set_size_disparity"))
    print(f"  {ax:10s} {d[0]:.4f} -> {d[1]:.4f}")

print("\n=== Table rq3targets: coverage gap by target, marg -> mond ===")
for alpha, tgt in [(0.2, "80%"), (0.1, "90%"), (0.05, "95%")]:
    row = {ax: (pooled(ax, "marginal", alpha, "coverage_gap"), pooled(ax, "mondrian", alpha, "coverage_gap")) for ax in ["sex", "age", "race"]}
    print(f"  {tgt}: " + "  ".join(f"{ax} {v[0]:.4f}->{v[1]:.4f}" for ax, v in row.items()))

print("\n=== per-dataset RACE gap @90%, marg -> mond ===")
for ds in ["ACSEmployment-CA", "ACSIncome-CA", "ACSPublicCoverage-CA"]:
    s = lac[(lac.axis == "race") & np.isclose(lac.alpha, 0.1) & (lac.dataset == ds)]
    m = s[s.method == "marginal"].groupby("seed")["coverage_gap"].mean().mean()
    d = s[s.method == "mondrian"].groupby("seed")["coverage_gap"].mean().mean()
    print(f"  {ds:24s} {m:.4f} -> {d:.4f}")

print("\n=== age set-size cost @90% (mond - marg) ===")
am = pooled("age", "marginal", 0.1, "mean_set_size")
ad = pooled("age", "mondrian", 0.1, "mean_set_size")
print(f"  {am:.4f} -> {ad:.4f}  (cost +{ad-am:.4f})")

print("\n=== WGC age @80%/95% marg->mond ===")
for alpha, tgt in [(0.2, "80%"), (0.05, "95%")]:
    print(f"  {tgt}: WGC {pooled('age','marginal',alpha,'worst_group_coverage'):.4f} -> {pooled('age','mondrian',alpha,'worst_group_coverage'):.4f}")
