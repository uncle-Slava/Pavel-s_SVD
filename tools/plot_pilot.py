"""Static scientific figure; full-spectrum and rank-k results are separate panels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = [json.loads(x) for x in Path("results/rtx4000-pilot.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x.get("type") == "measurement" and x["status"] == "ok"]
    def row(case, name):
        return next(x for x in rows if x["case"]["name"] == case and x["algorithm"]["name"] == name)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    methods = ["gesvd", "gesvdj", "gesvda", "gram"]
    colors = ["#275D8C", "#257F75", "#935E9D", "#C17C2F"]
    for name, c in zip(methods, colors):
        x = row("square1024_fp32", name)
        axes[0].scatter(x["resident_solver"]["device_event"]["median_ms"], x["metrics"]["worst"]["reconstruction_relative_fro"], s=65, color=c)
        axes[0].annotate(name, (x["resident_solver"]["device_event"]["median_ms"], x["metrics"]["worst"]["reconstruction_relative_fro"]), xytext=(5,5), textcoords="offset points")
    axes[0].set(xscale="log", yscale="log", xlabel="CUDA event median (ms)", ylabel="Relative reconstruction error", title="Full spectrum · Gaussian 1024²", xlim=(6,130))
    axes[0].grid(alpha=.2)
    low = row("lowrank1024_k32", "randomized")
    flat = row("square1024_fp32", "randomized")
    residual = [x["metrics"]["worst"]["reconstruction_relative_fro"] for x in (low,flat)]
    axes[1].bar(["Decaying / low rank", "Flat / Gaussian"], residual, color=["#257F75", "#C17C2F"],width=.55)
    axes[1].set(yscale="log", ylim=(5e-5, 5), ylabel="Relative reconstruction error", title="Same rank-32 algorithm · 1024²")
    for i,v in enumerate(residual):
        axes[1].text(i,v*1.3,f"{v:.3g}",ha="center")
    axes[1].grid(axis="y",alpha=.2)
    names=["gesvd","gesvdj","gesvda","randomized"]
    vals=[row("lowrank1024_k32",n)["memory"]["allocator_peak_increment_bytes"]/2**20 for n in names]
    axes[2].barh(["gesvd (all)","gesvdj (all)","gesvda (all)","randomized (k=32)"],vals,color=colors[:3]+["#334E68"])
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Peak allocator increase (MiB)", title="Memory · low-rank input 1024²",xlim=(0,75))
    for i,v in enumerate(vals):
        axes[2].text(v+1,i,f"{v:.1f}",va="center")
    axes[2].grid(axis="x",alpha=.2)
    fig.suptitle("SVD pilot · RTX PRO 4000 Blackwell SFF · FP32", fontsize=15)
    fig.text(.5,.01,"2 warm-ups, 5 measured calls, TF32 disabled. Pilot observations; outputs/accuracy differ. Native workspace is not inferred from allocator peaks.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92))
    fig.savefig("results/pilot.png",dpi=180)
    fig.savefig("results/pilot.svg")


if __name__ == "__main__":
    main()
