import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[
out = ROOT / "talk" / "figures" / "pre-asymptotic_analysis.png"
out.parent.mkdir(parents=True, exist_ok=True)

with open(ROOT / "result_values.json","r") as f:
    res = json.load(f)

n = np.array([1,2,4,8,16,32,64,128,256,512,1024], dtype=float)

ppm_p = res["PPM"]["parameters"]
def ppm_hn(n):
    return ppm_p["h"] + ppm_p["C"] * ppm_p["beta"] * (n ** (ppm_p["beta"] - 1.0))

def lm_ell(n, p):
    return p["L_inf"] + p["c"] * (n ** (p["beta"] - 1.0))

ppm_curve = ppm_hn(n)

models = {
    "1M (ctx=128)": res["1M_128"]["parameters"],
    "5M (ctx=512)": res["5M_512"]["parameters"],
    "10M (ctx=1024)": res["10M_1024"]["parameters"],
}

n1, n2 = 1.0, 1024.0
ppm1 = float(ppm_hn(np.array([n1]))[0])
ppm2 = float(ppm_hn(np.array([n2]))[0])

aligned = {}
params_ab = {}

for name, p in models.items():
    y1 = float(lm_ell(np.array([n1]), p)[0])
    y2 = float(lm_ell(np.array([n2]), p)[0])

    if abs(y2 - y1) < 1e-9:
        a = 1.0
        b = ppm1 - y1
    else:
        a = (ppm2 - ppm1) / (y2 - y1)
        b = ppm1 - a * y1

    params_ab[name] = {"a": a, "b": b}
    aligned[name] = a * lm_ell(n, p) + b

# Plot: affine-aligned visualization
plt.figure(figsize=(7.5, 5))
plt.plot(n, ppm_curve, linewidth=2.5, label="PPM baseline (conditional entropy fit)")

for name, y in aligned.items():
    plt.plot(n, y, "--", linewidth=2, label=name) 

plt.xscale("log")
plt.xlim(1, 1024)
plt.xlabel("Context length $n$ (tokens)")
plt.ylabel("Conditional entropy (bits)")
plt.title("Empirical Entropy Scaling vs Model Induced Entropy Scaling")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

# Save the alignment parameters for transparency
df = pd.DataFrame.from_dict(params_ab, orient="index")
df.index.name = "model"
csv_path = ROOT / "talk" / "figures" / "affine_alignment_params.csv"
df.to_csv(csv_path)

print(f"Saved figure to: {out}")
print(f"Saved params to: {csv_path}")
