import glob
import random
import math
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict, Counter
from math import log2


# Disjoint subset sampling

def sample_disjoint_subsets(pattern="shard*.parquet",
                            num_subsets=50,
                            subset_size=400_000):

    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} shards.")

    subsets = []

    for i in range(num_subsets):
        f = random.choice(files)
        table = pq.read_table(f, columns=["text"])
        texts = table["text"].to_pylist()

        # Shuffle for safety
        random.shuffle(texts)

        # Concatenate until we exceed subset_size
        buf = []
        total = 0
        for t in texts:
            buf.append(t)
            total += len(t)
            if total >= subset_size:
                break

        full = "".join(buf)

        # Choose disjoint slice inside this shard’s concatenated text
        if len(full) >= subset_size:
            start = random.randint(0, len(full) - subset_size)
            subset = full[start:start + subset_size]
        else:
            subset = full

        subsets.append(subset)
        print(f"Subset {i+1}/{num_subsets}: {len(subset)} chars from {f}")

    return subsets


# PPM-D model

class PPMD:
    """
    contexts[order][prefix_tuple][symbol] = count
    escape probability = D / (N + D)
    """

    def __init__(self, max_order=16):
        self.max_order = max_order
        self.contexts = [defaultdict(Counter) for _ in range(max_order+1)]

    def update(self, context, symbol):
        """Update counts in **all** context orders."""
        for order in range(self.max_order + 1):
            if order == 0:
                prefix = ()
            else:
                if len(context) < order:
                    continue
                prefix = tuple(context[-order:])
            self.contexts[order][prefix][symbol] += 1

    def symbol_prob(self, context, symbol):
        """
        True PPM-D backoff probability.
        """
        weight = 1.0
        p = 0.0

        for order in reversed(range(self.max_order + 1)):
            if order > 0 and len(context) < order:
                continue

            prefix = tuple(context[-order:]) if order > 0 else ()
            ctx = self.contexts[order][prefix]

            if not ctx:
                continue

            total = sum(ctx.values())
            distinct = len(ctx)
            escape = distinct / (total + distinct)

            if symbol in ctx:
                return p + weight * (ctx[symbol] / (total + distinct))
            else:
                weight *= escape

        # totally unseen symbol
        return p + weight * 1e-9

    def compress_prefix(self, seq, max_n):
        """
        Compute cumulative block entropy:
        H(n) = -sum_{t <= n} log2 p(x_t | prefix)

        Returns a numpy vector H[0:max_n].
        """
        H = np.zeros(max_n)
        total = 0.0
        context = []

        for t in range(max_n):
            sym = seq[t]
            prob = self.symbol_prob(context, sym)
            cost = -log2(prob)
            total += cost
            H[t] = total

            self.update(context, sym)
            context.append(sym)

        return H

    
# Master function

def hilberg_phase1_ppmd(subsets, max_n=400_000, order=16):

    H_all = []
    for i, seq in enumerate(subsets):
        print(f"\nPPM-D subset {i+1}/{len(subsets)}:")
        model = PPMD(max_order=order)
        H = model.compress_prefix(seq, max_n)
        H_all.append(H)

    H_all = np.array(H_all)   # shape = (50, max_n)

    # mean and confidence intervals
    mean_H = H_all.mean(axis=0)
    std_H  = H_all.std(axis=0)
    ci_low  = mean_H - 1.96 * (std_H / math.sqrt(len(subsets)))
    ci_high = mean_H + 1.96 * (std_H / math.sqrt(len(subsets)))

    return mean_H, ci_low, ci_high

# Run phase
if __name__ == "__main__":

    subsets = sample_disjoint_subsets(
        pattern="dataset_part*.parquet",
        num_subsets=50,
        subset_size=400_000,
    )

    mean_H, ci_low, ci_high = hilberg_phase1_ppmd(
        subsets=subsets,
        max_n=400_000,
        order=16
    )

    print("\n=== PHASE 1: Block Entropy Estimates (PPM-D) ===")
    for n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]:
        print(f"n={n:6d}:  H={mean_H[n]:10.3f}   CI=({ci_low[n]:.3f}, {ci_high[n]:.3f})")

    np.save("mean_H.npy", mean_H)
    np.save("ci_low.npy", ci_low)
    np.save("ci_high.npy", ci_high)
    print("\nSaved mean_H, ci_low, ci_high to disk.")
