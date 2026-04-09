"""
Synthetic data augmentation for FlipPhone trick data.

Based on Corrêa et al. (2017): triangle-based geometric signal modelling
with Gaussian parameter variation, moving-average smoothing, and noise injection.

Usage:
    python3 augment.py --input data/dataset.csv --output data/dataset_augmented.csv --per-class 50
    python3 augment.py --input data/dataset.csv --output data/dataset_augmented.csv --per-class 100 --seed 7
"""

import argparse
import os
import uuid

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
NORM_LEN = 100  # internal resolution for statistics


# ── Signal primitives ─────────────────────────────────────────────────


def interpolate_to_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate 1-D signal to target_len points."""
    if len(signal) == target_len:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal)


def triangle_pulse(length: int, peak_pos: float, half_width: float, amplitude: float) -> np.ndarray:
    """
    Single triangular pulse over [0, 1].

    peak_pos:   normalised position of the peak  (0 … 1)
    half_width: half-width of the triangle        (fraction of total length)
    amplitude:  signed peak amplitude
    """
    t = np.linspace(0, 1, length)
    left = peak_pos - half_width
    right = peak_pos + half_width
    sig = np.zeros(length)

    mask_rise = (t >= left) & (t <= peak_pos)
    if mask_rise.any():
        sig[mask_rise] = amplitude * (t[mask_rise] - left) / max(peak_pos - left, 1e-9)

    mask_fall = (t > peak_pos) & (t <= right)
    if mask_fall.any():
        sig[mask_fall] = amplitude * (right - t[mask_fall]) / max(right - peak_pos, 1e-9)

    return sig


# ── Per-axis statistics ───────────────────────────────────────────────


def _axis_stats(recordings: list[np.ndarray]) -> dict:
    """
    Derive triangle parameters from a set of real recordings (one axis).

    Returns dict with mean/std for: peak_pos, amplitude, half_width, baseline, noise.
    """
    normalized = [interpolate_to_length(r, NORM_LEN) for r in recordings]
    stack = np.stack(normalized)          # (n, NORM_LEN)
    mean_sig = stack.mean(axis=0)
    std_sig = stack.std(axis=0)

    # Dominant peak in the mean signal
    abs_mean = np.abs(mean_sig)
    peak_idx = int(np.argmax(abs_mean))

    # Per-recording peak positions and amplitudes
    peak_positions = []
    peak_amplitudes = []
    for r in normalized:
        abs_r = np.abs(r)
        pi = int(np.argmax(abs_r))
        peak_positions.append(pi / NORM_LEN)
        peak_amplitudes.append(r[pi])

    # Half-width from mean signal (half-power width)
    half_max = abs_mean[peak_idx] / 2
    above = np.where(abs_mean >= half_max)[0]
    half_width = (above[-1] - above[0]) / (2 * NORM_LEN) if len(above) > 1 else 0.1

    return {
        "peak_pos_mean": float(np.mean(peak_positions)),
        "peak_pos_std":  float(max(np.std(peak_positions), 0.03)),
        "amp_mean":      float(np.mean(peak_amplitudes)),
        "amp_std":       float(max(np.std(peak_amplitudes), abs(np.mean(peak_amplitudes)) * 0.1 + 0.05)),
        "width_mean":    float(half_width),
        "width_std":     float(half_width * 0.25),
        "baseline_mean": float(mean_sig.mean()),
        "baseline_std":  float(std_sig.mean() * 0.5),
        "noise_std":     float(max(std_sig.mean() * 0.35, 0.02)),
    }


# ── Core generator ────────────────────────────────────────────────────


def generate_synthetic_flip(
    reference_signals: list[list[dict]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[list[dict]]:
    """
    Generate synthetic IMU recordings for a single trick class.

    reference_signals : list of real recordings; each recording is a list of
                        row-dicts with keys t, ax, ay, az, gx, gy, gz.
    n_samples         : number of synthetic recordings to produce.
    rng               : seeded numpy Generator for reproducibility.

    Returns a list of n_samples synthetic recordings in the same format.
    """
    # Per-axis arrays from references
    axis_arrays: dict[str, list[np.ndarray]] = {
        ax: [np.array([row[ax] for row in rec]) for rec in reference_signals]
        for ax in AXES
    }
    axis_stats = {ax: _axis_stats(axis_arrays[ax]) for ax in AXES}

    t_arrays = [np.array([row["t"] for row in rec]) for rec in reference_signals]

    synthetic = []
    for _ in range(n_samples):
        # Base timing on a randomly picked reference recording
        ref_t = t_arrays[rng.integers(0, len(t_arrays))]
        n = len(ref_t)

        generated: dict[str, np.ndarray] = {}
        for ax in AXES:
            s = axis_stats[ax]

            # Sample triangle parameters with Gaussian variation
            peak_pos  = float(np.clip(rng.normal(s["peak_pos_mean"], s["peak_pos_std"]), 0.1, 0.9))
            amplitude = float(rng.normal(s["amp_mean"], s["amp_std"]))
            half_w    = float(np.clip(rng.normal(s["width_mean"], s["width_std"]), 0.04, 0.45))
            baseline  = float(rng.normal(s["baseline_mean"], s["baseline_std"]))

            # 1. Geometric signal (triangle)
            sig = baseline + triangle_pulse(n, peak_pos, half_w, amplitude)

            # 2. Smooth — moving average applied twice (2nd order)
            window = max(3, n // 20)
            sig = uniform_filter1d(sig, size=window)
            sig = uniform_filter1d(sig, size=window)

            # 3. Add realistic noise
            sig += rng.normal(0, s["noise_std"], size=n)

            generated[ax] = sig

        rows = [
            {"t": float(ref_t[i]), **{ax: float(generated[ax][i]) for ax in AXES}}
            for i in range(n)
        ]
        synthetic.append(rows)

    return synthetic


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment FlipPhone trick dataset")
    parser.add_argument("--input",     default="data/dataset.csv",            help="Input CSV")
    parser.add_argument("--output",    default="data/dataset_augmented.csv",  help="Output CSV")
    parser.add_argument("--per-class", type=int, default=50, dest="per_class",
                        help="Minimum recordings per trick after augmentation (default 50)")
    parser.add_argument("--seed",      type=int, default=42,                  help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    if "source" not in df.columns:
        df["source"] = "real"

    tricks = df["trick"].unique()
    print(f"Tricks found: {list(tricks)}")
    print(f"Target per class: {args.per_class}\n")

    synthetic_rows: list[dict] = []

    for trick in tricks:
        trick_df    = df[df["trick"] == trick]
        rec_ids     = trick_df["id"].unique()
        n_real      = len(rec_ids)
        n_needed    = max(0, args.per_class - n_real)

        print(f"  {trick:<20} {n_real:>3} real  →  +{n_needed} synthetic")

        if n_needed == 0:
            continue

        # Build reference signal list
        references = []
        for rid in rec_ids:
            rec_df = trick_df[trick_df["id"] == rid].sort_values("t")
            references.append(rec_df[["t"] + AXES].to_dict("records"))

        new_recordings = generate_synthetic_flip(references, n_needed, rng)

        # Attach metadata columns
        meta_pool = trick_df.groupby("id").first()[
            ["timestamp", "durationMs", "sampleRateHz"]
        ]
        for rows in new_recordings:
            new_id   = str(uuid.uuid4())
            meta_rid = rng.choice(rec_ids)
            meta     = meta_pool.loc[meta_rid]

            for row in rows:
                synthetic_rows.append({
                    "id":          new_id,
                    "trick":       trick,
                    "timestamp":   meta["timestamp"],
                    "durationMs":  meta["durationMs"],
                    "sampleCount": len(rows),
                    "sampleRateHz": meta["sampleRateHz"],
                    "collector":   "synthetic",
                    "source":      "synthetic",
                    **row,
                })

    if synthetic_rows:
        out_df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        out_df = df

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    n_synth_recs = len({r["id"] for r in synthetic_rows})
    print(f"\nSaved → {args.output}")
    print(f"  Real recordings:      {df['id'].nunique()}")
    print(f"  Synthetic recordings: {n_synth_recs}")
    print(f"  Total rows:           {len(out_df)}")


if __name__ == "__main__":
    main()
