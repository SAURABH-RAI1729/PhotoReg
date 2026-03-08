import os
import sys
import time
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset    import load_data, load_scaler
from evaluate        import load_model_from_checkpoint
from models.mlp      import count_parameters

# ============================================================
# infer.py
#
# Measures inference time for the trained model across a
# range of batch sizes.
#
# Protocol:
#   1. Warm up the model with WARMUP_BATCHES dummy forward passes
#      (important: first calls are slower due to JIT compilation
#       and memory allocation -- warmup gives a fair measurement)
#   2. Time N_REPEATS batches and average
#
# Reports:
#   - Hardware (CPU / GPU)
#   - Number of trainable parameters
#   - Per-batch and per-sample inference time for each batch size
#
# Usage:
#   python infer.py --config configs/config.yaml
# ============================================================

N_REPEATS = 200


def time_inference(model, X_tensor, batch_size, device, warmup=10):
    """
    Time how long the model takes per batch and per sample.

    Returns (avg_batch_ms, avg_sample_us).
    """

    model.eval()

    # Slice a fixed batch to use for timing
    batch = X_tensor[:batch_size].to(device)

    # Warmup -- not measured
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)

    # Synchronise GPU before starting the clock
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(N_REPEATS):
            _ = model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_s = time.perf_counter() - start

    avg_batch_ms  = (elapsed_s / N_REPEATS) * 1e3        # milliseconds
    avg_sample_us = avg_batch_ms / batch_size * 1e3       # microseconds

    return avg_batch_ms, avg_sample_us


def main(config_path):

    cfg = yaml.safe_load(open(config_path))

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch     = cfg["model"]["architecture"]
    ckpt_dir = cfg["checkpoints"]["dir"]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    ckpt_path = os.path.join(ckpt_dir, f"{arch}_best.pt")
    model, ckpt_cfg, arch = load_model_from_checkpoint(ckpt_path, device)
    model.eval()

    n_params = count_parameters(model)

    # --------------------------------------------------------
    # Prepare data tensor (just need raw inputs for timing)
    # --------------------------------------------------------

    mode_map = {"mlp": "flat", "cnn": "image", "deepsets": "deepsets"}

    _, val_ds, _ = load_data(
        data_path = cfg["data"]["path"],
        val_split = cfg["data"]["val_split"],
        seed      = seed,
        mode      = mode_map[arch]
    )

    # Pull out all X from val dataset as a single tensor
    X_val = val_ds.X   # already a tensor

    batch_sizes = cfg["inference"]["batch_sizes"]
    warmup      = cfg["inference"]["warmup_batches"]

    # Make sure we have enough samples for the largest batch
    max_bs = max(batch_sizes)
    if len(X_val) < max_bs:
        print(
            f"Warning: val set only has {len(X_val)} samples; "
            f"capping max batch size at {len(X_val)}"
        )
        batch_sizes = [bs for bs in batch_sizes if bs <= len(X_val)]

    # --------------------------------------------------------
    # Run timing benchmark
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"  Inference Benchmark")
    print(f"  Architecture : {arch.upper()}")
    print(f"  Hardware     : {device}")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Repeats      : {N_REPEATS}")
    print("=" * 65)
    print(
        f"  {'Batch size':>12} | "
        f"{'Avg time/batch (ms)':>22} | "
        f"{'Avg time/sample (µs)':>22}"
    )
    print("-" * 65)

    results = []

    for bs in batch_sizes:
        avg_batch_ms, avg_sample_us = time_inference(
            model, X_val, bs, device, warmup=warmup
        )
        print(
            f"  {bs:>12} | "
            f"{avg_batch_ms:>22.3f} | "
            f"{avg_sample_us:>22.3f}"
        )
        results.append((bs, avg_batch_ms, avg_sample_us))

    print("=" * 65)

    # Save results to text file
    out_path = os.path.join(cfg["plots"]["dir"], "inference_times.txt")
    os.makedirs(cfg["plots"]["dir"], exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"Architecture  : {arch.upper()}\n")
        f.write(f"Hardware      : {device}\n")
        f.write(f"Parameters    : {n_params:,}\n\n")
        f.write(
            f"{'Batch size':>12}  "
            f"{'Avg time/batch (ms)':>22}  "
            f"{'Avg time/sample (µs)':>22}\n"
        )
        f.write("-" * 60 + "\n")
        for bs, bms, sus in results:
            f.write(f"{bs:>12}  {bms:>22.3f}  {sus:>22.3f}\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
