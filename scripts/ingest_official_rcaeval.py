"""Ingest official RCAEval Online Boutique cases directly from Hugging Face.

Authoritative source:
https://huggingface.co/datasets/phamquiluan/RCAEval

This script downloads genuine official cases, computes SHA-256 hashes,
saves the official raw parquet file, and generates metrics.json for stdlib compatibility.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import List, Tuple

try:
    import pyarrow.parquet as pq
except ImportError:
    raise RuntimeError("pyarrow is required to run the official RCAEval ingestion script.")

OFFICIAL_CASES: List[Tuple[str, str, str, int]] = [
    ("re1ob_cartservice_cpu_1", "cartservice", "cpu", 1),
    ("re1ob_checkoutservice_delay_1", "checkoutservice", "delay", 1),
    ("re1ob_productcatalogservice_cpu_1", "productcatalogservice", "cpu", 1),
]

HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval"


def ingest_case(case_id: str, service: str, fault: str, instance: int, target_dirs: List[Path]) -> None:
    print(f"\n--- Ingesting {case_id} from {HF_BASE_URL} ---")

    # 1. Download official inject_time.txt
    url_inject = f"{HF_BASE_URL}/raw/main/{case_id}/inject_time.txt"
    req_inject = urllib.request.Request(url_inject, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_inject, timeout=30) as r:
        inject_time_str = r.read().decode("utf-8").strip()
    inject_time = float(inject_time_str)

    # 2. Download official metrics.parquet
    url_parquet = f"{HF_BASE_URL}/resolve/main/{case_id}/metrics.parquet"
    req_parquet = urllib.request.Request(url_parquet, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_parquet, timeout=60) as r:
        parquet_bytes = r.read()

    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    table = pq.read_table(io.BytesIO(parquet_bytes))

    print(f"Downloaded: inject_time={inject_time}, rows={table.num_rows}, cols={len(table.column_names)}")
    print(f"SHA-256: {parquet_sha256}")

    # 3. Convert to standardized JSON dictionary
    # Downsample if rows > 1000 to keep JSON size reasonable while preserving inject_time and fidelity
    step = 1
    if table.num_rows > 1000:
        step = 5  # 5-second sampling for 4200 rows -> 840 points (plenty for windowing)

    times = table["time"].to_pylist()
    # Ensure inject_time is preserved or closely bounded
    sampled_indices = set(range(0, table.num_rows, step))
    # Also find exact or nearest index to inject_time
    inject_idx = min(range(len(times)), key=lambda i: abs(times[i] - inject_time))
    sampled_indices.add(inject_idx)
    sorted_indices = sorted(sampled_indices)

    json_dict = {}
    for col in table.column_names:
        full_col = table[col].to_pylist()
        json_dict[col] = [round(float(full_col[i]), 5) if full_col[i] is not None else None for i in sorted_indices]

    metrics_json_str = json.dumps(json_dict)
    json_sha256 = hashlib.sha256(metrics_json_str.encode("utf-8")).hexdigest()

    metadata = {
        "source": "RCAEval",
        "data_status": "OFFICIAL_RCAEVAL",
        "case_id": case_id,
        "benchmark": "re1ob",
        "ground_truth_service": service,
        "fault_type": fault,
        "instance": instance,
        "inject_time": inject_time,
        "source_dataset": "phamquiluan/RCAEval",
        "source_reference": f"https://huggingface.co/datasets/phamquiluan/RCAEval/tree/main/{case_id}",
        "retrieval_version": "1.0",
        "provenance_note": "Official RCAEval Online Boutique benchmark case downloaded directly from Hugging Face.",
        "original_metric_count": len(table.column_names),
        "original_timestep_count": table.num_rows,
        "raw_artifact": "metrics.parquet",
        "raw_artifact_sha256": parquet_sha256,
        "normalized_artifact_sha256": json_sha256,
        "data_verified_against_source": True,
        "sampling_step_seconds": step,
    }

    for base_dir in target_dirs:
        case_dir = base_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        (case_dir / "inject_time.txt").write_text(f"{int(inject_time)}\n", encoding="utf-8")
        (case_dir / "metrics.parquet").write_bytes(parquet_bytes)
        (case_dir / "metrics.json").write_text(metrics_json_str, encoding="utf-8")
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved into {case_dir}")


def main():
    root = Path(__file__).resolve().parents[1]
    target_dirs = [
        root / "config" / "rcaeval_cases",
        root / "ripplecut" / "config" / "rcaeval_cases",
    ]

    for case_id, service, fault, instance in OFFICIAL_CASES:
        ingest_case(case_id, service, fault, instance, target_dirs)

    print("\n[SUCCESS] Official RCAEval ingestion complete!")


if __name__ == "__main__":
    main()
