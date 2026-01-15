#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join([f'"{c}"' if " " in c else c for c in cmd]))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline: normalize -> segment -> filter -> deduplicate"
    )
    parser.add_argument("corpus_dir", type=Path, help="Path to corpus root (contains period folders + metadata.csv)")
    parser.add_argument("out_dir", type=Path, help="Output directory for pipeline artifacts")

    parser.add_argument("--metadata", type=Path, default=None,
                        help="Path to metadata.csv (default: <corpus_dir>/metadata.csv)")
    parser.add_argument("--step1", type=Path, default=Path("step1_normalize.py"))
    parser.add_argument("--step2", type=Path, default=Path("step2_segment.py"))
    parser.add_argument("--step3", type=Path, default=Path("step3_filter.py"))
    parser.add_argument("--step4", type=Path, default=Path("step4_dedup.py"),
                        help="Path to step4 script (can contain spaces; pass in quotes)")

    parser.add_argument("--min-score", type=int, default=2, help="Min score for step3 filter (default: 2)")
    parser.add_argument("--all-years", action="store_true",
                        help="Pass --all-years to step3_filter.py (adds filter_all_years)")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85,
                        help="Fuzzy threshold for step4 (default: 0.85)")

    args = parser.parse_args()

    corpus_dir: Path = args.corpus_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.metadata or (corpus_dir / "metadata.csv")
    if not metadata_path.exists():
        print(f"ERROR: metadata.csv not found at: {metadata_path}", file=sys.stderr)
        return 2

    norm_dir = out_dir / "01_normalized"
    seg_dir  = out_dir / "02_segments"
    filt_dir = out_dir / "03_filtered"
    dedup_dir = out_dir / "04_dedup"

    norm_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    filt_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    segments_all_json = seg_dir / "segments_all.json"
    filtered_json = filt_dir / "segments_filtered.json"
    dedup_json = dedup_dir / "segments_final.json"

    py = sys.executable  # use current python

    run([py, str(args.step1), str(corpus_dir), str(norm_dir)])

    run([py, str(args.step2), str(norm_dir), str(metadata_path), str(seg_dir)])

    if not segments_all_json.exists():
        print(f"ERROR: step2 did not produce {segments_all_json}", file=sys.stderr)
        return 3

    step3_cmd = [py, str(args.step3), str(segments_all_json), str(filtered_json), str(args.min_score)]
    if args.all_years:
        step3_cmd.append("--all-years")
    run(step3_cmd)

    if not filtered_json.exists():
        print(f"ERROR: step3 did not produce {filtered_json}", file=sys.stderr)
        return 4

    run([py, str(args.step4), str(filtered_json), str(dedup_json), str(args.fuzzy_threshold)])

    if not dedup_json.exists():
        print(f"ERROR: step4 did not produce {dedup_json}", file=sys.stderr)
        return 5

    print(f"- Normalized corpus: {norm_dir}")
    print(f"- Segments JSON:      {segments_all_json}")
    print(f"- Filtered JSON:      {filtered_json}")
    print(f"- Final dedup JSON:   {dedup_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())