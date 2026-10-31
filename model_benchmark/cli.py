import argparse
from dataclasses import replace
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

from .config import ModelRunConfig
from .report import write_summary_report
from .runner import run_one
from .visdrone import prepare_visdrone_val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more YOLO models on the configured dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run model validation benchmarks")
    run_parser.add_argument("config", type=Path, help="single-run or multi-run JSON file")
    run_parser.add_argument(
        "--output", type=Path, default=Path("model_benchmark_results"),
        help="directory for per-model and summary reports",
    )
    run_parser.add_argument(
        "--repeats", type=int, help="override validation and measurement repetitions",
    )
    run_parser.add_argument(
        "--warmup-iterations", type=int, help="override measured-model warm-up calls",
    )
    run_parser.add_argument(
        "--inference-iterations", type=int,
        help="override timed predict calls per repetition",
    )
    run_parser.add_argument(
        "--min-map50-95", type=float, help="minimum acceptable mAP50-95",
    )
    run_parser.add_argument(
        "--max-latency-ms", type=float,
        help="maximum acceptable p95 end-to-end frame latency",
    )
    run_parser.add_argument(
        "--max-memory-mb", type=float,
        help="maximum acceptable benchmark process-tree peak RAM",
    )
    run_parser.add_argument(
        "--max-gpu-memory-mb", type=float,
        help="maximum acceptable GPU process allocation (or device memory) peak",
    )
    subparsers.add_parser(
        "prepare-visdrone",
        help="download and prepare the VisDrone validation split under datasets/",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-visdrone":
        try:
            result = prepare_visdrone_val()
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print("Could not prepare VisDrone validation data: {}".format(exc), file=sys.stderr)
            return 1
        action = "Prepared" if result["downloaded"] else "Already prepared"
        print(
            "{} {} images and labels at {}".format(
                action, result["images"], result["dataset_root"]
            )
        )
        return 0

    try:
        configs: List[ModelRunConfig] = ModelRunConfig.many_from_json(args.config)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Invalid run configuration: {}".format(exc), file=sys.stderr)
        return 2
    overrides = {
        "repeats": args.repeats,
        "warmup_iterations": args.warmup_iterations,
        "inference_iterations": args.inference_iterations,
        "min_map50_95": args.min_map50_95,
        "max_latency_ms": args.max_latency_ms,
        "max_memory_mb": args.max_memory_mb,
        "max_gpu_memory_mb": args.max_gpu_memory_mb,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    try:
        configs = [
            replace(config, **overrides) if overrides else config for config in configs
        ]
    except ValueError as exc:
        print("Invalid benchmark override: {}".format(exc), file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        run_one(config, args.output, index)
        for index, config in enumerate(configs, start=1)
    ]
    write_summary_report(args.output, results)

    for result in results:
        if result["status"] == "completed":
            decision = result.get("decision", {}).get("meets_requirements")
            suffix = (
                " (acceptance limits met)"
                if decision is True
                else " (acceptance limits not met)"
                if decision is False
                else ""
            )
            print(
                "Completed {}{}: {}".format(
                    result["name"], suffix, result["report_html"]
                )
            )
        else:
            error = result.get("error", {})
            print(
                "Failed {}: {}: {}".format(
                    result["name"], error.get("type", "Error"), error.get("message", "")
                ),
                file=sys.stderr,
            )
    print("Summary report: {}".format(args.output / "summary.html"))
    has_acceptance_limits = any(
        config.min_map50_95 is not None
        or config.max_latency_ms is not None
        or config.max_memory_mb is not None
        or config.max_gpu_memory_mb is not None
        for config in configs
    )
    any_acceptable_model = any(
        result.get("decision", {}).get("meets_requirements") is True
        for result in results
        if result["status"] == "completed"
    )
    if has_acceptance_limits and not any_acceptable_model:
        print("No completed model met all configured acceptance limits.", file=sys.stderr)
    all_completed = all(result["status"] == "completed" for result in results)
    acceptance_passed = not has_acceptance_limits or any_acceptable_model
    return 0 if all_completed and acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
