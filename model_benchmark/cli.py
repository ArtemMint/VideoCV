import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import ModelRunConfig
from .report import write_summary_report
from .runner import run_one


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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs: List[ModelRunConfig] = ModelRunConfig.many_from_json(args.config)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Invalid run configuration: {}".format(exc), file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        run_one(config, args.output, index)
        for index, config in enumerate(configs, start=1)
    ]
    write_summary_report(args.output, results)

    for result in results:
        if result["status"] == "completed":
            print("Completed {}: {}".format(result["name"], result["report_html"]))
        else:
            error = result.get("error", {})
            print(
                "Failed {}: {}: {}".format(
                    result["name"], error.get("type", "Error"), error.get("message", "")
                ),
                file=sys.stderr,
            )
    print("Summary report: {}".format(args.output / "summary.html"))
    return 0 if all(result["status"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
