import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _json_default(value: Any) -> str:
    return str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return "{:.4f}".format(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _table(rows: Iterable[Tuple[str, Any]]) -> str:
    body = "".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(str(key)), html.escape(_format_value(value))
        )
        for key, value in rows
    )
    return "<table><tbody>{}</tbody></table>".format(body)


def write_model_report(report_dir: Path, result: Dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    configuration = result["configuration"]
    sections = [
        ("Run", [("Status", result["status"]), ("Model", configuration["model"]),
                 ("Format", configuration["model_format"]),
                 ("Dataset", configuration["dataset"]),
                 ("Device", result.get("resolved_device") or configuration["device"]),
                 ("Parameters", configuration)]),
        ("Detection metrics", result.get("metrics", {}).items()),
        ("Timings", result.get("timings", {}).items()),
        ("Resource usage", result.get("resources", {}).items()),
        ("Hardware and runtime", result.get("runtime", {}).items()),
        ("Decision", result.get("decision", {}).items()),
        ("Comparison", result.get("comparison", {}).items()),
        ("Tracking", result.get("tracking_metrics", {}).items()),
        ("Notes", [("Note", note) for note in result.get("notes", [])]),
    ]
    if result.get("error"):
        sections.append(("Error", result["error"].items()))

    content = "".join(
        "<h2>{}</h2>{}".format(html.escape(title), _table(rows))
        for title, rows in sections
    )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Model evaluation report</title>
<style>body{{font:15px sans-serif;max-width:1000px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%;margin-bottom:2em}}
th,td{{text-align:left;border:1px solid #ccc;padding:.5em;vertical-align:top}}
th{{width:30%;background:#f4f4f4}}</style>
<h1>Model evaluation: {name}</h1>{content}</html>
""".format(name=html.escape(result["name"]), content=content)
    (report_dir / "report.html").write_text(document, encoding="utf-8")


def _performance_values(result: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    inference = result.get("metrics", {}).get("inference", {})
    latency = inference.get("end_to_end_frame", {})
    fps = inference.get("fps", {})
    return latency.get("p50_ms", ""), latency.get("p95_ms", ""), fps.get("p50_fps", "")


def _memory_dimensions(result: Dict[str, Any]) -> Dict[str, float]:
    resources = result.get("resources", {})
    dimensions = {}
    host_memory = resources.get("rss_peak_mb")
    if isinstance(host_memory, (int, float)):
        dimensions["host_ram_peak_mb"] = float(host_memory)
    gpu_memory = resources.get("gpu_process_memory", {})
    if gpu_memory.get("available"):
        dimensions["gpu_process_peak_allocated_mb"] = float(
            gpu_memory["peak_allocated_mb"]
        )
    else:
        device_memory = resources.get("gpu", {})
        if device_memory.get("available"):
            dimensions["gpu_device_peak_used_mb"] = float(
                device_memory["memory_used_peak_mb"]
            )
    return dimensions


def _annotate_comparisons(results: List[Dict[str, Any]]) -> None:
    for result in results:
        group = result.get("comparison_group")
        result["comparison"] = {
            "group": group,
            "pareto_optimal": None,
            "dominated_by": [],
            "reason": "No compatible completed runs in the same comparison group",
        }

    completed = [
        result
        for result in results
        if result.get("status") == "completed" and result.get("comparison_group")
    ]
    for candidate in completed:
        candidate_quality = candidate.get("metrics", {}).get("map50_95")
        candidate_latency = _performance_values(candidate)[1]
        candidate_memory = _memory_dimensions(candidate)
        if not isinstance(candidate_quality, (int, float)) or not isinstance(
            candidate_latency, (int, float)
        ):
            continue

        dominators = []
        for other in completed:
            if (
                other is candidate
                or other.get("comparison_group") != candidate.get("comparison_group")
            ):
                continue
            other_quality = other.get("metrics", {}).get("map50_95")
            other_latency = _performance_values(other)[1]
            other_memory = _memory_dimensions(other)
            if not isinstance(other_quality, (int, float)) or not isinstance(
                other_latency, (int, float)
            ):
                continue
            memory_axes = set(candidate_memory) & set(other_memory)
            no_worse = (
                other_quality >= candidate_quality
                and other_latency <= candidate_latency
                and all(other_memory[key] <= candidate_memory[key] for key in memory_axes)
            )
            strictly_better = (
                other_quality > candidate_quality
                or other_latency < candidate_latency
                or any(other_memory[key] < candidate_memory[key] for key in memory_axes)
            )
            if no_worse and strictly_better:
                dominators.append(other["name"])

        candidate["comparison"] = {
            "group": candidate["comparison_group"],
            "pareto_optimal": not dominators,
            "dominated_by": dominators,
            "memory_axes": sorted(candidate_memory),
            "reason": (
                "Not dominated on mAP50-95, p95 end-to-end latency, and measured memory"
                if not dominators
                else "Dominated by a run with no worse quality, p95 latency, and measured memory"
            ),
        }


def write_summary_report(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _annotate_comparisons(results)
    groups: Dict[str, int] = {}
    for result in results:
        group = result.get("comparison_group")
        if group:
            groups[group] = groups.get(group, 0) + 1
    summary = {"comparison_groups": groups, "runs": results}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    for result in results:
        report_path = result.get("report_json")
        if report_path:
            write_model_report(Path(report_path).parent, result)

    columns = (
        ("name", "Model"),
        ("status", "Status"),
        ("format", "Format"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("latency_p50", "Frame p50 ms"),
        ("latency_p95", "Frame p95 ms"),
        ("fps", "FPS median"),
        ("cpu", "CPU avg % (one core)"),
        ("ram", "Peak process-tree RAM MB"),
        ("gpu_ram", "Peak GPU memory MB"),
        ("requirements", "Meets limits"),
        ("pareto", "Pareto"),
        ("group", "Comparison group"),
    )
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        resources = result.get("resources", {})
        latency_p50, latency_p95, fps = _performance_values(result)
        gpu_memory = resources.get("gpu_process_memory", {})
        gpu_telemetry = resources.get("gpu", {})
        gpu_memory_peak = (
            gpu_memory.get("peak_allocated_mb")
            if gpu_memory.get("available")
            else gpu_telemetry.get("memory_used_peak_mb", "")
        )
        values = {
            "name": result["name"],
            "status": result["status"],
            "format": result["configuration"]["model_format"],
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "map50": metrics.get("map50", ""),
            "map50_95": metrics.get("map50_95", ""),
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
            "fps": fps,
            "cpu": resources.get("process_cpu_percent_avg", ""),
            "ram": resources.get("rss_peak_mb", ""),
            "gpu_ram": gpu_memory_peak,
            "requirements": result.get("decision", {}).get("meets_requirements", ""),
            "pareto": result.get("comparison", {}).get("pareto_optimal", ""),
            "group": result.get("comparison_group", ""),
        }
        rows.append("<tr>{}</tr>".format("".join(
            "<td>{}</td>".format(html.escape(_format_value(values[key])))
            for key, _ in columns
        )))
    header = "".join("<th>{}</th>".format(html.escape(title)) for _, title in columns)
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Model benchmark comparison</title>
<style>body{{font:15px sans-serif;margin:2em}}table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:.5em;text-align:left}}th{{background:#f4f4f4}}</style>
<h1>Model benchmark comparison</h1>
<p>Only runs sharing a comparison group are checked for Pareto dominance. Groups
require the same host, device model, dataset, input size, batch, thresholds,
precision mode, and configured export parameters.</p>
<p>GPU memory is process allocation when available; otherwise it is device-wide
NVML memory use.</p>
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></html>
""".format(header=header, rows="".join(rows))
    (output_dir / "summary.html").write_text(document, encoding="utf-8")
