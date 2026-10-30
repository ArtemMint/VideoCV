import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _json_default(value: Any) -> str:
    return str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return "{:.4f}".format(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _table(rows: Iterable[tuple[str, Any]]) -> str:
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
                 ("Device", result.get("resolved_device") or configuration["device"])]),
        ("Detection metrics", result.get("metrics", {}).items()),
        ("Performance", result.get("timings", {}).items()),
        ("Resource usage", result.get("resources", {}).items()),
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


def write_summary_report(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"runs": results}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    columns = (
        ("name", "Model"),
        ("status", "Status"),
        ("format", "Format"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("cpu", "CPU avg % (one core)"),
        ("ram", "Peak RAM MB"),
    )
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        resources = result.get("resources", {})
        values = {
            "name": result["name"],
            "status": result["status"],
            "format": result["configuration"]["model_format"],
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "map50": metrics.get("map50", ""),
            "map50_95": metrics.get("map50_95", ""),
            "cpu": resources.get("process_cpu_percent_avg", ""),
            "ram": resources.get("rss_peak_mb", ""),
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
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></html>
""".format(header=header, rows="".join(rows))
    (output_dir / "summary.html").write_text(document, encoding="utf-8")
