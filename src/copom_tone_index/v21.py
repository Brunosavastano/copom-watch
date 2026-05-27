from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copom_tone_index.config import get_paths, load_v2_settings
from copom_tone_index.market import empty_decision_expectations, empty_market_observations
from copom_tone_index.storage import export_tables, write_tables
from copom_tone_index.v2 import (
    V2CommandResult,
    append_run,
    file_sha256,
    make_run_id,
    read_optional_table,
    rule_engine_version,
    run_row,
    utc_now_naive,
    v2_benchmark_baseline_command,
)


V21_EVENT_PANEL_COLUMNS = [
    "meeting_id",
    "nro_reuniao",
    "data_referencia",
    "selic_decision_bps",
    "expected_selic_change_bps",
    "expected_selic_change_official_bps",
    "expected_selic_change_proxy_bps",
    "decision_surprise_bps",
    "decision_surprise_official_bps",
    "decision_surprise_proxy_bps",
    "decision_surprise_status",
    "communication_surprise_naive",
    "communication_surprise_model",
    "communication_surprise_model_status",
    "focus_feature_rows",
    "focus_delta_coverage",
    "focus_status",
    "market_window_rows",
    "market_ok_windows",
    "mean_abs_market_reaction",
    "market_status",
]


@dataclass(frozen=True)
class V21HealthResult:
    status: str
    json_path: Path
    html_path: Path
    warnings: int
    errors: int
    rows: int


def v21_build_event_panel_command() -> V2CommandResult:
    paths = get_paths()
    run_id = make_run_id("v21_event_panel")
    meetings = read_optional_table(paths.database, "v2_meetings", pd.DataFrame())
    scores = read_optional_table(paths.database, "v2_meeting_scores", pd.DataFrame())
    focus_features = read_optional_table(paths.database, "focus_event_features", pd.DataFrame())
    market_windows = read_optional_table(paths.database, "market_event_windows", empty_market_observations())
    expectations = read_optional_table(paths.database, "decision_expectations", empty_decision_expectations())
    panel = build_v21_event_panel(meetings, scores, focus_features, market_windows, expectations)
    runs = append_run(paths.database, run_row(run_id, "v21_build_event_panel", "completed", {"rows": len(panel)}))
    write_tables(paths.database, {"v2_runs": runs, "v21_event_panel": panel})
    export_tables(paths.database, paths.processed, ["v21_event_panel"])
    output_path = write_v21_outputs(panel, focus_features, market_windows)
    return V2CommandResult(paths.database, output_path, len(panel), "completed", run_id)


def v21_health_command() -> V21HealthResult:
    paths = get_paths()
    report, by_meeting, by_source = build_v21_acceptance_report(paths.database)
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    by_meeting_path = output_dir / "v21_acceptance_by_meeting.csv"
    by_source_path = output_dir / "v21_acceptance_by_source.csv"
    json_path = report_dir / "v21_acceptance_report.json"
    html_path = report_dir / "v21_acceptance_report.html"
    by_meeting.to_csv(by_meeting_path, index=False)
    by_source.to_csv(by_source_path, index=False)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=v21_json_default), encoding="utf-8")
    html_path.write_text(render_v21_acceptance_report_html(report, by_meeting, by_source), encoding="utf-8")
    return V21HealthResult(
        status=str(report["status"]),
        json_path=json_path,
        html_path=html_path,
        warnings=len(report["warnings"]),
        errors=len(report["errors"]),
        rows=len(by_meeting),
    )


def v21_freeze_release_command(version: str = "v2.1-public-focus-market-acceptance") -> V2CommandResult:
    """Freeze the already-built V2.1 analytical layer without fetching new data."""

    paths = get_paths()
    run_id = make_run_id("v21_freeze")
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    release_dir = report_dir / "releases" / version
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    from copom_tone_index.v2_health import v2_health_check_command

    benchmark = v2_benchmark_baseline_command()
    v2_health = v2_health_check_command()
    v21_health = v21_health_command()
    v21_report = json.loads(v21_health.json_path.read_text(encoding="utf-8")) if v21_health.json_path.exists() else {}
    release_errors = collect_v21_release_errors(output_dir, report_dir, benchmark.status, v2_health.errors, v21_health.errors, v21_report)
    status = "fail" if release_errors else "completed"

    artifact_paths = collect_v21_release_artifacts(output_dir, report_dir)
    copied_artifacts: list[dict[str, object]] = []
    for path in artifact_paths:
        if not path.exists() or not path.is_file():
            continue
        target = release_dir / path.name
        shutil.copy2(path, target)
        copied_artifacts.append({"path": str(target), "sha256": file_sha256(target), "bytes": target.stat().st_size})

    manifest = build_v21_release_manifest(
        version=version,
        status=status,
        run_id=run_id,
        database=paths.database,
        release_dir=release_dir,
        benchmark_status=benchmark.status,
        v2_health=v2_health,
        v21_health=v21_health,
        v21_report=v21_report,
        release_errors=release_errors,
        copied_artifacts=copied_artifacts,
    )
    manifest_path = report_dir / "v21_release_manifest.json"
    summary_path = report_dir / "v21_release_summary.html"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=v21_json_default), encoding="utf-8")
    summary_path.write_text(render_v21_release_summary_html(manifest), encoding="utf-8")
    for path in [manifest_path, summary_path]:
        shutil.copy2(path, release_dir / path.name)

    runs = append_run(paths.database, run_row(run_id, "freeze_v21", status, {"version": version, "artifacts": len(copied_artifacts)}))
    write_tables(paths.database, {"v2_runs": runs})
    return V2CommandResult(paths.database, manifest_path, len(copied_artifacts), status, run_id)


def collect_v21_release_errors(
    output_dir: Path,
    report_dir: Path,
    benchmark_status: str,
    v2_health_errors: int,
    v21_health_errors: int,
    v21_report: dict[str, Any],
) -> list[str]:
    required = [
        output_dir / "v21_event_panel.csv",
        output_dir / "focus_event_features.csv",
        output_dir / "market_event_study.csv",
        report_dir / "v21_acceptance_report.json",
    ]
    errors = [f"Required V2.1 artifact is missing: {path}" for path in required if not path.exists()]
    if benchmark_status == "fail":
        errors.append("V2 baseline benchmark is fail; V2.1 release cannot be frozen.")
    if v2_health_errors > 0:
        errors.append(f"V2 health-check has {v2_health_errors} errors.")
    if v21_health_errors > 0:
        errors.append(f"V2.1 health-check has {v21_health_errors} errors.")
    event_panel = pd.DataFrame()
    panel_path = output_dir / "v21_event_panel.csv"
    if panel_path.exists():
        event_panel = pd.read_csv(panel_path)
    contamination = v21_release_official_proxy_contamination(event_panel)
    if contamination > 0:
        errors.append(f"Proxy decision surprises filled official surprise fields in {contamination} rows.")
    for error in v21_report.get("errors", []):
        errors.append(f"V2.1 acceptance error: {error}")
    return errors


def v21_release_official_proxy_contamination(panel: pd.DataFrame) -> int:
    if panel.empty or not {"decision_surprise_official_bps", "decision_surprise_status"}.issubset(panel.columns):
        return 0
    official_value = pd.to_numeric(panel["decision_surprise_official_bps"], errors="coerce").notna()
    official_status = panel["decision_surprise_status"].astype(str).str.lower().eq("official")
    return int((official_value & ~official_status).sum())


def collect_v21_release_artifacts(output_dir: Path, report_dir: Path) -> list[Path]:
    return [
        report_dir / "v21_acceptance_report.html",
        report_dir / "v21_acceptance_report.json",
        report_dir / "v21_macro_market_report.html",
        report_dir / "focus_v21_report.html",
        report_dir / "public_market_coverage_report.html",
        report_dir / "market_event_study_report.html",
        output_dir / "v21_acceptance_by_meeting.csv",
        output_dir / "v21_acceptance_by_source.csv",
        output_dir / "v21_event_panel.csv",
        output_dir / "focus_v21_coverage.csv",
        output_dir / "public_market_coverage.csv",
        output_dir / "market_event_study.csv",
        output_dir / "decision_expectation_source_audit.csv",
    ]


def build_v21_release_manifest(
    version: str,
    status: str,
    run_id: str,
    database: Path,
    release_dir: Path,
    benchmark_status: str,
    v2_health: Any,
    v21_health: V21HealthResult,
    v21_report: dict[str, Any],
    release_errors: list[str],
    copied_artifacts: list[dict[str, object]],
) -> dict[str, Any]:
    v2_settings = load_v2_settings()
    focus = v21_report.get("focus", {})
    market = v21_report.get("market", {})
    decision = v21_report.get("decision_expectations", {})
    event_panel = v21_report.get("event_panel", {})
    warnings = list(v21_report.get("warnings", []))
    if decision.get("panel_official_surprises", 0) == 0:
        warnings.append("B3 Opcao de Copom has no structured public historical feed in this release; official decision surprise remains unavailable.")
    return {
        "version": version,
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "run_id": run_id,
        "database_path": str(database),
        "release_dir": str(release_dir),
        "rule_engine_version": rule_engine_version(v2_settings),
        "official_text_index": "deterministic_v2_baseline",
        "v2_0_release_dependency": "v2.0.4-holdout-stance-hardened",
        "benchmark_baseline_status": benchmark_status,
        "v2_health": {"status": v2_health.status, "warnings": v2_health.warnings, "errors": v2_health.errors},
        "v21_health": {"status": v21_health.status, "warnings": v21_health.warnings, "errors": v21_health.errors},
        "focus_metrics": {
            "status": focus.get("status"),
            "vintages": focus.get("vintages", 0),
            "event_features": focus.get("event_features", 0),
            "delta_coverage": focus.get("delta_coverage", 0.0),
        },
        "market_metrics": {
            "status": market.get("status"),
            "observations": market.get("observations", 0),
            "event_windows": market.get("event_windows", 0),
            "ok_windows": market.get("ok_windows", 0),
            "insufficient_market_observations": market.get("insufficient_market_observations", 0),
            "ambiguous_event_timing": market.get("ambiguous_event_timing", 0),
        },
        "surprise_metrics": {
            "official_expectation_rows": decision.get("official_rows", 0),
            "proxy_expectation_rows": decision.get("proxy_rows", 0),
            "official_surprises": event_panel.get("official_surprises", 0),
            "proxy_surprises": event_panel.get("proxy_surprises", 0),
            "not_available_surprises": event_panel.get("not_available_surprises", 0),
        },
        "limitations": [
            "V2.1 is descriptive and associative; event windows are not causal identification.",
            "Decision surprise official remains unavailable without structured public historical B3 Opcao de Copom data.",
            "Proxy decision surprises are exploratory and never fill decision_surprise_official_bps.",
            "V2.1 freezes processed public data already present locally; it does not fetch new data.",
        ],
        "warnings": sorted(set(warnings)),
        "errors": release_errors,
        "artifacts": copied_artifacts,
    }


def render_v21_release_summary_html(manifest: dict[str, Any]) -> str:
    status = str(manifest["status"]).upper()
    artifacts = pd.DataFrame(manifest.get("artifacts", []))
    artifact_table = artifacts.to_html(index=False, escape=True) if not artifacts.empty else "<p>Nenhum artefato copiado.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.1 Release Summary</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#e5e7eb}.fail{background:#fee2e2}.completed{background:#dcfce7}.muted{color:#6b7280}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.1 Release Summary</h1>",
            f"<p>Status: <span class='badge {manifest['status']}'>{status}</span></p>",
            f"<p>Version: <strong>{escape_text(str(manifest['version']))}</strong></p>",
            "<h2>Resumo executivo</h2>",
            dict_table(
                {
                    "rule_engine_version": manifest.get("rule_engine_version"),
                    "benchmark_baseline_status": manifest.get("benchmark_baseline_status"),
                    "v2_health": manifest.get("v2_health"),
                    "v21_health": manifest.get("v21_health"),
                }
            ),
            "<h2>Focus</h2>",
            dict_table(manifest.get("focus_metrics", {})),
            "<h2>Mercado</h2>",
            dict_table(manifest.get("market_metrics", {})),
            "<h2>Surpresas</h2>",
            dict_table(manifest.get("surprise_metrics", {})),
            "<h2>Warnings aceitos</h2>",
            html_list(manifest.get("warnings", [])),
            "<h2>Errors</h2>",
            html_list(manifest.get("errors", [])),
            "<h2>Limitações</h2>",
            html_list(manifest.get("limitations", [])),
            "<h2>Artefatos versionados</h2>",
            artifact_table,
            "<p class='muted'>V2.1 e uma camada analitica adicional. O indice textual oficial permanece o baseline deterministico V2.0.4.</p>",
            "</body></html>",
        ]
    )


def build_v21_acceptance_report(database: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    tables = load_v21_tables(database)
    by_meeting = build_v21_acceptance_by_meeting(tables)
    by_source = build_v21_acceptance_by_source(tables)
    focus_summary = v21_focus_summary(tables["focus_vintages"], tables["focus_event_features"])
    market_summary = v21_market_summary(tables["market_observations"], tables["market_event_windows"], tables["public_market_source_audit"])
    decision_summary = v21_decision_summary(tables["decision_expectations"], tables["decision_expectation_source_audit"], tables["v21_event_panel"])
    event_panel_summary = v21_event_panel_summary(tables["v21_event_panel"])
    idempotency = v21_idempotency_summary(tables)
    lookahead = v21_lookahead_summary(tables["focus_event_features"], tables["market_event_windows"], tables["decision_expectations"], tables["v21_event_panel"])
    health_status = v2_health_status(database)
    warnings: list[str] = []
    errors: list[str] = []
    collect_v21_findings(warnings, errors, focus_summary, market_summary, decision_summary, event_panel_summary, idempotency, lookahead, health_status)
    status = "fail" if errors else "warning" if warnings else "pass"
    report = {
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "database_path": str(database),
        "summary": {
            "meetings": int(len(by_meeting)),
            "focus_status": focus_summary["status"],
            "market_status": market_summary["status"],
            "decision_status": decision_summary["status"],
            "event_panel_status": event_panel_summary["status"],
            "v2_health_status": health_status["status"],
        },
        "focus": focus_summary,
        "market": market_summary,
        "decision_expectations": decision_summary,
        "event_panel": event_panel_summary,
        "idempotency": idempotency,
        "lookahead": lookahead,
        "v2_health": health_status,
        "outputs": {
            "by_meeting": "outputs/v2/v21_acceptance_by_meeting.csv",
            "by_source": "outputs/v2/v21_acceptance_by_source.csv",
            "json": "reports/v2/v21_acceptance_report.json",
            "html": "reports/v2/v21_acceptance_report.html",
        },
        "warnings": warnings,
        "errors": errors,
    }
    return report, by_meeting, by_source


def load_v21_tables(database: Path) -> dict[str, pd.DataFrame]:
    return {
        "focus_vintages": read_optional_table(database, "focus_vintages", pd.DataFrame()),
        "focus_event_features": read_optional_table(database, "focus_event_features", pd.DataFrame()),
        "market_observations": read_optional_table(database, "market_observations", empty_market_observations()),
        "market_event_windows": read_optional_table(database, "market_event_windows", pd.DataFrame()),
        "decision_expectations": read_optional_table(database, "decision_expectations", empty_decision_expectations()),
        "public_market_source_audit": read_optional_table(database, "public_market_source_audit", pd.DataFrame()),
        "decision_expectation_source_audit": read_optional_table(database, "decision_expectation_source_audit", pd.DataFrame()),
        "public_market_coverage": read_optional_table(database, "public_market_coverage", pd.DataFrame()),
        "v21_event_panel": read_optional_table(database, "v21_event_panel", pd.DataFrame()),
    }


def build_v21_acceptance_by_meeting(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = tables["v21_event_panel"].copy()
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "meeting_id",
                "nro_reuniao",
                "data_referencia",
                "focus_status",
                "focus_delta_coverage",
                "focus_missing_reasons",
                "market_status",
                "market_ok_windows",
                "market_problem_windows",
                "decision_surprise_status",
                "has_official_decision_surprise",
                "has_proxy_decision_surprise",
            ]
        )
    focus = tables["focus_event_features"]
    market = tables["market_event_windows"]
    panel["meeting_id"] = panel["meeting_id"].astype(str)
    if not focus.empty and "meeting_id" in focus:
        focus_reasons = (
            focus.assign(meeting_id=lambda data: data["meeting_id"].astype(str))
            .groupby("meeting_id")["missing_reason"]
            .apply(lambda values: "; ".join(sorted({str(value) for value in values.dropna() if str(value).strip()})))
            .rename("focus_missing_reasons")
        )
        panel = panel.merge(focus_reasons, on="meeting_id", how="left")
    else:
        panel["focus_missing_reasons"] = ""
    if not market.empty and "meeting_id" in market:
        problem = market[market.get("status", pd.Series(dtype=str)) != "ok"].copy()
        market_problem = problem.assign(meeting_id=lambda data: data["meeting_id"].astype(str)).groupby("meeting_id").size().rename("market_problem_windows")
        panel = panel.merge(market_problem, on="meeting_id", how="left")
    else:
        panel["market_problem_windows"] = 0
    panel["market_problem_windows"] = pd.to_numeric(panel["market_problem_windows"], errors="coerce").fillna(0).astype(int)
    panel["has_official_decision_surprise"] = panel.get("decision_surprise_official_bps", pd.Series(dtype=float)).notna()
    panel["has_proxy_decision_surprise"] = panel.get("decision_surprise_proxy_bps", pd.Series(dtype=float)).notna()
    columns = [
        "meeting_id",
        "nro_reuniao",
        "data_referencia",
        "focus_status",
        "focus_feature_rows",
        "focus_delta_coverage",
        "focus_missing_reasons",
        "market_status",
        "market_window_rows",
        "market_ok_windows",
        "market_problem_windows",
        "decision_surprise_status",
        "decision_surprise_official_bps",
        "decision_surprise_proxy_bps",
        "has_official_decision_surprise",
        "has_proxy_decision_surprise",
    ]
    for column in columns:
        if column not in panel:
            panel[column] = np.nan
    return panel[columns].sort_values("nro_reuniao" if "nro_reuniao" in panel else "meeting_id").reset_index(drop=True)


def build_v21_acceptance_by_source(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, count in value_counts_series(tables["focus_vintages"], "source").items():
        rows.append({"section": "focus_vintages", "source": source, "status": "ok", "rows": count, "detail": ""})
    for source, count in value_counts_series(tables["market_observations"], "source").items():
        rows.append({"section": "market_observations", "source": source, "status": "ok", "rows": count, "detail": ""})
    for source, count in value_counts_series(tables["decision_expectations"], "source").items():
        subset = tables["decision_expectations"][tables["decision_expectations"].get("source", pd.Series(dtype=str)).astype(str) == source]
        proxy_rows = int(v21_bool_series(subset.get("is_proxy", pd.Series(dtype=bool))).sum()) if not subset.empty else 0
        rows.append({"section": "decision_expectations", "source": source, "status": "proxy" if proxy_rows else "official", "rows": count, "detail": f"proxy_rows={proxy_rows}"})
    for frame_name, section in [
        ("public_market_source_audit", "public_market_source_audit"),
        ("decision_expectation_source_audit", "decision_expectation_source_audit"),
    ]:
        frame = tables[frame_name]
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "section": section,
                    "source": row.get("source", ""),
                    "status": row.get("status", ""),
                    "rows": row.get("rows", 0),
                    "detail": row.get("detail", ""),
                }
            )
    return pd.DataFrame(rows, columns=["section", "source", "status", "rows", "detail"])


def v21_focus_summary(vintages: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    delta_coverage = float(features[["delta_post_1", "delta_post_2"]].notna().any(axis=1).mean()) if not features.empty else 0.0
    status = "ready" if not vintages.empty and not features.empty and delta_coverage >= 0.40 else "partial" if not vintages.empty or not features.empty else "not_started"
    return {
        "status": status,
        "vintages": int(len(vintages)),
        "event_features": int(len(features)),
        "delta_coverage": delta_coverage,
        "events": value_counts_series(features, "event_type"),
        "indicators": value_counts_series(features, "indicator"),
        "horizons": value_counts_series(features, "horizon"),
        "statistics": value_counts_series(features, "statistic"),
        "missing_reasons": value_counts_series(features, "missing_reason"),
    }


def v21_market_summary(observations: pd.DataFrame, windows: pd.DataFrame, source_audit: pd.DataFrame) -> dict[str, Any]:
    ok_windows = int((windows.get("status", pd.Series(dtype=str)) == "ok").sum()) if not windows.empty else 0
    status = "ready" if not observations.empty and not windows.empty and ok_windows > 0 else "partial" if not observations.empty or not windows.empty else "no_market_data"
    return {
        "status": status,
        "observations": int(len(observations)),
        "event_windows": int(len(windows)),
        "ok_windows": ok_windows,
        "ambiguous_event_timing": int((windows.get("status", pd.Series(dtype=str)) == "ambiguous_event_timing").sum()) if not windows.empty else 0,
        "insufficient_market_observations": int((windows.get("status", pd.Series(dtype=str)) == "insufficient_market_observations").sum()) if not windows.empty else 0,
        "assets": value_counts_series(observations, "asset"),
        "windows_by_status": value_counts_series(windows, "status"),
        "windows_by_asset": value_counts_series(windows, "asset"),
        "source_status": value_counts_series(source_audit, "status"),
    }


def v21_decision_summary(expectations: pd.DataFrame, source_audit: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    is_proxy = v21_bool_series(expectations.get("is_proxy", pd.Series(dtype=bool))) if not expectations.empty else pd.Series(dtype=bool)
    official = int((~is_proxy).sum()) if not expectations.empty else 0
    proxy = int(is_proxy.sum()) if not expectations.empty else 0
    panel_official = int(panel.get("decision_surprise_official_bps", pd.Series(dtype=float)).notna().sum()) if not panel.empty else 0
    panel_proxy = int(panel.get("decision_surprise_proxy_bps", pd.Series(dtype=float)).notna().sum()) if not panel.empty else 0
    status = "official" if panel_official else "proxy" if panel_proxy else "not_available"
    return {
        "status": status,
        "expectation_rows": int(len(expectations)),
        "official_rows": official,
        "proxy_rows": proxy,
        "panel_official_surprises": panel_official,
        "panel_proxy_surprises": panel_proxy,
        "sources": value_counts_series(expectations, "source"),
        "proxy_methods": value_counts_series(expectations, "proxy_method"),
        "source_audit_status": value_counts_series(source_audit, "status"),
    }


def v21_event_panel_summary(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {"status": "missing", "rows": 0, "focus_ready_rows": 0, "market_ready_rows": 0, "official_surprises": 0, "proxy_surprises": 0}
    official = int(panel.get("decision_surprise_official_bps", pd.Series(dtype=float)).notna().sum())
    proxy = int(panel.get("decision_surprise_proxy_bps", pd.Series(dtype=float)).notna().sum())
    focus_ready = int((panel.get("focus_status", pd.Series(dtype=str)) == "ok").sum())
    market_ready = int((panel.get("market_status", pd.Series(dtype=str)) == "ready").sum())
    not_available = int(panel.get("decision_surprise_status", pd.Series(dtype=str)).astype(str).str.contains("not_available", na=False).sum())
    return {
        "status": "ready" if focus_ready and (official or proxy or market_ready) else "partial",
        "rows": int(len(panel)),
        "focus_ready_rows": focus_ready,
        "market_ready_rows": market_ready,
        "official_surprises": official,
        "proxy_surprises": proxy,
        "not_available_surprises": not_available,
        "decision_status_counts": value_counts_series(panel, "decision_surprise_status"),
        "focus_status_counts": value_counts_series(panel, "focus_status"),
        "market_status_counts": value_counts_series(panel, "market_status"),
    }


def v21_idempotency_summary(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {
        "focus_vintages": duplicate_count(tables["focus_vintages"], ["focus_release_date", "indicator", "reference_year", "horizon", "statistic", "source", "query_signature"]),
        "focus_event_features": duplicate_count(tables["focus_event_features"], ["meeting_id", "event_type", "indicator", "horizon", "statistic"]),
        "market_observations": duplicate_count(tables["market_observations"], ["asset", "vertex", "timestamp", "source", "source_file"]),
        "market_event_windows": duplicate_count(tables["market_event_windows"], ["meeting_id", "document_type", "asset", "vertex", "window"]),
        "decision_expectations": duplicate_count(tables["decision_expectations"], ["meeting_id", "as_of_timestamp", "expected_selic_change_bps", "source", "is_proxy"]),
        "v21_event_panel": duplicate_count(tables["v21_event_panel"], ["meeting_id"]),
    }


def v21_lookahead_summary(focus: pd.DataFrame, windows: pd.DataFrame, expectations: pd.DataFrame, panel: pd.DataFrame) -> dict[str, int]:
    focus_pre = 0
    focus_post = 0
    if not focus.empty:
        frame = focus.copy()
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        for column in ["pre_date", "post_1_date", "post_2_date"]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        focus_pre = int((frame["pre_date"].notna() & frame["event_date"].notna() & (frame["pre_date"] >= frame["event_date"])).sum())
        focus_post = int(
            (
                (frame["post_1_date"].notna() & frame["event_date"].notna() & (frame["post_1_date"] <= frame["event_date"]))
                | (frame["post_2_date"].notna() & frame["event_date"].notna() & (frame["post_2_date"] <= frame["event_date"]))
            ).sum()
        )
    market = 0
    if not windows.empty:
        ok = windows[windows.get("status", pd.Series(dtype=str)) == "ok"].copy()
        for column in ["pre_timestamp", "known_at_timestamp", "post_timestamp"]:
            ok[column] = pd.to_datetime(ok[column], errors="coerce")
        market = int((~(ok["pre_timestamp"] < ok["known_at_timestamp"]) | ~(ok["known_at_timestamp"] < ok["post_timestamp"])).sum()) if not ok.empty else 0
    decision = 0
    if not expectations.empty and not panel.empty:
        exp = expectations.copy()
        pan = panel[["meeting_id", "data_referencia"]].copy() if {"meeting_id", "data_referencia"}.issubset(panel.columns) else pd.DataFrame()
        if not pan.empty:
            exp["meeting_id"] = exp["meeting_id"].astype(str)
            pan["meeting_id"] = pan["meeting_id"].astype(str)
            merged = exp.merge(pan, on="meeting_id", how="left")
            merged["as_of_timestamp"] = pd.to_datetime(merged["as_of_timestamp"], errors="coerce")
            merged["event_cutoff"] = pd.to_datetime(merged["data_referencia"], errors="coerce").dt.normalize() + pd.Timedelta(hours=18, minutes=30)
            decision = int((merged["as_of_timestamp"].notna() & merged["event_cutoff"].notna() & (merged["as_of_timestamp"] >= merged["event_cutoff"])).sum())
    official_proxy_contamination = 0
    if not panel.empty and {"decision_surprise_official_bps", "decision_surprise_proxy_bps", "decision_surprise_status"}.issubset(panel.columns):
        official_proxy_contamination = int(
            (
                panel["decision_surprise_official_bps"].notna()
                & panel["decision_surprise_proxy_bps"].notna()
                & (panel["decision_surprise_status"].astype(str) == "proxy")
            ).sum()
        )
    return {
        "focus_pre_lookahead_violations": focus_pre,
        "focus_post_lookahead_violations": focus_post,
        "market_lookahead_violations": market,
        "decision_expectation_lookahead_violations": decision,
        "official_proxy_contamination": official_proxy_contamination,
    }


def v2_health_status(database: Path) -> dict[str, Any]:
    try:
        from copom_tone_index.v2_health import build_v2_acceptance_report

        health = build_v2_acceptance_report(database)
        return {"status": health.get("status", "unknown"), "warnings": len(health.get("warnings", [])), "errors": len(health.get("errors", []))}
    except Exception as exc:  # noqa: BLE001 - V2.1 report should surface health-check availability.
        return {"status": "unavailable", "warnings": 0, "errors": 1, "detail": str(exc)}


def collect_v21_findings(
    warnings: list[str],
    errors: list[str],
    focus: dict[str, Any],
    market: dict[str, Any],
    decision: dict[str, Any],
    event_panel: dict[str, Any],
    idempotency: dict[str, int],
    lookahead: dict[str, int],
    health: dict[str, Any],
) -> None:
    if health.get("errors", 0) > 0:
        errors.append("V2 health-check has errors; V2.1 cannot be accepted.")
    if focus["event_features"] == 0:
        errors.append("Focus event features are missing.")
    elif focus["status"] != "ready":
        warnings.append("Focus V2.1 coverage is partial.")
    if event_panel["rows"] == 0:
        errors.append("V2.1 event panel is missing.")
    if decision["panel_official_surprises"] == 0:
        warnings.append("No official decision surprise is available; B3 Opcao de Copom remains unavailable as structured public history.")
    if decision["proxy_rows"] > 0 and decision["official_rows"] == 0:
        warnings.append("Decision expectations are proxy-only and must be treated as exploratory.")
    if market["status"] != "ready":
        warnings.append(f"Market layer status is {market['status']}.")
    if market["ambiguous_event_timing"] > 0:
        warnings.append(f"{market['ambiguous_event_timing']} market windows have ambiguous event timing.")
    if market["insufficient_market_observations"] > 0:
        warnings.append(f"{market['insufficient_market_observations']} market windows have insufficient observations.")
    for table, count in idempotency.items():
        if count > 0:
            errors.append(f"V2.1 duplicate check failed for {table}: {count} duplicates.")
    for key, count in lookahead.items():
        if count > 0:
            errors.append(f"V2.1 look-ahead check failed for {key}: {count} violations.")


def value_counts_series(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].fillna("").astype(str).value_counts()
    return {str(index): int(value) for index, value in counts.items() if str(index).strip()}


def v21_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.astype(bool)
    clean = pd.Series(np.where(series.notna(), series, False), index=series.index)
    return clean.astype(str).str.lower().isin({"true", "1", "yes", "sim"})


def duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty or any(column not in frame for column in columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def render_v21_acceptance_report_html(report: dict[str, Any], by_meeting: pd.DataFrame, by_source: pd.DataFrame) -> str:
    status = str(report["status"]).upper()
    meeting_table = by_meeting.head(80).to_html(index=False, escape=True) if not by_meeting.empty else "<p>Sem painel por reuniao.</p>"
    source_table = by_source.to_html(index=False, escape=True) if not by_source.empty else "<p>Sem auditoria por fonte.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.1 Acceptance Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#e5e7eb}.fail{background:#fee2e2}.warning{background:#fef3c7}.pass{background:#dcfce7}.muted{color:#6b7280}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.1 Acceptance Report</h1>",
            f"<p>Status: <span class='badge {report['status']}'>{status}</span></p>",
            f"<p>Warnings: <strong>{len(report['warnings'])}</strong>. Errors: <strong>{len(report['errors'])}</strong>.</p>",
            "<h2>Resumo executivo</h2>",
            dict_table(report["summary"]),
            "<h2>Warnings</h2>",
            html_list(report["warnings"]),
            "<h2>Errors</h2>",
            html_list(report["errors"]),
            "<h2>Focus</h2>",
            dict_table(report["focus"]),
            "<h2>Mercado</h2>",
            dict_table(report["market"]),
            "<h2>Expectativas de decisao</h2>",
            dict_table(report["decision_expectations"]),
            "<h2>Painel de eventos</h2>",
            dict_table(report["event_panel"]),
            "<h2>Idempotencia</h2>",
            dict_table(report["idempotency"]),
            "<h2>Look-ahead</h2>",
            dict_table(report["lookahead"]),
            "<h2>Cobertura por fonte</h2>",
            source_table,
            "<h2>Cobertura por reuniao</h2>",
            meeting_table,
            "<p class='muted'>V2.1 e uma camada descritiva/associativa. Proxies de decisao nao substituem expectativa oficial de mercado.</p>",
            "</body></html>",
        ]
    )


def dict_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        rendered = json.dumps(value, ensure_ascii=False, default=v21_json_default) if isinstance(value, (dict, list)) else str(value)
        rows.append({"metric": key, "value": rendered})
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def html_list(items: list[str]) -> str:
    if not items:
        return "<p>Nenhum.</p>"
    return "<ul>" + "".join(f"<li>{escape_text(item)}</li>" for item in items) + "</ul>"


def v21_json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def build_v21_event_panel(
    meetings: pd.DataFrame,
    scores: pd.DataFrame,
    focus_features: pd.DataFrame,
    market_windows: pd.DataFrame,
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    if meetings.empty:
        return pd.DataFrame(columns=V21_EVENT_PANEL_COLUMNS)
    meetings = meetings.copy()
    scores = scores.copy()
    focus_features = focus_features.copy()
    market_windows = market_windows.copy()
    expectations = expectations.copy()
    if "data_referencia" in meetings:
        meetings["data_referencia"] = pd.to_datetime(meetings["data_referencia"], errors="coerce")
    if not scores.empty and "data_referencia" in scores:
        scores["data_referencia"] = pd.to_datetime(scores["data_referencia"], errors="coerce")
    score_cols = [col for col in ["meeting_id", "communication_surprise_naive"] if col in scores]
    score_lookup = scores[score_cols].drop_duplicates("meeting_id") if score_cols else pd.DataFrame(columns=["meeting_id"])
    rows: list[dict[str, Any]] = []
    for _, meeting in meetings.sort_values("nro_reuniao" if "nro_reuniao" in meetings else "meeting_id").iterrows():
        meeting_id = str(meeting.get("meeting_id", ""))
        selected_score = score_lookup[score_lookup["meeting_id"].astype(str) == meeting_id]
        communication_surprise = (
            selected_score["communication_surprise_naive"].iloc[0]
            if not selected_score.empty and "communication_surprise_naive" in selected_score
            else np.nan
        )
        expected = select_expected_decision(meeting, expectations)
        selic_decision_bps = selic_decision_bps_from_meeting(meeting)
        decision_surprise_official = (
            float(selic_decision_bps) - float(expected["expected_selic_change_official_bps"])
            if pd.notna(selic_decision_bps) and pd.notna(expected["expected_selic_change_official_bps"])
            else np.nan
        )
        decision_surprise_proxy = (
            float(selic_decision_bps) - float(expected["expected_selic_change_proxy_bps"])
            if pd.notna(selic_decision_bps) and pd.notna(expected["expected_selic_change_proxy_bps"])
            else np.nan
        )
        decision_surprise = decision_surprise_official if pd.notna(decision_surprise_official) else decision_surprise_proxy
        decision_status = "official" if pd.notna(decision_surprise_official) else "proxy" if pd.notna(decision_surprise_proxy) else expected["status"]
        meeting_focus = focus_features[focus_features.get("meeting_id", pd.Series(dtype=str)).astype(str) == meeting_id] if not focus_features.empty else pd.DataFrame()
        meeting_market = market_windows[market_windows.get("meeting_id", pd.Series(dtype=str)).astype(str) == meeting_id] if not market_windows.empty else pd.DataFrame()
        market_ok = meeting_market[meeting_market.get("status", pd.Series(dtype=str)) == "ok"] if not meeting_market.empty else pd.DataFrame()
        rows.append(
            {
                "meeting_id": meeting_id,
                "nro_reuniao": meeting.get("nro_reuniao", np.nan),
                "data_referencia": meeting.get("data_referencia", pd.NaT),
                "selic_decision_bps": selic_decision_bps,
                "expected_selic_change_bps": expected["expected_selic_change_bps"],
                "expected_selic_change_official_bps": expected["expected_selic_change_official_bps"],
                "expected_selic_change_proxy_bps": expected["expected_selic_change_proxy_bps"],
                "decision_surprise_bps": decision_surprise,
                "decision_surprise_official_bps": decision_surprise_official,
                "decision_surprise_proxy_bps": decision_surprise_proxy,
                "decision_surprise_status": decision_status,
                "communication_surprise_naive": communication_surprise,
                "communication_surprise_model": np.nan,
                "communication_surprise_model_status": "not_available_insufficient_macro_history",
                "focus_feature_rows": int(len(meeting_focus)),
                "focus_delta_coverage": focus_delta_coverage(meeting_focus),
                "focus_status": focus_status(meeting_focus),
                "market_window_rows": int(len(meeting_market)),
                "market_ok_windows": int(len(market_ok)),
                "mean_abs_market_reaction": safe_mean_abs(market_ok.get("market_reaction", pd.Series(dtype=float))),
                "market_status": market_status(meeting_market),
            }
        )
    return pd.DataFrame(rows, columns=V21_EVENT_PANEL_COLUMNS)


def select_expected_decision(meeting: pd.Series, expectations: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "expected_selic_change_bps": np.nan,
        "expected_selic_change_official_bps": np.nan,
        "expected_selic_change_proxy_bps": np.nan,
    }
    if expectations.empty:
        return {**empty, "status": "not_available_no_expectations"}
    meeting_id = str(meeting.get("meeting_id", ""))
    subset = expectations[expectations.get("meeting_id", pd.Series(dtype=str)).astype(str) == meeting_id].copy()
    if subset.empty:
        return {**empty, "status": "not_available_for_meeting"}
    subset["as_of_timestamp"] = pd.to_datetime(subset["as_of_timestamp"], errors="coerce")
    event_cutoff = pd.to_datetime(meeting.get("data_referencia"), errors="coerce")
    if pd.notna(event_cutoff):
        event_cutoff = event_cutoff.normalize() + pd.Timedelta(hours=18, minutes=30)
        subset = subset[subset["as_of_timestamp"] < event_cutoff]
    subset = subset.dropna(subset=["as_of_timestamp"]).sort_values("as_of_timestamp")
    if subset.empty:
        return {**empty, "status": "not_available_before_event"}
    if "is_proxy" in subset:
        is_proxy = subset["is_proxy"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "sim"})
    else:
        is_proxy = pd.Series(False, index=subset.index)
    official = latest_expected_change(subset[~is_proxy])
    proxy = latest_expected_change(subset[is_proxy])
    expected_value = official if pd.notna(official) else proxy
    if pd.notna(official):
        status = "official"
    elif pd.notna(proxy):
        status = "proxy"
    else:
        status = "not_available_missing_expected_change"
    return {
        "expected_selic_change_bps": expected_value,
        "expected_selic_change_official_bps": official,
        "expected_selic_change_proxy_bps": proxy,
        "status": status,
    }


def latest_expected_change(subset: pd.DataFrame) -> float:
    if subset.empty or "expected_selic_change_bps" not in subset:
        return np.nan
    latest_as_of = subset["as_of_timestamp"].max()
    latest = subset[subset["as_of_timestamp"] == latest_as_of].copy()
    expected_values = pd.to_numeric(latest["expected_selic_change_bps"], errors="coerce").dropna()
    return float(expected_values.iloc[-1]) if not expected_values.empty else np.nan


def selic_decision_bps_from_meeting(meeting: pd.Series) -> float | None:
    if "delta_selic" not in meeting or pd.isna(meeting.get("delta_selic")):
        return None
    return float(meeting["delta_selic"]) * 100.0


def focus_delta_coverage(features: pd.DataFrame) -> float:
    if features.empty:
        return 0.0
    return float(features[["delta_post_1", "delta_post_2"]].notna().any(axis=1).mean())


def focus_status(features: pd.DataFrame) -> str:
    coverage = focus_delta_coverage(features)
    if features.empty:
        return "not_available"
    if coverage >= 0.80:
        return "ok"
    if coverage >= 0.40:
        return "limited_data"
    return "invalid_for_inference"


def market_status(windows: pd.DataFrame) -> str:
    if windows.empty:
        return "no_market_data"
    ok_share = float((windows.get("status", pd.Series(dtype=str)) == "ok").mean())
    if ok_share >= 0.80:
        return "ready"
    if ok_share > 0:
        return "partial_market_data"
    return "no_usable_market_windows"


def safe_mean_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.abs().mean()) if not values.empty else None


def write_v21_outputs(panel: pd.DataFrame, focus_features: pd.DataFrame, market_windows: pd.DataFrame) -> Path:
    paths = get_paths()
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v21_event_panel.csv"
    panel.to_csv(output_path, index=False)
    panel.to_parquet(output_dir / "v21_event_panel.parquet", index=False)
    market_summary = market_reaction_summary(market_windows)
    market_summary.to_csv(output_dir / "market_reaction_summary.csv", index=False)
    report_path = report_dir / "v21_macro_market_report.html"
    report_path.write_text(render_v21_macro_market_report_html(panel, focus_features, market_windows), encoding="utf-8")
    return output_path


def market_reaction_summary(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty or "market_reaction" not in windows:
        return pd.DataFrame(columns=["asset", "vertex", "window", "ok_windows", "mean_reaction", "mean_abs_reaction"])
    ok = windows[windows.get("status", pd.Series(dtype=str)) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=["asset", "vertex", "window", "ok_windows", "mean_reaction", "mean_abs_reaction"])
    grouped = ok.groupby(["asset", "vertex", "window"], dropna=False)["market_reaction"]
    return grouped.agg(ok_windows="count", mean_reaction="mean", mean_abs_reaction=lambda values: values.abs().mean()).reset_index()


def render_v21_macro_market_report_html(panel: pd.DataFrame, focus_features: pd.DataFrame, market_windows: pd.DataFrame) -> str:
    status_counts = panel.get("market_status", pd.Series(dtype=str)).value_counts().reset_index().to_html(index=False, escape=True) if not panel.empty else "<p>Sem painel V2.1.</p>"
    focus_counts = panel.get("focus_status", pd.Series(dtype=str)).value_counts().reset_index().to_html(index=False, escape=True) if not panel.empty else "<p>Sem Focus V2.1.</p>"
    decision_official = int((panel.get("decision_surprise_status", pd.Series(dtype=str)) == "official").sum()) if not panel.empty else 0
    decision_proxy = int((panel.get("decision_surprise_status", pd.Series(dtype=str)) == "proxy").sum()) if not panel.empty else 0
    recent = panel.sort_values("data_referencia", ascending=False).head(25).to_html(index=False, escape=True) if not panel.empty else "<p>Sem dados.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.1 Macro Market Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.muted{color:#6b7280}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.1 Macro Market Report</h1>",
            "<p>Relatorio descritivo de decisao, comunicacao, Focus e reacao de mercado. Nao e interpretacao causal.</p>",
            f"<p>Reunioes no painel: <strong>{len(panel)}</strong>. Surpresa oficial: <strong>{decision_official}</strong>. Proxy: <strong>{decision_proxy}</strong>.</p>",
            f"<p>Focus event features: <strong>{len(focus_features)}</strong>. Market windows: <strong>{len(market_windows)}</strong>.</p>",
            "<h2>Status Focus</h2>",
            focus_counts,
            "<h2>Status Mercado</h2>",
            status_counts,
            "<h2>Painel recente</h2>",
            recent,
            "<p class='muted'>Communication surprise naive continua sendo mudanca textual contra a reuniao anterior; surpresa de mercado exige expectativas importadas.</p>",
            "</body></html>",
        ]
    )


def escape_text(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
