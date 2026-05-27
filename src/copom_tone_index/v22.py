from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from copom_tone_index.config import ROOT, get_paths, load_v2_settings
from copom_tone_index.storage import write_tables
from copom_tone_index.v2 import V2CommandResult, file_sha256, make_run_id, read_optional_table, rule_engine_version, utc_now_naive, v2_benchmark_baseline_command


PUBLIC_DATA_DIR = ROOT / "app_data"
PUBLIC_DATA_DB = PUBLIC_DATA_DIR / "copom_watch_public.duckdb"
PUBLIC_DATA_MANIFEST = PUBLIC_DATA_DIR / "public_data_manifest.json"

PUBLIC_TABLES: dict[str, bool] = {
    "copom_scores": False,
    "copom_meetings": False,
    "copom_topic_scores": False,
    "evidence_sentences": False,
    "focus_revisions": False,
    "v2_meeting_scores": True,
    "v2_subindices": True,
    "v2_evidence": True,
    "v2_redline": True,
    "v2_model_audit": False,
    "v2_model_audit_details": False,
    "focus_event_features": False,
    "market_event_windows": False,
    "decision_expectations": False,
    "v21_event_panel": True,
    "semantic_chunks": True,
}

FORBIDDEN_PUBLIC_TABLES = {
    "v2_labels",
    "focus_vintages",
    "focus_observations",
    "v2_documents",
    "v2_sentences",
    "v2_sentence_scores",
    "v2_model_predictions",
}


@dataclass(frozen=True)
class V22HealthResult:
    status: str
    json_path: Path
    html_path: Path
    warnings: int
    errors: int


def package_public_data_command() -> V2CommandResult:
    paths = get_paths()
    run_id = make_run_id("v22_package")
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PUBLIC_DATA_DB.exists():
        PUBLIC_DATA_DB.unlink()
    wal_path = PUBLIC_DATA_DB.with_suffix(PUBLIC_DATA_DB.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}
    for table, required in PUBLIC_TABLES.items():
        frame = read_optional_table(paths.database, table, pd.DataFrame())
        if frame.empty and len(frame.columns) == 0:
            skipped.append({"table": table, "required": required, "reason": "missing"})
            continue
        tables[table] = frame
        included.append({"table": table, "required": required, "rows": int(len(frame)), "columns": list(frame.columns)})

    manifests = collect_deploy_manifests(paths.reports.parent / "v2")
    if not manifests.empty:
        tables["app_manifests"] = manifests
        included.append({"table": "app_manifests", "required": False, "rows": int(len(manifests)), "columns": list(manifests.columns)})

    if tables:
        write_tables(PUBLIC_DATA_DB, tables)

    copied_manifests = copy_deploy_manifest_files(paths.reports.parent / "v2", PUBLIC_DATA_DIR / "manifests")
    package_manifest = {
        "generated_at": utc_now_naive().isoformat(),
        "run_id": run_id,
        "source_database": str(paths.database),
        "package_database": str(PUBLIC_DATA_DB),
        "package_sha256": file_sha256(PUBLIC_DATA_DB) if PUBLIC_DATA_DB.exists() else "",
        "tables_included": included,
        "tables_skipped": skipped,
        "forbidden_tables": sorted(FORBIDDEN_PUBLIC_TABLES),
        "copied_manifests": copied_manifests,
        "status": "completed" if package_required_tables_present(PUBLIC_DATA_DB) else "partial",
    }
    PUBLIC_DATA_MANIFEST.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows = sum(int(item["rows"]) for item in included)
    return V2CommandResult(paths.database, PUBLIC_DATA_MANIFEST, rows, str(package_manifest["status"]), run_id)


def v22_health_command() -> V22HealthResult:
    paths = get_paths()
    report_dir = paths.reports.parent / "v2"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_v22_acceptance_report()
    json_path = report_dir / "v22_acceptance_report.json"
    html_path = report_dir / "v22_acceptance_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_v22_acceptance_report_html(report), encoding="utf-8")
    return V22HealthResult(
        status=str(report["status"]),
        json_path=json_path,
        html_path=html_path,
        warnings=len(report["warnings"]),
        errors=len(report["errors"]),
    )


def v22_freeze_release_command(version: str = "v2.2-product-rag-deploy") -> V2CommandResult:
    paths = get_paths()
    run_id = make_run_id("v22_freeze")
    report_dir = paths.reports.parent / "v2"
    release_dir = report_dir / "releases" / version
    report_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    from copom_tone_index.v2_health import v2_health_check_command
    from copom_tone_index.v21 import v21_health_command

    benchmark = v2_benchmark_baseline_command()
    v2_health = v2_health_check_command()
    v21_health = v21_health_command()
    v22_health = v22_health_command()
    v22_report = json.loads(v22_health.json_path.read_text(encoding="utf-8")) if v22_health.json_path.exists() else {}
    release_errors = collect_v22_release_errors(benchmark.status, v2_health.errors, v21_health.errors, v22_health.errors, v22_report)
    status = "fail" if release_errors else "completed"

    artifacts = collect_v22_release_artifacts(paths.reports.parent / "v2")
    copied_artifacts = copy_v22_release_artifacts(artifacts, release_dir)
    manifest = build_v22_release_manifest(
        version=version,
        status=status,
        run_id=run_id,
        database=paths.database,
        release_dir=release_dir,
        benchmark_status=benchmark.status,
        v2_health=v2_health,
        v21_health=v21_health,
        v22_health=v22_health,
        v22_report=v22_report,
        release_errors=release_errors,
        artifacts=copied_artifacts,
    )
    manifest_path = report_dir / "v22_release_manifest.json"
    summary_path = report_dir / "v22_release_summary.html"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_path.write_text(render_v22_release_summary_html(manifest), encoding="utf-8")
    for path in [manifest_path, summary_path]:
        target = release_dir / path.name
        shutil.copy2(path, target)
    return V2CommandResult(paths.database, manifest_path, len(copied_artifacts), status, run_id)


def build_v22_acceptance_report() -> dict[str, Any]:
    files = v22_file_status()
    package = inspect_public_package(PUBLIC_DATA_DB)
    manifests = inspect_deploy_manifests()
    semantic = semantic_package_status(package)
    warnings: list[str] = []
    errors: list[str] = []
    for path, exists in files.items():
        if not exists:
            errors.append(f"Required deploy file is missing: {path}")
    if not PUBLIC_DATA_DB.exists():
        errors.append("Public DuckDB package is missing.")
    if not PUBLIC_DATA_MANIFEST.exists():
        errors.append("Public data manifest is missing.")
    for table, required in PUBLIC_TABLES.items():
        if required and table not in package["tables"]:
            errors.append(f"Required public package table is missing: {table}")
    forbidden = sorted(set(package["tables"]) & FORBIDDEN_PUBLIC_TABLES)
    if forbidden:
        errors.append(f"Forbidden tables found in public package: {', '.join(forbidden)}")
    if semantic["status"] != "ready":
        errors.append("Semantic/RAG index is missing from public package.")
    if not manifests["v20_release_manifest"]:
        warnings.append("V2.0 release manifest was not found in app_data/manifests.")
    if not manifests["v21_release_manifest"]:
        warnings.append("V2.1 release manifest was not found in app_data/manifests.")
    if manifests["b3_official_unavailable"]:
        warnings.append("B3 official decision surprise remains unavailable; V2.2 must show proxy/availability status.")
    status = "fail" if errors else "warning" if warnings else "pass"
    return {
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "public_database": str(PUBLIC_DATA_DB),
        "public_manifest": str(PUBLIC_DATA_MANIFEST),
        "files": files,
        "package": package,
        "manifests": manifests,
        "semantic": semantic,
        "warnings": warnings,
        "errors": errors,
    }


def collect_v22_release_errors(
    benchmark_status: str,
    v2_health_errors: int,
    v21_health_errors: int,
    v22_health_errors: int,
    v22_report: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if benchmark_status == "fail":
        errors.append("V2 baseline benchmark is fail; V2.2 cannot be frozen.")
    if v2_health_errors > 0:
        errors.append(f"V2 health-check has {v2_health_errors} errors.")
    if v21_health_errors > 0:
        errors.append(f"V2.1 health-check has {v21_health_errors} errors.")
    if v22_health_errors > 0:
        errors.append(f"V2.2 health-check has {v22_health_errors} errors.")
    for error in v22_report.get("errors", []):
        errors.append(f"V2.2 acceptance error: {error}")
    files = v22_file_status()
    for path, exists in files.items():
        if not exists:
            errors.append(f"Required V2.2 release file is missing: {path}")
    package = inspect_public_package(PUBLIC_DATA_DB)
    if semantic_package_status(package)["status"] != "ready":
        errors.append("Semantic/RAG index is missing from public package.")
    return sorted(set(errors))


def collect_v22_release_artifacts(report_dir: Path) -> list[tuple[str, Path]]:
    artifacts: list[tuple[str, Path]] = [
        ("copom_watch_public.duckdb", PUBLIC_DATA_DB),
        ("public_data_manifest.json", PUBLIC_DATA_MANIFEST),
        ("streamlit_app.py", ROOT / "streamlit_app.py"),
        ("requirements.txt", ROOT / "requirements.txt"),
        ("streamlit_config.toml", ROOT / ".streamlit" / "config.toml"),
        ("v22_acceptance_report.json", report_dir / "v22_acceptance_report.json"),
        ("v22_acceptance_report.html", report_dir / "v22_acceptance_report.html"),
        ("semantic_ask_report.html", report_dir / "semantic_ask_report.html"),
        ("acceptance_report.json", report_dir / "acceptance_report.json"),
        ("acceptance_report.html", report_dir / "acceptance_report.html"),
        ("v21_acceptance_report.json", report_dir / "v21_acceptance_report.json"),
        ("v21_acceptance_report.html", report_dir / "v21_acceptance_report.html"),
        ("release_manifest.json", report_dir / "release_manifest.json"),
        ("v21_release_manifest.json", report_dir / "v21_release_manifest.json"),
    ]
    manifest_dir = PUBLIC_DATA_DIR / "manifests"
    for path in sorted(manifest_dir.glob("*.json")) if manifest_dir.exists() else []:
        artifacts.append((f"app_data_manifest_{path.name}", path))
    return artifacts


def copy_v22_release_artifacts(artifacts: list[tuple[str, Path]], release_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    seen_targets: set[Path] = set()
    for name, path in artifacts:
        if not path.exists() or not path.is_file():
            continue
        target = release_dir / name
        if target in seen_targets:
            continue
        shutil.copy2(path, target)
        seen_targets.add(target)
        copied.append(
            {
                "name": name,
                "source_path": str(path),
                "path": str(target),
                "sha256": file_sha256(target),
                "bytes": target.stat().st_size,
            }
        )
    return copied


def build_v22_release_manifest(
    version: str,
    status: str,
    run_id: str,
    database: Path,
    release_dir: Path,
    benchmark_status: str,
    v2_health: Any,
    v21_health: Any,
    v22_health: V22HealthResult,
    v22_report: dict[str, Any],
    release_errors: list[str],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    package = v22_report.get("package", inspect_public_package(PUBLIC_DATA_DB))
    semantic = v22_report.get("semantic", semantic_package_status(package))
    warnings = list(v22_report.get("warnings", []))
    if inspect_deploy_manifests().get("b3_official_unavailable"):
        warnings.append("B3 official decision surprise remains unavailable; V2.2 exposes proxy/availability status.")
    return {
        "version": version,
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "run_id": run_id,
        "database_path": str(database),
        "release_dir": str(release_dir),
        "rule_engine_version": rule_engine_version(load_v2_settings()),
        "official_text_index": "deterministic_v2_baseline",
        "deploy_entrypoint": "streamlit_app.py",
        "streamlit_command": "streamlit run streamlit_app.py --server.port 8502 --server.address localhost",
        "benchmark_baseline_status": benchmark_status,
        "v2_health": {"status": v2_health.status, "warnings": v2_health.warnings, "errors": v2_health.errors},
        "v21_health": {"status": v21_health.status, "warnings": v21_health.warnings, "errors": v21_health.errors},
        "v22_health": {"status": v22_health.status, "warnings": v22_health.warnings, "errors": v22_health.errors},
        "public_package": {
            "path": str(PUBLIC_DATA_DB),
            "sha256": package.get("sha256", ""),
            "bytes": package.get("bytes", 0),
            "tables": package.get("tables", []),
            "rows": package.get("rows", {}),
        },
        "rag": semantic,
        "methodology": [
            "V2.2 freezes product, RAG and deploy packaging only.",
            "V2.2 does not change the V2.0.4 textual index, formula, taxonomy or calibration.",
            "RAG is local/extractive and does not require an LLM or paid API key.",
            "Event-study and market outputs remain descriptive/associative, not causal inference.",
        ],
        "warnings": sorted(set(warnings)),
        "errors": release_errors,
        "artifacts": artifacts,
    }


def render_v22_release_summary_html(manifest: dict[str, Any]) -> str:
    artifacts = pd.DataFrame(manifest.get("artifacts", []))
    artifact_table = artifacts.to_html(index=False, escape=True) if not artifacts.empty else "<p>Nenhum artefato copiado.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.2 Release Summary</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#e5e7eb}.completed{background:#dcfce7}.fail{background:#fee2e2}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.2 Release Summary</h1>",
            f"<p>Status: <span class='badge {manifest['status']}'>{escape_html(str(manifest['status']).upper())}</span></p>",
            f"<p>Version: <strong>{escape_html(manifest['version'])}</strong></p>",
            "<h2>Resumo</h2>",
            dict_table(
                {
                    "rule_engine_version": manifest.get("rule_engine_version"),
                    "deploy_entrypoint": manifest.get("deploy_entrypoint"),
                    "benchmark_baseline_status": manifest.get("benchmark_baseline_status"),
                    "v2_health": manifest.get("v2_health"),
                    "v21_health": manifest.get("v21_health"),
                    "v22_health": manifest.get("v22_health"),
                }
            ),
            "<h2>Pacote publico</h2>",
            dict_table(manifest.get("public_package", {})),
            "<h2>RAG local</h2>",
            dict_table(manifest.get("rag", {})),
            "<h2>Warnings</h2>",
            html_list(manifest.get("warnings", [])),
            "<h2>Errors</h2>",
            html_list(manifest.get("errors", [])),
            "<h2>Metodologia</h2>",
            html_list(manifest.get("methodology", [])),
            "<h2>Artefatos</h2>",
            artifact_table,
            "</body></html>",
        ]
    )


def v22_file_status() -> dict[str, bool]:
    return {
        "streamlit_app.py": (ROOT / "streamlit_app.py").exists(),
        "requirements.txt": (ROOT / "requirements.txt").exists(),
        ".streamlit/config.toml": (ROOT / ".streamlit" / "config.toml").exists(),
        "app_data/copom_watch_public.duckdb": PUBLIC_DATA_DB.exists(),
        "app_data/public_data_manifest.json": PUBLIC_DATA_MANIFEST.exists(),
    }


def inspect_public_package(database: Path) -> dict[str, Any]:
    if not database.exists():
        return {"exists": False, "tables": [], "rows": {}, "sha256": "", "bytes": 0}
    with duckdb.connect(str(database), read_only=True) as con:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        rows = {table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
    return {"exists": True, "tables": sorted(tables), "rows": rows, "sha256": file_sha256(database), "bytes": database.stat().st_size}


def semantic_package_status(package: dict[str, Any]) -> dict[str, Any]:
    rows = int(package.get("rows", {}).get("semantic_chunks", 0))
    return {"status": "ready" if rows > 0 else "missing", "semantic_chunks": rows}


def inspect_deploy_manifests() -> dict[str, Any]:
    manifest_dir = PUBLIC_DATA_DIR / "manifests"
    v20 = manifest_dir / "release_manifest.json"
    v21 = manifest_dir / "v21_release_manifest.json"
    b3_unavailable = False
    if v21.exists():
        try:
            payload = json.loads(v21.read_text(encoding="utf-8"))
            b3_unavailable = int(payload.get("surprise_metrics", {}).get("official_surprises", 0)) == 0
        except json.JSONDecodeError:
            b3_unavailable = False
    return {
        "v20_release_manifest": v20.exists(),
        "v21_release_manifest": v21.exists(),
        "b3_official_unavailable": b3_unavailable,
    }


def package_required_tables_present(database: Path) -> bool:
    if not database.exists():
        return False
    package = inspect_public_package(database)
    tables = set(package["tables"])
    return all(table in tables for table, required in PUBLIC_TABLES.items() if required)


def collect_deploy_manifests(report_dir: Path) -> pd.DataFrame:
    rows = []
    for name in ["release_manifest.json", "v21_release_manifest.json", "acceptance_report.json", "v21_acceptance_report.json"]:
        path = report_dir / name
        if not path.exists():
            continue
        rows.append(
            {
                "manifest_name": name,
                "source_path": str(path),
                "content": path.read_text(encoding="utf-8"),
                "sha256": file_sha256(path),
                "loaded_at": utc_now_naive(),
            }
        )
    return pd.DataFrame(rows, columns=["manifest_name", "source_path", "content", "sha256", "loaded_at"])


def copy_deploy_manifest_files(report_dir: Path, target_dir: Path) -> list[dict[str, Any]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["release_manifest.json", "v21_release_manifest.json", "acceptance_report.json", "v21_acceptance_report.json"]:
        source = report_dir / name
        if not source.exists():
            continue
        target = target_dir / name
        shutil.copy2(source, target)
        copied.append({"name": name, "path": str(target), "sha256": file_sha256(target), "bytes": target.stat().st_size})
    return copied


def render_v22_acceptance_report_html(report: dict[str, Any]) -> str:
    status = str(report["status"]).upper()
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.2 Acceptance Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#e5e7eb}.fail{background:#fee2e2}.warning{background:#fef3c7}.pass{background:#dcfce7}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.2 Acceptance Report</h1>",
            f"<p>Status: <span class='badge {report['status']}'>{status}</span></p>",
            f"<p>Warnings: <strong>{len(report['warnings'])}</strong>. Errors: <strong>{len(report['errors'])}</strong>.</p>",
            "<h2>Arquivos de deploy</h2>",
            dict_table(report["files"]),
            "<h2>Pacote publico</h2>",
            dict_table(report["package"]),
            "<h2>Manifests</h2>",
            dict_table(report["manifests"]),
            "<h2>RAG local</h2>",
            dict_table(report["semantic"]),
            "<h2>Warnings</h2>",
            html_list(report["warnings"]),
            "<h2>Errors</h2>",
            html_list(report["errors"]),
            "</body></html>",
        ]
    )


def dict_table(data: dict[str, Any]) -> str:
    rows = [{"metric": key, "value": json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)} for key, value in data.items()]
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def html_list(items: list[str]) -> str:
    if not items:
        return "<p>Nenhum.</p>"
    return "<ul>" + "".join(f"<li>{escape_html(item)}</li>" for item in items) + "</ul>"


def escape_html(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
