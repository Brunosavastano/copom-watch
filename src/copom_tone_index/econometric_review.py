from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from copom_tone_index.config import get_paths, load_topics
from copom_tone_index.focus import (
    audit_focus_coverage,
    build_focus_missing_events,
    empty_focus_observations,
    read_optional_table,
)
from copom_tone_index.scoring import classify_index
from copom_tone_index.storage import read_table


MIN_FORMAL_OBSERVATIONS = 30
MIN_EXPLORATORY_OBSERVATIONS = 8


@dataclass
class EconometricReviewResult:
    status: str
    report_path: Path
    output_dir: Path
    valid_regressions: int
    blocked_regressions: int


def run_econometric_review(
    database: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    manual_sample_size: int = 80,
    min_formal_obs: int = MIN_FORMAL_OBSERVATIONS,
    min_exploratory_obs: int = MIN_EXPLORATORY_OBSERVATIONS,
) -> EconometricReviewResult:
    paths = get_paths()
    database = database or paths.database
    output_dir = output_dir or paths.processed.parent / "econometrics"
    report_path = report_path or paths.reports.parent / "econometric_accuracy_review.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tables = _load_review_tables(database)
    data_quality = audit_data_quality(tables)
    focus_coverage = audit_focus_coverage(tables["focus_revisions"])
    focus_missing = build_focus_missing_events(tables["focus_revisions"])
    sensitivity = build_index_sensitivity(
        tables["copom_scores"],
        tables["copom_sentences"],
        tables["copom_documents"],
        load_topics(),
    )
    manual_metrics, sample_path = build_or_evaluate_manual_sentence_sample(
        tables["copom_sentences"],
        output_dir / "manual_sentence_audit_sample.csv",
        sample_size=manual_sample_size,
    )
    regression_diagnostics = run_regression_suite(
        tables["copom_scores"],
        tables["focus_revisions"],
        min_formal_obs=min_formal_obs,
        min_exploratory_obs=min_exploratory_obs,
    )

    _write_frame(data_quality, output_dir / "data_quality_audit")
    _write_frame(focus_coverage, output_dir / "focus_coverage_audit")
    _write_frame(focus_missing, output_dir / "focus_missing_events")
    _write_frame(sensitivity, output_dir / "index_sensitivity")
    _write_frame(manual_metrics, output_dir / "manual_validation_metrics")
    _write_frame(regression_diagnostics, output_dir / "regression_diagnostics")

    status = determine_overall_status(data_quality, regression_diagnostics)
    report = render_review_report(
        status=status,
        data_quality=data_quality,
        focus_coverage=focus_coverage,
        sensitivity=sensitivity,
        manual_metrics=manual_metrics,
        regression_diagnostics=regression_diagnostics,
        sample_path=sample_path,
        output_dir=output_dir,
    )
    report_path.write_text(report, encoding="utf-8")
    return EconometricReviewResult(
        status=status,
        report_path=report_path,
        output_dir=output_dir,
        valid_regressions=int((regression_diagnostics["status"] == "estimated").sum())
        if not regression_diagnostics.empty
        else 0,
        blocked_regressions=int((regression_diagnostics["status"] == "blocked").sum())
        if not regression_diagnostics.empty
        else 0,
    )


def audit_data_quality(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    meetings = tables.get("copom_meetings", pd.DataFrame())
    documents = tables.get("copom_documents", pd.DataFrame())
    sentences = tables.get("copom_sentences", pd.DataFrame())
    scores = tables.get("copom_scores", pd.DataFrame())
    focus = tables.get("focus_revisions", pd.DataFrame())
    focus_observations = tables.get("focus_observations", pd.DataFrame())
    rows: list[dict[str, Any]] = []

    meeting_count = len(meetings)
    operational_count = int(meetings.get("in_operational_window", pd.Series(dtype=bool)).fillna(False).sum())
    rows.append(
        _audit_row(
            "data",
            "meeting_coverage",
            meeting_count,
            "limited_sample" if meeting_count < MIN_FORMAL_OBSERVATIONS else "ok",
            f"{operational_count} meetings in the operational window.",
        )
    )

    observation_count = len(focus_observations)
    if focus_observations.empty or "source" not in focus_observations:
        source_detail = "No Focus observations are stored. Run fetch-focus or import-focus-snapshot."
    else:
        source_counts = focus_observations["source"].value_counts().to_dict()
        source_detail = f"Stored Focus observations by source: {source_counts}."
    rows.append(
        _audit_row(
            "data",
            "focus_observation_coverage",
            observation_count,
            "invalid_for_inference" if observation_count == 0 else "ok",
            source_detail,
        )
    )

    expected_documents = meeting_count * 2
    document_count = len(documents)
    missing_documents = max(expected_documents - document_count, 0)
    rows.append(
        _audit_row(
            "data",
            "document_coverage",
            document_count,
            "limited_data" if missing_documents else "ok",
            f"{missing_documents} expected comunicado/ata documents are missing.",
        )
    )

    sentence_count = len(sentences)
    missing_scores = int(sentences.get("stance_score", pd.Series(dtype=float)).isna().sum()) if not sentences.empty else 0
    rows.append(
        _audit_row(
            "text",
            "sentence_scoring",
            sentence_count,
            "limited_data" if sentence_count == 0 or missing_scores else "ok",
            f"{missing_scores} sentences have missing stance_score.",
        )
    )

    focus_total = len(focus)
    focus_delta_available = int(
        focus[["delta_post_comunicado", "delta_post_ata"]].notna().any(axis=1).sum()
    ) if focus_total and {"delta_post_comunicado", "delta_post_ata"}.issubset(focus.columns) else 0
    focus_coverage = focus_delta_available / focus_total if focus_total else 0.0
    rows.append(
        _audit_row(
            "data",
            "focus_revision_coverage",
            round(focus_coverage, 4),
            "invalid_for_inference" if focus_coverage == 0 else "limited_data" if focus_coverage < 0.8 else "ok",
            f"{focus_delta_available}/{focus_total} Focus revision rows have at least one usable delta.",
        )
    )

    selic_variation = _non_null_nunique(scores, "delta_selic")
    rows.append(
        _audit_row(
            "data",
            "delta_selic_variation",
            selic_variation,
            "invalid_for_inference" if selic_variation <= 1 else "ok",
            "Regressions using delta_selic are blocked when the series has no variation.",
        )
    )

    title_alignment = audit_selic_title_alignment(meetings if not meetings.empty else scores)
    rows.append(title_alignment)

    surprise_available = int(scores.get("communication_surprise", pd.Series(dtype=float)).notna().sum())
    surprise_status = "ok"
    surprise_detail = f"{surprise_available} non-missing communication_surprise observations."
    if surprise_available < MIN_FORMAL_OBSERVATIONS:
        surprise_status = "limited_sample"
        surprise_detail += " Not enough observations for formal validation."
    if selic_variation <= 1:
        surprise_status = "invalid_for_inference"
        surprise_detail += " Residual interpretation is not defensible because delta_selic has no variation."
    rows.append(_audit_row("model", "communication_surprise_validity", surprise_available, surprise_status, surprise_detail))
    return pd.DataFrame(rows)


def audit_selic_title_alignment(meetings: pd.DataFrame) -> dict[str, Any]:
    if meetings.empty or "titulo_comunicado" not in meetings or "selic_pos" not in meetings:
        return _audit_row("data", "selic_title_alignment", np.nan, "limited_data", "Missing titles or selic_pos.")
    mismatches = 0
    comparable = 0
    for _, row in meetings.iterrows():
        title_rate = parse_selic_rate_from_title(row.get("titulo_comunicado"))
        if title_rate is None or pd.isna(row.get("selic_pos")):
            continue
        comparable += 1
        if not np.isclose(float(title_rate), float(row["selic_pos"]), atol=0.01):
            mismatches += 1
    if comparable == 0:
        return _audit_row("data", "selic_title_alignment", 0, "limited_data", "No comparable title rates found.")
    return _audit_row(
        "data",
        "selic_title_alignment",
        comparable - mismatches,
        "limited_data" if mismatches else "ok",
        f"{mismatches}/{comparable} title-implied Selic rates differ from selic_pos.",
    )


def parse_selic_rate_from_title(title: object) -> float | None:
    if not isinstance(title, str):
        return None
    match = re.search(r"(\d{1,2}(?:,\d{1,2}|\.\d{1,2})?)\s*%", title)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def build_index_sensitivity(
    scores: pd.DataFrame,
    sentences: pd.DataFrame,
    documents: pd.DataFrame,
    topics: dict[str, Any],
) -> pd.DataFrame:
    if scores.empty or sentences.empty:
        return pd.DataFrame(columns=["variant", "rank_corr", "classification_agreement", "mean_abs_index_diff", "status"])

    base = scores[["meeting_id", "tone_raw", "copom_tone_index", "classification"]].copy()
    variants = {
        "equal_topic_weights": _alternative_index(sentences, documents, topics, topic_weight_mode="equal"),
        "document_weights_50_50": _alternative_index(sentences, documents, topics, comunicado_weight=0.50, ata_weight=0.50),
        "document_weights_80_20": _alternative_index(sentences, documents, topics, comunicado_weight=0.80, ata_weight=0.20),
        "comunicado_only": _alternative_index(sentences, documents, topics, comunicado_weight=1.0, ata_weight=0.0),
        "ata_only": _alternative_index(sentences, documents, topics, comunicado_weight=0.0, ata_weight=1.0),
        "trim_top_abs_5pct_sentences": _alternative_index(sentences, documents, topics, trim_abs_quantile=0.95),
    }
    rows: list[dict[str, Any]] = []
    for variant, alt in variants.items():
        joined = base.merge(alt, on="meeting_id", how="inner")
        rank_corr = _safe_corr(joined["copom_tone_index"], joined["copom_tone_index_alt"], method="spearman")
        agreement = (
            float((joined["classification"] == joined["classification_alt"]).mean()) if not joined.empty else np.nan
        )
        mean_abs_diff = (
            float((joined["copom_tone_index"] - joined["copom_tone_index_alt"]).abs().mean())
            if not joined.empty
            else np.nan
        )
        status = "ok"
        if pd.isna(rank_corr) or rank_corr < 0.70 or agreement < 0.50:
            status = "sensitivity_risk"
        elif rank_corr < 0.85 or agreement < 0.70:
            status = "limited_stability"
        rows.append(
            {
                "variant": variant,
                "observations": len(joined),
                "rank_corr": rank_corr,
                "classification_agreement": agreement,
                "mean_abs_index_diff": mean_abs_diff,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_or_evaluate_manual_sentence_sample(
    sentences: pd.DataFrame,
    sample_path: Path,
    sample_size: int = 80,
) -> tuple[pd.DataFrame, Path]:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:
        sample = _build_manual_sample(sentences, sample_size)
        sample.to_csv(sample_path, index=False)
    metrics = evaluate_manual_sentence_labels(sample)
    return metrics, sample_path


def evaluate_manual_sentence_labels(sample: pd.DataFrame) -> pd.DataFrame:
    required = {"model_topic", "model_stance", "human_topic", "human_stance"}
    if sample.empty or not required.issubset(sample.columns):
        return pd.DataFrame(
            [
                {
                    "dimension": "manual_sentence_validation",
                    "status": "pending",
                    "observations": 0,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "cohen_kappa": np.nan,
                    "detail": "Manual audit sample exists but human labels are missing.",
                }
            ]
        )
    completed = sample.dropna(subset=["human_topic", "human_stance"]).copy()
    completed = completed[(completed["human_topic"].astype(str).str.strip() != "") & (completed["human_stance"].astype(str).str.strip() != "")]
    if completed.empty:
        return pd.DataFrame(
            [
                {
                    "dimension": "manual_sentence_validation",
                    "status": "pending",
                    "observations": 0,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "cohen_kappa": np.nan,
                    "detail": "Fill human_topic and human_stance to compute manual accuracy.",
                }
            ]
        )
    topic_metrics = _classification_metrics(completed["human_topic"], completed["model_topic"])
    stance_metrics = _classification_metrics(completed["human_stance"], completed["model_stance"])
    return pd.DataFrame(
        [
            {"dimension": "topic", "status": "estimated", "observations": len(completed), **topic_metrics},
            {"dimension": "stance", "status": "estimated", "observations": len(completed), **stance_metrics},
        ]
    )


def run_regression_suite(
    scores: pd.DataFrame,
    focus_revisions: pd.DataFrame,
    min_formal_obs: int = MIN_FORMAL_OBSERVATIONS,
    min_exploratory_obs: int = MIN_EXPLORATORY_OBSERVATIONS,
) -> pd.DataFrame:
    if scores.empty or focus_revisions.empty:
        return pd.DataFrame([_blocked_regression_row("all", "all", "all", "all", "missing_scores_or_focus")])
    panel = focus_revisions.merge(scores, on="meeting_id", how="left")
    specs = [
        ("intercept_only", []),
        ("delta_selic", ["delta_selic"]),
        ("tone_raw", ["tone_raw"]),
        ("delta_tone", ["delta_tone"]),
        ("communication_surprise", ["communication_surprise"]),
    ]
    target_columns = ["delta_post_comunicado", "delta_post_ata"]
    rows: list[dict[str, Any]] = []
    for variable in sorted(panel["variable"].dropna().unique()):
        for reference_year in sorted(panel.loc[panel["variable"] == variable, "reference_year"].dropna().unique()):
            subset = panel[(panel["variable"] == variable) & (panel["reference_year"] == reference_year)].copy()
            for target in target_columns:
                for model_name, predictors in specs:
                    rows.append(
                        fit_regression_review(
                            subset,
                            target=target,
                            predictors=predictors,
                            model_name=model_name,
                            variable=variable,
                            reference_year=int(reference_year),
                            min_formal_obs=min_formal_obs,
                            min_exploratory_obs=min_exploratory_obs,
                        )
                    )
    return pd.DataFrame(rows)


def fit_regression_review(
    data: pd.DataFrame,
    target: str,
    predictors: list[str],
    model_name: str,
    variable: str,
    reference_year: int,
    min_formal_obs: int = MIN_FORMAL_OBSERVATIONS,
    min_exploratory_obs: int = MIN_EXPLORATORY_OBSERVATIONS,
) -> dict[str, Any]:
    block_reason = validate_regression_inputs(
        data,
        target,
        predictors,
        min_exploratory_obs=min_exploratory_obs,
        min_formal_obs=min_formal_obs,
    )
    base = {
        "variable": variable,
        "reference_year": reference_year,
        "target": target,
        "model": model_name,
        "predictors": ",".join(predictors) if predictors else "const",
    }
    if block_reason:
        return base | _blocked_regression_fields(block_reason)

    cols = [target] + predictors
    clean = data[cols].dropna().copy()
    y = clean[target].astype(float)
    x = _design_matrix(clean, predictors)
    fit = sm.OLS(y, x).fit()
    robust = fit.get_robustcov_results(cov_type="HC3")
    predictions = pd.Series(fit.predict(x), index=y.index)
    loo = leave_one_out_metrics(clean, target, predictors)
    coefficient_name = predictors[0] if predictors else "const"
    coefficient_position = list(x.columns).index(coefficient_name)
    influence = OLSInfluence(fit)
    bootstrap = wild_bootstrap_coefficient_stability(clean, target, predictors, coefficient_name)
    status = "estimated"
    inference_flag = "formal_candidate" if len(clean) >= min_formal_obs else "exploratory_only"
    if model_name == "communication_surprise" and len(clean) < min_formal_obs:
        inference_flag = "not_validated_less_than_30_obs"
    return base | {
        "status": status,
        "block_reason": "",
        "inference_flag": inference_flag,
        "observations": len(clean),
        "r_squared": fit.rsquared,
        "rmse": _rmse(y, predictions),
        "mae": _mae(y, predictions),
        "directional_accuracy": _directional_accuracy(y, predictions),
        "spearman_corr": _safe_corr(y, predictions, method="spearman"),
        "loo_rmse": loo["rmse"],
        "loo_mae": loo["mae"],
        "loo_directional_accuracy": loo["directional_accuracy"],
        "coef_name": coefficient_name,
        "coef": float(fit.params.iloc[coefficient_position]),
        "hc3_se": float(robust.bse[coefficient_position]),
        "hc3_t": float(robust.tvalues[coefficient_position]),
        "hc3_pvalue": float(robust.pvalues[coefficient_position]),
        "max_cooks_d": float(np.nanmax(influence.cooks_distance[0])),
        "bootstrap_coef_std": bootstrap["coef_std"],
        "bootstrap_sign_stability": bootstrap["sign_stability"],
    }


def validate_regression_inputs(
    data: pd.DataFrame,
    target: str,
    predictors: list[str],
    min_exploratory_obs: int = MIN_EXPLORATORY_OBSERVATIONS,
    min_formal_obs: int = MIN_FORMAL_OBSERVATIONS,
) -> str | None:
    missing_columns = [column for column in [target] + predictors if column not in data.columns]
    if missing_columns:
        return f"missing_columns:{','.join(missing_columns)}"
    clean = data[[target] + predictors].dropna()
    if len(clean) < min_exploratory_obs:
        return f"insufficient_observations:{len(clean)}"
    if _non_null_nunique(clean, target) <= 1:
        return f"target_no_variation:{target}"
    for predictor in predictors:
        if _non_null_nunique(clean, predictor) <= 1:
            return f"predictor_no_variation:{predictor}"
    if "communication_surprise" in predictors and len(clean) < min_formal_obs:
        return f"communication_surprise_requires_min_{min_formal_obs}_obs:{len(clean)}"
    return None


def leave_one_out_metrics(data: pd.DataFrame, target: str, predictors: list[str]) -> dict[str, float]:
    clean = data[[target] + predictors].dropna().reset_index(drop=True)
    y_true: list[float] = []
    y_pred: list[float] = []
    if len(clean) < 3:
        return {"rmse": np.nan, "mae": np.nan, "directional_accuracy": np.nan}
    for idx in clean.index:
        train = clean.drop(index=idx)
        test = clean.loc[[idx]]
        try:
            x_train = _design_matrix(train, predictors)
            x_test = _design_matrix(test, predictors)
            fit = sm.OLS(train[target].astype(float), x_train).fit()
            pred = float(fit.predict(x_test).iloc[0])
        except Exception:  # noqa: BLE001 - diagnostics should continue even if one fold is singular.
            continue
        y_true.append(float(test[target].iloc[0]))
        y_pred.append(pred)
    if not y_true:
        return {"rmse": np.nan, "mae": np.nan, "directional_accuracy": np.nan}
    return {
        "rmse": _rmse(pd.Series(y_true), pd.Series(y_pred)),
        "mae": _mae(pd.Series(y_true), pd.Series(y_pred)),
        "directional_accuracy": _directional_accuracy(pd.Series(y_true), pd.Series(y_pred)),
    }


def wild_bootstrap_coefficient_stability(
    data: pd.DataFrame,
    target: str,
    predictors: list[str],
    coefficient_name: str,
    draws: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    clean = data[[target] + predictors].dropna().reset_index(drop=True)
    if len(clean) < MIN_EXPLORATORY_OBSERVATIONS:
        return {"coef_std": np.nan, "sign_stability": np.nan}
    x = _design_matrix(clean, predictors)
    y = clean[target].astype(float)
    try:
        fit = sm.OLS(y, x).fit()
    except Exception:  # noqa: BLE001
        return {"coef_std": np.nan, "sign_stability": np.nan}
    if coefficient_name not in x.columns:
        return {"coef_std": np.nan, "sign_stability": np.nan}
    rng = np.random.default_rng(seed)
    position = list(x.columns).index(coefficient_name)
    base_sign = np.sign(fit.params.iloc[position])
    coefs: list[float] = []
    fitted = pd.Series(fit.predict(x), index=y.index)
    residuals = y - fitted
    for _ in range(draws):
        multipliers = rng.choice([-1.0, 1.0], size=len(clean))
        y_star = fitted + residuals * multipliers
        try:
            boot = sm.OLS(y_star, x).fit()
            coefs.append(float(boot.params.iloc[position]))
        except Exception:  # noqa: BLE001
            continue
    if not coefs:
        return {"coef_std": np.nan, "sign_stability": np.nan}
    signs = np.sign(coefs)
    sign_stability = float((signs == base_sign).mean()) if base_sign != 0 else np.nan
    return {"coef_std": float(np.std(coefs, ddof=1)) if len(coefs) > 1 else 0.0, "sign_stability": sign_stability}


def determine_overall_status(data_quality: pd.DataFrame, regression_diagnostics: pd.DataFrame) -> str:
    if not data_quality.empty and (data_quality["status"] == "invalid_for_inference").any():
        return "INVALID_FOR_INFERENCE"
    if regression_diagnostics.empty or not (regression_diagnostics["status"] == "estimated").any():
        return "INVALID_FOR_INFERENCE"
    if not data_quality.empty and (data_quality["status"] == "limited_data").any():
        return "LIMITED_BY_DATA"
    if not data_quality.empty and (data_quality["status"] == "limited_sample").any():
        return "LIMITED_BY_SAMPLE"
    exploratory = regression_diagnostics.get("inference_flag", pd.Series(dtype=str)).astype(str).str.contains("exploratory|not_validated").any()
    if exploratory:
        return "LIMITED_BY_SAMPLE"
    return "APPROVED"


def render_review_report(
    status: str,
    data_quality: pd.DataFrame,
    focus_coverage: pd.DataFrame,
    sensitivity: pd.DataFrame,
    manual_metrics: pd.DataFrame,
    regression_diagnostics: pd.DataFrame,
    sample_path: Path,
    output_dir: Path,
) -> str:
    estimated = int((regression_diagnostics["status"] == "estimated").sum()) if not regression_diagnostics.empty else 0
    blocked = int((regression_diagnostics["status"] == "blocked").sum()) if not regression_diagnostics.empty else 0
    lines = [
        "# Econometric Accuracy Review",
        "",
        f"**Overall status:** `{status}`",
        "",
        "This review is diagnostic. It does not convert the COPOM Tone Index into a causal estimate.",
        "",
        "## Data Quality Flags",
        _markdown_table(data_quality),
        "",
        "## Focus Coverage",
        _markdown_table(focus_coverage, max_rows=30),
        "",
        "## Index Sensitivity",
        _markdown_table(sensitivity),
        "",
        "## Manual Sentence Validation",
        f"Manual audit sample: `{_portable_path(sample_path)}`",
        "",
        _markdown_table(manual_metrics),
        "",
        "## Regression Diagnostics",
        f"Estimated specifications: {estimated}. Blocked specifications: {blocked}.",
        "",
        _regression_summary(regression_diagnostics),
        "",
        "## Interpretation Rules",
        "- Regressions with missing Focus targets are blocked rather than imputed.",
        "- Predictors with no variation, such as a flat `delta_selic`, block the affected specification.",
        "- `communication_surprise` is not treated as validated with fewer than 30 usable observations.",
        "- HC3 errors, leave-one-out metrics and bootstrap sign stability are diagnostic, not proof of causality.",
        "",
        "## Output Tables",
        f"- `{_portable_path(output_dir / 'data_quality_audit.csv')}`",
        f"- `{_portable_path(output_dir / 'focus_coverage_audit.csv')}`",
        f"- `{_portable_path(output_dir / 'focus_missing_events.csv')}`",
        f"- `{_portable_path(output_dir / 'index_sensitivity.csv')}`",
        f"- `{_portable_path(output_dir / 'manual_validation_metrics.csv')}`",
        f"- `{_portable_path(output_dir / 'regression_diagnostics.csv')}`",
    ]
    return "\n".join(lines) + "\n"


def _load_review_tables(database: Path) -> dict[str, pd.DataFrame]:
    table_names = [
        "copom_meetings",
        "copom_documents",
        "copom_sentences",
        "copom_scores",
        "focus_revisions",
    ]
    tables = {table: read_table(database, table) for table in table_names}
    tables["focus_observations"] = read_optional_table(database, "focus_observations", empty_focus_observations())
    return tables


def _alternative_index(
    sentences: pd.DataFrame,
    documents: pd.DataFrame,
    topics: dict[str, Any],
    topic_weight_mode: str = "configured",
    comunicado_weight: float = 0.60,
    ata_weight: float = 0.40,
    trim_abs_quantile: float | None = None,
) -> pd.DataFrame:
    frame = sentences.copy()
    weights = {topic: float(details.get("weight", 1.0)) for topic, details in topics.items()}
    frame["topic_weight"] = 1.0 if topic_weight_mode == "equal" else frame["topic"].map(weights).fillna(1.0)
    frame["weighted_score"] = frame["stance_score"].astype(float) * frame["topic_weight"]
    if trim_abs_quantile is not None and not frame.empty:
        cutoff = frame["weighted_score"].abs().quantile(trim_abs_quantile)
        frame = frame[frame["weighted_score"].abs() <= cutoff]
    doc_scores = (
        frame.groupby(["document_id", "meeting_id", "document_type"], as_index=False)
        .agg(weighted_sum=("weighted_score", "sum"), weight_sum=("topic_weight", "sum"))
    )
    doc_scores["document_tone"] = np.where(doc_scores["weight_sum"] > 0, doc_scores["weighted_sum"] / doc_scores["weight_sum"], np.nan)
    rows = []
    for meeting_id, group in doc_scores.groupby("meeting_id"):
        comunicado = _doc_tone(group, "comunicado")
        ata = _doc_tone(group, "ata")
        if pd.notna(comunicado) and pd.notna(ata):
            tone_raw = comunicado_weight * comunicado + ata_weight * ata
        elif pd.notna(comunicado):
            tone_raw = comunicado
        else:
            tone_raw = ata
        rows.append({"meeting_id": meeting_id, "tone_raw_alt": tone_raw})
    alt = pd.DataFrame(rows)
    alt = _normalize_alt_index(alt)
    return alt


def _normalize_alt_index(alt: pd.DataFrame) -> pd.DataFrame:
    if alt.empty:
        return pd.DataFrame(columns=["meeting_id", "tone_raw_alt", "copom_tone_index_alt", "classification_alt"])
    valid = alt["tone_raw_alt"].dropna()
    std = valid.std(ddof=0) if len(valid) > 1 else 0.0
    if std == 0 or pd.isna(std):
        alt["copom_tone_index_alt"] = 50.0
    else:
        alt["copom_tone_index_alt"] = 50 + 10 * ((alt["tone_raw_alt"] - valid.mean()) / std)
    alt["classification_alt"] = alt["copom_tone_index_alt"].map(classify_index)
    return alt


def _build_manual_sample(sentences: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sentences.empty:
        return pd.DataFrame(
            columns=[
                "sentence_id",
                "meeting_id",
                "document_type",
                "text",
                "model_topic",
                "model_stance",
                "model_score",
                "confidence",
                "human_topic",
                "human_stance",
                "reviewer_notes",
            ]
        )
    ranked = sentences.copy()
    ranked["abs_score"] = ranked["stance_score"].abs()
    material = ranked.sort_values(["abs_score", "confidence"], ascending=[False, False]).head(max(sample_size // 2, 1))
    random_part = ranked.drop(index=material.index, errors="ignore").sample(
        n=min(sample_size - len(material), max(len(ranked) - len(material), 0)),
        random_state=42,
    )
    sample = pd.concat([material, random_part], ignore_index=True).head(sample_size)
    sample = sample.rename(columns={"topic": "model_topic", "stance": "model_stance", "stance_score": "model_score"})
    sample["human_topic"] = ""
    sample["human_stance"] = ""
    sample["reviewer_notes"] = ""
    return sample[
        [
            "sentence_id",
            "meeting_id",
            "document_type",
            "text",
            "model_topic",
            "model_stance",
            "model_score",
            "confidence",
            "human_topic",
            "human_stance",
            "reviewer_notes",
        ]
    ]


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | str]:
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = float((y_true == y_pred).mean())
    f1_values = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
    macro_f1 = float(np.mean(f1_values)) if f1_values else np.nan
    kappa = _cohen_kappa(y_true, y_pred, labels)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "cohen_kappa": kappa, "detail": ""}


def _cohen_kappa(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    observed = float((y_true == y_pred).mean())
    expected = 0.0
    n = len(y_true)
    for label in labels:
        expected += float((y_true == label).sum() / n) * float((y_pred == label).sum() / n)
    if math.isclose(1.0, expected):
        return np.nan
    return (observed - expected) / (1.0 - expected)


def _design_matrix(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    if predictors:
        return sm.add_constant(data[predictors].astype(float), has_constant="add")
    return pd.DataFrame({"const": np.ones(len(data))}, index=data.index)


def _doc_tone(document_scores: pd.DataFrame, document_type: str) -> float:
    subset = document_scores[document_scores["document_type"] == document_type]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["document_tone"])


def _audit_row(layer: str, check: str, value: Any, status: str, detail: str) -> dict[str, Any]:
    return {"layer": layer, "check": check, "value": value, "status": status, "detail": detail}


def _non_null_nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].dropna().nunique())


def _blocked_regression_row(variable: str, reference_year: Any, target: str, model: str, reason: str) -> dict[str, Any]:
    return {
        "variable": variable,
        "reference_year": reference_year,
        "target": target,
        "model": model,
        "predictors": "",
    } | _blocked_regression_fields(reason)


def _blocked_regression_fields(reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "block_reason": reason,
        "inference_flag": "not_estimated",
        "observations": 0,
        "r_squared": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "directional_accuracy": np.nan,
        "spearman_corr": np.nan,
        "loo_rmse": np.nan,
        "loo_mae": np.nan,
        "loo_directional_accuracy": np.nan,
        "coef_name": "",
        "coef": np.nan,
        "hc3_se": np.nan,
        "hc3_t": np.nan,
        "hc3_pvalue": np.nan,
        "max_cooks_d": np.nan,
        "bootstrap_coef_std": np.nan,
        "bootstrap_sign_stability": np.nan,
    }


def _rmse(y: pd.Series, predictions: pd.Series) -> float:
    return float(np.sqrt(np.mean((y.astype(float).to_numpy() - predictions.astype(float).to_numpy()) ** 2)))


def _mae(y: pd.Series, predictions: pd.Series) -> float:
    return float(np.mean(np.abs(y.astype(float).to_numpy() - predictions.astype(float).to_numpy())))


def _directional_accuracy(y: pd.Series, predictions: pd.Series) -> float:
    y_sign = np.sign(y.astype(float).to_numpy())
    pred_sign = np.sign(predictions.astype(float).to_numpy())
    non_zero = y_sign != 0
    if not non_zero.any():
        return np.nan
    return float((y_sign[non_zero] == pred_sign[non_zero]).mean())


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
        return np.nan
    return float(frame["left"].corr(frame["right"], method=method))


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    display = display.fillna("")
    columns = list(display.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(rows)


def _regression_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No regression diagnostics generated._"
    summary = (
        frame.groupby(["status", "block_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status", "count"], ascending=[True, False])
    )
    return _markdown_table(summary, max_rows=30)


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
