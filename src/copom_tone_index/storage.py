from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


TABLE_EXPORTS = [
    "copom_meetings",
    "copom_documents",
    "copom_sentences",
    "copom_document_scores",
    "copom_topic_scores",
    "copom_scores",
    "focus_observations",
    "focus_revisions",
    "focus_vintages",
    "focus_event_features",
    "evidence_sentences",
    "event_calendar",
    "v2_runs",
    "v2_meetings",
    "v2_documents",
    "v2_sentences",
    "v2_sentence_scores",
    "v2_document_scores",
    "v2_meeting_scores",
    "v2_subindices",
    "v2_calibration",
    "v2_taxonomy_versions",
    "v2_lexicon_versions",
    "v2_redline",
    "v2_evidence",
    "v2_labels",
    "v2_model_predictions",
    "v2_supervised_predictions",
    "v2_supervised_model_audit",
    "v2_model_audit",
    "v2_model_audit_details",
    "market_observations",
    "market_event_windows",
    "decision_expectations",
    "public_market_source_audit",
    "decision_expectation_source_audit",
    "public_market_coverage",
    "v21_event_panel",
    "semantic_chunks",
]


def write_tables(database: Path, tables: dict[str, pd.DataFrame]) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as con:
        for name, frame in tables.items():
            con.register("frame_view", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame_view")
            con.unregister("frame_view")


def read_table(database: Path, table: str) -> pd.DataFrame:
    with duckdb.connect(str(database), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table}").df()


def export_tables(database: Path, output_dir: Path, tables: list[str] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = tables or TABLE_EXPORTS
    with duckdb.connect(str(database), read_only=True) as con:
        for table in tables:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
            if not exists:
                continue
            frame = con.execute(f"SELECT * FROM {table}").df()
            frame.to_csv(output_dir / f"{table}.csv", index=False)
            frame.to_parquet(output_dir / f"{table}.parquet", index=False)
