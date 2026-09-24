"""English presentation copy. Analytical identifiers and source documents stay unchanged."""
from .ui_copy import PageCopy

APP_POSITIONING = (
    "COPOM Watch explains how Brazil's central bank communication has changed, "
    "which topics drive that change and which official passages support the analysis."
)
METHODOLOGY_OVERVIEW = (
    "The app reads official Copom statements and minutes, splits the text into sentences, "
    "classifies each sentence by topic and policy direction, and aggregates the results into "
    "indicators that can be compared across meetings. The analysis is descriptive: it organizes "
    "textual evidence, expectations and market reactions. It does not forecast the next Selic "
    "decision, establish causality or replace economic judgment."
)
GLOSSARY = {
    "Raw tone": "Average direction of informative sentences before normalization. Positive values indicate more hawkish communication; negative values indicate a more dovish tone.",
    "Tone index": "Normalized tone measure for comparing meetings over time. Read it alongside the sentences and topics that explain the aggregate number.",
    "Textual surprise": "Difference between the current meeting's tone and the previous meeting's tone. This measures a marginal change in communication, not a market surprise.",
    "Intensity": "Share of a document with relevant directional content. It distinguishes strongly directional communication from a largely neutral text.",
    "Calibration": "Fixed reference window used to put tone on a comparable scale. This prevents the historical index from changing whenever a new meeting is added.",
    "Subindices": "Topic-level breakdowns of tone, including inflation, expectations, activity, fiscal conditions and the external environment.",
    "Text change": "Comparison with the equivalent document from the previous meeting. It identifies added, removed, maintained and rewritten passages.",
    "Key sentences": "Official sentences that help explain the index. They connect the aggregate indicator to its underlying textual evidence.",
    "Focus": "Market expectations collected by Brazil's central bank. Here they measure observable revisions before and after Copom events.",
    "Market reaction": "Asset movements within event windows. These associations should not be interpreted as evidence of causality.",
    "Audit": "Overview of validation, human labels, classification metrics and known methodological limitations.",
    "Evidence search": "Local search of official Copom documents. Answers use retrieved passages and cite the meeting, document and sentence.",
    "Analysis period": "The recent operational window focuses on the period used most often in day-to-day analysis. Full history includes all available meetings.",
}
PAGE_COPY = {
    "latest": PageCopy(
        "Latest meeting",
        "A summary of the selected meeting: tone, change from the previous meeting, dominant topics and supporting sentences.",
        ("Start with the tone index and textual surprise to distinguish level from change.", "Use subindices to identify the topics behind the aggregate reading.", "Read the key sentences before drawing strong economic conclusions."),
        ("The analysis is textual and descriptive; it does not forecast the next Selic decision.", "Aggregate numbers can hide ambiguous sentences, so textual evidence is essential."),
        ("Raw tone", "Tone index", "Textual surprise", "Intensity", "Calibration", "Subindices", "Key sentences"),
    ),
    "timeline": PageCopy(
        "Tone over time", "How the tone index has evolved across Copom meetings.",
        ("Compare persistent movements rather than isolated observations.", "Use the table to locate specific meetings and relevant changes in wording.", "Interpret the index in the context of the monetary cycle and the economy at the time."),
        ("The index measures official communication, not the decision itself.", "Changes in the central bank's writing style can affect long-term comparisons."),
        ("Tone index", "Textual surprise", "Intensity", "Calibration"),
    ),
    "decomposition": PageCopy(
        "Topic breakdown", "Which topics made communication more hawkish or more dovish?",
        ("Compare inflation, expectations, activity, fiscal and external subindices.", "A stable headline index can hide important changes in composition.", "Use sentence counts to assess whether a topic reading is broad or concentrated."),
        ("A sentence can cover several topics; this view summarizes its primary classification.", "Read subindices based on few sentences with caution."),
        ("Subindices", "Raw tone", "Tone index"),
    ),
    "text_changes": PageCopy(
        "Text changes", "What changed compared with the equivalent document from the previous meeting?",
        ("Added passages identify new points of attention.", "Removed passages identify signals that no longer appear.", "Changes in tone identify similar sentences with different directional intensity."),
        ("A change in wording does not automatically imply a change in policy.", "Rewriting can reflect editorial style as well as economic developments."),
        ("Text change", "Textual surprise"),
    ),
    "evidence": PageCopy(
        "Key sentences", "Official sentences that support the index reading.",
        ("Filter by document, topic and stance to audit the classification.", "Use citations to return to the original meeting document.", "Review high-tone, high-confidence sentences while checking ambiguous passages in context."),
        ("Selected evidence does not replace reading the full statement or minutes.", "Ambiguous sentences may require human judgment."),
        ("Key sentences", "Tone index", "Subindices"),
    ),
    "focus": PageCopy(
        "Focus expectations", "Revisions to market expectations before and after Copom events.",
        ("Compare pre-event values with the first and second post-event observations.", "Missing observations are not imputed; a zero change differs from missing data.", "Compare IPCA, Selic, GDP and exchange-rate expectations at the same horizon."),
        ("Focus measures reported expectations, not real-time market prices.", "Revisions may reflect other shocks within the same event window."),
        ("Focus",),
    ),
    "market": PageCopy(
        "Market reaction", "Asset movements around the statement and minutes release.",
        ("Use event windows to observe temporal associations with communication.", "Check each window's status before interpreting the reaction.", "Compare assets and tenors where data coverage is sufficient."),
        ("These associations are not causal; other events may have moved prices.", "The market layer is optional and depends on available public or imported data."),
        ("Market reaction",),
    ),
    "audit": PageCopy(
        "Audit", "Methodological quality and limitations of the textual classification.",
        ("Use accuracy and F1 scores to assess classification performance.", "Review human labels where available.", "Treat warnings as methodological limitations, not automatic application failures."),
        ("Formal auditing is limited when there are too few human labels.", "Text-classification accuracy does not establish macroeconomic causality."),
        ("Audit",),
    ),
    "ask": PageCopy(
        "Questions with official evidence", "Search historical Copom documents using answers grounded in retrieved citations.",
        ("Ask about language, topics and historical cases.", "Each answer cites a meeting, document and sentence.", "Use the examples to start with questions within the app's scope."),
        ("Search does not forecast the next Selic decision or provide opinions outside the retrieved evidence.", "When evidence is insufficient, the app reports that rather than inventing an answer."),
        ("Evidence search", "Key sentences"),
    ),
    "reports": PageCopy(
        "Reports", "Reports and manifests produced by the analytical pipeline.",
        ("Use reports for auditing, acceptance and methodological documentation.", "Manifests record versions, hashes and the status of processed data."),
        ("Reports reflect the pipeline at the time they were generated.", "Missing files indicate that the corresponding stage has not run in this environment."),
        ("Audit",),
    ),
    "legacy": PageCopy(
        "Classic view", "The original project view, retained for compatibility and comparison.",
        ("Use this view to compare with the earlier methodology.", "Use the main interface for current analysis."),
        ("The classic view has fewer explanations, less decomposition and narrower methodological coverage.",),
        ("Tone index",),
    ),
}
ASK_EXAMPLES = (
    "When did Copom discuss unanchored expectations?",
    "Which sentences signaled more hawkish communication?",
    "Show references to fiscal risk in recent meetings.",
    "Compare language about economic activity during rate-cutting cycles.",
)
