"""
End-to-end take-home report for one participant folder.

    video/CSV --(DataProcessor)--> 3 key frames --(SwingScorer)--> score
             --(build_report_context + FIX_LIBRARY)--> draft --(Claude polish)--> report

Usage:
    python generate_report.py <folder>            # template only (no model call)
    python generate_report.py <folder> --polish   # + Claude personalisation pass
"""
import sys
from process_swing import DataProcessor, SwingScorer
from report_content import build_report_context, render_template_report, polish_report


def report_for_folder(folder, player="Player", rank=None, total=None, polish=False):
    dp = DataProcessor(folder)
    dp.load_data()
    dp.preprocess_data()
    scorer = SwingScorer(dp)

    ctx = build_report_context(scorer, player=player, rank=rank, total=total)
    draft = render_template_report(ctx)
    if polish:
        text, used_llm = polish_report(ctx, draft)
        return text, used_llm, ctx
    return draft, False, ctx


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else "test_classes"
    polish = "--polish" in sys.argv

    text, used_llm, ctx = report_for_folder(
        folder, player="Priya", rank=7, total=42, polish=polish
    )
    banner = "LLM-POLISHED" if used_llm else ("TEMPLATE (polish requested but Claude "
                                              "unavailable - fell back)" if polish else "TEMPLATE")
    print(f"===== {banner} =====\n")
    print(text)
