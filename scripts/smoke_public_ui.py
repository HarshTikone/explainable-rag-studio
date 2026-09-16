"""Run all four public pages through Streamlit's headless AppTest harness."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from backend.config import SETTINGS


def _require_clean(page: AppTest, label: str) -> None:
    if list(page.exception):
        raise AssertionError(f"{label} raised: {[item.message for item in page.exception]}")


def main() -> None:
    if not SETTINGS.low_memory_demo or SETTINGS.security_mode != "demo":
        raise SystemExit("LOW_MEMORY_DEMO=true and SECURITY_MODE=demo are required.")
    app = AppTest.from_file(str(ROOT / "app" / "Home.py")).run(timeout=30)
    _require_clean(app, "Home")
    if not any(item.value == "Public read-only demo" for item in app.sidebar.caption):
        raise AssertionError("Home did not identify the public read-only profile.")

    ask_query_passed = False
    for path, label in (
        ("pages/1_What_is_RAG.py", "What is RAG"),
        ("pages/3_Ask_and_Explain.py", "Ask & Explain"),
        ("pages/9_Results.py", "Results"),
    ):
        app.switch_page(path).run(timeout=30)
        _require_clean(app, label)
        if label == "Ask & Explain":
            if not any(item.label == "Your question" for item in app.text_area):
                raise AssertionError("Ask & Explain did not render its question input.")
            question = next(item for item in app.text_area if item.label == "Your question")
            run_button = next(item for item in app.button if item.label == "Run grounded query")
            question.set_value("How long are Aegis audit events retained?")
            run_button.click()
            app.run(timeout=30)
            _require_clean(app, "Ask & Explain query")
            if not any("400 days" in str(item.value) for item in app.markdown):
                raise AssertionError("The public UI sample did not render the expected 400-day answer.")
            ask_query_passed = True
    if not ask_query_passed:
        raise AssertionError("Ask & Explain query was not exercised.")

    router_source = (ROOT / "app" / "Home.py").read_text(encoding="utf-8")
    public_registry = router_source.split("if SETTINGS.low_memory_demo:", 1)[0]
    for restricted in (
        "pages/2_Ingest_and_Index.py", "pages/5_Evaluation.py",
        "pages/6_Latency_Dashboard.py", "pages/8_Security_Center.py",
    ):
        if restricted in public_registry:
            raise AssertionError(f"Restricted page was registered in public navigation: {restricted}")
    print("All four public Streamlit pages passed AppTest.")


if __name__ == "__main__":
    main()
