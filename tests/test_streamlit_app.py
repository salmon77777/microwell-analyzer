"""Optional real Streamlit smoke test. Skips cleanly when Streamlit is absent."""
from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

def test_home_and_synthetic_flow():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
    assert not app.exception
    app.checkbox(key='demo_mode').set_value(True).run()
    assert not app.exception
    app.button(key='run_analysis').click().run()
    assert not app.exception
    app.checkbox(key='paper_view').set_value(True).run()
    assert not app.exception
