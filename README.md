# WellScope LAMP

Microwell fluorescence analysis — research-use software, v1.0.0.

**한국어 설치·사용 안내: [README_KO.md](README_KO.md)**  
브라우저에서 읽는 안내서는 `START_HERE_KO.html`을 내려받아 여세요.

기존 Streamlit 앱을 교체할 때는 `app.py`, `wellscope_core.py`, `wellscope_report.py`, `requirements.txt`를 같은 폴더에 올리고, `.streamlit/config.toml`도 적용하세요. ZIP 파일 자체를 업로드하는 것이 아닙니다.

No API key or AI service is used. Measurements remain in native pixels; the app separates exploratory thresholding, locked profiles and optional user-supplied calibration. Positive-well fraction is not GMO content. No non-GMO certification is generated.

See [QA_REPORT_KO.md](QA_REPORT_KO.md) for what was and was not tested. Do not commit unpublished experiment images, analysis ZIPs, secrets or calibration data to a public repository.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
