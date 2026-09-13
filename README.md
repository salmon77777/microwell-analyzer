# WellScope LAMP v1.1.0

Microwell fluorescence measurement, shared-reference thresholding, and exploratory study screening.

## GitHub / Streamlit update

Keep the existing repository and `.devcontainer`. Upload the CONTENTS of this package to the directory containing the existing `app.py`. Replace same-named files in one commit; do not delete the existing app first. Do not upload the outer ZIP as an executable app.

Runtime files (keep together):

```text
app.py
wellscope_core.py
wellscope_report.py
wellscope_batch.py
wellscope_standards_ui.py
wellscope_figure6.py
requirements.txt
.streamlit/config.toml
```

The start path remains `app.py`. The added modules are required even for the home page. No API key is needed. Images uploaded in Community Cloud are sent to that server; this is not a local-only service.

## Quick workflow

1. Start with `표준 이미지 일괄 분석 / Standard-image series`.
2. Upload 3–30 real standard images including a 0% reference and standards below, at, and above the study boundary.
3. Check the proposed labels in the editable table. Percentages in filenames are only label suggestions, never prediction inputs.
4. Run the series. Default: use the median of the 99th-percentile corrected signals from the zero-reference images as ONE well cutoff for every image. This is an exploratory rule, not a validated assay false-positive rate.
5. Review all grids and matching acquisition, assay and input conditions. Keep the default sample feature unless feature selection is explicitly part of model development.
6. Save `screening_model.json`, then use `단일 시료 분석 / Single-image analysis` and `저장된 연구 선별 모델 / Saved study screening model`.
7. Upload the model and an image. Run analysis and confirm comparable conditions before viewing the study screen.
8. Use `분석 결과 내보내기 / Export analysis results` to save CSV, SVG, HTML and ZIP files.

## What changed

- Explicit foreground and background colors for metric cards, including nested metric values: fixes white text on pale cards under a dark user theme.
- A shared well cutoff replaces per-image Otsu classification when a study model is applied.
- Image-specific Otsu output is still available, clearly marked exploratory.
- Batch measurements include positive fraction, mean/median corrected signal, P90, top-decile mean and saturation summaries.
- The sample screening cutoff is separate from the well cutoff and is fitted only when the selected feature separates the training classes.
- The JSON model locks photometry, cutoff, compatibility checks and training-image fingerprints. Renaming a training image does not turn it into validation.
- Full numerical mixture estimation is withheld for nonmonotonic/flat standards or saturation. No forced monotonic rearrangement, extrapolation or invented confidence interval.
- A4 portrait SVG export uses actual uploaded images and measurements, not generated microscopy.
- The manuscript-method paragraph was removed from the UI/report. Auditable settings and QC remain in JSON.

## Scientific scope

A threshold-positive WELL fraction is not GMO mixture percentage. A study-screen-negative sample is below a research boundary, not proof of GMO absence or non-GMO certification. Assigned mixture labels are user-supplied; the software does not know whether they represent mass, copy ratio or another preparation basis.

The built-in screening model is always DEVELOPMENT ONLY until independently evaluated. Well counts are not independent biological/experimental replicates. Software QC does not establish well filling, amplification specificity or assay validity. Automatic grids in dim or damaged images require human review. No automatic alignment between separate images is claimed.

Measurement is native disc-mean minus annulus-median signal. Pixels are not rescaled for measurement. Exposure, gain, DNA input/dilution, amplification conditions and pixel registration must be controlled externally. Saturated pixels are reported, not silently removed.

Old v1.0 profiles/models must be rebuilt for v1.1. Image sizes differing by up to two pixels from a training image are allowed by the screening-model compatibility check only; this does not authorize arbitrary resampling. Geometry templates still require exact registration.

## Local run / tests

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For tests, additionally install pytest and run `python -m pytest -q`.

See `QA_REPORT_KO.md` for the actual testing scope. Do not interpret a software test pass as analytical validation. This package contains no private experiment images or fitted study model; keep study outputs outside public GitHub.
