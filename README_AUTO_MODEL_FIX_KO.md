# WellScope LAMP v1.1.0 — Auto-model + SVG preview fix

## 이번 패치의 목적
1. `논문용 결과 보기`에서 깨진 이미지 아이콘과 `0`이 나타나는 SVG 미리보기 문제를 수정했습니다.
2. screening model을 매번 업로드하지 않아도 되도록 기본 모델을 앱에 내장했습니다.
3. 필요할 때는 별도의 `screening_model.json`을 직접 업로드할 수도 있습니다.
4. 업로드한 모델은 같은 Streamlit 세션 동안 session_state에도 유지됩니다.

## GitHub에서 교체/추가할 파일
- `app.py` : 교체
- `wellscope_report.py` : 교체(동기화용)
- `models/screening_model_DEFAULT.json` : 새로 추가

기존 `wellscope_core.py`, `wellscope_batch.py`는 이전 동기화 핫픽스 버전을 그대로 유지하세요.
분석 엔진은 변경하지 않았으므로 현재 development model과 hash 호환성이 유지됩니다.

## 사용
`분석 방식`에서 `저장된 연구 선별 모델 · 권장`을 선택하면
기본적으로 `내장 연구 선별 모델 · 권장`이 선택됩니다.
따라서 JSON 파일을 매번 업로드할 필요가 없습니다.

다른 모델을 시험할 때만 `다른 screening_model.json 불러오기`를 선택하세요.

## 1% / 3% 기대 결과 (현재 development model)
- 1% 이미지: q≈0.643%, Ts≈3.054% -> `<3% class`
- 3% 이미지: q≈4.932%, Ts≈3.054% -> `>=3% class / GMO screen positive`

정확한 GMO 혼합비 수치 정량은 현재 `Not quantified`로 남는 것이 정상입니다.

## 중요
내장된 JSON은 development model입니다.
코드와 분석법을 최종 확정한 뒤 표준 시리즈를 다시 분석하여
`screening_model_DEFAULT.json`을 최종 locked model로 교체하세요.
