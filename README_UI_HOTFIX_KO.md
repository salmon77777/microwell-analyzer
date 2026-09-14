# WellScope LAMP v1.1.0 UI screening hotfix

이 패치는 분석 엔진(wellscope_core.py, wellscope_batch.py)을 변경하지 않습니다.
따라서 기존 screening_model_DEVELOPMENT.json과 모델 hash 호환성이 유지됩니다.

## 변경
1. 저장된 연구 선별 모델이 로드되면 Tw / Ts / 3% study boundary를 사이드바에 표시합니다.
2. 분석 후 '격자 검토'와 '촬영·반응·DNA-input/희석 조건 비교 가능' 확인을 모두 해야 screening classification이 적용된다는 점을 크게 표시합니다.
3. screening 결과가 적용되면 4번째 metric은 Estimated mixture 대신 Sample classification을 보여줍니다.
4. SVG/HTML 보고서의 우측 상단 badge가 FIXED THRESHOLD가 아니라 LOCKED SCREENING MODEL로 바뀝니다.
5. 보고서의 4번째 metric도 <3% class 또는 >=3% class로 표시됩니다.
6. 정확한 GMO 혼합비 정량이 보류된 경우, screening classification과 numerical quantification이 다른 출력임을 명확히 표시합니다.

## GitHub 교체 파일
- app.py
- wellscope_report.py

나머지 core/batch 파일은 이전 동기화 핫픽스 그대로 유지하세요.

## 사용
1. 3%.png 업로드
2. '저장된 연구 선별 모델 · 권장'
3. screening_model_DEVELOPMENT.json 업로드
4. 분석 실행
5. '격자 전체와 확대 영역...' 확인
6. '표준과 현재 시료의 촬영·반응·DNA 투입/희석 조건...' 확인
7. 3% training image에서는 현재 개발 모델 기준 'GMO screen positive / >=3% study class'가 표시되어야 합니다.
