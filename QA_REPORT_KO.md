# WellScope LAMP v1.1.0 — 실제 점검 범위

## 실행 환경
Python 3.13.5; numpy 2.3.5; scipy 1.17.0; pandas 2.2.3; Pillow 12.3.0; opencv-python-headless 4.13.0.92; matplotlib 3.10.8.

## 수행한 점검
- Python 모듈 문법 컴파일.
- 자동 시험 62개 통과: 기존 core/export 35개, 새 batch/model/export 27개.
- 공통 threshold 적용, 원본 배열 불변, 양성/음성 합계, 자료형 유지, 잘못된 입력 거부.
- 훈련 이미지 이름 변경 시에도 hash로 학습 이미지 판별, 파일명을 예측값으로 사용하지 않음.
- 모델 checksum/engine 버전, 광학/측정 설정, bit depth, 허용 이미지 크기, pitch 호환성 점검.
- 모델 클래스 중첩, 비단조/포화 정량 보류, 변경 threshold/프로필 거부.
- 실제 사용자가 제공한 이미지 10장으로 계산 및 전체 결과 묶음 생성.
- UI Python 분기 6개를 테스트 대역으로 점검: 시작, 단일 자동, 모델, 논문용 화면, 표준 일괄, 모델 미업로드.
- Chromium으로 다크 배경의 metric CSS 컴포넌트와 독립 HTML 보고서 표시 확인. Metric value rgb(25,51,72); card background rgb(247,249,251).
- A4 세로 SVG/PDF/PPTX 생성; PDF page size 210 x 297 mm 확인; PyMuPDF와 LibreOffice 렌더링으로 배치 확인.

## 수행하지 못한 점검
- Streamlit 패키지를 설치할 네트워크 연결이 없어 실제 Streamlit AppTest 1개를 건너뜀.
- UI 테스트 대역과 CSS 컴포넌트 검사는 실제 Streamlit 브라우저 및 Cloud 배포 검사가 아님.
- 사용자의 GitHub 저장소와 Community Cloud는 직접 수정/배포하지 않음.
- 실제 well 정답 위치/수동 계수와의 정확도 비교, 독립 시료의 assay 성능 검증은 수행하지 않음.

## 배포 후 확인할 항목
v1.1.0 제목, 업로드→분석 실행, 상단 숫자 대비, 표준 일괄 분석, 모델 저장/재업로드, 단일 이미지 선별, 입력 변경 후 이전 결과 숨김, CSV/SVG/HTML/ZIP 다운로드를 확인하세요.

## 해석 제한
소프트웨어 시험 통과는 실험 검증 통과가 아닙니다. 표준을 학습/출력에 함께 쓰는 것은 개발 예시입니다. well 수를 독립 반응 시료 수로 세지 않고, 확보하지 않은 오차 막대·정확도·신뢰구간을 생성하지 않습니다.

기존 방법 문장 UI와 자동 methods.txt는 요청에 따라 제거했습니다. 함수 method_text는 호환성을 위해 코드에만 남아 있으며 앱 화면에 표시되지 않습니다. 감사용 측정 설정/QC는 JSON으로 보존됩니다.
