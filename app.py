import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Grid Analyzer", layout="wide")
st.title("🔬 격자 맞춤형 Microwell 분석기")
st.markdown("---")
st.info("💡 **결과 보는 법**: 파란색 원 = Positive(형광), 빨간색 원 = Negative(비형광)")

# 1. 사이드바: 격자 배치 설정
st.sidebar.header("📏 격자 설정 (Grid Setup)")
col_count = st.sidebar.number_input("가로 우물 개수", 1, 100, 23)
row_count = st.sidebar.number_input("세로 우물 개수", 1, 100, 24)

st.sidebar.markdown("---")
st.sidebar.write("📍 위치 및 간격 조절")

# 좌표 직접 입력
start_x = st.sidebar.number_input("첫 번째 우물 X 좌표", 0, 2000, 5)
start_y = st.sidebar.number_input("첫 번째 우물 Y 좌표", 0, 2000, 7)

# 간격 및 반지름 슬라이더 (소수점 지원)
gap_x = st.sidebar.slider("가로 간격 (Spacing X)", 1.0, 100.0, 14.2, step=0.1)
gap_y = st.sidebar.slider("세로 간격 (Spacing Y)", 1.0, 100.0, 9.8, step=0.1)
radius = st.sidebar.slider("우물 반지름", 1, 50, 5)

st.sidebar.markdown("---")
# 임계값 설정
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 50, help="이 값보다 밝으면 파란색, 어두우면 빨간색으로 표시됩니다.")

# 2. 사진 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    display_img = img_rgb.copy()
    
    pos_count = 0
    total_wells = col_count * row_count
    
    # 격자 생성 및 분석
    for r in range(row_count):
        for c in range(col_count):
            # 간격 계산 (정수 좌표로 변환)
            center_x = int(start_x + (
