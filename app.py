import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Auto Analyzer", layout="wide")
st.title("🔬 자동 인식형 Microwell 분석기")
st.markdown("---")

# 1. 사이드바: 설정
st.sidebar.header("⚙️ 분석 설정")

# 회전 및 전처리 설정
rotation = st.sidebar.slider("📸 사진 회전", -10.0, 10.0, 0.0, step=0.1)
min_dist = st.sidebar.slider("📍 우물 간 최소 거리", 5, 50, 10)
param2 = st.sidebar.slider("🎯 인식 민감도 (낮을수록 많이 찾음)", 1, 30, 12)
min_rad = st.sidebar.slider("📏 최소 반지름", 1, 20, 3)
max_rad = st.sidebar.slider("📏 최대 반지름", 5, 30, 8)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 60)

# 2. 사진 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # [회전 보정]
    h, w = img
