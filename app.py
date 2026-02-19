import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 Microwell Grid & GMO Analyzer")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
st.sidebar.info("눈금자와 중앙 가이드라인을 참고하여 좌표를 입력하세요.")
col1, col2 = st.sidebar.columns(2)
tl_x = col1.number_input("좌상 X", 0, 5000, 150)
tl_y = col2.number_input("좌상 Y", 0, 5000, 200)
tr_x = col1.number_input("우상 X", 0, 5000, 2300)
tr_y = col2.number_input("우상 Y", 0, 5000, 200)
bl_x = col1.number_input("좌하 X", 0, 5000, 150)
bl_y = col2.number_input("좌하 Y", 0, 5000, 2300)
br_x = col1.number_input("우하 X", 0, 5000, 2300)
br_y = col2.number_input("우하 Y", 0, 5000, 2300)

st.sidebar.header("🔢 3단계: Well 개수 설정")
auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
if not auto_mode:
    manual_cols = st.sidebar.number_input("가로 Well 개수", 1, 100, 23)
    manual_rows = st.sidebar.number_input("세로 Well 개수", 1, 100, 24)

st.sidebar.header("🧪 4단계: 판정 및 크기")
radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0, step=0.1)

st.sidebar.header("🧬 5단계: GMO 판정 설정")
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# 유틸리티 함수
def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    ruler_img = img.copy()
    guide_color = (0, 0, 255) # 빨간색 가이드라인
    cv2.line(ruler_img, (0, h//2), (w, h//2), guide_color, 1)
    cv2.line(ruler_img, (w//2, 0), (w//2, h), guide_color, 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w, 100):
        cv2.line(ruler_img, (x, 0), (x, 30), (255, 255, 255), 2)
        cv2.putText(ruler_img, str(x), (x+5, 25), font, 0.5, (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(ruler_img, (0, y), (30, y), (255, 255, 255), 2)
