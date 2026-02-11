import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 점선 원 그리기 함수 (OpenCV에는 기본 점선 함수가 없어서 직접 구현합니다)
def draw_dotted_circle(img, center, radius, color, thickness=2, gap=8):
    circumference = 2 * np.pi * radius
    num_dots = int(circumference / gap)
    if num_dots < 4: num_dots = 4 # 최소 점 개수 보장
    for i in range(num_dots):
        angle_start = (2 * np.pi / num_dots) * i
        angle_end = angle_start + (np.pi / num_dots) # 점선 하나의 길이
        
        # 타원 호를 그리는 방식으로 점선을 표현합니다.
        cv2.ellipse(img, center, (radius, radius), 0, np.degrees(angle_start), np.degrees(angle_end), color, thickness)

st.set_page_config(page_title="Microwell Analyzer", layout="wide")
st.title("🔬 정밀 Microwell 분석기 (파란색 점선 표시)")

# 1. 설정 사이드바
st.sidebar.header("🔍 분석 설정 (세밀 조정)")
st.sidebar.info("먼저 '최소/최대 반지름'을 사진 속 우물 크기에 맞추고, '원 검출 임계값'으로 개수를 조절하세요.")

threshold = st.sidebar.slider("형광 감도 (임계값)", 0, 255, 150, help="이 값보다 밝으면 Positive로 카운트합니다.")
min_dist = st.sidebar.slider("우물 간 최소 거리", 5, 50, 12, help="우물 중심 사이의 최소 픽셀 거리입니다.")
min_rad = st.sidebar.number_input("최소 반지름 (픽셀)", 1, 50, 4)
max_rad = st.sidebar
