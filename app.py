import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 페이지 기본 설정
st.set_page_config(page_title="Microwell Analyzer", layout="wide")

# 점선 원 그리기 함수 (파란색, 얇게 수정됨)
def draw_dotted_circle(img, center, radius, color, thickness=1, gap=8):
    circumference = 2 * np.pi * radius
    num_dots = int(circumference / gap)
    for i in range(num_dots):
        start_angle = (360 / num_dots) * i
        end_angle = start_angle + (180 / num_dots)
        # OpenCV는 BGR 색상 체계를 사용합니다. 파란색은 (255, 0, 0)입니다.
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

st.title("🔬 정밀 Microwell 분석기")
st.markdown("---")

# 2. 업로드 버튼
uploaded_file = st.file_uploader("1. 분석할 Microwell 사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

# 3. 사이드바 설정
st.sidebar.header("⚙️ 분석 세부 설정")
st.sidebar.write("파란색 점선이 우물 테두리에 맞게 조절하세요.")

# 파라미터들
min_rad = st.sidebar.number_input("우물 최소 반지름 (픽셀)", 1, 50, 5)
max_rad = st.sidebar.number_input("우물 최대 반지름 (픽셀)", 1, 100, 15)
param2_val = st.sidebar.slider("원 인식 감도 (낮을수록 많이 찾음)", 5, 50, 20)
min_dist = st.sidebar.slider("우물 간 최소 거리", 5, 100, 15)
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 130)

if uploaded_file is not None:
    # 이미지 처리 시작
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 원형 감지
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=min_dist, 
        param1=50, param2=param2_val, 
        minRadius=min_rad, maxRadius=max_rad
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        pos_count = 0
        total_count = len(circles[0])

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # 1. 모든 인식된 원을 [얇은 파란색 점선]으로 표시
            # 색상: (255, 0, 0) - BGR 기준 순수 파란색
            # 두께: 1 (최소 두께)
            # 간격: 8 (점선 간격을 넓혀 더 얇아 보이게 함)
            draw_dotted_circle(display_img, center, radius, (255, 0, 0), thickness=1, gap=8)

            # 2. 녹색(Green) 채널 분석
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_val = cv2.mean(img_rgb, mask=mask)
            green_val = mean_val[1]

            # 3. 임계값 이상이면 내부에 작은 초록 점 표시
            if green_val > threshold:
                pos_count += 1
                cv2.circle(display_img, center, 2, (0, 255, 0), -1)

        # 결과 이미지 출력
        st.image(display_img, caption='파란 점선: 인식된 구역 / 중앙 초록점: Positive 판정', use_container
