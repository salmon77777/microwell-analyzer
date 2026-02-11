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
max_rad = st.sidebar.number_input("최대 반지름 (픽셀)", 1, 100, 10)
# 새로 추가된 중요한 설정입니다!
param2_val = st.sidebar.slider("원 검출 임계값 (높을수록 엄격)", 10, 100, 25, help="이 값을 높이면 더 완벽한 원 모양만 찾습니다. 너무 많이 잡히면 값을 올려보세요.")

uploaded_file = st.file_uploader("Microwell 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    # 이미지 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 노이즈를 줄여서 원 인식률을 높입니다.
    gray = cv2.GaussianBlur(gray, (5, 5), 1) 
    
    # 원형 감지 알고리즘
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.1, 
        minDist=min_dist, 
        param1=60, # 엣지 검출 임계값
        param2=param2_val, # 사용자가 설정한 원 검출 임계값 적용
        minRadius=min_rad, maxRadius=max_rad
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        pos_count = 0
        total_count = len(circles[0])

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # 1. 인식된 모든 우물을 '파란색 점선'으로 표시 (OpenCV는 BGR이므로 파란색=(255,0,0))
            draw_dotted_circle(display_img, center, radius, (255, 0, 0), thickness=2)

            # 2. 형광 판정 (Positive 카운팅)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_val = cv2.mean(img_rgb, mask=mask)
            green_avg = mean_val[1]

            if green_avg > threshold:
                pos_count += 1
                # (선택사항) Positive인 경우 원 중심에 작은 초록색 점을 찍어 구분을 도울 수 있습니다.
                # cv2.circle(display_img, center, 1, (0, 255, 0), -1) 

        st.image(display_img, caption='분석 결과 (파란색 점선: 인식된 우물)', use_container_width=True)
        
        percent = (pos_count / total_count) * 100 if total_count > 0 else 0
        
        st.subheader("📊 분석 리포트")
        c1, c2, c3 = st.columns(3)
        c1.metric("인식된 우물 (파란 점선)", f"{total_count}개")
        c2.metric("Positive (임계값 이상)", f"{pos_count}개")
        c3.metric("비율", f"{percent:.1f}%")

        # 결과 저장
        buffered = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("결과 이미지 저장", data=buffered, file_name="result_dotted.png", mime="image/png")
    else:
        st.error("우물을 찾지 못했습니다. '원 검출 임계값'을 낮추거나 '반지름 범위'를 조절해보세요.")
