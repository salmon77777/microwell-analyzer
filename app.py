import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Analyzer", layout="wide")
st.title("🔬 정밀 Microwell 분석기")

# 1. 설정 사이드바
st.sidebar.header("🔍 분석 설정 (세밀 조정)")
# 실제 사진 속 우물은 매우 작으므로 범위를 작게 설정합니다.
threshold = st.sidebar.slider("형광 감도 (임계값)", 0, 255, 180)
min_dist = st.sidebar.slider("우물 간 최소 거리", 5, 50, 10)
min_rad = st.sidebar.number_input("최소 반지름", 1, 50, 3)
max_rad = st.sidebar.number_input("최대 반지름", 1, 100, 12)

uploaded_file = st.file_uploader("Microwell 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    # 이미지 전처리 (작은 원 인식률 향상)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # 노이즈 제거
    
    # 원형 감지 알고리즘 조절
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, 
        minDist=min_dist, 
        param1=50, param2=15, # param2를 낮추면 더 작은 원도 잘 찾습니다.
        minRadius=min_rad, maxRadius=max_rad
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        pos_count = 0
        total_count = len(circles[0])

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # 우물 영역 마스크 생성
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            # 녹색(Green) 채널의 평균값 계산
            mean_val = cv2.mean(img_rgb, mask=mask)
            green_avg = mean_val[1]

            if green_avg > threshold:
                color = (0, 255, 0) # Positive
                pos_count += 1
            else:
                color = (255, 0, 0) # Negative
            
            cv2.circle(display_img, center, radius, color, 1)

        st.image(display_img, caption='분석 결과 (확대해서 원이 정확한지 확인하세요)', use_container_width=True)
        
        percent = (pos_count / total_count) * 100 if total_count > 0 else 0
        
        st.subheader("📊 분석 리포트")
        c1, c2, c3 = st.columns(3)
        c1.metric("검출된 전체 우물", f"{total_count}개")
        c2.metric("Positive (형광)", f"{pos_count}개")
        c3.metric("비율", f"{percent:.1f}%")

        # 결과 저장
        res_img = Image.fromarray(display_img)
        buffered = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("결과 이미지 저장", data=buffered, file_name="result.png", mime="image/png")
    else:
        st.error("우물을 찾지 못했습니다. '최소 반지름'을 줄이거나 '감도'를 조절해보세요.")
