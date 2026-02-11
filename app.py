import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Analyzer", layout="wide")
st.title("🔬 형광 Microwell 분석기")

# 1. 설정 사이드바 (실시간 조절 가능)
st.sidebar.header("🔍 분석 설정")
threshold = st.sidebar.slider("형광 감도 (임계값)", 0, 255, 120, help="이 값보다 녹색이 밝으면 Positive로 인식합니다.")
min_dist = st.sidebar.slider("우물 간 최소 거리", 10, 100, 30, help="우물 사이의 간격을 조절하세요.")
circle_size = st.sidebar.slider("우물 크기 범위", 5, 100, (15, 30), help="찾고자 하는 우물의 최소/최대 반지름입니다.")

uploaded_file = st.file_uploader("Microwell 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    # 원형 우물 감지 로직
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=min_dist, 
        param1=50, param2=30, 
        minRadius=circle_size[0], maxRadius=circle_size[1]
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        pos_count = 0
        total_count = len(circles[0])

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # 각 우물 영역의 평균 녹색(Green) 값 계산
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_val = cv2.mean(img_rgb, mask=mask)
            green_avg = mean_val[1]  # RGB 중 G 인덱스는 1

            # 임계값에 따른 판정 및 표시
            if green_avg > threshold:
                color = (0, 255, 0) # Positive: 초록색 테두리
                pos_count += 1
            else:
                color = (255, 0, 0) # Negative: 빨간색 테두리
            
            cv2.circle(display_img, center, radius, color, 2)

        # 결과 화면 출력
        st.image(display_img, caption='분석 결과 (초록: Positive, 빨강: Negative)', use_column_width=True)
        
        # 실제 통계 계산
        percent = (pos_count / total_count) * 100 if total_count > 0 else 0
        
        st.subheader("📊 실시간 분석 리포트")
        col1, col2, col3 = st.columns(3)
        col1.metric("전체 Well 감지", f"{total_count}개")
        col2.metric("Positive (형광)", f"{pos_count}개")
        col3.metric("비율 (%)", f"{percent:.1f}%")

        # 결과 저장 기능
        res_img = Image.fromarray(display_img)
        buffered = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("분석 결과 이미지 다운로드", data=buffered, file_name="analysis_result.png", mime="image/png")
    else:
        st.error("우물을 찾지 못했습니다. 왼쪽 '설정'에서 우물 크기나 거리를 조절해 보세요.")
