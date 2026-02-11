import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Grid Analyzer", layout="wide")
st.title("🔬 정밀 보정형 Microwell 분석기")
st.markdown("---")

# 1. 사이드바: 설정
st.sidebar.header("📏 격자 및 보정 설정")

# [보정] 사진 회전 기능 추가
rotation = st.sidebar.slider("📸 사진 회전 (Rotation)", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.markdown("---")
col_count = st.sidebar.number_input("가로 우물 개수", 1, 100, 23)
row_count = st.sidebar.number_input("세로 우물 개수", 1, 100, 24)

st.sidebar.write("📍 위치 및 간격 조절")
start_x = st.sidebar.number_input("첫 번째 우물 X 좌표", 0.0, 2000.0, 5.0, step=1.0)
start_y = st.sidebar.number_input("첫 번째 우물 Y 좌표", 0.0, 2000.0, 7.0, step=1.0)
gap_x = st.sidebar.number_input("가로 간격 (Spacing X)", 1.0, 100.0, 14.2, step=0.01)
gap_y = st.sidebar.number_input("세로 간격 (Spacing Y)", 1.0, 100.0, 9.8, step=0.01)
radius = st.sidebar.slider("우물 반지름", 1, 50, 5)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 50)

# 2. 사진 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # [회전 보정 실행]
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
    img = cv2.warpAffine(img, matrix, (w, h))
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    pos_count = 0
    neg_count = 0
    total_wells = col_count * row_count
    
    for r in range(row_count):
        for c in range(col_count):
            center_x = int(start_x + (c * gap_x))
            center_y = int(start_y + (r * gap_y))
            
            if center_x < w and center_y < h:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                mean_val = cv2.mean(img_rgb, mask=mask)
                green_val = mean_val[1]
                
                if green_val > threshold:
                    pos_count += 1
                    border_color = (0, 0, 255) # Positive (Blue)
                else:
                    neg_count += 1
                    border_color = (255, 0, 0) # Negative (Red)
                
                cv2.circle(display_img, (center_x, center_y), radius, border_color, 1)

    st.image(display_img, caption=f"분석 결과 (회전 보정: {rotation}도)", use_container_width=True)
    
    # 리포트 출력
    st.subheader("📊 분석 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 우물 수", f"{total_wells}개")
    c2.metric("Positive (파랑)", f"{pos_count}개")
    c3.metric("Negative (빨강)", f"{neg_count}개")
    c4.metric("형광 발현 비율", f"{(pos_count/total_wells*100):.1f}%" if total_wells > 0 else "0%")

    res_bytes = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button("📸 보정된 분석 이미지 저장", data=res_bytes, file_name="calibrated_analysis.png")
