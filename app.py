import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Grid Analyzer", layout="wide")
st.title("🔬 격자 맞춤형 Microwell 분석기")

# 1. 사이드바: 격자 배치 설정
st.sidebar.header("📏 격자 설정 (Grid Setup)")
col_count = st.sidebar.number_input("가로 우물 개수", 1, 100, 20)
row_count = st.sidebar.number_input("세로 우물 개수", 1, 100, 15)

st.sidebar.markdown("---")
st.sidebar.write("📍 위치 및 간격 조절")
start_x = st.sidebar.slider("첫 번째 우물 X 좌표", 0, 1500, 100)
start_y = st.sidebar.slider("첫 번째 우물 Y 좌표", 0, 1500, 100)
gap_x = st.sidebar.slider("가로 간격 (Spacing X)", 1, 100, 25)
gap_y = st.sidebar.slider("세로 간격 (Spacing Y)", 1, 100, 25)
radius = st.sidebar.slider("우물 반지름", 1, 50, 8)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 120)

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
            center_x = start_x + (c * gap_x)
            center_y = start_y + (r * gap_y)
            
            # 이미지 범위 내에 있는 경우만 분석
            if center_x < w and center_y < h:
                # 개별 우물 영역 분석
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                mean_val = cv2.mean(img_rgb, mask=mask)
                green_val = mean_val[1] # Green 채널 평균값
                
                # 임계값 판정
                if green_val > threshold:
                    pos_count += 1
                    cv2.circle(display_img, (center_x, center_y), 2, (0, 255, 0), -1) # 중앙 초록점
                
                # 파란색 얇은 테두리 표시
                cv2.circle(display_img, (center_x, center_y), radius, (50, 150, 255), 1)

    # 결과 화면 출력
    st.image(display_img, caption="격자 분석 결과 (파란 원: 격자 구역 / 초록 점: Positive)", use_container_width=True)
    
    # 3. 리포트
    percent = (pos_count / total_wells) * 100 if total_wells > 0 else 0
    st.subheader("📊 분석 결과 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 우물 수", f"{total_wells}개")
    c2.metric("Positive (형광)", f"{pos_count}개")
    c3.metric("형광 발현 비율", f"{percent:.1f}%")

    # 결과 저장
    res_bytes = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button("📸 분석 이미지 저장", data=res_bytes, file_name="grid_analysis.png")
else:
    st.info("사진을 업로드하면 격자 조절 화면이 나타납니다.")
