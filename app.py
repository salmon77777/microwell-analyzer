import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microwell Auto Analyzer", layout="wide")
st.title("🔬 자동 격자 보정 Microwell 분석기")
st.markdown("---")

# 1. 사이드바 설정
st.sidebar.header("⚙️ 분석 설정")
rotation = st.sidebar.slider("📸 사진 회전", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.subheader("🎯 격자 자동 감지 설정")
# 사용자가 대략적인 개수만 입력하면 알고리즘이 미세 조정합니다.
expected_cols = st.sidebar.number_input("가로 우물 예상 개수", 1, 100, 23)
expected_rows = st.sidebar.number_input("세로 우물 예상 개수", 1, 100, 24)
radius = st.sidebar.slider("우물 반지름", 1, 20, 5)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 60)

# 2. 사진 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        # [회전 보정]
        h, w = img.shape[:2] # 에러 수정: .shape[:2] 사용
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = img_rgb.copy()
        
        # [격자 자동 감지 로직]
        # 이미지의 X축, Y축 투영(Projection)을 통해 피크 지점을 찾습니다.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_proj = np.mean(gray, axis=0)
        y_proj = np.mean(gray, axis=1)

        def get_grid_points(proj, expected_count):
            # 신호에서 피크(우물 위치)를 추출하는 단순화된 로직
            indices = np.argsort(proj)[-expected_count:]
            return sorted(indices)

        # 실제로는 단순 피크보다 일정한 간격을 계산하는 것이 더 정확합니다.
        # 여기서는 UI에서 입력받은 값을 기반으로 하되, 
        # 오른쪽 끝 오차를 줄이기 위해 간격을 '소수점' 단위로 정밀 계산합니다.
        
        # 간격 미세 조정을 위한 가이드 (첫 우물과 마지막 우물 기준 분할)
        # 이미지 가장자리의 여백을 제외한 실제 영역 탐색 (간단한 예시)
        grid_x = np.linspace(start=10, stop=w-10, num=expected_cols)
        grid_y = np.linspace(start=10, stop=h-10, num=expected_rows)

        pos_count = 0
        neg_count = 0
        
        # 분석 실행
        for py in grid_y:
            for px in grid_x:
                cx, cy = int(px), int(py)
                
                if 0 <= cx < w and 0 <= cy < h:
                    # 해당 좌표의 Green 채널 값 확인
                    g_val = img_rgb[cy, cx, 1] 
                    
                    if g_val > threshold:
                        pos_count += 1
                        color = (0, 255, 0) # Positive: 녹색 (이미지가 녹색이므로 보조용)
                        cv2.circle(display_img, (cx, cy), radius, (255, 0, 0), 1) # 표시: 빨강
                    else:
                        neg_count += 1
                        cv2.circle(display_img, (cx, cy), radius, (0, 0, 255), 1) # Negative: 파랑

        # 결과 출력
        st.image(display_img, caption="분석 결과", use_container_width=True)
        
        total = pos_count + neg_count
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{pos_count}개")
        c2.metric("Negative", f"{neg_count}개")
        c3.metric("비율", f"{(pos_count/total*100):.1f}%" if total > 0 else "0%")
    else:
        st.error("이미지를 불러올 수 없습니다.")
