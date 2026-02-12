import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Corner Analyzer", layout="wide")
st.title("🔬 4점 보정형 Microwell 분석기")

# 1. 사이드바: 격자 설정
st.sidebar.header("📍 모서리 좌표 설정 (Pixel)")

col_count = st.sidebar.number_input("가로 우물 개수", 1, 100, 23)
row_count = st.sidebar.number_input("세로 우물 개수", 1, 100, 24)

# 이미지의 대략적인 크기를 미리 알 수 없으므로 초기값은 적절히 배정
st.sidebar.subheader("📐 네 모서리 지정")
tl_x = st.sidebar.number_input("좌측 상단(Top-Left) X", 0, 3000, 50)
tl_y = st.sidebar.number_input("좌측 상단(Top-Left) Y", 0, 3000, 50)

tr_x = st.sidebar.number_input("우측 상단(Top-Right) X", 0, 3000, 400)
tr_y = st.sidebar.number_input("우측 상단(Top-Right) Y", 0, 3000, 50)

bl_x = st.sidebar.number_input("좌측 하단(Bottom-Left) X", 0, 3000, 50)
bl_y = st.sidebar.number_input("좌측 하단(Bottom-Left) Y", 0, 3000, 400)

br_x = st.sidebar.number_input("우측 하단(Bottom-Right) X", 0, 3000, 400)
br_y = st.sidebar.number_input("우측 하단(Bottom-Right) Y", 0, 3000, 400)

st.sidebar.markdown("---")
radius = st.sidebar.slider("우물 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 60)

# 2. 사진 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = img_rgb.copy()

        # 모서리 좌표 정의
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=float)

        pos_count = 0
        neg_count = 0
        total_wells = col_count * row_count

        # 선형 보간을 통한 격자 생성 루프
        for r in range(row_count):
            # 세로축 비율 (0.0 ~ 1.0)
            v_ratio = r / (row_count - 1) if row_count > 1 else 0
            
            # 왼쪽 변과 오른쪽 변의 해당 높이 지점 계산
            left_edge = (1 - v_ratio) * pts_src[0] + v_ratio * pts_src[3]
            right_edge = (1 - v_ratio) * pts_src[1] + v_ratio * pts_src[2]
            
            for c in range(col_count):
                # 가로축 비율 (0.0 ~ 1.0)
                h_ratio = c / (col_count - 1) if col_count > 1 else 0
                
                # 최종 우물 중심 좌표 (X, Y)
                center = (1 - h_ratio) * left_edge + h_ratio * right_edge
                cx, cy = int(center[0]), int(center[1])

                if 0 <= cx < w and 0 <= cy < h:
                    # 해당 위치 색상 추출
                    g_val = img_rgb[cy, cx, 1]
                    
                    if g_val > threshold:
                        pos_count += 1
                        color = (255, 0, 0) # Positive: Blue
                    else:
                        neg_count += 1
                        color = (0, 0, 255) # Negative: Red
                    
                    cv2.circle(display_img, (cx, cy), radius, color, 1)

        # 모서리 영역 표시 (가이드 라인)
        cv2.polylines(display_img, [pts_src.astype(int)], True, (255, 255, 0), 2)

        st.image(display_img, caption="4점 보정 분석 결과", use_container_width=True)
        
        # 리포트
        st.subheader("📊 분석 결과 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive (파랑)", f"{pos_count}개")
        c2.metric("Negative (빨강)", f"{neg_count}개")
        c3.metric("형광 비율", f"{(pos_count/total_wells*100):.1f}%")
