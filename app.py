import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 정밀 회전 및 4점 보정 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 사진 회전")
rotation = st.sidebar.slider("사진 기울기 조절", -180.0, 180.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 (Pixel)")
col_count = st.sidebar.number_input("가로 우물 개수", 1, 100, 23)
row_count = st.sidebar.number_input("세로 우물 개수", 1, 100, 24)

# 초기 좌표값 (이미지 업로드 전 가이드용)
tl_x = st.sidebar.number_input("좌측 상단 X", 0, 5000, 50)
tl_y = st.sidebar.number_input("좌측 상단 Y", 0, 5000, 50)
tr_x = st.sidebar.number_input("우측 상단 X", 0, 5000, 600)
tr_y = st.sidebar.number_input("우측 상단 Y", 0, 5000, 50)
bl_x = st.sidebar.number_input("좌측 하단 X", 0, 5000, 50)
bl_y = st.sidebar.number_input("좌측 하단 Y", 0, 5000, 600)
br_x = st.sidebar.number_input("우측 하단 X", 0, 5000, 600)
br_y = st.sidebar.number_input("우측 하단 Y", 0, 5000, 600)

st.sidebar.header("🧪 3단계: 판정 설정")
radius = st.sidebar.slider("우물 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)

# 2. 사진 업로드 및 처리
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        # [회전 보정 실행]
        h, w = raw_img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        # 회전 시 잘림 방지를 위해 결과 이미지 크기 유지
        img = cv2.warpAffine(raw_img, rot_matrix, (w, h))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = img_rgb.copy()

        # 모서리 좌표 정의
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=float)

        pos_count = 0
        neg_count = 0
        total_wells = col_count * row_count

        # 바리센트릭 보간 격자 생성
        for r in range(row_count):
            v_ratio = r / (row_count - 1) if row_count > 1 else 0
            left_edge = (1 - v_ratio) * pts_src[0] + v_ratio * pts_src[3]
            right_edge = (1 - v_ratio) * pts_src[1] + v_ratio * pts_src[2]
            
            for c in range(col_count):
                h_ratio = c / (col_count - 1) if col_count > 1 else 0
                center_pt = (1 - h_ratio) * left_edge + h_ratio * right_edge
                cx, cy = int(center_pt[0]), int(center_pt[1])

                if 0 <= cx < w and 0 <= cy < h:
                    # 중심 픽셀 주변의 평균 G값 계산 (단일 픽셀보다 정확함)
                    sample = img_rgb[max(0, cy-1):cy+2, max(0, cx-1):cx+2, 1]
                    g_val = np.mean(sample)
                    
                    if g_val > threshold:
                        pos_count += 1
                        color = (0, 255, 255) # Positive: Cyan (눈에 잘 띄게)
                    else:
                        neg_count += 1
                        color = (255, 0, 0) # Negative: Red
                    
                    cv2.circle(display_img, (cx, cy), radius, color, 1)

        # 영역 가이드 라인 (노란색 사각형)
        cv2.polylines(display_img, [pts_src.astype(int)], True, (255, 255, 0), 2)

        st.image(display_img, caption=f"회전 {rotation}° 및 4점 보정 적용 결과", use_container_width=True)
        
        # 결과 대시보드
        st.subheader("📊 데이터 분석 요약")
        cols = st.columns(4)
        cols[0].metric("총 우물", f"{total_wells}개")
        cols[1].metric("Positive", f"{pos_count}개")
        cols[2].metric("Negative", f"{neg_count}개")
        cols[3].metric("형광 발현율", f"{(pos_count/total_wells*100):.1f}%")
