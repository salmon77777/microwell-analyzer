import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. 페이지 설정
st.set_page_config(page_title="Well 분석기", layout="wide")
st.title("🔬 Microwell 피크 분석기")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 인식 설정")
peak_min_val = st.sidebar.slider("최소 밝기(배경 제거)", 0, 255, 30)
min_dist = st.sidebar.slider("Well 간 최소 거리", 1, 100, 20)
well_r = st.sidebar.slider("표시 반지름", 1, 50, 12)
threshold_g = st.sidebar.slider("형광 임계값(Positive)", 0, 255, 65)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 선택하세요 (스마트폰 사진 가능)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # [수정] PIL을 사용하여 안전하게 이미지 로드
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    
    # 분석을 위해 OpenCV 포맷(BGR)으로 복사
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # Green 채널 추출
    green_ch = img_bgr[:,:,1]
    
    # 픽셀 피크 탐색 (안정적인 dilate 방식)
    kernel_size = max(3, min_dist if min_dist % 2 != 0 else min_dist + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    local_max = cv2.dilate(green_ch, kernel, iterations=1)
    peak_mask = (green_ch == local_max) & (green_ch > peak_min_val)
    
    y_coords, x_coords = np.where(peak_mask)
    
    res_img = img_rgb.copy()
    valid_wells = []
    pos_cnt = 0
    centers = []

    # 중복 제거 및 분석
    for cx, cy in zip(x_coords, y_coords):
        if cx < 5 or cx > w-5 or cy < 5 or cy > h-5:
            continue
        
        too_close = False
        for ox, oy in centers:
            if np.sqrt((cx-ox)**2 + (cy-oy)**2) < min_dist:
                too_close = True
                break
        if too_close: continue
        
        centers.append((cx, cy))
        valid_wells.append((cx, cy))
        
        is_pos = green_ch[cy, cx] > threshold_g
        if is_pos:
            pos_cnt += 1
        
        # 노란색 원과 초록색 점 그리기
        cv2.circle(res_img, (cx, cy), well_r, (255, 255, 0), 2) # 노란색 테두리
        if is_pos:
            cv2.circle(res_img, (cx, cy), max(1, int(well_r*0.5)), (0, 255, 0), -1)

    # [수정] 이미지 출력 방식 변경
    st.image(res_img, caption="분석 결과 화면", use_container_width=True)
    
    total = len(valid_wells)
    if total > 0:
        ratio = (pos_cnt / total * 100)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 Well", f"{total}개")
        c2.metric("Positive", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
    else:
        st.warning("설정값 내에서 Well을 찾지 못했습니다. '최소 밝기'를 낮춰보세요.")

else:
    st.info("스마트폰으로 촬영한 Microwell 사진을 업로드해 주세요.")
