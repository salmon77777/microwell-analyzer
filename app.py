import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="정밀 격자 제어기", layout="wide")
st.title("📏 사용자 정의 격자 분석기")
st.info("사이드바의 'Well 반지름'과 'Well 간격'을 조절하여 실제 사진의 구멍 크기와 맞추세요.")

# --- 사이드바: 사용자가 직접 사이즈 결정 ---
st.sidebar.header("📏 1. 격자 사이즈 설정")
# 실제 Well의 크기
well_radius = st.sidebar.slider("Well 표시 반지름", 2, 50, 10)
# Well 중심과 다음 중심 사이의 거리
spacing_x = st.sidebar.slider("가로 간격 (Pitch X)", 5.0, 100.0, 20.0, step=0.1)
spacing_y = st.sidebar.slider("세로 간격 (Pitch Y)", 5.0, 100.0, 20.0, step=0.1)

st.sidebar.header("🧪 2. 판정 설정")
min_brightness = st.sidebar.slider("배경 노이즈 제거", 0, 255, 40)
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석 기준 해상도 고정 (1000px 가로 기준)
    scale = 1000 / w
    target_w, target_h = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    green_ch = img_bgr[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 1. 기준점 찾기 (가장 밝은 웰 하나를 기준으로 격자 시작)
    # 전체를 다 찾는게 아니라 '격자의 시작점'만 찾습니다.
    k_size = int(well_radius * 2)
    if k_size % 2 == 0: k_size += 1
    local_max = cv2.dilate(blurred, np.ones((k_size, k_size), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_p, x_p = np.where(peak_mask)

    if len(x_p) > 0:
        # 가장 밝은 점을 격자의 원점(Origin)으로 설정
        idx = np.argmax(blurred[y_p, x_p])
        origin_x, origin_y = x_p[idx], y_p[idx]

        # 2. 사용자 설정 간격으로 격자망 생성 (이미지 전체 영역)
        res_img = img_small.copy()
        pos_cnt = 0
        total_count = 0

        # 원점으로부터 좌우/상하로 격자 전개
        # 이미지 전체를 덮도록 범위를 계산합니다.
        x_start = origin_x % spacing_x
        y_start = origin_y % spacing_y
        
        for ty in np.arange(y_start, target_h, spacing_y):
            for tx in np.arange(x_start, target_w, spacing_x):
                cx, cy = int(tx), int(ty)
                
                # 가장자리 마진 제외
                if cx < 5 or cx > target_w-5 or cy < 5 or cy > target_h-5:
                    continue
                
                total_count += 1
                # 격자 포인트의 밝기 분석
                val = blurred[cy, cx]
                is_pos = val > threshold_g
                
                if is_pos:
                    pos_cnt += 1
                    # Positive: 초록색 원
                    cv2.circle(res_img, (cx, cy), well_radius, (0, 255, 0), 2)
                else:
                    # Negative: 노란색 원
                    cv2.circle(res_img, (cx, cy), well_radius, (255, 255, 0), 1)

        st.image(res_img, use_container_width=True)
        
        # 결과 대시보드
        ratio = (pos_cnt / total_count * 100) if total_count > 0 else 0
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("격자 내 전체 Well", f"{total_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        st.caption("💡 팁: 노란색 원이 실제 Well보다 크거나 작으면 '반지름'을, 간격이 어긋나면 '가로/세로 간격'을 조절하세요.")
    else:
        st.warning("사진에서 Well의 위치를 파악할 수 없습니다. '배경 노이즈 제거'를 낮춰주세요.")
