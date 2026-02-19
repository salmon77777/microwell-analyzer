import streamlit as st
import numpy as np
import cv2
from PIL import Image
from collections import Counter

st.set_page_config(page_title="자동 간격 분석기", layout="wide")
st.title("🤖 간격 자동 인식 Well 분석기")
st.info("보이는 Well들을 분석하여 전체 격자 간격을 스스로 계산합니다.")

# --- 사이드바: 최소한의 설정 ---
st.sidebar.header("⚙️ 기본 설정")
well_radius = st.sidebar.slider("Well 표시 반지름", 2, 30, 10)
min_brightness = st.sidebar.slider("배경 노이즈 제거", 0, 255, 50)
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 1. 전처리 및 고속 탐지
    scale = 1000 / w
    target_w, target_h = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    green_ch = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 2. 확실한 Well들 추출 (Seed Points)
    local_max = cv2.dilate(blurred, np.ones((15, 15), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_p, x_p = np.where(peak_mask)
    
    if len(x_p) > 10:
        # 3. [핵심] 간격 자동 계산 (Auto-Spacing Logic)
        def get_auto_spacing(coords):
            coords = np.sort(coords)
            diffs = np.diff(coords)
            # 너무 작은 노이즈 간격 제외 (5px 이상만)
            valid_diffs = diffs[(diffs > 10) & (diffs < 50)]
            if len(valid_diffs) == 0: return 20.0 # 기본값
            # 가장 빈번하게 나타나는 간격을 선택
            counts = np.bincount(valid_diffs.astype(int))
            return np.argmax(counts)

        auto_dx = get_auto_spacing(x_p)
        auto_dy = get_auto_spacing(y_p)
        
        # 4. 격자 원점 설정 및 전체 확장
        # 가장 많은 Well이 발견되는 라인을 기준으로 원점 보정
        origin_x = np.median(x_p % auto_dx)
        origin_y = np.median(y_p % auto_dy)
        
        res_img = img_small.copy()
        pos_cnt = 0
        total_count = 0
        
        # 생성된 자동 간격으로 격자 그리기
        for ty in np.arange(origin_y, target_h, auto_dy):
            for tx in np.arange(origin_x, target_w, auto_dx):
                cx, cy = int(tx), int(ty)
                
                if cx < 5 or cx > target_w-5 or cy < 5 or cy > target_h-5:
                    continue
                
                total_count += 1
                val = blurred[cy, cx]
                is_pos = val > threshold_g
                
                if is_pos:
                    pos_cnt += 1
                    cv2.circle(res_img, (cx, cy), well_radius, (0, 255, 0), 2)
                else:
                    cv2.circle(res_img, (cx, cy), well_radius, (255, 255, 0), 1)

        st.image(res_img, use_container_width=True)
        
        # 결과 대시보드
        ratio = (pos_cnt / total_count * 100) if total_count > 0 else 0
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("탐지된 격자 (자동)", f"{total_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        st.write(f"📏 **자동 계산된 간격:** 가로 {auto_dx}px, 세로 {auto_dy}px")
    else:
        st.warning("분석할 만큼 충분한 Well이 보이지 않습니다. '배경 노이즈 제거'를 낮춰보세요.")
