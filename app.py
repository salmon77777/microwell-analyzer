import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="격자 복원형 분석기", layout="wide")
st.title("🧬 격자 복원형 Microwell 분석기 (안정화 버전)")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 인식 설정")
min_brightness = st.sidebar.slider("배경 노이즈 제거", 0, 255, 60)
min_distance = st.sidebar.slider("Well 사이 최소 거리", 5, 100, 18)

st.sidebar.header("🧪 2. 판정 및 격자 설정")
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)
grid_reconstruct = st.sidebar.checkbox("빈 공간 격자 복원 활성화", value=True)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석 속도를 위한 리사이즈
    scale = 1000 / w
    target_w, target_h = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    green_ch = img_bgr[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 1. 일차 탐지 (가장 밝은 지점들 찾기)
    k_size = max(3, min_distance)
    if k_size % 2 == 0: k_size += 1
    local_max = cv2.dilate(blurred, np.ones((k_size, k_size), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_p, x_p = np.where(peak_mask)
    
    found_pts = []
    if len(x_p) > 0:
        used_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        sorted_idx = np.argsort(blurred[y_p, x_p])[::-1]
        
        for i in sorted_idx:
            cx, cy = x_p[i], y_p[i]
            if used_mask[cy, cx] > 0: continue
            cv2.circle(used_mask, (cx, cy), int(min_distance * 0.7), 255, -1)
            found_pts.append([cx, cy])

    final_wells = []
    # 2. 격자 복원 알고리즘 (에러 방지 로직 포함)
    if grid_reconstruct and len(found_pts) > 10:
        pts = np.array(found_pts)
        
        # X, Y 간격 추정 (중앙값 사용)
        def estimate_spacing(coords):
            coords_unique = np.sort(np.unique(coords))
            diffs = np.diff(coords_unique)
            # 너무 좁은 간격은 노이즈로 간주하고 필터링
            valid_diffs = diffs[diffs > min_distance * 0.5]
            return np.median(valid_diffs) if len(valid_diffs) > 0 else min_distance

        dx = estimate_spacing(pts[:, 0])
        dy = estimate_spacing(pts[:, 1])
        
        # 격자 범위 설정
        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        
        # 격자망 생성
        for ty in np.arange(min_y, max_y + 0.1, dy):
            for tx in np.arange(min_x, max_x + 0.1, dx):
                final_wells.append([int(tx), int(ty)])
    else:
        final_wells = found_pts

    # 3. 최종 분석 및 시각화
    res_img = img_small.copy()
    pos_cnt = 0
    total_count = 0

    if len(final_wells) > 0:
        for cx, cy in final_wells:
            if 0 <= cx < target_w and 0 <= cy < target_h:
                total_count += 1
                # 격자 중심의 밝기 확인
                val = blurred[cy, cx]
                is_pos = val > threshold_g
                
                if is_pos:
                    pos_cnt += 1
                    # Positive: 초록색 원 (테두리 두껍게)
                    cv2.circle(res_img, (cx, cy), 7, (0, 255, 0), 2)
                else:
                    # Negative: 노란색 원 (테두리 얇게)
                    cv2.circle(res_img, (cx, cy), 7, (255, 255, 0), 1)

        st.image(res_img, use_container_width=True)
        
        ratio = (pos_cnt / total_count * 100)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 Well (격자 포함)", f"{total_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        if ratio >= 50:
            st.success("🧬 판정 결과: GMO Positive")
        else:
            st.error("🧬 판정 결과: Non-GMO")
    else:
        st.warning("Well을 탐지하지 못했습니다. 사이드바의 '배경 노이즈 제거'를 낮춰보세요.")
