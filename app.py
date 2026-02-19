import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="정밀 Well 분석기", layout="wide")
st.title("🔬 Microwell 정밀 분석기 (에러 수정 및 개수 최적화)")

# --- 사이드바: 노이즈 및 개수 조절 ---
st.sidebar.header("⚙️ 인식 설정")
# 1. 배경 제거: 5만 개씩 잡힌다면 이 값을 70~100까지 높이세요.
min_brightness = st.sidebar.slider("배경 노이즈 제거 (최소 밝기)", 0, 255, 60)
# 2. Well 간격: Well 하나에 여러 원이 생긴다면 이 값을 높이세요.
min_distance = st.sidebar.slider("Well 사이 최소 거리", 5, 100, 15)
# 3. 판정 기준
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 1. 이미지 로드 및 리사이즈
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    scale = 1000 / w
    target_w = 1000
    target_h = int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    
    # 2. Green 채널 노이즈 제거
    green_ch = img_bgr[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 3. 피크 탐색 (고속 Dilate 방식)
    k_size = max(3, min_distance)
    if k_size % 2 == 0: k_size += 1
    kernel = np.ones((k_size, k_size), np.uint8)
    
    local_max = cv2.dilate(blurred, kernel, iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_coords, x_coords = np.where(peak_mask)
    
    # 4. 중복 제거 및 결과 그리기
    res_img = img_small.copy()
    valid_pts = []
    pos_cnt = 0
    
    # [에러 수정된 마스크] uint8 타입으로 명시적 생성
    used_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # 밝기 순으로 정렬하여 신뢰도 높은 점부터 선점
    sorted_idx = np.argsort(blurred[y_coords, x_coords])[::-1]

    for i in sorted_idx:
        cx, cy = x_coords[i], y_coords[i]
        
        # 이미 처리된 영역(used_mask가 255인 곳)이면 건너뜀
        if used_mask[cy, cx] > 0:
            continue
        
        # [에러 해결] used_mask에 원을 그려 주변 중복 탐지 방지
        cv2.circle(used_mask, (cx, cy), int(min_distance * 0.8), 255, -1)
        
        valid_pts.append((cx, cy))
        
        # 형광 판정
        val = blurred[cy, cx]
        is_pos = val > threshold_g
        
        if is_pos:
            pos_cnt += 1
            # 양성: 두꺼운 초록색 테두리
            cv2.circle(res_img, (cx, cy), 7, (0, 255, 0), 2)
        else:
            # 음성: 얇은 노란색 테두리
            cv2.circle(res_img, (cx, cy), 7, (255, 255, 0), 1)

    # 5. 결과 시각화
    st.image(res_img, use_container_width=True)
    
    total = len(valid_pts)
    if total > 0:
        ratio = (pos_cnt / total * 100)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("실제 탐지된 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        if ratio >= 50:
            st.success("🧬 판정 결과: GMO Positive")
        else:
            st.error("🧬 판정 결과: Non-GMO")
    else:
        st.warning("Well이 감지되지 않았습니다. '배경 노이즈 제거' 값을 낮춰보세요.")
