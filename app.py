import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="정밀 Well 분석기", layout="wide")
st.title("🔬 Microwell 정밀 분석기 (노이즈 필터 강화)")

# --- 사이드바: 노이즈 잡는 핵심 설정 ---
st.sidebar.header("⚙️ 인식 정밀도 조절")
# 1. 최소 밝기: 배경 노이즈를 걸러내는 첫 번째 장벽 (값이 너무 낮으면 5만 개씩 잡힘)
min_brightness = st.sidebar.slider("배경 노이즈 제거 (최소 밝기)", 0, 255, 50)
# 2. Well 사이 간격: 이 값보다 가까운 점들은 하나로 합쳐버림
min_distance = st.sidebar.slider("Well 사이 최소 거리", 5, 100, 15)
# 3. 형광 판정: Positive로 볼 밝기 기준
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 1. 이미지 로드 및 리사이즈 (연산 속도와 노이즈 뭉개기 효과)
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석용 이미지 (가로 1000px로 고정하여 거리 계산을 일정하게 유지)
    scale = 1000 / w
    img_small = cv2.resize(img_rgb, (1000, int(h * scale)))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    
    # 2. 노이즈 제거 (가우시안 블러를 강하게 주어 미세 노이즈를 지움)
    green_ch = img_bgr[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 3. 고속 피크 탐색 + 중복 제거 강화
    kernel_size = max(3, min_distance)
    if kernel_size % 2 == 0: kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    local_max = cv2.dilate(blurred, kernel, iterations=1)
    # 진짜 피크점만 추출
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_coords, x_coords = np.where(peak_mask)
    
    res_img = img_small.copy()
    valid_pts = []
    pos_cnt = 0
    
    # 중복 제거 로직 (이미 잡힌 좌표 주변은 건너뜀)
    sorted_idx = np.argsort(blurred[y_coords, x_coords])[::-1] # 밝은 순서대로 정렬
    used_mask = np.zeros_like(peak_mask)

    for i in sorted_idx:
        cx, cy = x_coords[i], y_coords[i]
        
        if used_mask[cy, cx]: continue # 이미 처리된 영역이면 패스
        
        # 주변 영역 사용 처리 (min_distance 만큼 마킹)
        cv2.circle(used_mask, (cx, cy), int(min_distance/2), True, -1)
        
        valid_pts.append((cx, cy))
        
        # 형광 판독 (점 하나가 아닌 주변 평균값으로 판독하여 정확도 향상)
        roi = blurred[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2)]
        avg_val = np.mean(roi)
        
        is_pos = avg_val > threshold_g
        if is_pos:
            pos_cnt += 1
            # 양성은 '초록색 테두리' (가운데 점 없음)
            cv2.circle(res_img, (cx, cy), 8, (0, 255, 0), 2)
        else:
            # 음성은 '노란색 테두리'
            cv2.circle(res_img, (cx, cy), 8, (255, 255, 0), 1)

    # 4. 결과 출력
    st.image(res_img, use_container_width=True, caption="노란색: 탐지된 Well / 초록색: Positive 판정")
    
    total = len(valid_pts)
    if total > 0:
        ratio = (pos_cnt / total * 100)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("실제 탐지된 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율(%)", f"{ratio:.1f}%")
        
        if ratio >= 50:
            st.success("🧬 판정 결과: GMO Positive")
        else:
            st.error("🧬 판정 결과: Non-GMO")
    else:
        st.warning("Well이 감지되지 않았습니다. '배경 노이즈 제거' 수치를 낮춰보세요.")
