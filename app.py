import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="격자 복원형 분석기", layout="wide")
st.title("🧬 격자 복원형 Microwell 분석기")
st.info("보이는 Well의 위치를 분석해 신호가 없는 빈 칸까지 모두 찾아냅니다.")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 인식 설정 (패턴 찾기)")
min_brightness = st.sidebar.slider("배경 노이즈 제거", 0, 255, 60)
min_distance = st.sidebar.slider("Well 사이 최소 거리", 5, 100, 15)

st.sidebar.header("🧪 2. 판정 및 격자 설정")
threshold_g = st.sidebar.slider("GMO 양성 판정 기준", 0, 255, 80)
grid_reconstruct = st.sidebar.checkbox("빈 공간 격자 복원 활성화", value=True)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    scale = 1000 / w
    target_w, target_h = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    green_ch = img_bgr[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 1. 1차 탐지: 보이는(밝은) Well들 먼저 찾기
    k_size = max(3, min_distance)
    if k_size % 2 == 0: k_size += 1
    local_max = cv2.dilate(blurred, np.ones((k_size, k_size), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_p, x_p = np.where(peak_mask)
    
    # 중복 제거 후 유효한 Well 좌표 리스트 생성
    found_pts = []
    used_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    sorted_idx = np.argsort(blurred[y_p, x_p])[::-1]
    
    for i in sorted_idx:
        cx, cy = x_p[i], y_p[i]
        if used_mask[cy, cx] > 0: continue
        cv2.circle(used_mask, (cx, cy), int(min_distance * 0.8), 255, -1)
        found_pts.append([cx, cy])

    # 2. 격자 복원 로직 (Grid Reconstruction)
    final_wells = []
    if grid_reconstruct and len(found_pts) > 20:
        pts = np.array(found_pts)
        # X, Y 좌표별로 정렬하여 평균 간격 추출
        ux = np.sort(pts[:, 0])
        uy = np.sort(pts[:, 1])
        
        # 델타(간격)의 중앙값 계산 (노이즈에 강함)
        dx = np.median(np.diff(ux)[np.diff(ux) > min_distance*0.8])
        dy = np.median(np.diff(uy)[np.diff(uy) > min_distance*0.8])
        
        # 실제 격자 좌표 생성 (발견된 영역 내에서)
        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        
        # 안전장치가 포함된 격자 루프
        for ty in np.arange(min_y, max_y + 1, dy):
            for tx in np.arange(min_x, max_x + 1, dx):
                final_wells.append([int(tx), int(ty)])
    else:
        final_wells = found_pts

    # 3. 최종 판정 및 시각화
    res_img = img_small.copy()
    pos_cnt = 0
    analyzed_count = 0

    for cx, cy in final_wells:
        if 0 <= cx < target_w and 0 <= cy < target_h:
            analyzed_count += 1
            # 해당 좌표 밝기 확인
            val = blurred[cy, cx]
            is_pos = val > threshold_g
            
            if is_pos:
                pos_cnt += 1
                # Positive: 초록색 두꺼운 원
                cv2.circle(res_img, (cx, cy), 7, (0, 255, 0), 2)
            else:
                # Negative: 노란색 얇은 원 (신호가 없어도 그 자리에 그려짐)
                cv2.circle(res_img, (cx, cy), 7, (255, 255, 0), 1)

    # 4. 결과 출력
    st.image(res_img, use_container_width=True)
    
    if analyzed_count > 0:
        ratio = (pos_cnt / analyzed_count * 100)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 Well (격자 복원 포함)", f"{analyzed_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        # 판정 결과 안내
        st.info(f"💡 현재 전체 {analyzed_count}개의 Well 중 {pos_cnt}개에서 신호가 감지되었습니다.")
