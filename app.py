import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="최종 안정화 분석기", layout="wide")
st.title("🔬 픽셀 피크 기반 자동 분석기 (안정 버전)")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 인식 강도 조절")
# 최소 밝기: 이 값보다 밝은 점들 중에서 피크를 찾습니다.
peak_min_val = st.sidebar.slider("최소 밝기 문턱값", 0, 255, 30)
# 최소 거리: 점들 사이의 간격입니다. (너무 작으면 한 Well에 여러 점이 찍힘)
min_dist = st.sidebar.slider("Well 간 최소 거리", 1, 100, 20)
# 시각화용 반지름
well_r = st.sidebar.slider("표시될 Well 반지름", 1, 50, 12)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 65)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Green 채널 추출 (형광 분석의 핵심)
        green_ch = img_bgr[:,:,1] 
        
        # [핵심] Scipy 없이 피크 탐색 (OpenCV의 Dilate 사용)
        # 주변에서 가장 밝은 값을 확장한 뒤 원본과 비교하여 '꼭짓점' 추출
        kernel_size = max(3, min_dist if min_dist % 2 != 0 else min_dist + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        local_max = cv2.dilate(green_ch, kernel, iterations=1)
        peak_mask = (green_ch == local_max) & (green_ch > peak_min_val)
        
        # 피크 좌표 추출
        y_coords, x_coords = np.where(peak_mask)
        
        res_img = img_rgb.copy()
        valid_wells = []
        pos_cnt = 0
        
        # 중복 제거 및 시각화
        # dilate로도 중복이 생길 수 있으므로 거리를 한 번 더 체크
        centers = []
        for cx, cy in zip(x_coords, y_coords):
            # 가장자리 제외 (5px 마진)
            if cx < 5 or cx > w-5 or cy < 5 or cy > h-5:
                continue
            
            # 너무 붙어있는 점들 필터링
            too_close = False
            for ox, oy in centers:
                if np.sqrt((cx-ox)**2 + (cy-oy)**2) < min_dist:
                    too_close = True
                    break
            if too_close: continue
            
            centers.append((cx, cy))
            valid_wells.append((cx, cy))
            
            # 형광 판정 및 그리기
            is_pos = green_ch[cy, cx] > threshold_g
            if is_pos:
                pos_cnt += 1
            
            # 노란색 원: 모든 Well / 초록색 점: Positive
            cv2.circle(res_img, (cx, cy), well_r, (255, 255, 0), 1)
            if is_pos:
                cv2.circle(res_img, (cx, cy), max(1, int(well_r*0.5)), (0, 255, 0), -1)

        st.image(res_img, use_container_width=True)
        
        total = len(valid_wells)
        ratio = (pos_cnt / total * 100) if total > 0 else 0
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("탐지된 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        if ratio >= 50: # 예시 기준값
            st.success("🧬 판정: GMO Positive")
        else:
            st.error("🧬 판정: Non-GMO")
            
    else:
        st.error("이미지를 읽을 수 없습니다.")
