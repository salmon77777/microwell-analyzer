import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="정밀 격자 분석기", layout="wide")
st.title("🧪 정밀 격자 자동 정렬 분석기")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 인식 및 표시 설정")
well_radius = st.sidebar.slider("Well 표시 크기", 2, 30, 10)
min_brightness = st.sidebar.slider("인식 감도 (배경 제거)", 0, 255, 50)
threshold_g = st.sidebar.slider("형광 임계값 (Positive)", 0, 255, 80)

uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 1. 전처리 및 고속 탐지 (가로 1000px 기준)
    scale = 1000 / w
    target_w, target_h = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    green_ch = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 2. 확실한 씨앗 Well(Seed Points) 찾기
    local_max = cv2.dilate(blurred, np.ones((15, 15), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    y_p, x_p = np.where(peak_mask)
    
    if len(x_p) > 20:
        # 3. 자동 간격 및 "기울기(Angle)" 분석
        pts = np.column_stack((x_p, y_p))
        
        # 간격 계산 (가장 빈번한 거리 측정)
        def get_spacing(coords):
            coords = np.sort(coords)
            diffs = np.diff(coords)
            valid = diffs[(diffs > 10) & (diffs < 60)]
            return np.median(valid) if len(valid) > 0 else 22.0

        auto_dx = get_spacing(x_p)
        auto_dy = get_spacing(y_p)

        # 4. 격자 생성 및 시각화 (모든 Well은 노란색)
        res_img = img_small.copy()
        pos_cnt = 0
        total_count = 0
        
        # 격자 원점 보정 (평균 편차 적용)
        origin_x = np.median(x_p % auto_dx)
        origin_y = np.median(y_p % auto_dy)
        
        # 격자 생성 루프
        for ty in np.arange(origin_y, target_h, auto_dy):
            for tx in np.arange(origin_x, target_w, auto_dx):
                cx, cy = int(tx), int(ty)
                
                if cx < 5 or cx > target_w-5 or cy < 5 or cy > target_h-5:
                    continue
                
                total_count += 1
                
                # 모든 인식된 Well은 노란색 테두리 (요청사항)
                cv2.circle(res_img, (cx, cy), well_radius, (255, 255, 0), 1)
                
                # 중앙부 밝기 확인 (형광 판정)
                val = blurred[cy, cx]
                if val > threshold_g:
                    pos_cnt += 1
                    # Positive인 경우 안쪽에 초록색 점 추가
                    cv2.circle(res_img, (cx, cy), int(well_radius*0.5), (0, 255, 0), -1)

        st.image(res_img, use_container_width=True, caption="노란색 원: 탐지된 모든 Well / 초록색 채움: 양성 신호")
        
        # 결과 요약
        ratio = (pos_cnt / total_count * 100) if total_count > 0 else 0
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 Well 개수", f"{total_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        st.caption(f"📏 자동 계산된 간격: 가로 {auto_dx:.1f}px / 세로 {auto_dy:.1f}px")
    else:
        st.error("Well을 충분히 찾지 못했습니다. '인식 감도'를 낮춰보세요.")
        st.image(blurred, caption="현재 인식용 흑백 화면 (점들이 보여야 합니다)")
