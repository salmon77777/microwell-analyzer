import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

st.set_page_config(page_title="최종 병기 분석기", layout="wide")
st.title("🔬 초강력 픽셀 피크 분석기")

# --- 사이드바: 파라미터 극단적 단순화 ---
st.sidebar.header("⚙️ 인식 강도 조절")
st.sidebar.info("자동 인식이 안 될 때 사용하는 최후의 수단입니다.")

# 피크 탐색 민감도 (낮을수록 아주 미세한 점도 다 잡음)
peak_min_val = st.sidebar.slider("최소 밝기 문턱값", 0, 255, 30)
min_dist = st.sidebar.slider("Well 간 최소 거리", 1, 100, 15)
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
        # Green 채널이 가장 정보가 많으므로 이를 분석용으로 사용
        gray = img_bgr[:,:,1] 
        
        # 1. Local Maximum Filter (주변에서 가장 밝은 점 찾기)
        # 이 필터는 수학적 원을 무시하고 그냥 '밝은 지점'의 중심을 찾습니다.
        data_max = maximum_filter(gray, size=min_dist)
        maxima = (gray == data_max)
        
        # 2. 배경 노이즈 제거 (문턱값 이하 제외)
        maxima[gray < peak_min_val] = False
        
        # 3. 좌표 추출
        y_coords, x_coords = np.where(maxima)
        
        res_img = img_rgb.copy()
        pos_cnt = 0
        valid_wells = []

        for cx, cy in zip(x_coords, y_coords):
            # 가장자리 제외
            if cx < 5 or cx > w-5 or cy < 5 or cy > h-5:
                continue
                
            valid_wells.append((cx, cy))
            
            # 형광 판정 (해당 피크 지점의 밝기)
            is_pos = gray[cy, cx] > threshold_g
            if is_pos:
                pos_cnt += 1
            
            # 노란색 원: 탐지된 Well / 초록색 점: Positive
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
        
        with st.expander("인식 보조 화면"):
            st.image(gray, caption="분석에 사용된 Green 채널 원본")
    else:
        st.error("이미지를 읽을 수 없습니다.")
