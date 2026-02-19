import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="강제 피크 분석기", layout="wide")
st.title("🚀 초강력 강제 Well 탐지기")

# --- 사이드바: 파라미터 극단적 단순화 ---
st.sidebar.header("⚙️ 인식 강도 조절")
st.sidebar.info("형태와 상관없이 '밝은 지점'을 강제로 찾아냅니다.")

# 1. 픽셀 탐색 민감도 (이걸 낮추면 무조건 잡힙니다)
min_brightness = st.sidebar.slider("최소 밝기 (낮을수록 다 잡음)", 0, 255, 20)
# 2. Well 사이의 간격 (너무 낮으면 한 곳에 여러 개 찍힘)
min_distance = st.sidebar.slider("Well 사이 간격", 1, 100, 15)
# 3. 형광 판정 기준
threshold_g = st.sidebar.slider("형광 임계값 (Positive 기준)", 0, 255, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 1. 이미지 로드 및 고속 처리를 위한 리사이즈
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 속도를 위해 가로 1000px로 축소
    scale = 1000 / w
    img_small = cv2.resize(img_rgb, (1000, int(h * scale)))
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
    green_ch = img_bgr[:,:,1] # Green 채널만 집중 분석
    
    # 2. [핵심] 고속 피크 탐색 (오브젝트 분석 대신 픽셀 최대값 찾기)
    # 주변에서 가장 밝은 픽셀들을 골라냅니다.
    kernel_size = max(3, min_distance)
    if kernel_size % 2 == 0: kernel_size += 1
    
    # Dilate 연산을 이용한 Local Maximum 찾기
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    local_max = cv2.dilate(green_ch, kernel, iterations=1)
    # 원본과 확장 이미지가 같으면서 문턱값보다 높은 곳이 '피크'
    peak_mask = (green_ch == local_max) & (green_ch > min_brightness)
    
    y_coords, x_coords = np.where(peak_mask)
    
    res_img = img_small.copy()
    valid_pts = []
    pos_cnt = 0
    
    # 3. 결과 그리기
    for cx, cy in zip(x_coords, y_coords):
        # 가장자리 마진
        if cx < 5 or cx > 995 or cy < 5 or cy > (int(h*scale)-5):
            continue
            
        valid_pts.append((cx, cy))
        
        # 형광 판정
        is_pos = green_ch[cy, cx] > threshold_g
        if is_pos:
            pos_cnt += 1
        
        # 노란색 원(탐지), 초록색 점(Positive)
        cv2.circle(res_img, (cx, cy), 8, (255, 255, 0), 1)
        if is_pos:
            cv2.circle(res_img, (cx, cy), 4, (0, 255, 0), -1)

    # 4. 결과 출력
    st.image(res_img, use_container_width=True)
    
    total = len(valid_pts)
    if total > 0:
        ratio = (pos_cnt / total * 100)
        st.markdown(f"### 분석 결과: {'GMO Positive' if ratio >= 50 else 'Non-GMO'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("탐지된 전체 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
    else:
        st.error("Well을 하나도 찾지 못했습니다. '최소 밝기'를 더 낮춰보세요.")
        st.image(green_ch, caption="분석용 흑백 이미지 (여기에 밝은 점이 보여야 합니다)")
