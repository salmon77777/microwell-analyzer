import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="오브젝트 분석기", layout="wide")
st.title("🎯 오브젝트 기반 고속 Well 분석기")

# --- 사이드바: 픽셀 대신 '크기'와 '모양'으로 제어 ---
st.sidebar.header("📦 오브젝트 필터")
st.sidebar.info("픽셀을 훑지 않고 덩어리(Object)를 직접 찾습니다.")

min_area = st.sidebar.slider("Well 최소 면적", 10, 500, 50)
max_area = st.sidebar.slider("Well 최대 면적", 500, 5000, 1500)
circularity_threshold = st.sidebar.slider("원형도 (1에 가까울수록 정원)", 0.1, 1.0, 0.5)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (평균 G)", 0, 255, 70)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 선택하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 1. 이미지 로드 (속도를 위해 적당한 크기로 리사이즈)
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석 속도를 위해 가로 1200px 기준 최적화
    if w > 1200:
        new_w = 1200
        new_h = int(h * (1200 / w))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        h, w = new_h, new_w

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 이진화 (오브젝트 추출 준비)
    # 블러로 노이즈를 지우고 적응형 이진화로 덩어리 경계선을 땁니다.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # 3. 덩어리(Contour) 찾기 - 이 방식이 픽셀 루프보다 훨씬 빠릅니다.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res_img = img_rgb.copy()
    valid_wells = []
    pos_cnt = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 면적 필터링
        if min_area < area < max_area:
            # 원형도 계산 (진짜 Well인지 판별)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > circularity_threshold:
                # Well의 중심과 반지름 계산
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cx, cy, r = int(cx), int(cy), int(r)
                
                # 가장자리 잘린 것 제외
                if cx < 5 or cx > w-5 or cy < 5 or cy > h-5:
                    continue
                
                # 해당 오브젝트 영역의 평균 녹색값 추출
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = cv2.mean(img_bgr[:,:,1], mask=mask)[0]
                
                is_pos = mean_val > threshold_g
                if is_pos:
                    pos_cnt += 1
                
                # 시각화
                color = (0, 255, 0) if is_pos else (255, 255, 0)
                cv2.drawContours(res_img, [cnt], -1, color, 2)
                valid_wells.append((cx, cy))

    # 4. 결과 출력
    st.image(res_img, use_container_width=True)
    
    total = len(valid_wells)
    if total > 0:
        ratio = (pos_cnt / total * 100)
        st.markdown(f"### 분석 결과: {'GMO Positive' if ratio >= 50 else 'Non-GMO'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("탐지된 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
    else:
        st.warning("오브젝트를 찾지 못했습니다. '최소 면적'을 낮추거나 '원형도'를 조절해 보세요.")
        with st.expander("인식 과정 보기"):
            st.image(thresh, caption="이진화된 이미지 (하얀 덩어리가 Well입니다)")
