import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# --- 헬퍼 함수 ---
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria):
    # 1. 이미지 변환
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 2. SimpleBlobDetector 설정 (밝은 스팟 찾기)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255 
    
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.thresholdStep = 5
    
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    
    params.filterByCircularity = True
    params.minCircularity = circularity
    
    params.filterByConvexity = True
    params.minConvexity = convexity
    
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)

    # 3. Positive Well 필터링
    positive_wells = []
    margin = 2 # 가장자리 마진 최소화
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if margin < x < (img_w - margin) and margin < y < (img_h - margin):
            positive_wells.append((x, y, r))

    num_positive = len(positive_wells)
    total_wells = 0
    num_negative = 0
    ratio = 0.0

    # 4. 회전 각도를 고려한 자동 전체 개수 추정 알고리즘
    if num_positive > 10:
        # 4-1. 검출된 스팟 간의 가장 가까운 거리(Pitch) 계산
        nearest_distances = []
        for i in range(num_positive):
            p1 = positive_wells[i]
            min_d = float('inf')
            for j in range(num_positive):
                if i == j: continue
                p2 = positive_wells[j]
                d = calculate_distance((p1[0], p1[1]), (p2[0], p2[1]))
                if d < min_d: min_d = d
            nearest_distances.append(min_d)
        
        pitch = np.median(nearest_distances)

        # 4-2. 회전된 바운딩 박스(minAreaRect)를 이용한 격자 크기 추정
        if pitch > 0:
            points = np.array([[w[0], w[1]] for w in positive_wells], dtype=np.float32)
            # 이미지의 미세한 회전을 보정하여 스팟들을 감싸는 가장 작은 직사각형을 찾음
            rect = cv2.minAreaRect(points)
            rect_width, rect_height = rect[1]
            
            # 폭과 높이를 간격으로 나누어 대략적인 행/열 개수 추정
            estimated_cols = round(rect_width / pitch) + 1
            estimated_rows = round(rect_height / pitch) + 1
            
            total_wells = int(estimated_cols * estimated_rows)
            total_wells = max(total_wells, num_positive) # 최소한 검출된 것보단 많아야 함
            
            num_negative = total_wells - num_positive
            ratio = (num_positive / total_wells * 100)

    is_gmo = ratio >= gmo_criteria

    # 5. 결과 이미지 그리기 (가장 얇은 테두리)
    result_img = image_rgb.copy()
    for x, y, r in positive_wells:
        # 두께 1로 아주 얇은 테두리 적용
        cv2.circle(result_img, (x, y), r, (255, 255, 0), 1)

    return result_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("⚙️ 자동 분석 설정")
    
    with st.expander("1️⃣ 판정 기준 설정", expanded=True):
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)

    with st.expander("2️⃣ 밝기 설정", expanded=True):
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 26)
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)

    with st.expander("3️⃣ 스팟 형태 필터링", expanded=True):
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=10, step=5)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=10, max_value=50000, value=50, step=10)
        circularity = st.slider("최소 원형도 (Circularity)", 0.0, 1.0, 0.1, step=0.05)
        convexity = st.slider("최소 볼록성 (Convexity)", 0.0, 1.0, 0.3, step=0.05)

    uploaded_file = st.file_uploader("✨ 분석할 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("이미지를 분석 중입니다..."):
            result_img, total, pos, neg, ratio, is_gmo = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            st.subheader("📊 자동 분석 결과 리포트")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("전체 Well (자동추정)", f"{total:,} 개")
            m2.metric("Positive (검출됨)", f"{pos:,} 개")
            m3.metric("Negative (계산됨)", f"{neg:,} 개")
            m4.metric("Positive 비율", f"{ratio:.1f} %")
            
            if total > 0:
                if is_gmo:
                    st.error(f"🚨 **판정 결과: GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                else:
                    st.success(f"✅ **판정 결과: Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.warning("⚠️ 검출된 Well이 너무 적어 전체 개수를 추정할 수 없습니다.")
                
            st.image(result_img, caption="분석 결과 (노란색 얇은 테두리: 검출된 Positive Well)", use_column_width=True)
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
