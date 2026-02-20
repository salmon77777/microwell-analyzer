import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# --- 헬퍼 함수: 두 점 사이의 거리 계산 ---
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- 핵심 분석 함수 ---
def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria):
    # 1. 이미지 변환
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 2. SimpleBlobDetector 설정
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255 # 밝은 스팟 찾기
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
    margin = 5
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if margin < x < (img_w - margin) and margin < y < (img_h - margin):
            positive_wells.append((x, y, r))

    num_positive = len(positive_wells)

    # 4. [수정됨] 격자 간격 기반 전체 및 Negative Well 추정
    num_negative = 0
    total_wells = 0
    ratio = 0.0

    # 통계적 추정을 위해 최소한의 샘플 개수 필요 (예: 15개 이상)
    if num_positive > 15:
        # 4-1. 가장 가까운 이웃 간의 거리(Pitch) 계산
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
        
        # 중간값(median)을 사용하여 대표적인 격자 간격(pitch) 결정
        pitch = np.median(nearest_distances)

        # 4-2. 간격을 바탕으로 전체 그리드 크기 추정
        if pitch > 0:
            xs = [w[0] for w in positive_wells]
            ys = [w[1] for w in positive_wells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 검출된 영역의 가로/세로 폭 계산
            width = max_x - min_x
            height = max_y - min_y

            # 폭을 간격으로 나누어 대략적인 행/열 개수 추정 (반올림)
            estimated_cols = round(width / pitch) + 1
            estimated_rows = round(height / pitch) + 1
            
            # 전체 개수 추정
            estimated_total = int(estimated_cols * estimated_rows)
            
            # 안전장치: 추정된 전체 개수가 실제 검출된 Positive보다 적을 순 없음
            total_wells = max(estimated_total, num_positive)
            
            num_negative = total_wells - num_positive
            ratio = (num_positive / total_wells * 100)

    is_gmo = ratio >= gmo_criteria

    # 5. [수정됨] 결과 이미지 그리기 (얇은 테두리만 표시)
    result_img = image_rgb.copy()
    for x, y, r in positive_wells:
        # 두께(thickness)를 2로 설정하여 얇은 테두리만 그림
        cv2.circle(result_img, (x, y), r, (255, 255, 0), 2)

    return result_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("⚙️ 고급 분석 설정")
    
    with st.expander("1️⃣ 밝기 및 GMO 기준 설정", expanded=True):
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 60, help="기본값 60 내외에서 조절해보세요.")
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)

    with st.expander("2️⃣ 스팟 형태 필터링 (중요)", expanded=True):
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=30, step=10)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=50, max_value=50000, value=200, step=50)
        circularity = st.slider("최소 원형도 (Circularity)", 0.0, 1.0, 0.2, step=0.1, help="낮을수록 찌그러진 원도 검출합니다.")
        convexity = st.slider("최소 볼록성 (Convexity)", 0.0, 1.0, 0.3, step=0.1, help="낮을수록 울퉁불퉁한 형태도 검출합니다.")

    uploaded_file = st.file_uploader("✨ 분석할 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("이미지를 분석 중입니다..."):
            result_img, total, pos, neg, ratio, is_gmo = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            st.subheader("📊 분석 결과 리포트")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("추정 전체 Well", f"{total:,} 개", help="검출된 스팟들의 간격을 기반으로 추정한 대략적인 전체 개수입니다.")
            m2.metric("Positive (검출됨)", f"{pos:,} 개")
            m3.metric("Negative (추정됨)", f"{neg:,} 개")
            m4.metric("Positive 비율", f"{ratio:.1f} %")
            
            if total > 0:
                if is_gmo:
                    st.error(f"🚨 **판정 결과: GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                else:
                    st.success(f"✅ **판정 결과: Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.warning("⚠️ 검출된 Well이 너무 적어 통계적 추정이 불가능합니다. 설정을 조절하여 더 많은 스팟을 검출해보세요.")
                
            st.image(result_img, caption="분석 결과 이미지 (노란색 얇은 테두리: 검출된 Positive Well)", use_column_width=True)
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
