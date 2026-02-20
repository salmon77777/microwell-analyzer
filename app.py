import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria):
    # 1. 이미지를 3채널 RGB로 변환 후 Grayscale로 변환
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 2. SimpleBlobDetector 파라미터 설정
    params = cv2.SimpleBlobDetector_Params()

    # ★ 핵심 수정: 어두운 배경에서 '밝은 스팟(형광)'을 찾도록 명시적 설정
    params.filterByColor = True
    params.blobColor = 255 

    # 밝기 임계값 설정
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.thresholdStep = 5

    # 크기(면적) 필터링 설정
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # 원형도 필터링
    params.filterByCircularity = True
    params.minCircularity = circularity

    # 볼록성 필터링
    params.filterByConvexity = True
    params.minConvexity = convexity
    
    # 관성 비율 필터링
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # 감지기 생성 및 실행
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)

    # 3. Positive Well 필터링 (테두리 여백 제외)
    positive_wells = []
    margin = 5
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if margin < x < (img_w - margin) and margin < y < (img_h - margin):
            positive_wells.append((x, y, r))

    num_positive = len(positive_wells)

    # 4. 전체 및 Negative Well 추정
    num_negative = 0
    total_wells = 0
    ratio = 0.0

    if num_positive > 5:
        avg_radius = np.mean([w[2] for w in positive_wells])
        avg_spot_area = np.pi * (avg_radius ** 2)
        
        xs = [w[0] for w in positive_wells]
        ys = [w[1] for w in positive_wells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        bbox_area = (max_x - min_x + avg_radius*2) * (max_y - min_y + avg_radius*2)
        
        fill_factor = 1.8 
        if avg_spot_area > 0:
            estimated_total = bbox_area / (avg_spot_area * fill_factor)
            total_wells = int(round(estimated_total))
        
        total_wells = max(total_wells, num_positive)
        num_negative = total_wells - num_positive
        ratio = (num_positive / total_wells * 100) if total_wells > 0 else 0

    is_gmo = ratio >= gmo_criteria

    # 5. 결과 이미지 그리기 (노란색 테두리 원)
    result_img = image_rgb.copy()
    for x, y, r in positive_wells:
        cv2.circle(result_img, (x, y), r, (255, 255, 0), 3)

    return result_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("⚙️ 고급 분석 설정")
    
    with st.expander("1️⃣ 밝기 및 GMO 기준 설정", expanded=True):
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 30)
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)

    with st.expander("2️⃣ 스팟 형태 필터링 (중요)", expanded=True):
        # ★ 수정: 사용자가 더 작은 면적도 테스트할 수 있도록 min_value를 1로 변경
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=20, step=10)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=50, max_value=50000, value=5000, step=100)
        circularity = st.slider("최소 원형도 (Circularity)", 0.0, 1.0, 0.3, step=0.1)
        convexity = st.slider("최소 볼록성 (Convexity)", 0.0, 1.0, 0.5, step=0.1)

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
            m1.metric("추정 전체 Well", f"{total:,} 개")
            m2.metric("Positive (검출됨)", f"{pos:,} 개")
            m3.metric("Negative (추정됨)", f"{neg:,} 개")
            m4.metric("Positive 비율", f"{ratio:.1f} %")
            
            if total > 0:
                if is_gmo:
                    st.error(f"🚨 **판정 결과: GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                else:
                    st.success(f"✅ **판정 결과: Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.warning("⚠️ 검출된 Well이 없습니다. 좌측 설정을 조절해보세요.")
                
            st.image(result_img, caption="분석 결과 이미지 (노란색 테두리: 검출된 Positive Well)", use_column_width=True)
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
