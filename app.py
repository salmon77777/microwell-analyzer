import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# --- 새로운 핵심 함수: Blob Detection 기반 분석 ---
def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria):
    # 1. 이미지 변환 (RGB -> Grayscale)
    image_rgb = np.array(image_pil.convert('RGB'))
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 2. SimpleBlobDetector 파라미터 설정 (강력한 스팟 검출 엔진)
    params = cv2.SimpleBlobDetector_Params()

    # 임계값 설정 (밝기 기반 필터링)
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.thresholdStep = 5

    # 크기(면적) 필터링 설정
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # 원형도 필터링 (찌그러진 정도)
    params.filterByCircularity = True
    params.minCircularity = circularity

    # 볼록성 필터링 (오목한 부분 확인)
    params.filterByConvexity = True
    params.minConvexity = convexity
    
    # 관성 비율 필터링 (길쭉한 정도) - 기본값 사용
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # 감지기 생성 및 실행
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)

    # 3. Positive Well 필터링 (테두리 근처 제외)
    positive_wells = []
    margin = 5 # 테두리 여백
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if margin < x < (img_w - margin) and margin < y < (img_h - margin):
            positive_wells.append((x, y, r))

    num_positive = len(positive_wells)

    # 4. 통계적 방법을 통한 전체 및 Negative Well 추정
    num_negative = 0
    total_wells = 0
    ratio = 0.0

    if num_positive > 5: # 최소한의 샘플이 확보되었을 때 추정 수행
        # 검출된 스팟들의 평균 면적 및 반지름 계산
        avg_radius = np.mean([w[2] for w in positive_wells])
        avg_spot_area = np.pi * (avg_radius ** 2)
        
        # 스팟들이 분포하는 전체 영역(Bounding Box) 계산
        xs = [w[0] for w in positive_wells]
        ys = [w[1] for w in positive_wells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Bounding Box의 전체 면적 계산 (약간의 여백 포함)
        bbox_area = (max_x - min_x + avg_radius*2) * (max_y - min_y + avg_radius*2)
        
        # 전체 Well 개수 추정 (전체 면적 / (스팟 면적 + 간격 고려))
        # * 간격 보정 계수(1.5 ~ 2.0)를 사용하여 스팟 사이의 빈 공간을 반영
        fill_factor = 1.8 # 경험적 보정 계수 (조절 가능)
        estimated_total = bbox_area / (avg_spot_area * fill_factor)
        
        total_wells = int(round(estimated_total))
        # 추정된 전체 개수가 실제 Positive 개수보다 적을 경우 보정
        total_wells = max(total_wells, num_positive)
        
        num_negative = total_wells - num_positive
        ratio = (num_positive / total_wells * 100)

    is_gmo = ratio >= gmo_criteria

    # 5. 결과 이미지 그리기 (요청하신 대로 테두리만 표시)
    result_img = image_rgb.copy()
    for x, y, r in positive_wells:
        # 노란색(RGB: 255, 255, 0), 두께 3의 테두리 원 그리기
        cv2.circle(result_img, (x, y), r, (255, 255, 0), 3)

    return result_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("⚙️ 고급 분석 설정")
    st.info("새로운 알고리즘이 적용되었습니다. 아래 설정들을 조절하여 최적의 검출 결과를 찾아보세요.")
    
    with st.expander("1️⃣ 밝기 및 GMO 기준 설정", expanded=True):
        min_threshold = st.slider("최소 밝기 임계값 (Min Threshold)", 0, 255, 50, help="이 값보다 어두운 영역은 무시합니다. 낮을수록 어두운 스팟도 검출합니다.")
        max_threshold = st.slider("최대 밝기 임계값 (Max Threshold)", 0, 255, 255, help="이 값보다 밝은 영역은 무시합니다. 보통 최대로 설정합니다.")
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50, help="Positive 비율이 이 수치 이상이면 GMO로 판정합니다.")

    with st.expander("2️⃣ 스팟 형태 필터링 (중요)", expanded=True):
        st.write("검출하려는 스팟의 크기와 모양을 정의합니다.")
        min_area = st.number_input("최소 면적 (픽셀)", min_value=10, max_value=5000, value=100, step=50, help="이보다 작은 노이즈는 제거합니다.")
        max_area = st.number_input("최대 면적 (픽셀)", min_value=100, max_value=50000, value=5000, step=100, help="이보다 큰 뭉친 영역은 제외합니다.")
        circularity = st.slider("최소 원형도 (Circularity)", 0.1, 1.0, 0.5, step=0.1, help="1.0에 가까울수록 완벽한 원만 검출합니다. 찌그러진 모양을 검출하려면 낮추세요.")
        convexity = st.slider("최소 볼록성 (Convexity)", 0.1, 1.0, 0.7, step=0.1, help="1.0에 가까울수록 매끈한 형태만 검출합니다. 울퉁불퉁하면 낮추세요.")

    uploaded_file = st.file_uploader("✨ 분석할 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("🔥 새로운 엔진으로 이미지를 분석 중입니다... 잠시만 기다려주세요."):
            # 분석 함수 실행
            result_img, total, pos, neg, ratio, is_gmo = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            st.subheader("📊 분석 결과 리포트")
            
            # 주요 지표 메트릭 표시
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("추정 전체 Well", f"{total:,} 개", help="통계적으로 추정된 전체 Well의 대략적인 개수입니다.")
            m2.metric("Positive (검출됨)", f"{pos:,} 개", help="확실하게 형광 신호가 검출된 Well 개수입니다.")
            m3.metric("Negative (추정됨)", f"{neg:,} 개", help="전체에서 Positive를 뺀, 신호가 없는 것으로 추정되는 개수입니다.")
            m4.metric("Positive 비율", f"{ratio:.1f} %")
            
            # GMO 판정 결과 표시
            if total > 0:
                if is_gmo:
                    st.error(f"🚨 **판정 결과: GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                else:
                    st.success(f"✅ **판정 결과: Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.warning("⚠️ 검출된 Well이 없습니다. 좌측의 '최소 밝기'를 낮추거나 '면적/형태 필터링' 설정을 조절해보세요.")
                
            # 결과 이미지 출력
            st.image(result_img, caption="분석 결과 이미지 (노란색 테두리: 검출된 Positive Well)", use_column_width=True)
            st.caption("💡 참고: 정확도를 위해 확실하게 검출된 Positive Well에만 노란색 테두리가 표시됩니다. Negative Well은 위치가 불확실하여 이미지에 표시하지 않고 수치로만 제공됩니다.")

    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
