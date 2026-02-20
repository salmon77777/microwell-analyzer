import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_image(image, threshold_val, min_r, max_r, gmo_criteria):
    # 1. 이미지를 OpenCV 형식(RGB)으로 변환
    img_rgb = np.array(image)
    img_h, img_w = img_rgb.shape[:2]
    
    # 2. 형광 신호 추출을 위해 Grayscale 변환 및 노이즈 제거
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding (민감도 조절에 따라 빛나는 부분만 분리)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    
    # 4. 윤곽선(Contour) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    positive_wells = []
    
    # 5. Positive Well 필터링 (크기, 테두리 제외)
    for cnt in contours:
        # 원형 근사화
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)
        
        # 조건 1: 사용자가 설정한 반지름 크기 내에 들어오는가?
        if min_r <= radius <= max_r:
            # 조건 2: 사진 테두리에 걸쳐 있는 불완전한 스팟은 제외 (요구사항 2)
            margin = radius + 5
            if margin < x < (img_w - margin) and margin < y < (img_h - margin):
                positive_wells.append((x, y, radius))
                
    # 6. 전체 Well 유추 및 Negative Well 찾기 (요구사항 3)
    negative_wells = []
    if len(positive_wells) >= 2:
        # 스팟 간의 평균 최소 거리(간격) 계산
        distances = []
        for i in range(len(positive_wells)):
            min_dist = float('inf')
            for j in range(len(positive_wells)):
                if i != j:
                    dist = calculate_distance(positive_wells[i][:2], positive_wells[j][:2])
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)
        
        avg_pitch = np.median(distances) # 중간값을 사용하여 튀는 값 방지
        
        # 검출된 Positive Well들의 전체 영역(Bounding Box) 파악
        xs = [w[0] for w in positive_wells]
        ys = [w[1] for w in positive_wells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 가상의 격자를 생성하여 비어있는 곳(Negative) 찾기
        avg_radius = int(np.mean([w[2] for w in positive_wells]))
        
        # 평균 간격(avg_pitch)을 바탕으로 가상 그리드 탐색
        for grid_y in np.arange(min_y, max_y + avg_pitch/2, avg_pitch):
            for grid_x in np.arange(min_x, max_x + avg_pitch/2, avg_pitch):
                is_positive = False
                # 해당 가상 그리드 위치 근처에 Positive 스팟이 있는지 확인
                for px, py, pr in positive_wells:
                    if calculate_distance((grid_x, grid_y), (px, py)) < (avg_pitch * 0.6):
                        is_positive = True
                        break
                
                if not is_positive:
                    negative_wells.append((int(grid_x), int(grid_y), avg_radius))

    # 7. 결과 이미지에 원 그리기 (요구사항 7 - 내부는 비우고 테두리만)
    output_img = img_rgb.copy()
    
    # Positive는 노란색 (RGB: 255, 255, 0), 두께 2
    for x, y, r in positive_wells:
        cv2.circle(output_img, (x, y), r, (255, 255, 0), 2)
        
    # Negative는 빨간색 (RGB: 255, 0, 0), 두께 2
    for x, y, r in negative_wells:
        cv2.circle(output_img, (x, y), r, (255, 0, 0), 2)

    # 8. 결과 수치 계산 (요구사항 5 & 6)
    num_positive = len(positive_wells)
    num_negative = len(negative_wells)
    total_wells = num_positive + num_negative
    ratio = (num_positive / total_wells * 100) if total_wells > 0 else 0
    is_gmo = ratio >= gmo_criteria

    return output_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.title("🦠 Microwell 형광 자동 분석기")
st.write("형광 결과 사진을 업로드하면 자동으로 스팟을 인식하고 비율을 계산합니다.")

# 사이드바 설정 (요구사항 5)
st.sidebar.header("⚙️ 분석 설정")
st.sidebar.write("결과가 잘 나오지 않는다면 아래 수치들을 조절해 보세요.")
threshold_val = st.sidebar.slider("인식 감도 (Threshold)", 0, 255, 100, help="값이 낮을수록 어두운 신호도 잡아냅니다.")
min_r = st.sidebar.slider("최소 Well 반지름", 1, 50, 10)
max_r = st.sidebar.slider("최대 Well 반지름", 10, 150, 40)
gmo_criteria = st.sidebar.slider("GMO 판정 기준 (%)", 1, 100, 50, help="Positive 비율이 이 수치 이상이면 GMO로 판정합니다.")

uploaded_file = st.file_uploader("형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_column_width=True)
    
    with st.spinner("이미지 분석 중..."):
        result_img, total, pos, neg, ratio, is_gmo = process_image(image, threshold_val, min_r, max_r, gmo_criteria)
        
        st.subheader("📊 분석 결과")
        
        # 지표 출력 (요구사항 5)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체 Well", f"{total} 개")
        col2.metric("Positive (노란색)", f"{pos} 개")
        col3.metric("Negative (빨간색)", f"{neg} 개")
        col4.metric("Positive 비율", f"{ratio:.1f} %")
        
        # GMO 판정 결과 (요구사항 6)
        if total > 0:
            if is_gmo:
                st.error(f"🚨 판정 결과: **GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.success(f"✅ 판정 결과: **Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
        else:
            st.warning("인식된 Well이 없습니다. 좌측의 설정(크기, 감도)을 조절해 보세요.")
            
        # 결과 이미지 출력 (요구사항 7)
        st.image(result_img, caption="분석 완료 이미지 (노란색: 인식됨, 빨간색: 비어있음)", use_column_width=True)
