import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_image(image_pil, threshold_val, min_r, max_r, gmo_criteria):
    # [수정됨] 1. 이미지를 무조건 3채널 RGB로 변환 (투명도 채널 제거)
    image_rgb_pil = image_pil.convert('RGB')
    img_rgb = np.array(image_rgb_pil)
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
            # 조건 2: 사진 테두리에 걸쳐 있는 불완전한 스팟은 제외
            margin = radius + 5
            if margin < x < (img_w - margin) and margin < y < (img_h - margin):
                positive_wells.append((x, y, radius))
                
    # 6. 전체 Well 유추 및 Negative Well 찾기
    negative_wells = []
    # 평균 반지름 계산 (Positive가 하나도 없을 경우를 대비한 예외 처리)
    avg_radius = int(np.mean([w[2] for w in positive_wells])) if positive_wells else (min_r + max_r) // 2
    
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
        
        if avg_pitch > 0: # 간격이 계산된 경우에만 실행
            # 검출된 Positive Well들의 전체 영역(Bounding Box) 파악
            xs = [w[0] for w in positive_wells]
            ys = [w[1] for w in positive_wells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # 평균 간격(avg_pitch)을 바탕으로 가상 그리드 탐색
            # 범위에 약간의 여유(margin)를 주어 테두리 근처의 Negative도 찾도록 함
            grid_margin = avg_pitch * 0.5
            for grid_y in np.arange(min_y - grid_margin, max_y + grid_margin, avg_pitch):
                for grid_x in np.arange(min_x - grid_margin, max_x + grid_margin, avg_pitch):
                    # 이미지 범위를 벗어나는 좌표는 제외
                    if not (avg_radius < grid_x < img_w - avg_radius and avg_radius < grid_y < img_h - avg_radius):
                        continue

                    is_positive = False
                    # 해당 가상 그리드 위치 근처에 Positive 스팟이 있는지 확인
                    for px, py, pr in positive_wells:
                        if calculate_distance((grid_x, grid_y), (px, py)) < (avg_pitch * 0.5): # 인식 범위를 간격의 반으로 설정
                            is_positive = True
                            break
                    
                    if not is_positive:
                        negative_wells.append((int(grid_x), int(grid_y), avg_radius))

    # 7. 결과 이미지에 원 그리기
    output_img = img_rgb.copy()
    
    # [수정됨] Positive는 노란색 (RGB: 255, 255, 0), 두께를 3으로 증가
    for x, y, r in positive_wells:
        cv2.circle(output_img, (x, y), r, (255, 255, 0), 3)
        
    # [수정됨] Negative는 빨간색 (RGB: 255, 0, 0), 두께를 3으로 증가
    for x, y, r in negative_wells:
        cv2.circle(output_img, (x, y), r, (255, 0, 0), 3)

    # 8. 결과 수치 계산
    num_positive = len(positive_wells)
    num_negative = len(negative_wells)
    total_wells = num_positive + num_negative
    ratio = (num_positive / total_wells * 100) if total_wells > 0 else 0
    is_gmo = ratio >= gmo_criteria

    return output_img, total_wells, num_positive, num_negative, ratio, is_gmo

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide") # 넓은 레이아웃 사용
st.title("🦠 Microwell 형광 자동 분석기")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ 분석 설정")
    st.write("결과가 잘 나오지 않는다면 아래 수치들을 조절해 보세요.")
    threshold_val = st.slider("인식 감도 (Threshold)", 0, 255, 100, help="값이 낮을수록 어두운 신호도 잡아냅니다.")
    min_r = st.slider("최소 Well 반지름", 1, 50, 10)
    max_r = st.slider("최대 Well 반지름", 10, 150, 40)
    gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50, help="Positive 비율이 이 수치 이상이면 GMO로 판정합니다.")

    uploaded_file = st.file_uploader("형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("이미지 분석 중..."):
            result_img, total, pos, neg, ratio, is_gmo = process_image(image_pil, threshold_val, min_r, max_r, gmo_criteria)
            
            st.header("📊 분석 결과")
            
            # 지표 출력
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("전체 Well", f"{total} 개")
            m2.metric("Positive (노란색)", f"{pos} 개")
            m3.metric("Negative (빨간색)", f"{neg} 개")
            m4.metric("Positive 비율", f"{ratio:.1f} %")
            
            # GMO 판정 결과
            if total > 0:
                if is_gmo:
                    st.error(f"🚨 판정 결과: **GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                else:
                    st.success(f"✅ 판정 결과: **Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
            else:
                st.warning("인식된 Well이 없습니다. 좌측의 설정(크기, 감도)을 조절해 보세요.")
                
            # 결과 이미지 출력
            st.image(result_img, caption="분석 완료 이미지 (노란색: 인식됨, 빨간색: 비어있음)", use_column_width=True)
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드해주세요.")
