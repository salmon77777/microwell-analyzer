import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# --- 헬퍼 함수 ---
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity_thresh, convexity_thresh, gmo_criteria):
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 1. 윤곽선(Contour) 및 무게중심(Moments) 기반의 초정밀 스팟 검출
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, min_threshold, max_threshold, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_positive_wells = []
    margin = 5
    
    # 각 스팟의 기하학적 특성 필터링 및 정확한 중앙점 계산
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            convexity = area / hull_area

            # 사용자가 설정한 형태 기준을 통과한 경우에만
            if circularity >= circularity_thresh and convexity >= convexity_thresh:
                # Moments를 이용한 픽셀 단위의 완벽한 무게중심 계산
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    _, radius = cv2.minEnclosingCircle(cnt)
                    
                    # 가장자리 제외
                    if margin < cx < (img_w - margin) and margin < cy < (img_h - margin):
                        raw_positive_wells.append((cx, cy, radius))

    num_raw_positive = len(raw_positive_wells)
    
    grid_img = image_rgb.copy()
    result_img = image_rgb.copy()
    
    total_wells = 0
    matched_pos_count = 0
    matched_neg_count = 0
    ratio = 0.0
    is_gmo = False
    cols = 0
    rows = 0

    # 2. 좌표 투영 및 보간법을 통한 가상 격자 생성
    if num_raw_positive > 10:
        nearest_distances = []
        for p1 in raw_positive_wells:
            min_d = float('inf')
            for p2 in raw_positive_wells:
                if p1 == p2: continue
                d = calculate_distance((p1[0], p1[1]), (p2[0], p2[1]))
                if d < min_d: min_d = d
            if min_d != float('inf'):
                nearest_distances.append(min_d)
        pitch = np.median(nearest_distances)

        if pitch > 0:
            angles = []
            for i, p1 in enumerate(raw_positive_wells):
                for j, p2 in enumerate(raw_positive_wells):
                    if i == j: continue
                    d = calculate_distance(p1[:2], p2[:2])
                    if d < pitch * 1.5: 
                        dy = p2[1] - p1[1]
                        dx = p2[0] - p1[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        angle = angle % 90
                        if angle > 45: angle -= 90 
                        angles.append(angle)
            
            grid_angle = np.median(angles) if angles else 0.0

            center = np.mean([w[:2] for w in raw_positive_wells], axis=0)
            M_rot = cv2.getRotationMatrix2D(tuple(center), grid_angle, 1.0)
            
            pts = np.array([w[:2] for w in raw_positive_wells])
            ones = np.ones(shape=(len(pts), 1))
            pts_ones = np.hstack([pts, ones])
            rotated_pts = M_rot.dot(pts_ones.T).T
            
            xs = rotated_pts[:, 0]
            ys = rotated_pts[:, 1]

            def find_grid_lines(coords, pitch):
                sorted_coords = np.sort(coords)
                lines = []
                curr_group = [sorted_coords[0]]
                
                for c in sorted_coords[1:]:
                    if c - curr_group[-1] <= pitch * 0.5:
                        curr_group.append(c)
                    else:
                        lines.append(np.mean(curr_group))
                        curr_group = [c]
                lines.append(np.mean(curr_group))
                
                if len(lines) < 2: return lines
                interpolated = [lines[0]]
                for i in range(1, len(lines)):
                    gap = lines[i] - lines[i-1]
                    steps = int(round(gap / pitch))
                    if steps > 1:
                        step_size = gap / steps
                        for j in range(1, steps):
                            interpolated.append(lines[i-1] + j * step_size)
                    interpolated.append(lines[i])
                return interpolated

            grid_xs = find_grid_lines(xs, pitch)
            grid_ys = find_grid_lines(ys, pitch)
            
            cols = len(grid_xs)
            rows = len(grid_ys)
            total_wells = cols * rows

            ideal_grid = []
            for gx in grid_xs:
                for gy in grid_ys:
                    ideal_grid.append([gx, gy])
            ideal_grid = np.array(ideal_grid)
            
            M_rot_inv = cv2.getRotationMatrix2D(tuple(center), -grid_angle, 1.0)
            ones_grid = np.ones(shape=(len(ideal_grid), 1))
            grid_ones = np.hstack([ideal_grid, ones_grid])
            final_grid_points = M_rot_inv.dot(grid_ones.T).T

            # 3. 시각화 및 판정 로직
            avg_radius = int(round(np.mean([w[2] for w in raw_positive_wells])))

            for gx, gy in final_grid_points:
                # 미세한 오프셋 방지를 위해 round 적용
                gx, gy = int(round(gx)), int(round(gy))
                
                # 인식된 가상 격자를 노란색으로 일괄 표기
                cv2.circle(grid_img, (gx, gy), avg_radius, (255, 255, 0), 1)
                
                is_pos = False
                for px, py, pr in raw_positive_wells:
                    if calculate_distance((gx, gy), (px, py)) < (pitch * 0.5):
                        is_pos = True
                        break
                
                if is_pos:
                    matched_pos_count += 1
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 255, 0), 1)
                else:
                    matched_neg_count += 1
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 0, 0), 1)

            ratio = (matched_pos_count / total_wells * 100) if total_wells > 0 else 0
            is_gmo = ratio >= gmo_criteria

    return grid_img, result_img, total_wells, matched_pos_count, matched_neg_count, ratio, is_gmo, cols, rows

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2.5])

with col1:
    st.subheader("⚙️ 분석 설정")
    
    with st.expander("1️⃣ 판정 기준 및 밝기", expanded=True):
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)
        # 기본 임계값 조정 (새로운 윤곽선 엔진에 맞춤)
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 50)
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)

    with st.expander("2️⃣ 스팟 형태 필터링", expanded=True):
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=10, step=5)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=10, max_value=50000, value=200, step=10)
        circularity = st.slider("최소 원형도", 0.0, 1.0, 0.1, step=0.05)
        convexity = st.slider("최소 볼록성", 0.0, 1.0, 0.3, step=0.05)

    uploaded_file = st.file_uploader("✨ 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("초정밀 스팟 중심점을 계산하여 격자를 매핑 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            tab1, tab2 = st.tabs(["📌 1. 전체 Well 인식 확인", "📊 2. Positive 분석 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 매핑 결과")
                st.write("각 형광 스팟의 정확한 무게중심을 계산하여 중앙에 정렬된 바둑판 격자를 생성했습니다.")
                
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="노란색 얇은 테두리: 프로그램이 추출한 정중앙 좌표점", use_column_width=True)
                else:
                    st.warning("스팟이 충분히 검출되지 않아 전체 영역을 추정할 수 없습니다. '최소 밝기'를 낮춰보세요.")
                    
            with tab2:
                st.subheader("Positive / Negative 분류 결과")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("전체 Well", f"{total:,} 개")
                m2.metric("Positive (노란색)", f"{pos:,} 개")
                m3.metric("Negative (빨간색)", f"{neg:,} 개")
                m4.metric("Positive 비율", f"{ratio:.1f} %")
                
                if total > 0:
                    if is_gmo:
                        st.error(f"🚨 **판정 결과: GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                    else:
                        st.success(f"✅ **판정 결과: Non-GMO 입니다.** (기준: {gmo_criteria}%, 현재: {ratio:.1f}%)")
                    
                    st.image(result_img, caption="노란색: Positive, 빨간색: Negative (정확하게 겹쳐진 테두리)", use_column_width=True)
                else:
                    st.warning("분석할 결과가 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
