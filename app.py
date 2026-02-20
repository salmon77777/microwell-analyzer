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

    # 1. 윤곽선 및 무게중심 계산으로 실제 스팟(Positive) 찾기
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, min_threshold, max_threshold, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_positive_wells = []
    margin = 5
    
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

            if circularity >= circularity_thresh and convexity >= convexity_thresh:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    _, radius = cv2.minEnclosingCircle(cnt)
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

    # 2. X축, Y축 독립 간격 계산 및 가상 격자 생성
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
        rough_pitch = np.median(nearest_distances)

        if rough_pitch > 0:
            # 각도 보정
            angles = []
            for i, p1 in enumerate(raw_positive_wells):
                for j, p2 in enumerate(raw_positive_wells):
                    if i == j: continue
                    d = calculate_distance(p1[:2], p2[:2])
                    if d < rough_pitch * 1.5: 
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

            # 좌표 클러스터링을 통해 겹치는 선 찾기
            def cluster_coords(coords, min_gap):
                sorted_c = np.sort(coords)
                clusters = []
                curr = [sorted_c[0]]
                for c in sorted_c[1:]:
                    if c - np.mean(curr) < min_gap:
                        curr.append(c)
                    else:
                        clusters.append(np.mean(curr))
                        curr = [c]
                clusters.append(np.mean(curr))
                return clusters

            x_clusters = cluster_coords(xs, rough_pitch * 0.5)
            y_clusters = cluster_coords(ys, rough_pitch * 0.5)

            # X, Y 각각의 정밀 Pitch 계산
            def get_precise_pitch(clusters, fallback):
                gaps = np.diff(clusters)
                valid_gaps = [g for g in gaps if g < fallback * 1.5]
                return np.median(valid_gaps) if valid_gaps else fallback

            pitch_x = get_precise_pitch(x_clusters, rough_pitch)
            pitch_y = get_precise_pitch(y_clusters, rough_pitch)

            # 누락된 선 보간
            def interpolate_lines(clusters, pitch):
                if len(clusters) < 2: return clusters
                lines = [clusters[0]]
                for i in range(1, len(clusters)):
                    gap = clusters[i] - clusters[i-1]
                    steps = int(round(gap / pitch))
                    if steps > 1:
                        step_size = gap / steps
                        for j in range(1, steps):
                            lines.append(clusters[i-1] + j * step_size)
                    lines.append(clusters[i])
                return lines

            grid_xs = interpolate_lines(x_clusters, pitch_x)
            grid_ys = interpolate_lines(y_clusters, pitch_y)
            
            cols = len(grid_xs)
            rows = len(grid_ys)
            total_wells = cols * rows

            # 보간된 1D 선들을 2D 격자점으로 조합
            ideal_grid = []
            for gx in grid_xs:
                for gy in grid_ys:
                    ideal_grid.append([gx, gy])
            ideal_grid = np.array(ideal_grid)
            
            # 원래 각도로 회전 복구
            M_rot_inv = cv2.getRotationMatrix2D(tuple(center), -grid_angle, 1.0)
            ones_grid = np.ones(shape=(len(ideal_grid), 1))
            grid_ones = np.hstack([ideal_grid, ones_grid])
            final_grid_points = M_rot_inv.dot(grid_ones.T).T

            # 3. 자석 스냅(Magnetic Snapping) 시각화 및 판정 로직
            avg_radius = int(round(np.mean([w[2] for w in raw_positive_wells])))
            used_positives = set() # 중복 인식 방지

            for gx, gy in final_grid_points:
                gx, gy = int(round(gx)), int(round(gy))
                
                # Tab 1: 순수 가상 격자 위치 표기 (청록색)
                cv2.circle(grid_img, (gx, gy), avg_radius, (0, 255, 255), 1)
                
                # 현재 가상 격자와 가장 가까운 실제 스팟 찾기
                best_pos_idx = -1
                min_dist = rough_pitch * 0.45 # 자석처럼 끌어당길 허용 반경
                
                for i, (px, py, pr) in enumerate(raw_positive_wells):
                    if i in used_positives: continue
                    dist = calculate_distance((gx, gy), (px, py))
                    if dist < min_dist:
                        min_dist = dist
                        best_pos_idx = i
                
                if best_pos_idx != -1:
                    # [매칭 성공 - Positive] 가상 좌표가 아닌 실제 스팟 좌표에 원을 그림!
                    px, py, pr = raw_positive_wells[best_pos_idx]
                    matched_pos_count += 1
                    used_positives.add(best_pos_idx)
                    # Tab 2: 완벽하게 일치하는 노란색 원
                    cv2.circle(result_img, (int(px), int(py)), avg_radius, (255, 255, 0), 1)
                else:
                    # [매칭 실패 - Negative] 형광이 없으므로 가상 좌표에 빨간색 원을 그림
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
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 30, help="형광 신호가 잘 안 잡히면 이 값을 낮추세요.")
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)

    with st.expander("2️⃣ 스팟 형태 필터링", expanded=True):
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=15, step=5)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=10, max_value=50000, value=200, step=10)
        circularity = st.slider("최소 원형도", 0.0, 1.0, 0.1, step=0.05)
        convexity = st.slider("최소 볼록성", 0.0, 1.0, 0.3, step=0.05)

    uploaded_file = st.file_uploader("✨ 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("렌즈 왜곡을 보정하고 스냅핑을 적용 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            tab1, tab2 = st.tabs(["📌 1. 가상 격자 계산", "📊 2. 최종 스냅핑 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 계산 확인")
                st.write("청록색 원은 프로그램이 렌즈 왜곡을 보정하여 1차로 추정한 기준 좌표입니다.")
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="청록색: 보간법으로 생성된 1차 가상 기준점", use_column_width=True)
                else:
                    st.warning("스팟이 충분히 검출되지 않아 전체 영역을 추정할 수 없습니다.")
                    
            with tab2:
                st.subheader("Positive / Negative 최종 분류 결과")
                st.write("가상 기준점 근처의 형광 스팟을 감지하면 **실제 스팟의 중심으로 원을 끌어당겨(Snap)** 오차 없이 표시합니다.")
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
                    
                    st.image(result_img, caption="노란색: 정확히 일치된 Positive, 빨간색: 비어있는 Negative", use_column_width=True)
                else:
                    st.warning("분석할 결과가 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
