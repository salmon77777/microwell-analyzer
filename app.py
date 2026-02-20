import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
from scipy.spatial import cKDTree

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity_thresh, convexity_thresh, gmo_criteria):
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 1. 윤곽선 기반 실제 스팟(Positive) 찾기
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

    grid_img = image_rgb.copy()
    result_img = image_rgb.copy()
    
    total_wells = 0
    matched_pos_count = 0
    matched_neg_count = 0
    ratio = 0.0
    is_gmo = False
    cols = 0
    rows = 0

    if len(raw_positive_wells) > 10:
        pts = np.array([w[:2] for w in raw_positive_wells])
        radii = [w[2] for w in raw_positive_wells]
        avg_radius = int(round(np.mean(radii)))

        # 2. KD-Tree를 이용한 초고속 간격(Pitch) 파악
        tree = cKDTree(pts)
        distances, _ = tree.query(pts, k=2)
        rough_pitch = np.median(distances[:, 1])

        # 각도 계산
        pairs = tree.query_pairs(r=rough_pitch * 1.5)
        angles = []
        for i, j in pairs:
            p1 = pts[i]
            p2 = pts[j]
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            angle = np.degrees(np.arctan2(dy, dx))
            a_mod = angle % 90
            if a_mod > 45: a_mod -= 90
            angles.append(a_mod)
        grid_angle = np.median(angles) if angles else 0.0

        # 3. 배열 평탄화 회전
        rad = np.radians(-grid_angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        rot_pts = np.empty_like(pts)
        rot_pts[:, 0] = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        rot_pts[:, 1] = pts[:, 0] * sin_a + pts[:, 1] * cos_a

        # 4. 축 투영법
        def build_axes(coords, pitch):
            sorted_c = np.sort(coords)
            clusters = []
            curr = [sorted_c[0]]
            for c in sorted_c[1:]:
                if c - np.mean(curr) < pitch * 0.5:
                    curr.append(c)
                else:
                    clusters.append(np.mean(curr))
                    curr = [c]
            clusters.append(np.mean(curr))

            gaps = np.diff(clusters)
            valid_gaps = [g for g in gaps if g < pitch * 1.5]
            local_pitch = np.median(valid_gaps) if valid_gaps else pitch

            final_axis = [clusters[0]]
            for i in range(1, len(clusters)):
                gap = clusters[i] - clusters[i-1]
                steps = int(round(gap / local_pitch))
                if steps > 1:
                    step_size = gap / steps
                    for j in range(1, steps):
                        final_axis.append(clusters[i-1] + j * step_size)
                final_axis.append(clusters[i])
            return np.array(final_axis)

        final_cols = build_axes(rot_pts[:, 0], rough_pitch)
        final_rows = build_axes(rot_pts[:, 1], rough_pitch)

        cols = len(final_cols)
        rows = len(final_rows)
        total_wells = cols * rows

        # 5. ★ 추가됨: 엄격한 노이즈 필터링 및 정밀 매핑
        detected_grid = {}
        for idx, (rx, ry) in enumerate(rot_pts):
            c_idx = np.argmin(np.abs(final_cols - rx))
            r_idx = np.argmin(np.abs(final_rows - ry))
            
            # 가상 격자의 정중앙 좌표
            ideal_rx = final_cols[c_idx]
            ideal_ry = final_rows[r_idx]
            
            # 정중앙으로부터 얼마나 벗어나 있는지 오차 계산
            error_dist = math.hypot(rx - ideal_rx, ry - ideal_ry)
            
            # 허용 오차: 스팟 간격(pitch)의 40% 이내인 경우만 정상 스팟으로 인정 (그 이상은 웰 사이의 먼지/노이즈로 간주)
            if error_dist < rough_pitch * 0.40:
                # 이미 해당 자리에 다른 스팟이 매핑되었다면, 중앙에 더 가까운 진짜 스팟으로 덮어쓰기
                if (c_idx, r_idx) not in detected_grid:
                    detected_grid[(c_idx, r_idx)] = (pts[idx], error_dist)
                else:
                    if error_dist < detected_grid[(c_idx, r_idx)][1]:
                        detected_grid[(c_idx, r_idx)] = (pts[idx], error_dist)

        # 6. 복원 및 렌더링
        inv_rad = np.radians(grid_angle)
        inv_cos = np.cos(inv_rad)
        inv_sin = np.sin(inv_rad)

        for c in range(cols):
            for r in range(rows):
                if (c, r) in detected_grid:
                    # Positive: 검증된 실제 스팟
                    px, py = detected_grid[(c, r)][0] # 좌표만 추출
                    px, py = int(round(px)), int(round(py))
                    
                    cv2.circle(grid_img, (px, py), avg_radius, (0, 255, 255), 1)
                    cv2.circle(result_img, (px, py), avg_radius, (255, 255, 0), 1)
                    matched_pos_count += 1
                else:
                    # Negative: 스팟이 없거나 먼지만 있었던 빈 자리
                    rx = final_cols[c]
                    ry = final_rows[r]
                    px = rx * inv_cos - ry * inv_sin
                    py = rx * inv_sin + ry * inv_cos
                    px, py = int(round(px)), int(round(py))
                    
                    cv2.circle(grid_img, (px, py), avg_radius, (0, 255, 255), 1)
                    cv2.circle(result_img, (px, py), avg_radius, (255, 0, 0), 1)
                    matched_neg_count += 1

        ratio = (matched_pos_count / total_wells * 100) if total_wells > 0 else 0
        is_gmo = ratio >= gmo_criteria

    return grid_img, result_img, total_wells, matched_pos_count, matched_neg_count, ratio, is_gmo, cols, rows

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (초고속 Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2.5])

with col1:
    st.subheader("⚙️ 분석 설정")
    
    with st.expander("1️⃣ 판정 기준 및 밝기", expanded=True):
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 30)
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
        
        with st.spinner("노이즈를 필터링하며 맵핑 중입니다... (약 0.1초 소요)"):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            tab1, tab2 = st.tabs(["📌 1. 왜곡 보정 가상 격자", "📊 2. 최종 분석 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 계산 확인")
                st.write("웰 사이의 먼지나 노이즈를 식별하여 제거하고, 완벽한 바둑판 배열만 표시합니다.")
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="청록색: 노이즈가 제거된 정밀 가상 격자점", use_column_width=True)
                else:
                    st.warning("스팟이 충분히 검출되지 않았습니다. 밝기나 면적 설정을 조절해주세요.")
                    
            with tab2:
                st.subheader("Positive / Negative 최종 분류 결과")
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
                    
                    st.image(result_img, caption="노란색: 일치된 Positive, 빨간색: 비어있는 Negative", use_column_width=True)
                else:
                    st.warning("분석할 결과가 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
