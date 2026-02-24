import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import collections
from scipy.spatial import cKDTree

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity_thresh, convexity_thresh, gmo_criteria, signal_thresh, min_pitch):
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    green_channel = image_rgb[:, :, 1]
    
    img_h, img_w = gray_img.shape[:2]

    # 1. 격자 기준점 찾기 (모양 필터링 적용)
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

    # 2. 강력한 중복/노이즈 제거
    if len(raw_positive_wells) > 10:
        raw_positive_wells.sort(key=lambda x: x[2], reverse=True)
        filtered_wells = []
        for w in raw_positive_wells:
            is_dup = False
            for fw in filtered_wells:
                if calculate_distance(w[:2], fw[:2]) < min_pitch * 0.8:
                    is_dup = True
                    break
            if not is_dup:
                filtered_wells.append(w)
                
        raw_positive_wells = filtered_wells

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

        # 3. KD-Tree를 이용한 진짜 간격(Pitch) 파악
        tree = cKDTree(pts)
        
        distances, _ = tree.query(pts, k=min(6, len(pts)))
        valid_pitches = []
        for i in range(len(pts)):
            for k in range(1, distances.shape[1]):
                if distances[i, k] >= min_pitch * 0.8:
                    valid_pitches.append(distances[i, k])
                    break
                    
        rough_pitch = np.median(valid_pitches) if valid_pitches else min_pitch
        
        # ★ 핵심 추가: 웰 크기를 기반으로 한 "자동 안전 마진" 계산 (파라미터 불필요)
        auto_margin = rough_pitch * 0.7

        pairs = tree.query_pairs(r=rough_pitch * 1.5)
        angles = []
        for i, j in pairs:
            dy, dx = pts[j][1] - pts[i][1], pts[j][0] - pts[i][0]
            angle = np.degrees(np.arctan2(dy, dx))
            a_mod = angle % 90
            if a_mod > 45: a_mod -= 90
            angles.append(a_mod)
        grid_angle = np.median(angles) if angles else 0.0

        right_vecs, down_vecs = [], []
        for i, j in pairs:
            dy, dx = pts[j][1] - pts[i][1], pts[j][0] - pts[i][0]
            for v_dx, v_dy in [(dx, dy), (-dx, -dy)]:
                a = np.degrees(np.arctan2(v_dy, v_dx))
                r_a = (a - grid_angle + 360) % 360
                if r_a > 180: r_a -= 360
                if math.hypot(v_dx, v_dy) < rough_pitch * 0.5: continue
                if -45 <= r_a <= 45: right_vecs.append((v_dx, v_dy))
                elif 45 < r_a <= 135: down_vecs.append((v_dx, v_dy))

        vec_right = np.median(right_vecs, axis=0) if right_vecs else (rough_pitch, 0)
        vec_down = np.median(down_vecs, axis=0) if down_vecs else (0, rough_pitch)

        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        bound_margin = rough_pitch * 5

        _, start_idx = tree.query([img_w / 2, img_h / 2])
        start_x, start_y = pts[start_idx]

        visited = set([(0, 0)])
        used_spots = set([start_idx])
        grid_dict = {(0, 0): (start_x, start_y)}
        queue = collections.deque([(0, 0, start_x, start_y)])

        dirs = [(1, 0, vec_right[0], vec_right[1]), (-1, 0, -vec_right[0], -vec_right[1]),
                (0, 1, vec_down[0], vec_down[1]), (0, -1, -vec_down[0], -vec_down[1])]

        while queue:
            c, r, cx, cy = queue.popleft()
            for dc, dr, vx, vy in dirs:
                nc, nr = c + dc, r + dr
                if (nc, nr) in visited: continue
                
                ex, ey = cx + vx, cy + vy
                
                # ★ 변경: 0이 아니라 자동 안전 마진(auto_margin) 밖으로 나가면 즉시 컷
                if ex < auto_margin or ex > img_w - auto_margin or ey < auto_margin or ey > img_h - auto_margin:
                    continue

                visited.add((nc, nr))
                d, idx = tree.query([ex, ey])

                if d < rough_pitch * 0.45:
                    if idx not in used_spots:
                        ax, ay = pts[idx]
                        grid_dict[(nc, nr)] = (ax, ay)
                        used_spots.add(idx)
                        queue.append((nc, nr, ax, ay))
                    else:
                        grid_dict[(nc, nr)] = (ex, ey)
                        queue.append((nc, nr, ex, ey))
                else:
                    grid_dict[(nc, nr)] = (ex, ey)
                    queue.append((nc, nr, ex, ey))

        min_c = min(c for c, r in grid_dict.keys())
        max_c = max(c for c, r in grid_dict.keys())
        min_r = min(r for c, r in grid_dict.keys())
        max_r = max(r for c, r in grid_dict.keys())

        for c in range(min_c, max_c + 1):
            for r in range(min_r, max_r + 1):
                if (c, r) not in grid_dict:
                    best_k = min(grid_dict.keys(), key=lambda k: abs(c - k[0]) + abs(r - k[1]))
                    kx, ky = grid_dict[best_k]
                    dc, dr = c - best_k[0], r - best_k[1]
                    ex = kx + dc * vec_right[0] + dr * vec_down[0]
                    ey = ky + dc * vec_right[1] + dr * vec_down[1]
                    
                    # ★ 빈자리 채울 때도 자동 마진 확인
                    if ex < auto_margin or ex > img_w - auto_margin or ey < auto_margin or ey > img_h - auto_margin:
                        continue
                    
                    d, idx = tree.query([ex, ey])
                    if d < rough_pitch * 0.45 and idx not in used_spots:
                        ax, ay = pts[idx]
                        grid_dict[(c, r)] = (ax, ay)
                        used_spots.add(idx)
                    else:
                        grid_dict[(c, r)] = (ex, ey)

        # ★ 3-2. 최종적으로 grid_dict 한 번 더 정리 및 행/열 재계산
        final_grid_dict = {}
        for (c, r), (px, py) in grid_dict.items():
            if auto_margin <= px <= img_w - auto_margin and auto_margin <= py <= img_h - auto_margin:
                final_grid_dict[(c, r)] = (px, py)
                
        grid_dict = final_grid_dict
        
        if grid_dict:
            min_c = min(c for c, r in grid_dict.keys())
            max_c = max(c for c, r in grid_dict.keys())
            min_r = min(r for c, r in grid_dict.keys())
            max_r = max(r for c, r in grid_dict.keys())
            cols = max_c - min_c + 1
            rows = max_r - min_r + 1
            total_wells = len(grid_dict) # 정확하게 남은 웰 개수만 계산
        else:
            cols = rows = total_wells = 0

        # 4. 신호 정밀 측정 및 판정
        r_int = max(1, int(round(avg_radius * 0.5))) 
        
        for (c, r), (px, py) in grid_dict.items():
            px, py = int(round(px)), int(round(py))
            cv2.circle(grid_img, (px, py), avg_radius, (0, 255, 255), 1)
            
            y1, y2 = max(0, py - r_int), min(img_h, py + r_int)
            x1, x2 = max(0, px - r_int), min(img_w, px + r_int)
            roi_green = green_channel[y1:y2, x1:x2]
            
            if roi_green.size > 0:
                intensity = np.mean(roi_green)
            else:
                intensity = 0
                
            is_pos = intensity >= signal_thresh
            
            if is_pos:
                cv2.circle(result_img, (px, py), avg_radius, (255, 255, 0), 1)
                matched_pos_count += 1
            else:
                cv2.circle(result_img, (px, py), avg_radius, (255, 0, 0), 1)
                matched_neg_count += 1

        ratio = (matched_pos_count / total_wells * 100) if total_wells > 0 else 0
        is_gmo = ratio >= gmo_criteria

    return grid_img, result_img, total_wells, matched_pos_count, matched_neg_count, ratio, is_gmo, cols, rows

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (신호 정밀분석 Pro)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2.5])

with col1:
    st.subheader("⚙️ 분석 설정")
    
    with st.expander("1️⃣ 판정 기준 및 밝기", expanded=True):
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)
        
        st.markdown("---")
        signal_thresh = st.slider("✨ Positive 판정 밝기 기준 (Signal)", 0, 255, 40)
        
        st.markdown("---")
        min_threshold = st.slider("격자 탐색 최소 밝기", 0, 255, 30)
        max_threshold = st.slider("격자 탐색 최대 밝기", 0, 255, 255)

    with st.expander("2️⃣ 스팟 형태 필터링 (격자용)", expanded=True):
        min_pitch = st.number_input("최소 웰 간격 (Pitch - 픽셀)", min_value=5, max_value=200, value=20, step=1, help="웰 중심과 다음 웰 중심 사이의 최소 거리를 픽셀 단위로 입력하세요. 격자가 너무 촘촘하게(뻥튀기) 잡힐 때 이 값을 올리면 해결됩니다.")
        
        st.markdown("---")
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=5, step=5)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=10, max_value=50000, value=200, step=10)
        circularity = st.slider("최소 원형도", 0.0, 1.0, 0.3, step=0.05)
        convexity = st.slider("최소 볼록성", 0.0, 1.0, 1.0, step=0.05)

    uploaded_file = st.file_uploader("✨ 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("형광 신호를 정밀 측정 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria, signal_thresh, min_pitch
            )
            
            tab1, tab2 = st.tabs(["📌 1. 왜곡 보정 가상 격자", "📊 2. 형광 신호 측정 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 계산 확인")
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="청록색: 누적 오차 및 노이즈 중복 없이 추적된 정밀 가상 격자점", use_column_width=True)
                else:
                    st.warning("스팟이 충분히 검출되지 않았습니다. 격자 탐색 밝기나 면적 설정을 조절해주세요.")
                    
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
                    
                    st.image(result_img, caption="노란색: 신호 강함(Positive), 빨간색: 신호 없음(Negative)", use_column_width=True)
                else:
                    st.warning("분석할 결과가 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
