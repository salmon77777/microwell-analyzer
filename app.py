import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import collections

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
        # 2. 기초 간격(Pitch) 파악 및 노이즈 필터링
        nearest_distances = []
        for p1 in raw_positive_wells:
            min_d = float('inf')
            for p2 in raw_positive_wells:
                if p1 == p2: continue
                d = calculate_distance(p1[:2], p2[:2])
                if d < min_d: min_d = d
            if min_d != float('inf'):
                nearest_distances.append(min_d)
        pitch = np.median(nearest_distances)

        # 주변에 동료가 없는 독립된 노이즈 스팟 제거
        filtered_wells = []
        for p1 in raw_positive_wells:
            for p2 in raw_positive_wells:
                if p1 == p2: continue
                if calculate_distance(p1[:2], p2[:2]) < pitch * 2:
                    filtered_wells.append(p1)
                    break
        raw_positive_wells = filtered_wells

        if len(raw_positive_wells) > 10:
            # 3. 회전 각도 및 가로/세로 독립 간격 정밀 계산
            angles = []
            for p1 in raw_positive_wells:
                for p2 in raw_positive_wells:
                    if p1 == p2: continue
                    d = calculate_distance(p1[:2], p2[:2])
                    if d < pitch * 1.5:
                        dy = p2[1] - p1[1]
                        dx = p2[0] - p1[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        a_mod = angle % 90
                        if a_mod > 45: a_mod -= 90
                        angles.append(a_mod)
            grid_angle = np.median(angles) if angles else 0.0

            dist_x, dist_y = [], []
            for p1 in raw_positive_wells:
                for p2 in raw_positive_wells:
                    if p1 == p2: continue
                    d = calculate_distance(p1[:2], p2[:2])
                    if d < pitch * 1.5:
                        angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
                        rel_angle = (angle - grid_angle + 360) % 360
                        if rel_angle > 180: rel_angle -= 360
                        if -30 < rel_angle < 30 or rel_angle > 150 or rel_angle < -150:
                            dist_x.append(d)
                        elif 60 < rel_angle < 120 or -120 < rel_angle < -60:
                            dist_y.append(d)
            
            pitch_x = np.median(dist_x) if dist_x else pitch
            pitch_y = np.median(dist_y) if dist_y else pitch

            rad = np.radians(grid_angle)
            vec_r = (pitch_x * np.cos(rad), pitch_x * np.sin(rad))
            vec_d = (pitch_y * np.cos(rad + np.pi/2), pitch_y * np.sin(rad + np.pi/2))

            # 4. 거미줄 확장 알고리즘 (BFS) - 곡면 왜곡 완벽 흡수
            min_x = min(w[0] for w in raw_positive_wells)
            max_x = max(w[0] for w in raw_positive_wells)
            min_y = min(w[1] for w in raw_positive_wells)
            max_y = max(w[1] for w in raw_positive_wells)
            margin_x, margin_y = pitch_x * 0.8, pitch_y * 0.8

            avg_radius = int(round(np.mean([w[2] for w in raw_positive_wells])))
            unmatched_spots = set(range(len(raw_positive_wells)))
            visited_cr = set()
            grid_dict = {}

            ref_spot = min(raw_positive_wells, key=lambda w: w[0] + w[1])
            ref_x, ref_y = ref_spot[0], ref_spot[1]

            while unmatched_spots:
                start_idx = unmatched_spots.pop()
                unmatched_spots.add(start_idx)
                sx, sy, sr = raw_positive_wells[start_idx]

                # 글로벌 좌표 보정
                dx, dy = sx - ref_x, sy - ref_y
                rad_inv = np.radians(-grid_angle)
                rot_x = dx * np.cos(rad_inv) - dy * np.sin(rad_inv)
                rot_y = dx * np.sin(rad_inv) + dy * np.cos(rad_inv)

                start_c = int(round(rot_x / pitch_x))
                start_r = int(round(rot_y / pitch_y))

                queue = collections.deque([(start_c, start_r, sx, sy)])

                while queue:
                    c, r, px, py = queue.popleft()
                    if (c, r) in visited_cr: continue
                    
                    # 스팟 구역을 벗어나면 거미줄 확장 중지
                    if px < min_x - margin_x or px > max_x + margin_x or py < min_y - margin_y or py > max_y + margin_y:
                        continue

                    visited_cr.add((c, r))

                    best_idx = -1
                    min_d = pitch * 0.45
                    for idx in list(unmatched_spots):
                        wx, wy, _ = raw_positive_wells[idx]
                        d = calculate_distance((wx, wy), (px, py))
                        if d < min_d:
                            min_d = d; best_idx = idx

                    # 실제 스팟이 있으면 중심으로 끌어당김(Snap), 없으면 유추된 위치 사용
                    if best_idx != -1:
                        wx, wy, _ = raw_positive_wells[best_idx]
                        grid_dict[(c, r)] = (wx, wy, True)
                        unmatched_spots.remove(best_idx)
                        cx, cy = wx, wy
                    else:
                        grid_dict[(c, r)] = (px, py, False)
                        cx, cy = px, py

                    # 동서남북으로 거미줄 뻗기
                    queue.append((c+1, r, cx + vec_r[0], cy + vec_r[1]))
                    queue.append((c-1, r, cx - vec_r[0], cy - vec_r[1]))
                    queue.append((c, r+1, cx + vec_d[0], cy + vec_d[1]))
                    queue.append((c, r-1, cx - vec_d[0], cy - vec_d[1]))

            # 5. 비어있는 모서리 영역까지 직사각형 형태로 완벽하게 채우기
            min_c = min(c for c, r in grid_dict.keys())
            max_c = max(c for c, r in grid_dict.keys())
            min_r = min(r for c, r in grid_dict.keys())
            max_r = max(r for c, r in grid_dict.keys())

            for c in range(min_c, max_c + 1):
                for r in range(min_r, max_r + 1):
                    if (c, r) not in grid_dict:
                        best_dist = float('inf')
                        best_k = None
                        # 가장 가까운 이미 찾은 스팟의 곡률을 빌려와서 예측
                        for (kc, kr) in grid_dict.keys():
                            dist = abs(c - kc) + abs(r - kr)
                            if dist < best_dist:
                                best_dist = dist; best_k = (kc, kr)
                        kx, ky, _ = grid_dict[best_k]
                        dc, dr = c - best_k[0], r - best_k[1]
                        px = kx + dc * vec_r[0] + dr * vec_d[0]
                        py = ky + dc * vec_r[1] + dr * vec_d[1]
                        grid_dict[(c, r)] = (px, py, False)

            # 6. 통계 및 렌더링
            total_wells = len(grid_dict)
            cols = max_c - min_c + 1
            rows = max_r - min_r + 1

            for (c, r), (px, py, is_pos) in grid_dict.items():
                px, py = int(round(px)), int(round(py))
                cv2.circle(grid_img, (px, py), avg_radius, (0, 255, 255), 1)
                if is_pos:
                    matched_pos_count += 1
                    cv2.circle(result_img, (px, py), avg_radius, (255, 255, 0), 1)
                else:
                    matched_neg_count += 1
                    cv2.circle(result_img, (px, py), avg_radius, (255, 0, 0), 1)

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
        
        with st.spinner("렌즈 왜곡을 흡수하며 초정밀 거미줄 격자를 생성 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            tab1, tab2 = st.tabs(["📌 1. 왜곡 보정 가상 격자", "📊 2. 최종 분석 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 계산 확인")
                st.write("알고리즘이 렌즈의 휘어짐을 자동으로 추적하며 물결 현상 없이 스팟에 밀착된 격자를 그립니다.")
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="청록색: 거미줄 방식으로 렌즈 왜곡에 맞춰 밀착된 기준점", use_column_width=True)
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
