import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# --- 헬퍼 함수 ---
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_microwells(image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria):
    image_rgb_pil = image_pil.convert('RGB')
    image_rgb = np.array(image_rgb_pil)
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_img.shape[:2]

    # 1. 스팟 검출 (밝은 스팟)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255 
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

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)

    # 2. 확실한 양성 스팟 필터링
    raw_positive_wells = []
    margin = 5
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if margin < x < (img_w - margin) and margin < y < (img_h - margin):
            raw_positive_wells.append((x, y, r))

    num_raw_positive = len(raw_positive_wells)
    
    grid_img = image_rgb.copy()
    result_img = image_rgb.copy()
    
    total_wells = 0
    matched_pos_count = 0
    matched_neg_count = 0
    ratio = 0.0
    is_gmo = False

    # 3. 새로운 로직: 좌표 투영 및 보간법을 통한 완벽한 격자 생성
    if num_raw_positive > 10:
        # 3-1. 스팟 간 평균 최소 거리(Pitch) 계산
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
            # 3-2. 전체 이미지의 미세한 기울기(Angle) 파악
            angles = []
            for i, p1 in enumerate(raw_positive_wells):
                for j, p2 in enumerate(raw_positive_wells):
                    if i == j: continue
                    d = calculate_distance(p1[:2], p2[:2])
                    if d < pitch * 1.5: # 인접한 스팟 사이의 각도만 계산
                        dy = p2[1] - p1[1]
                        dx = p2[0] - p1[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        angle = angle % 90
                        if angle > 45: angle -= 90 # -45 ~ 45도 사이로 정규화
                        angles.append(angle)
            
            grid_angle = np.median(angles) if angles else 0.0

            # 3-3. 스팟들을 똑바르게(회전) 펴기
            center = np.mean([w[:2] for w in raw_positive_wells], axis=0)
            M_rot = cv2.getRotationMatrix2D(tuple(center), grid_angle, 1.0)
            
            pts = np.array([w[:2] for w in raw_positive_wells])
            ones = np.ones(shape=(len(pts), 1))
            pts_ones = np.hstack([pts, ones])
            rotated_pts = M_rot.dot(pts_ones.T).T
            
            xs = rotated_pts[:, 0]
            ys = rotated_pts[:, 1]

            # 3-4. 축소 및 보간 함수 (빈 열/행 채워넣기)
            def find_grid_lines(coords, pitch):
                sorted_coords = np.sort(coords)
                lines = []
                curr_group = [sorted_coords[0]]
                
                # 좌표들을 묶어서 실제 존재하는 선(Line) 찾기
                for c in sorted_coords[1:]:
                    if c - curr_group[-1] <= pitch * 0.5:
                        curr_group.append(c)
                    else:
                        lines.append(np.mean(curr_group))
                        curr_group = [c]
                lines.append(np.mean(curr_group))
                
                # 비어있는 선(Line)을 간격(pitch)을 이용해 수학적으로 채워넣기
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
            
            # 가로/세로 전체 개수
            cols = len(grid_xs)
            rows = len(grid_ys)
            total_wells = cols * rows

            # 3-5. 완벽한 바둑판 포인트 생성 후 다시 원래 각도로 되돌리기
            ideal_grid = []
            for gx in grid_xs:
                for gy in grid_ys:
                    ideal_grid.append([gx, gy])
            ideal_grid = np.array(ideal_grid)
            
            M_rot_inv = cv2.getRotationMatrix2D(tuple(center), -grid_angle, 1.0)
            ones_grid = np.ones(shape=(len(ideal_grid), 1))
            grid_ones = np.hstack([ideal_grid, ones_grid])
            final_grid_points = M_rot_inv.dot(grid_ones.T).T

            # 4. 시각화 및 판정 로직
            avg_radius = int(np.mean([w[2] for w in raw_positive_wells]))

            for gx, gy in final_grid_points:
                gx, gy = int(gx), int(gy)
                
                # Tab 1용: 파란색 원
                cv2.circle(grid_img, (gx, gy), avg_radius, (0, 255, 255), 1)
                
                # 실제 스팟과 매칭 (가까운 곳에 형광이 있는가?)
                is_pos = False
                for px, py, pr in raw_positive_wells:
                    if calculate_distance((gx, gy), (px, py)) < (pitch * 0.5):
                        is_pos = True
                        break
                
                if is_pos:
                    matched_pos_count += 1
                    # Tab 2용: 노란색 테두리 (두께 1)
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 255, 0), 1)
                else:
                    matched_neg_count += 1
                    # Tab 2용: 빨간색 테두리 (두께 1)
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 0, 0), 1)

            ratio = (matched_pos_count / total_wells * 100) if total_wells > 0 else 0
            is_gmo = ratio >= gmo_criteria

    return grid_img, result_img, total_wells, matched_pos_count, matched_neg_count, ratio, is_gmo, len(grid_xs) if 'grid_xs' in locals() else 0, len(grid_ys) if 'grid_ys' in locals() else 0

# --- Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="Microwell 분석기 Pro")

st.title("🦠 Microwell 형광 자동 분석기 (Pro 버전)")
st.markdown("---")

col1, col2 = st.columns([1.2, 2.5])

with col1:
    st.subheader("⚙️ 분석 설정")
    
    with st.expander("1️⃣ 판정 기준 및 밝기", expanded=True):
        gmo_criteria = st.slider("GMO 판정 기준 (%)", 1, 100, 50)
        min_threshold = st.slider("최소 밝기 임계값", 0, 255, 26)
        max_threshold = st.slider("최대 밝기 임계값", 0, 255, 255)

    with st.expander("2️⃣ 스팟 형태 필터링", expanded=True):
        min_area = st.number_input("최소 면적 (픽셀)", min_value=1, max_value=5000, value=10, step=5)
        max_area = st.number_input("최대 면적 (픽셀)", min_value=10, max_value=50000, value=50, step=10)
        circularity = st.slider("최소 원형도", 0.0, 1.0, 0.1, step=0.05)
        convexity = st.slider("최소 볼록성", 0.0, 1.0, 0.3, step=0.05)

    uploaded_file = st.file_uploader("✨ 형광 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        with st.spinner("가상 격자를 정밀 매핑 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo, cols, rows = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            tab1, tab2 = st.tabs(["📌 1. 전체 Well 인식 확인", "📊 2. Positive 분석 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 매핑 결과")
                st.write("계단 현상을 제거하고 배열을 보간하여 완벽한 바둑판 격자를 생성했습니다.")
                
                # 인식된 배열 형태 추가 출력
                col_a, col_b = st.columns(2)
                col_a.metric("추정된 배열 형태", f"가로 {cols} x 세로 {rows} 줄")
                col_b.metric("계산된 전체 Well 개수", f"{total:,} 개")
                
                if total > 0:
                    st.image(grid_img, caption="파란색 얇은 테두리: 프로그램이 추정한 완벽한 Well 위치", use_column_width=True)
                else:
                    st.warning("스팟이 충분히 검출되지 않아 전체 영역을 추정할 수 없습니다.")
                    
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
                    
                    st.image(result_img, caption="노란색: Positive, 빨간색: Negative (두께 1의 얇은 테두리)", use_column_width=True)
                else:
                    st.warning("분석할 결과가 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 이미지를 업로드하면 분석이 시작됩니다.")
