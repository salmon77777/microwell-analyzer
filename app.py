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

    # 1. 스팟 검출 설정 (밝은 스팟)
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

    # 2. 확실한 양성 스팟 필터링 (테두리 제외)
    raw_positive_wells = []
    margin = 2
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

    # 3. 가상 격자(Virtual Grid) 생성 및 시각화
    if num_raw_positive > 10:
        # 스팟 간 최소 거리(Pitch) 계산
        nearest_distances = []
        for i in range(num_raw_positive):
            p1 = raw_positive_wells[i]
            min_d = float('inf')
            for j in range(num_raw_positive):
                if i == j: continue
                p2 = raw_positive_wells[j]
                d = calculate_distance((p1[0], p1[1]), (p2[0], p2[1]))
                if d < min_d: min_d = d
            nearest_distances.append(min_d)
        
        pitch = np.median(nearest_distances)

        if pitch > 0:
            # 회전된 최소 면적 사각형(Bounding Box) 구하기
            points = np.array([[w[0], w[1]] for w in raw_positive_wells], dtype=np.float32)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            
            # 사각형 꼭짓점 정렬 (좌상, 우상, 우하, 좌하)
            box = box[np.argsort(box[:, 0])]
            left_pts = box[:2]
            right_pts = box[2:]
            tl = left_pts[np.argmin(left_pts[:, 1])]
            bl = left_pts[np.argmax(left_pts[:, 1])]
            tr = right_pts[np.argmin(right_pts[:, 1])]
            br = right_pts[np.argmax(right_pts[:, 1])]
            
            # 가로, 세로 개수 추정
            width_px = np.linalg.norm(tr - tl)
            height_px = np.linalg.norm(bl - tl)
            cols = int(round(width_px / pitch)) + 1
            rows = int(round(height_px / pitch)) + 1
            total_wells = cols * rows
            
            avg_radius = int(np.mean([w[2] for w in raw_positive_wells]))

            # 가상 격자의 좌표 벡터 생성
            u_vec = (tr - tl) / max(1, cols - 1) if cols > 1 else np.array([0,0])
            v_vec = (bl - tl) / max(1, rows - 1) if rows > 1 else np.array([0,0])
            
            all_grid_points = []
            
            # 전체 격자점 계산
            for i in range(cols):
                for j in range(rows):
                    pt = tl + i * u_vec + j * v_vec
                    gx, gy = int(pt[0]), int(pt[1])
                    all_grid_points.append((gx, gy))
                    # Tab 1용 이미지: 파란색 원으로 전체 격자 그리기 (두께 1)
                    cv2.circle(grid_img, (gx, gy), avg_radius, (0, 255, 255), 1) 

            # 4. 생성된 격자점과 실제 스팟 매칭 (Positive/Negative 분류)
            for gx, gy in all_grid_points:
                is_pos = False
                for px, py, pr in raw_positive_wells:
                    # 격자점과 실제 스팟이 충분히 가까우면 Positive로 판정
                    if calculate_distance((gx, gy), (px, py)) < (pitch * 0.5):
                        is_pos = True
                        break
                
                if is_pos:
                    matched_pos_count += 1
                    # Tab 2용 이미지: 노란색 테두리 (두께 1)
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 255, 0), 1)
                else:
                    matched_neg_count += 1
                    # Tab 2용 이미지: 빨간색 테두리 (두께 1)
                    cv2.circle(result_img, (gx, gy), avg_radius, (255, 0, 0), 1)

            ratio = (matched_pos_count / total_wells * 100) if total_wells > 0 else 0
            is_gmo = ratio >= gmo_criteria

    return grid_img, result_img, total_wells, matched_pos_count, matched_neg_count, ratio, is_gmo

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
        
        with st.spinner("이미지를 분석 중입니다..."):
            grid_img, result_img, total, pos, neg, ratio, is_gmo = analyze_microwells(
                image_pil, min_threshold, max_threshold, min_area, max_area, circularity, convexity, gmo_criteria
            )
            
            # ★ 탭 생성 (전체 인식 / 판정 결과)
            tab1, tab2 = st.tabs(["📌 1. 전체 Well 인식 확인", "📊 2. Positive 분석 결과"])
            
            with tab1:
                st.subheader("가상 격자(Virtual Grid) 매핑 결과")
                st.write("프로그램이 전체 Well 영역으로 추정한 격자점(파란색)입니다. 이 영역을 기준으로 분석이 진행됩니다.")
                st.metric("추정된 전체 Well 개수", f"{total:,} 개")
                if total > 0:
                    st.image(grid_img, caption="파란색 얇은 테두리: 프로그램이 추정한 전체 Well 위치", use_column_width=True)
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
