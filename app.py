import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Ruler Analyzer", layout="wide")
st.title("🔬 눈금 가이드형 자동 Microwell 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

# 2. 사진 업로드 (이미지 크기를 알아야 눈금 범위를 정할 수 있음)
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        # [회전 보정]
        h, w = raw_img.shape[:2]
        rot_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        img = cv2.warpAffine(raw_img, rot_matrix, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 사이드바: 눈금 기반 영역 설정
        st.sidebar.header("📏 2단계: 눈금 영역 설정")
        st.sidebar.info("눈금선(Cyan)을 가장 바깥쪽 우물 중심에 맞추세요.")
        
        # 가로 눈금 (X축 범위)
        x_range = st.sidebar.slider("가로 범위 (Left - Right)", 0, w, (int(w*0.1), int(w*0.9)))
        # 세로 눈금 (Y축 범위)
        y_range = st.sidebar.slider("세로 범위 (Top - Bottom)", 0, h, (int(h*0.1), int(h*0.9)))
        
        # 미세 조정 (사다리꼴 왜곡 대비)
        skew_x = st.sidebar.slider("좌우 비대칭 보정 (Skew X)", -50, 50, 0)
        skew_y = st.sidebar.slider("상하 비대칭 보정 (Skew Y)", -50, 50, 0)

        # 4점 좌표 자동 계산 (눈금 기반)
        tl = [x_range[0], y_range[0]]
        tr = [x_range[1], y_range[0] + skew_y]
        bl = [x_range[0] + skew_x, y_range[1]]
        br = [x_range[1], y_range[1]]
        pts_src = np.array([tl, tr, br, bl], dtype=np.float32)

        # [개수 자동 인식 로직]
        def get_auto_count(roi_gray, sens=0.5):
            x_proj = np.mean(roi_gray, axis=0)
            y_proj = np.mean(roi_gray, axis=1)
            def count_peaks(proj):
                avg = np.mean(proj)
                std = np.std(proj)
                # 평균보다 높은 피크 감지
                thresh = avg + std * sens
                return len([i for i in range(1, len(proj)-1) if proj[i] > thresh and proj[i] > proj[i-1] and proj[i] > proj[i+1]])
            return max(1, count_peaks(x_proj)), max(1, count_peaks(y_proj))

        # 원근 변환 및 개수 감지
        tw, th = 1000, 1000
        M = cv2.getPerspectiveTransform(pts_src, np.array([[0,0], [tw, 0], [tw, th], [0, th]], dtype=np.float32))
        warped = cv2.warpPerspective(img, M, (tw, th))
        auto_cols, auto_rows = get_auto_count(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

        # [시각화 및 분석]
        display_img = img_rgb.copy()
        
        # 가이드 눈금선 그리기 (Cyan 색상)
        line_color = (0, 255, 255)
        cv2.line(display_img, (x_range[0], 0), (x_range[0], h), line_color, 2)
        cv2.line(display_img, (x_range[1], 0), (x_range[1], h), line_color, 2)
        cv2.line(display_img, (0, y_range[0]), (w, y_range[0]), line_color, 2)
        cv2.line(display_img, (0, y_range[1]), (w, y_range[1]), line_color, 2)

        # 분석 진행
        threshold = st.sidebar.slider("형광 임계값", 0, 255, 60)
        radius = st.sidebar.slider("표시 반지름", 1, 15, 5)
        
        pos_count = 0
        for r in range(auto_rows):
            v = r / (auto_rows - 1) if auto_rows > 1 else 0
            edge_l = (1-v)*pts_src[0] + v*pts_src[3]
            edge_r = (1-v)*pts_src[1] + v*pts_src[2]
            for c in range(auto_cols):
                h_rat = c / (auto_cols - 1) if auto_cols > 1 else 0
                pt = (1-h_rat)*edge_l + h_rat*edge_r
                cx, cy = int(pt[0]), int(pt[1])
                if 0 <= cx < w and 0 <= cy < h:
                    g_val = img_rgb[cy, cx, 1]
                    is_pos = g_val > threshold
                    if is_pos: pos_count += 1
                    cv2.circle(display_img, (cx, cy), radius, (0, 255, 0) if is_pos else (255, 0, 0), 1)

        st.image(display_img, caption=f"감지된 격자: {auto_cols} x {auto_rows}", use_container_width=True)
        
        # 결과 리포트
        st.subheader("📊 분석 결과")
        st.write(f"자동 인식된 우물 개수: **{auto_cols * auto_rows}개**")
        st.metric("Positive (녹색)", f"{pos_count}개")
