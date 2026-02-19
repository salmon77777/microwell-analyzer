import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="AI 패턴 매칭 분석기", layout="wide")
st.title("🔬 AI 자동 Well 탐지 분석기")

# --- 사이드바: 감도 설정 ---
st.sidebar.header("⚙️ 분석 민감도")
st.sidebar.info("수동 입력이 필요 없습니다. 노란색 원이 Well 위치에 잘 오도록 조절하세요.")

# Well을 더 뚜렷하게 보이게 하는 파라미터
contrast = st.sidebar.slider("이미지 대비 강도", 1.0, 3.0, 1.5)
blur_size = st.sidebar.slider("노이즈 제거 강도", 1, 15, 5, step=2)

st.sidebar.header("📏 Well 크기 설정")
well_radius = st.sidebar.slider("Well 반지름(픽셀)", 5, 50, 15)
min_dist = st.sidebar.slider("Well 사이 최소 거리", 10, 100, 30)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        # 1. 전처리: 대비 향상 및 노이즈 제거
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=contrast, beta=0)
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # 2. 특징점 추출 (Blob Detection 방식)
        # 신호가 있는 Well과 없는 Well 모두를 잡기 위해 적응형 이진화 사용
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # 3. 모든 Well 후보지 자동 탐지 (Hough보다 강력한 Blob 탐지)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = (well_radius ** 2) * 0.5
        params.maxArea = (well_radius ** 2) * 4
        params.filterByCircularity = False # 기울어져도 잡히도록 끔
        params.minDistBetweenBlobs = min_dist
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray) # 원본 그레이에서 특징점 탐색
        
        res_img = img_rgb.copy()
        pos_cnt = 0
        valid_wells = []

        # 4. 결과 시각화 및 분석
        for kp in keypoints:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            r = well_radius
            
            # [필터링] 사진 끝에 잘린 Well 제외
            if (cx - r < 5) or (cx + r > w - 5) or (cy - r < 5) or (cy + r > h - 5):
                continue
            
            valid_wells.append((cx, cy))
            
            # 모든 인식된 Well은 노란색
            cv2.circle(res_img, (cx, cy), r, (255, 255, 0), 1)
            
            # 중심부 녹색값(형광) 체크
            roi_g = img_rgb[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3), 1]
            avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
            
            if avg_g > threshold_g:
                pos_cnt += 1
                # Positive는 초록색 점
                cv2.circle(res_img, (cx, cy), int(r*0.5), (0, 255, 0), -1)

        st.image(res_img, use_container_width=True)
        
        total = len(valid_wells)
        ratio = (pos_cnt / total * 100) if total > 0 else 0
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("자동 탐지된 Well", f"{total}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("GMO 신호율", f"{ratio:.1f}%")

        if ratio >= gmo_thresh:
            st.success("### 🧬 판정 결과: GMO Positive")
        else:
            st.error("### 🧬 판정 결과: Non-GMO")
            
        # 디버깅용 (안 잡힐 때 확인)
        if st.checkbox("인식용 이진화 이미지 보기"):
            st.image(thresh)
    else:
        st.error("이미지를 불러올 수 없습니다.")
