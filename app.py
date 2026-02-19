import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Auto-Detector", layout="wide")
st.title("🚀 Microwell 완전 자동 분석기 (Auto-Coordinate)")

# --- 함수 정의: 좌표 자동 감지 로직 ---
def auto_detect_coords(img):
    """이미지 분석을 통해 Well 격자의 4개 모서리 좌표를 자동으로 추출"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. 노이즈 제거 및 이진화
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. 윤곽선 감지
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 3. 가장 큰 사각형 영역 찾기
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # 사각형(점 4개)으로 근사화된 경우 해당 좌표 반환
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        # 좌표 정렬 (좌상, 우상, 우하, 좌하 순서)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    return None

# --- 사이드바 ---
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    f_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    
    if img_raw is not None:
        # 회전 적용
        h, w = img_raw.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img_rot = cv2.warpAffine(img_raw, M, (w, h))
        
        # [핵심] 좌표 자동 감지 시도
        auto_pts = auto_detect_coords(img_rot)
        
        st.sidebar.header("📍 2단계: 영역 좌표 (자동 감지됨)")
        # 자동 감지된 값이 있으면 기본값으로 사용, 없으면 기존 기본값 사용
        def_tl = auto_pts[0] if auto_pts is not None else [150, 200]
        def_tr = auto_pts[1] if auto_pts is not None else [2300, 200]
        def_br = auto_pts[2] if auto_pts is not None else [2300, 2300]
        def_bl = auto_pts[3] if auto_pts is not None else [150, 2300]

        sc1, sc2 = st.sidebar.columns(2)
        tl_x = sc1.number_input("좌상 X", 0, w, int(def_tl[0]))
        tl_y = sc2.number_input("좌상 Y", 0, h, int(def_tl[1]))
        tr_x = sc1.number_input("우상 X", 0, w, int(def_tr[0]))
        tr_y = sc2.number_input("우상 Y", 0, h, int(def_tr[1]))
        bl_x = sc1.number_input("좌하 X", 0, w, int(def_bl[0]))
        bl_y = sc2.number_input("좌하 Y", 0, h, int(def_bl[1]))
        br_x = sc1.number_input("우하 X", 0, w, int(def_br[0]))
        br_y = sc2.number_input("우하 Y", 0, h, int(def_br[1]))

        # 이후 분석 로직 (Well 개수 자동 인식 등 이전과 동일)
        st.sidebar.header("🔢 3단계: Well & GMO 설정")
        auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
        radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
        threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
        sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0)
        gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

        pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

        # 분석 진행
        tab1, tab2 = st.tabs(["📝 좌표 확인", "📊 분석 결과"])
        
        # (중략: 이전과 동일한 분석 및 시각화 로직 적용)
        # ... (이전 코드의 tab1, tab2 내부 로직 실행) ...
        
        with tab2:
            # (Well 개수 계산 및 원 그리기 로직 생략 - 이전과 동일하게 유지)
            st.write("자동 감지가 완료되었습니다. 좌표가 맞지 않으면 사이드바에서 수정하세요.")
