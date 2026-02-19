import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="Microwell Auto Analyzer", layout="wide")
st.title("🚀 Microwell 자동 감지 및 GMO 분석기")

# --- 유틸리티 함수: 좌표 자동 감지 ---
def detect_well_area(img):
    """이미지 내에서 Well들이 모여있는 가장 큰 사각형 영역을 감지"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러로 노이즈 제거 후 이진화
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 가장 면적이 큰 윤곽선 선택
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # 사각형 형태(점 4개)인 경우 좌표 정렬 후 반환
    if len(approx) >= 4:
        pts = approx.reshape(-1, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # 좌상
        rect[2] = pts[np.argmax(s)] # 우하
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 우상
        rect[3] = pts[np.argmax(diff)] # 좌하
        return rect
    return None

def get_auto_count(roi_gray, sens):
    _, th_img = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x_p = np.mean(th_img, axis=0)
    y_p = np.mean(th_img, axis=1)
    def count_p(proj):
        limit = np.mean(proj) * sens
        cnt, peak = 0, False
        for v in proj:
            if v > limit and not peak:
                cnt += 1; peak = True
            elif v < limit: peak = False
        return cnt
    return max(1, count_p(x_p)), max(1, count_p(y_p))

def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    r_img = img.copy()
    cv2.line(r_img, (0, h//2), (w, h//2), (255, 0, 0), 2) # Red Center Line
    cv2.line(r_img, (w//2, 0), (w//2, h), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(h, w) / 2000.0
    for x in range(0, w, 200):
        cv2.line(r_img, (x, 0), (x, int(40*scale)), (255, 255, 255), int(3*scale))
        cv2.putText(r_img, str(x), (x, int(80*scale)), font, scale, (255, 255, 255), int(2*scale))
    for y in range(0, h, 200):
        cv2.line(r_img, (0, y), (int(40*scale), y), (255, 255, 255), int(3*scale))
        cv2.putText(r_img, str(y), (int(10*scale), y), font, scale, (255, 255, 255), int(2*scale))
    return r_img

# --- 메인 프로세스 ---
uploaded_file = st.file_uploader("사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    f_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        # 1. 초기 자동 감지 실행 (한 번만 실행되도록 설정 가능)
        auto_coords = detect_well_area(img_bgr)
        
        # 2. 사이드바 설정 (자동 감지된 값을 기본값으로 세팅)
        st.sidebar.header("🔄 1단계: 수평 보정")
        rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

        st.sidebar.header("📍 2단계: 영역 좌표 (자동 입력됨)")
        # 자동 감지 좌표가 있으면 사용, 없으면 임의의 기본값 사용
        def_pts = auto_coords if auto_coords is not None else [[500, 500], [2000, 500], [2000, 2000], [500, 2000]]
        
        sc1, sc2 = st.sidebar.columns(2)
        tl_x = sc1.number_input("좌상 X", 0, 8000, int(def_pts[0][0]))
        tl_y = sc2.number_input("좌상 Y", 0, 8000, int(def_pts[0][1]))
        tr_x = sc1.number_input("우상 X", 0, 8000, int(def_pts[1][0]))
        tr_y = sc2.number_input("우상 Y", 0, 8000, int(def_pts[1][1]))
        br_x = sc1.number_input("우하 X", 0, 8000, int(def_pts[2][0]))
        br_y = sc2.number_input("우하 Y", 0, 8000, int(def_pts[2][1]))
        bl_x = sc1.number_input("좌하 X", 0, 8000, int(def_pts[3][0]))
        bl_y = sc2.number_input("좌하 Y", 0, 8000, int(def_pts[3][1]))

        st.sidebar.header("🧪 3단계: 분석 설정")
        auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
        radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
        threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
        sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0)
        gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

        # 3. 이미지 회전 처리
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img_rot = cv2.warpAffine(img_bgr, M, (w, h))
        img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)

        # 4. 결과 출력 탭
        tab1, tab2 = st.tabs(["📝 좌표 확인 (Red Guide)", "📊 분석 결과"])
        
        with tab1:
            st.image(draw_ruler_and_guide(img_rgb), use_container_width=True)
            if auto_coords is not None:
                st.success("✅ Well 영역이 자동으로 감지되었습니다. 필요시 좌표를 수정하세요.")
            else:
                st.warning("⚠️ 자동 감지에 실패했습니다. 눈금자를 보고 수동으로 입력하세요.")

        with tab2:
            pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
            
            # Well 개수 파악
            if auto_mode:
                M_p = cv2.getPerspectiveTransform(pts, np.array([[0,0],[1000,0],[1000,1000],[0,1000]], dtype=np.float32))
                warped = cv2.cvtColor(cv2.warpPerspective(img_rot, M_p, (1000, 1000)), cv2.COLOR_BGR2GRAY)
                f_cols, f_rows = get_auto_count(warped, sensitivity)
            else:
                f_cols, f_rows = 23, 24 # 수동 입력창 생략 시 기본값

            # 시각화 및 분석
            res_img = img_rgb.copy()
            pos_cnt = 0
            for r in range(f_rows):
                v_r = r/(f_rows-1) if f_rows > 1 else 0
                lp, rp = (1-v_r)*pts[0] + v_r*pts[3], (1-v_r)*pts[1] + v_r*pts[2]
                for c in range(f_cols):
                    h_r = c/(f_cols-1) if f_cols > 1 else 0
                    cp = (1-h_r)*lp + h_r*rp
                    cx, cy = int(cp[0]), int(cp[1])
                    if 0 <= cx < w and 0 <= cy < h:
                        is_pos = img_rgb[cy, cx, 1] > threshold
                        if is_pos: pos_cnt += 1
                        cv2.circle(res_img, (cx, cy), radius, (0,255,0) if is_pos else (255,0,0), 1)
            
            cv2.polylines(res_img, [pts.astype(int)], True, (255, 255, 0), 2)
            st.image(res_img, use_container_width=True)
            
            # 결과 대시보드
            total = f_cols * f_rows
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            st.markdown("---")
            st.info(f"📊 **Grid Info:** {f_cols}개(Col) × {f_rows}개(Row) = 총 {total} Well")
            if ratio >= gmo_thresh:
                st.success(f"### 🧬 판정 결과: GMO Positive ({ratio:.1f}%)")
            else:
                st.error(f"### 🧬 판정 결과: Non-GMO ({ratio:.1f}%)")
