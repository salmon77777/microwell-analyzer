import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="Microwell Grid Analyzer", layout="wide")
st.title("🔬 Microwell Well & GMO Analyzer")

# --- 2. 사이드바: 설정 영역 ---
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
sc1, sc2 = st.sidebar.columns(2)
tl_x = sc1.number_input("좌상 X", 0, 8000, 150)
tl_y = sc2.number_input("좌상 Y", 0, 8000, 200)
tr_x = sc1.number_input("우상 X", 0, 8000, 2300)
tr_y = sc2.number_input("우상 Y", 0, 8000, 200)
bl_x = sc1.number_input("좌하 X", 0, 8000, 150)
bl_y = sc2.number_input("좌하 Y", 0, 8000, 2300)
br_x = sc1.number_input("우하 X", 0, 8000, 2300)
br_y = sc2.number_input("우하 Y", 0, 8000, 2300)

st.sidebar.header("🔢 3단계: Well 개수 설정")
auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
manual_cols, manual_rows = 23, 24
if not auto_mode:
    mc1, mc2 = st.sidebar.columns(2)
    manual_cols = mc1.number_input("가로 Well", 1, 150, 23)
    manual_rows = mc2.number_input("세로 Well", 1, 150, 24)

st.sidebar.header("🧪 4단계: 판정 및 크기")
radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0, step=0.1)

st.sidebar.header("🧬 5단계: GMO 판정")
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 3. 유틸리티 함수 정의 ---
def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    r_img = img.copy()
    # 중앙 가이드라인 (Red) - RGB 기준 (255, 0, 0)
    cv2.line(r_img, (0, h//2), (w, h//2), (255, 0, 0), 2)
    cv2.line(r_img, (w//2, 0), (w//2, h), (255, 0, 0), 2)
    # 눈금자
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(h, w) / 2000.0
    for x in range(0, w, 200):
        cv2.line(r_img, (x, 0), (x, int(40*scale)), (255, 255, 255), int(3*scale))
        cv2.putText(r_img, str(x), (x, int(80*scale)), font, scale, (255, 255, 255), int(2*scale))
    for y in range(0, h, 200):
        cv2.line(r_img, (0, y), (int(40*scale), y), (255, 255, 255), int(3*scale))
        cv2.putText(r_img, str(y), (int(10*scale), y), font, scale, (255, 255, 255), int(2*scale))
    return r_img

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

# --- 4. 메인 화면 로직 ---
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    f_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        # 회전 보정
        M = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img_rot = cv2.warpAffine(img_bgr, M, (w, h))
        img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
        
        tab1, tab2 = st.tabs(["📝 좌표 확인 (Red Guide)", "📊 분석 결과"])
        
        with tab1:
            # 눈금자와 가이드라인이 있는 이미지 생성
            ruler_view = draw_ruler_and_guide(img_rgb)
            st.image(ruler_view, use_container_width=True, caption="빨간 중앙선을 기준으로 수평을 맞추고 눈금 좌표를 입력하세요.")
            
        with tab2:
            # 4점 좌표 설정
            pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
            
            # Well 개수 결정
            if auto_mode:
                M_p = cv2.getPerspectiveTransform(pts, np.array([[0,0],[1000,0],[1000,1000],[0,1000]], dtype=np.float32))
                warped = cv2.cvtColor(cv2.warpPerspective(img_rot, M_p, (1000, 1000)), cv2.COLOR_BGR2GRAY)
                f_cols, f_rows = get_auto_count(warped, sensitivity)
            else:
                f_cols, f_rows = manual_cols, manual_rows
            
            # 격자 생성 및 분석 시각화
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
                        # 원 테두리 두께 1로 얇게
                        cv2.circle(res_img, (cx, cy), radius, (0,255,0) if is_pos else (255,0,0), 1)
            
            # 노란색 선택 영역 표시
            cv2.polylines(res_img, [pts.astype(int)], True, (255, 255, 0), 2)
            st.image(res_img, use_container_width=True)
            
            # 결과 지표 계산
            total = f_cols * f_rows
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            is_gmo = ratio >= gmo_thresh
            
            st.markdown("---")
            st.info(f"📊 **Grid Info:** 가로(Column) **{f_cols}**개 × 세로(Row) **{f_rows}**개 (총 {total} Well)")

            if is_gmo:
                st.success(f"### 🧬 판정 결과: GMO Positive (발현율: {ratio:.1f}%)")
            else:
                st.error(f"### 🧬 판정 결과: Non-GMO (발현율: {ratio:.1f}%)")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Well", f"{total}")
            c2.metric("Positive", f"{pos_cnt}")
            c3.metric("Ratio", f"{ratio:.1f}%")
            c4.metric("Threshold", f"{gmo_thresh}%")
else:
    st.info("💡 분석을 시작하려면 사진을 업로드해 주세요.")
