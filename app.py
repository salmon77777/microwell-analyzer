import streamlit as st
import cv2
import numpy as np

# 페이지 설정 (전체 화면 넓게 사용)
st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 Microwell Grid & GMO Analyzer")

# --- 1. 사이드바: 설정 영역 (공간 최적화) ---
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
# 사이드바 내부에서도 컬럼을 나누어 입력창 크기를 줄임
sc1, sc2 = st.sidebar.columns(2)
tl_x = sc1.number_input("좌상 X", 0, 5000, 150)
tl_y = sc2.number_input("좌상 Y", 0, 5000, 200)
tr_x = sc1.number_input("우상 X", 0, 5000, 2300)
tr_y = sc2.number_input("우상 Y", 0, 5000, 200)
bl_x = sc1.number_input("좌하 X", 0, 5000, 150)
bl_y = sc2.number_input("좌하 Y", 0, 5000, 2300)
br_x = sc1.number_input("우하 X", 0, 5000, 2300)
br_y = sc2.number_input("우하 Y", 0, 5000, 2300)

st.sidebar.header("🔢 3단계: Well 개수 설정")
auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
manual_cols, manual_rows = 23, 24
if not auto_mode:
    mc1, mc2 = st.sidebar.columns(2)
    manual_cols = mc1.number_input("가로 Well", 1, 100, 23)
    manual_rows = mc2.number_input("세로 Well", 1, 100, 24)

st.sidebar.header("🧪 4단계: 판정 및 크기")
radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0, step=0.1)

st.sidebar.header("🧬 5단계: GMO 판정")
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 2. 유틸리티 함수 ---
def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    ruler_img = img.copy()
    guide_color = (0, 0, 255) # 빨간색 가이드라인
    cv2.line(ruler_img, (0, h//2), (w, h//2), guide_color, 2)
    cv2.line(ruler_img, (w//2, 0), (w//2, h), guide_color, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w, 200): # 눈금 간격 조정
        cv2.line(ruler_img, (x, 0), (x, 50), (255, 255, 255), 3)
        cv2.putText(ruler_img, str(x), (x+10, 45), font, 1.2, (255, 255, 255), 3)
    for y in range(0, h, 200):
        cv2.line(ruler_img, (0, y), (50, y), (255, 255, 255), 3)
        cv2.putText(ruler_img, str(y), (10, y-10), font, 1.2, (255, 255, 255), 3)
    return ruler_img

def get_auto_count(roi_gray, sens):
    _, thresh_img = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x_proj = np.mean(thresh_img, axis=0)
    y_proj = np.mean(thresh_img, axis=1)
    def count_peaks(proj):
        p_thresh = np.mean(proj) * sens
        peaks, in_p = 0, False
        for v in proj:
            if v > p_thresh and not in_p:
                peaks += 1; in_p = True
            elif v < p_thresh: in_p = False
        return peaks
    return max(1, count_peaks(x_proj)), max(1, count_peaks(y_proj))

# --- 3. 메인 화면 로직 ---
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        h, w = raw_img.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img = cv2.warpAffine(raw_img, M_rot, (w, h))
