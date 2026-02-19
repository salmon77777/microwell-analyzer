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
        cnt, peak =
