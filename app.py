import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 정밀 보정형 Microwell 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
col1, col2 = st.sidebar.columns(2)
tl_x = col1.number_input("좌상 X", 0, 5000, 150)
tl_y = col2.number_input("좌상 Y", 0, 5000, 200)
tr_x = col1.number_input("우상 X", 0, 5000, 2300)
tr_y = col2.number_input("우상 Y", 0, 5000, 200)
bl_x = col1.number_input("좌하 X", 0, 5000, 150)
bl_y = col2.number_input("좌하 Y", 0, 5000, 2300)
br_x = col1.number_input("우하 X", 0, 5000, 2300)
br_y = col2.number_input("우하 Y", 0, 5000, 2300)

st.sidebar.header("🔢 3단계: 격자 개수 설정")
auto_mode = st.sidebar.checkbox("우물 개수 자동 인식", value=True)
if not auto_mode:
    manual_cols = st.sidebar.number_input("가로 개수 수동 입력", 1, 100, 23)
    manual_rows = st.sidebar.number_input("세로 개수 수동 입력", 1, 100, 24)

st.sidebar.header("🧪 4단계: 판정 및 크기")
radius = st.sidebar.slider("우물 반지름", 1, 30, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.0, step=0.1)

# 유틸리티 함수
def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    ruler_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(ruler_img, (0, h//2), (w, h//2), (0, 255, 0), 1)
    cv2.line(ruler_img, (w//2, 0), (w//2, h), (0, 255, 0), 1)
    for x in range(0, w, 100):
        cv2.line(ruler_img, (x, 0), (x, 30), (255, 255, 255), 2)
        cv2.putText(ruler_img, str(x), (x+5, 25), font, 0.5, (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(ruler_img, (0, y), (30, y), (255, 255, 255), 2)
        cv2.putText(ruler_img, str(y), (5, y-5), font, 0.5, (255, 255, 255), 1)
    return ruler_img

def get_auto_count(roi_gray, sens):
    # 이진화를 통해 피크를 더 명확하게 분리
    _, thresh_img = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x_proj = np.mean(thresh_img, axis=0)
    y_proj = np.mean(thresh_img, axis=1)
    
    def count_peaks(proj):
        avg = np.mean(proj)
        # 민감도를 적용한 임계값 설정
        p_thresh = avg * sens
        peaks = 0
        in_peak = False
        for val in proj:
            if val > p_thresh and not in_peak:
                peaks += 1
                in_peak = True
            elif val < p_thresh:
                in_peak = False
        return peaks
    
    return max(1, count_peaks(x_proj)), max(1, count_peaks(y_proj))

# 메인 프로세스
uploaded_file = st.file_uploader("사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        h, w = raw_img.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img = cv2.warpAffine(raw_img, M_rot, (w, h))
        
        ruler_guide_img = draw_ruler_and_guide(img)
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

        # 개수 결정
        if auto_mode:
            tw, th = 1000, 1000
            M_persp = cv2.getPerspectiveTransform(pts_src, np.array([[0,0], [tw, 0], [tw, th], [0, th]], dtype=np.float32))
            warped_gray = cv2.cvtColor(cv2.warpPerspective(img, M_persp, (tw, th)), cv2.COLOR_BGR2GRAY)
            final_cols, final_rows = get_auto_count(warped_gray, sensitivity)
        else:
            final_cols, final_rows = manual_cols, manual_rows

        st.info(f"현재 설정된 격자 크기: 가로 {final_cols}개 x 세로 {final_rows}개")

        # 결과 분석 및 시각화
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pos_count = 0
        total_wells = final_cols * final_rows
        
        for r in range(final_rows):
            v = r / (final_rows - 1) if final_rows > 1 else 0
            l_edge = (1-v)*pts_src[0] + v*pts_src[3]
            r_edge = (1-v)*pts_src[1] + v*pts_src[2]
            for c in range(final_cols):
                h_r = c / (final_cols - 1) if final_cols > 1 else 0
                pt = (1-h_r)*l_edge + h_r*r_edge
                cx, cy = int(pt[0]), int(pt[1])
                
                if 0 <= cx < w and 0 <= cy < h:
                    g_val = display_img[cy, cx, 1]
                    is_pos = g_val > threshold
                    if is_pos: pos_count += 1
                    cv2.circle(display_img, (cx, cy), radius, (0, 255, 0) if is_pos else (255, 0, 0), 1)

        cv2.polylines(display_img, [pts_src.astype(int)], True, (255, 255, 0), 2)

        tab1, tab2 = st.tabs(["📝 좌표/수평 확인", "📊 분석 결과"])
        with tab1:
            st.image(ruler_guide_img, use_container_width=True)
        with tab2:
            st.image(display_img, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 우물", f"{total_wells}개")
            c2.metric("Positive", f"{pos_count}개")
            c3.metric("비율", f"{(pos_count/total_wells*100):.1f}%" if total_wells > 0 else "0%")
