import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 Microwell Well & GMO Analyzer")

# --- 2. 사이드바: 설정 영역 ---
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 영역 좌표 설정")
st.sidebar.info("이미지에 최적화된 기본값이 설정되었습니다.")

# 사용자 요청 순서: 좌상 -> 우상 -> 좌하 -> 우하
sc1, sc2 = st.sidebar.columns(2)

# 이미지 분석 기반 추천 Default 값 적용
tl_x = sc1.number_input("1. 좌상 X (Top-Left)", 0, 8000, 310)
tl_y = sc2.number_input("1. 좌상 Y (Top-Left)", 0, 8000, 410)

tr_x = sc1.number_input("2. 우상 X (Top-Right)", 0, 8000, 2750)
tr_y = sc2.number_input("2. 우상 Y (Top-Right)", 0, 8000, 410)

bl_x = sc1.number_input("3. 좌하 X (Bottom-Left)", 0, 8000, 310)
bl_y = sc2.number_input("3. 좌하 Y (Bottom-Left)", 0, 8000, 2900)

br_x = sc1.number_input("4. 우하 X (Bottom-Right)", 0, 8000, 2750)
br_y = sc2.number_input("4. 우하 Y (Bottom-Right)", 0, 8000, 2900)

st.sidebar.header("🔢 3단계: Well & 분석 설정")
auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
manual_cols = st.sidebar.number_input("가로 Well (수동)", 1, 100, 23) if not auto_mode else 23
manual_rows = st.sidebar.number_input("세로 Well (수동)", 1, 100, 24) if not auto_mode else 24

radius = st.sidebar.slider("Well 반지름", 1, 30, 8) # 조금 더 잘 보이게 8로 조정
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.1)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 3. 유틸리티 함수 ---
def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    r_img = img.copy()
    cv2.line(r_img, (0, h//2), (w, h//2), (255, 0, 0), 2) # 중앙 Red 가이드선
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

# --- 4. 메인 분석 로직 ---
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    f_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img_rot = cv2.warpAffine(img_bgr, M_rot, (w, h))
        img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
        
        tab1, tab2 = st.tabs(["📝 좌표 확인 (Red Guide)", "📊 분석 결과"])
        
        with tab1:
            st.image(draw_ruler_and_guide(img_rgb), use_container_width=True)
            
        with tab2:
            # 입력 좌표 (정렬 순서 준수)
            src_pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype="float32")
            
            # Well 개수 인식 (원근 보정 적용)
            target_size = 1200
            dst_pts = np.array([[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]], dtype="float32")
            M_p = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_rot, M_p, (target_size, target_size))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            if auto_mode:
                _, th_img = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_p, y_p = np.mean(th_img, axis=0), np.mean(th_img, axis=1)
                def count_peaks(proj, sens):
                    lim = np.mean(proj) * sens
                    cnt, peak = 0, False
                    for v in proj:
                        if v > lim and not peak: cnt += 1; peak = True
                        elif v < lim: peak = False
                    return cnt
                f_cols, f_rows = count_peaks(x_p, sensitivity), count_peaks(y_p, sensitivity)
            else:
                f_cols, f_rows = manual_cols, manual_rows

            # 분석 시각화
            res_img = img_rgb.copy()
            pos_cnt = 0
            for r in range(f_rows):
                v_f = r/(f_rows-1) if f_rows > 1 else 0.5
                lp, rp = (1-v_f)*src_pts[0] + v_f*src_pts[3], (1-v_f)*src_pts[1] + v_f*src_pts[2]
                for c in range(f_cols):
                    h_f = c/(f_cols-1) if f_cols > 1 else 0.5
                    cp = (1-h_f)*lp + h_f*rp
                    cx, cy = int(cp[0]), int(cp[1])
                    if 0 <= cx < w and 0 <= cy < h:
                        is_pos = img_rgb[cy, cx, 1] > threshold
                        if is_pos: pos_cnt += 1
                        cv2.circle(res_img, (cx, cy), radius, (0,255,0) if is_pos else (255,0,0), 1)

            cv2.polylines(res_img, [src_pts.astype(int)], True, (255, 255, 0), 2)
            st.image(res_img, use_container_width=True)
            
            total = f_cols * f_rows
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            st.info(f"📊 **Grid Info:** 가로 {f_cols}개 x 세로 {f_rows}개 (총 {total} Well)")
            if ratio >= gmo_thresh:
                st.success(f"### 🧬 판정 결과: GMO Positive ({ratio:.1f}%)")
            else:
                st.error(f"### 🧬 판정 결과: Non-GMO ({ratio:.1f}%)")
