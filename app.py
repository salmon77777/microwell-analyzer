import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Precision Analyzer", layout="wide")
st.title("🔬 정밀 원근 보정형 Well 분석기")

# --- 유틸리티 함수 ---
def order_points(pts):
    """좌표를 좌상, 우상, 우하, 좌하 순서로 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_ruler_and_guide(img):
    h, w = img.shape[:2]
    r_img = img.copy()
    cv2.line(r_img, (0, h//2), (w, h//2), (255, 0, 0), 2)
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
        h, w = img_bgr.shape[:2]
        
        # 사이드바 설정
        st.sidebar.header("🔄 1단계: 수평 및 영역 설정")
        rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)
        
        # 초기 좌표값 제안 (이미지 크기의 10% 여백)
        sc1, sc2 = st.sidebar.columns(2)
        tl_x = sc1.number_input("좌상 X", 0, w, int(w*0.1))
        tl_y = sc2.number_input("좌상 Y", 0, h, int(h*0.1))
        tr_x = sc1.number_input("우상 X", 0, w, int(w*0.9))
        tr_y = sc2.number_input("우상 Y", 0, h, int(h*0.1))
        br_x = sc1.number_input("우하 X", 0, w, int(w*0.9))
        br_y = sc2.number_input("우하 Y", 0, h, int(h*0.9))
        bl_x = sc1.number_input("좌하 X", 0, w, int(w*0.1))
        bl_y = sc2.number_input("좌하 Y", 0, h, int(h*0.9))

        st.sidebar.header("🧪 2단계: 분석 설정")
        auto_mode = st.sidebar.checkbox("Well 개수 자동 인식", value=True)
        manual_cols = st.sidebar.number_input("가로 Well (수동)", 1, 100, 23) if not auto_mode else 23
        manual_rows = st.sidebar.number_input("세로 Well (수동)", 1, 100, 24) if not auto_mode else 24
        
        radius = st.sidebar.slider("Well 반지름", 1, 30, 5)
        threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
        sensitivity = st.sidebar.slider("인식 민감도", 0.1, 2.0, 1.1)
        gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

        # 이미지 회전
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img_rot = cv2.warpAffine(img_bgr, M_rot, (w, h))
        img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)

        tab1, tab2 = st.tabs(["📝 좌표 확인", "📊 분석 결과"])

        with tab1:
            st.image(draw_ruler_and_guide(img_rgb), use_container_width=True)

        with tab2:
            # 입력된 4점 좌표
            src_pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype="float32")
            
            # [핵심] 원근 변환을 통해 격자만 똑바로 펴기 (Warp Perspective)
            target_w, target_h = 1200, 1200
            dst_pts = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype="float32")
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_rot, M_persp, (target_w, target_h))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # 펴진 이미지에서 Well 개수 자동 분석
            if auto_mode:
                _, th_img = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_proj = np.mean(th_img, axis=0)
                y_proj = np.mean(th_img, axis=1)
                
                def count_peaks(proj, sens):
                    limit = np.mean(proj) * sens
                    cnt, in_peak = 0, False
                    for v in proj:
                        if v > limit and not in_peak:
                            cnt += 1; in_peak = True
                        elif v < limit: in_peak = False
                    return cnt
                
                f_cols = count_peaks(x_proj, sensitivity)
                f_rows = count_peaks(y_proj, sensitivity)
            else:
                f_cols, f_rows = manual_cols, manual_rows

            # 분석 결과 시각화 (원본 이미지에 매핑)
            res_img = img_rgb.copy()
            pos_cnt = 0
            
            # 4개 모서리 사이를 선형 보간하여 좌표 계산 (Perspective-aware)
            for r in range(f_rows):
                # 세로비율
                v_frac = r / (f_rows - 1) if f_rows > 1 else 0.5
                left_edge = (1 - v_frac) * src_pts[0] + v_frac * src_pts[3]
                right_edge = (1 - v_frac) * src_pts[1] + v_frac * src_pts[2]
                
                for c in range(f_cols):
                    # 가로비율
                    h_frac = c / (f_cols - 1) if f_cols > 1 else 0.5
                    well_center = (1 - h_frac) * left_edge + h_frac * right_edge
                    cx, cy = int(well_center[0]), int(well_center[1])
                    
                    if 0 <= cx < w and 0 <= cy < h:
                        # 중심점의 Green 값 확인
                        is_pos = img_rgb[cy, cx, 1] > threshold
                        if is_pos: pos_cnt += 1
                        cv2.circle(res_img, (cx, cy), radius, (0,255,0) if is_pos else (255,0,0), 1)

            cv2.polylines(res_img, [src_pts.astype(int)], True, (255, 255, 0), 2)
            st.image(res_img, use_container_width=True)
            
            # 결과 표시
            total = f_cols * f_rows
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            st.info(f"📊 **인식 결과:** 가로 {f_cols}개 x 세로 {f_rows}개 (총 {total} Well)")
            if ratio >= gmo_thresh:
                st.success(f"### 🧬 판정: GMO Positive ({ratio:.1f}%)")
            else:
                st.error(f"### 🧬 판정: Non-GMO ({ratio:.1f}%)")
