import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Auto-Grid Analyzer", layout="wide")
st.title("🔬 스마트 자동 격자 Microwell 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 사진 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 분석 영역(4점) 설정")
# 업로드된 이미지 크기에 맞춰 조절할 수 있도록 범위를 넉넉히 설정
tl_x = st.sidebar.number_input("좌상 X", 0, 5000, 100)
tl_y = st.sidebar.number_input("좌상 Y", 0, 5000, 100)
tr_x = st.sidebar.number_input("우상 X", 0, 5000, 1000)
tr_y = st.sidebar.number_input("우상 Y", 0, 5000, 100)
bl_x = st.sidebar.number_input("좌하 X", 0, 5000, 100)
bl_y = st.sidebar.number_input("좌하 Y", 0, 5000, 800)
br_x = st.sidebar.number_input("우하 X", 0, 5000, 1000)
br_y = st.sidebar.number_input("우하 Y", 0, 5000, 800)

st.sidebar.header("🧪 3단계: 판정 및 감도")
radius = st.sidebar.slider("우물 표시 크기", 1, 20, 5)
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도 (Peak)", 0.1, 1.0, 0.5)

# 2. 이미지 처리 함수
def get_auto_count(roi_gray, sensitivity):
    """이미지 투영을 통해 행/열 개수를 자동 계산"""
    # X축(가로) 및 Y축(세로) 평균 밝기 계산
    x_proj = np.mean(roi_gray, axis=0)
    y_proj = np.mean(roi_gray, axis=1)
    
    # 간단한 피크 카운팅 로직 (평균값 이상을 피크로 간주)
    def count_peaks(proj):
        avg = np.mean(proj)
        peaks = 0
        is_peak = False
        threshold_val = avg + (np.max(proj) - avg) * (1 - sensitivity)
        for val in proj:
            if val > threshold_val and not is_peak:
                peaks += 1
                is_peak = True
            elif val < threshold_val:
                is_peak = False
        return max(1, peaks)

    return count_peaks(x_proj), count_peaks(y_proj)

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
        
        # [영역 잘라내기 및 개수 자동 파악]
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
        
        # 원근 변환(Perspective Transform)을 통해 영역을 평면으로 펴서 개수 분석
        target_w, target_h = 800, 800 # 분석용 임시 해상도
        pts_dst = np.array([[0,0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # 행/열 개수 자동 감지
        auto_cols, auto_rows = get_auto_count(warped_gray, sensitivity)
        
        st.info(f"🔎 시스템이 감지한 격자 크기: 가로 {auto_cols}개 x 세로 {auto_rows}개")

        # [격자 그리기 및 분석]
        display_img = img_rgb.copy()
        pos_count = 0
        total_wells = auto_cols * auto_rows

        for r in range(auto_rows):
            v_ratio = r / (auto_rows - 1) if auto_rows > 1 else 0
            left = (1 - v_ratio) * pts_src[0] + v_ratio * pts_src[3]
            right = (1 - v_ratio) * pts_src[1] + v_ratio * pts_src[2]
            
            for c in range(auto_cols):
                h_ratio = c / (auto_cols - 1) if auto_cols > 1 else 0
                center = (1 - h_ratio) * left + h_ratio * right
                cx, cy = int(center[0]), int(center[1])

                if 0 <= cx < w and 0 <= cy < h:
                    g_val = img_rgb[cy, cx, 1]
                    color = (0, 255, 0) if g_val > threshold else (255, 0, 0)
                    if g_val > threshold: pos_count += 1
                    cv2.circle(display_img, (cx, cy), radius, color, 1)

        # 가이드 라인 및 결과 출력
        cv2.polylines(display_img, [pts_src.astype(int)], True, (255, 255, 0), 2)
        st.image(display_img, use_container_width=True)
        
        # 분석 요약
        c1, c2, c3 = st.columns(3)
        c1.metric("감지된 우물 수", f"{total_wells}개")
        c2.metric("Positive", f"{pos_count}개")
        c3.metric("비율", f"{(pos_count/total_wells*100):.1f}%" if total_wells > 0 else "0%")
