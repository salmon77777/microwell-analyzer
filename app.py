import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Ruler Analyzer", layout="wide")
st.title("🔬 눈금자 가이드형 자동 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
st.sidebar.info("이미지의 눈금을 보고 좌표를 입력하세요.")
col1, col2 = st.sidebar.columns(2)
tl_x = col1.number_input("좌상 X", 0, 5000, 150)
tl_y = col2.number_input("좌상 Y", 0, 5000, 200)
tr_x = col1.number_input("우상 X", 0, 5000, 2300)
tr_y = col2.number_input("우상 Y", 0, 5000, 200)
bl_x = col1.number_input("좌하 X", 0, 5000, 150)
bl_y = col2.number_input("좌하 Y", 0, 5000, 2300)
br_x = col1.number_input("우하 X", 0, 5000, 2300)
br_y = col2.number_input("우하 Y", 0, 5000, 2300)

st.sidebar.header("🧪 3단계: 판정 설정")
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 1.0, 0.5)

# 2. 이미지 처리 함수들
def draw_ruler(img):
    """이미지 상단과 좌측에 픽셀 눈금자를 그리는 함수"""
    h, w = img.shape[:2]
    ruler_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255) # 흰색 눈금
    bg_color = (0, 0, 0)    # 검은색 배경 (글자 가독성)

    # 상단 가로 눈금 (X축)
    for x in range(0, w, 100):
        cv2.line(ruler_img, (x, 0), (x, 30), color, 2)
        cv2.putText(ruler_img, str(x), (x+5, 25), font, font_scale, color, thickness)
    
    # 좌측 세로 눈금 (Y축)
    for y in range(0, h, 100):
        cv2.line(ruler_img, (0, y), (30, y), color, 2)
        cv2.putText(ruler_img, str(y), (5, y-5), font, font_scale, color, thickness)
    
    return ruler_img

def get_auto_count(roi_gray, sens):
    x_proj = np.mean(roi_gray, axis=0)
    y_proj = np.mean(roi_gray, axis=1)
    def count_peaks(proj):
        avg = np.mean(proj)
        std = np.std(proj)
        thresh = avg + std * sens
        peaks = [i for i in range(1, len(proj)-1) if proj[i] > thresh and proj[i] > proj[i-1] and proj[i] > proj[i+1]]
        return len(peaks)
    return max(1, count_peaks(x_proj)), max(1, count_peaks(y_proj))

# 3. 메인 로직
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        # 회전 적용
        h, w = raw_img.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img = cv2.warpAffine(raw_img, M_rot, (w, h))
        
        # 눈금자가 그려진 이미지 생성 (사용자 입력 가이드용)
        ruler_guide_img = draw_ruler(img)
        
        # 실제 분석용 4점 좌표
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

        # 영역 내부 개수 자동 파악
        tw, th = 1000, 1000
        M_persp = cv2.getPerspectiveTransform(pts_src, np.array([[0,0], [tw, 0], [tw, th], [0, th]], dtype=np.float32))
        warped = cv2.warpPerspective(img, M_persp, (tw, th))
        auto_cols, auto_rows = get_auto_count(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), sensitivity)

        # 결과 시각화
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pos_count = 0
        for r in range(auto_rows):
            v = r / (auto_rows - 1) if auto_rows > 1 else 0
            line_l = (1-v)*pts_src[0] + v*pts_src[3]
            line_r = (1-v)*pts_src[1] + v*pts_src[2]
            for c in range(auto_cols):
                h_r = c / (auto_cols - 1) if auto_cols > 1 else 0
                pt = (1-h_r)*line_l + h_r*line_r
                cx, cy = int(pt[0]), int(pt[1])
                if 0 <= cx
