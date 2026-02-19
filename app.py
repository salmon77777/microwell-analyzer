import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Microwell Ruler Analyzer", layout="wide")
st.title("🔬 정밀 가이드형 자동 Microwell 분석기")

# 1. 사이드바: 설정
st.sidebar.header("🔄 1단계: 수평 보정")
rotation = st.sidebar.slider("사진 기울기 조절", -10.0, 10.0, 0.0, step=0.1)

st.sidebar.header("📍 2단계: 모서리 좌표 입력")
st.sidebar.info("눈금자(Ruler)와 중앙 가이드선을 보고 좌표를 입력하세요.")
col1, col2 = st.sidebar.columns(2)
tl_x = col1.number_input("좌상 X", 0, 5000, 150)
tl_y = col2.number_input("좌상 Y", 0, 5000, 200)
tr_x = col1.number_input("우상 X", 0, 5000, 2300)
tr_y = col2.number_input("우상 Y", 0, 5000, 200)
bl_x = col1.number_input("좌하 X", 0, 5000, 150)
bl_y = col2.number_input("좌하 Y", 0, 5000, 2300)
br_x = col1.number_input("우하 X", 0, 5000, 2300)
br_y = col2.number_input("우하 Y", 0, 5000, 2300)

st.sidebar.header("🧪 3단계: 판정 및 크기 설정")
radius = st.sidebar.slider("우물 반지름 (Radius)", 1, 30, 5) # 반지름 설정 부활
threshold = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
sensitivity = st.sidebar.slider("인식 민감도", 0.1, 1.0, 0.5)

# 2. 유틸리티 함수
def draw_ruler_and_guide(img):
    """눈금자와 중앙 십자 가이드선을 그리는 함수"""
    h, w = img.shape[:2]
    ruler_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- 중앙 가이드선 (수평/수직 맞춤용) ---
    guide_color = (0, 255, 0) # 녹색 가이드선
    cv2.line(ruler_img, (0, h//2), (w, h//2), guide_color, 1) # 중앙 가로선
    cv2.line(ruler_img, (w//2, 0), (w//2, h), guide_color, 1) # 중앙 세로선

    # --- 눈금자 (Ruler) ---
    color = (255, 255, 255) # 흰색
    for x in range(0, w, 100):
        cv2.line(ruler_img, (x, 0), (x, 30), color, 2)
        cv2.putText(ruler_img, str(x), (x+5, 25), font, 0.5, color, 1)
    
    for y in range(0, h, 100):
        cv2.line(ruler_img, (0, y), (30, y), color, 2)
        cv2.putText(ruler_img, str(y), (5, y-5), font, 0.5, color, 1)
    
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

# 3. 메인 프로세스
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is not None:
        # [회전 보정]
        h, w = raw_img.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), rotation, 1.0)
        img = cv2.warpAffine(raw_img, M_rot, (w, h))
        
        # [눈금자 및 가이드선 이미지]
        ruler_guide_img = draw_ruler_and_guide(img)
        
        # [4점 좌표 설정]
        pts_src = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

        # [영역 내부 개수 자동 파악]
        tw, th = 1000, 1000
        M_persp = cv2.getPerspectiveTransform(pts_src, np.array([[0,0], [tw, 0], [tw, th], [0, th]], dtype=np.float32))
        warped = cv2.warpPerspective(img, M_persp, (tw, th))
        auto_cols, auto_rows = get_auto_count(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), sensitivity)

        # [결과 시각화 및 판정]
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pos_count = 0
        
        # 격자 생성 및 원 그리기
        for r in range(auto_rows):
            v = r / (auto_rows - 1) if auto_rows > 1 else 0
            line_l = (1-v)*pts_src[0] + v*pts_src[3]
            line_r = (1-v)*pts_src[1] + v*pts_src[2]
            for c in range(auto_cols):
                h_r = c / (auto_cols - 1) if auto_cols > 1 else 0
                pt = (1-h_r)*line_l + h_r*line_r
                cx, cy = int(pt[0]), int(pt[1])
                
                if 0 <= cx < w and 0 <= cy < h:
                    # 반지름을 고려한 평균 밝기 추출을 위해 간단한 ROI 설정
                    g_val = display_img[cy, cx, 1]
                    is_pos = g_val > threshold
                    if is_pos: pos_count += 1
                    
                    # 사용자가 설정한 반지름(radius)으로 원 그리기
                    cv2.circle(display_img, (cx, cy), radius, (0, 255, 0) if is_pos else (255, 0, 0), 1)

        # 분석 영역 테두리 표시
        cv2.polylines(display_img, [pts_src.astype(int)], True, (255, 255, 0), 2)

        # [UI 출력]
        tab1, tab2 = st.tabs(["📝 좌표 확인 (눈금자 & 가이드)", "📊 분석 결과"])
        
        with tab1:
            st.image(ruler_guide_img, caption="중앙 녹색선을 수평 기준으로 삼고, 눈금자를 보고 좌표를 입력하세요.", use_container_width=True)
        
        with tab2:
            st.image(display_img, caption=f"감지된 격자: {auto_cols} x {auto_rows}", use_container_width=True)
            
            total = auto_cols * auto_rows
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 우물 수", f"{total}개")
            c2.metric("Positive (녹색)", f"{pos_count}개")
            c3.metric("형광 발현율", f"{(pos_count/total*100):.1f}%" if total > 0 else "0%")

        # 데이터 저장 버튼
        res_bytes = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 결과 이미지 다운로드", data=res_bytes, file_name="microwell_analysis.png")
