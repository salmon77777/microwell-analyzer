import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="격자 강제 생성 분석기", layout="wide")
st.title("🧬 격자 배치형 Microwell 분석기")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 인식 정밀도 (Well 찾기)")
block_size = st.sidebar.slider("이진화 블록 크기", 3, 99, 31, step=2)
offset = st.sidebar.slider("이진화 보정치", 0, 50, 10)

st.sidebar.header("📏 2. Well 면적 필터")
min_area = st.sidebar.slider("Well 최소 면적", 1, 500, 50)
max_area = st.sidebar.slider("Well 최대 면적", 10, 2000, 800)

st.sidebar.header("🗺️ 3. 격자 설정 (중요)")
# 이미지를 보고 가로/세로 Well 개수를 직접 지정합니다.
cols_num = st.sidebar.number_input("가로 Well 개수", 1, 100, 23)
rows_num = st.sidebar.number_input("세로 Well 개수", 1, 100, 24)

st.sidebar.header("🧪 4. 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 일차적으로 보이는 Well들 찾기 (영역 파악용)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        found_pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                found_pts.append([cx, cy])

        if len(found_pts) >= 4:
            found_pts = np.array(found_pts)
            # 발견된 Well들의 외곽 범위를 기준으로 격자 영역 설정
            min_x, max_x = found_pts[:, 0].min(), found_pts[:, 0].max()
            min_y, max_y = found_pts[:, 1].min(), found_pts[:, 1].max()
            
            # 격자 좌표 생성
            grid_wells = []
            x_coords = np.linspace(min_x, max_x, cols_num)
            y_coords = np.linspace(min_y, max_y, rows_num)
            
            # 반지름은 인식된 원들의 평균값 혹은 기본값 10 사용
            avg_r = 10 
            
            res_img = img_rgb.copy()
            pos_cnt = 0
            
            # 2. 생성된 격자 순회하며 분석
            for gy in y_coords:
                for gx in x_coords:
                    cx, cy = int(gx), int(gy)
                    # 노란색 격자 그리기
                    cv2.circle(res_img, (cx, cy), avg_r, (255, 255, 0), 1)
                    
                    # 중심부 녹색값 체크
                    roi_g = img_rgb[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2), 1]
                    avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
                    
                    if avg_g > threshold_g:
                        pos_cnt += 1
                        cv2.circle(res_img, (cx, cy), int(avg_r*0.6), (0, 255, 0), -1)

            st.image(res_img, use_container_width=True)
            
            total = cols_num * rows_num
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 Well (격자)", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")
            
            # 보정 팁
            st.caption("💡 팁: 격자가 Well 위치와 안 맞으면 '이진화 보정치'를 조절해 외곽 범위를 먼저 잡으세요.")
        else:
            st.warning("분석 영역을 잡기 위해 최소 4개 이상의 Well이 먼저 인식되어야 합니다. '최소 면적'을 낮추거나 '이진화 보정치'를 낮춰보세요.")
