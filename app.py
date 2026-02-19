import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="격자 복원형 분석기", layout="wide")
st.title("🧬 격자 복원형 Microwell 분석기")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 인식 정밀도")
block_size = st.sidebar.slider("이진화 블록 크기", 3, 99, 31, step=2)
offset = st.sidebar.slider("이진화 보정치", 0, 50, 10)

st.sidebar.header("📏 2. Well 면적 필터")
min_area = st.sidebar.slider("Well 최소 면적", 1, 500, 50)
max_area = st.sidebar.slider("Well 최대 면적", 10, 2000, 800)

st.sidebar.header("🧪 3. 판정 및 분석")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
# 복원 기능을 사용자가 끄고 켤 수 있게 함
grid_fix = st.sidebar.checkbox("빈 Well 격자 자동 복원", value=True)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 일차적 패턴 인식
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        found_wells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                found_wells.append([int(cx), int(cy), int(r)])

        if found_wells:
            found_wells = np.array(found_wells)
            final_wells = []

            # 2. 격자 복원 로직 (무한 루프 방지 안전장치 추가)
            if grid_fix and len(found_wells) >= 5:
                all_x = found_wells[:, 0]
                all_y = found_wells[:, 1]
                avg_r = int(np.mean(found_wells[:, 2]))
                
                # 중복되지 않는 좌표들 사이의 최소 간격 추정
                ux = np.sort(np.unique(all_x))
                uy = np.sort(np.unique(all_y))
                
                # 간격 계산 (최소 10픽셀 이상으로 제한하여 무한 루프 방지)
                dx = max(10, np.median(np.diff(ux))) if len(ux) > 1 else 30
                dy = max(10, np.median(np.diff(uy))) if len(uy) > 1 else 30
                
                # 격자 생성 (이미지 범위를 벗어나지 않도록 안전하게 생성)
                start_x, end_x = all_x.min(), all_x.max()
                start_y, end_y = all_y.min(), all_y.max()
                
                # 개수가 너무 많아지는 것을 방지 (최대 100x100)
                num_cols = min(100, int((end_x - start_x) / dx) + 1)
                num_rows = min(100, int((end_y - start_y) / dy) + 1)
                
                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        final_wells.append([int(start_x + c_idx * dx), int(start_y + r_idx * dy), avg_r])
            else:
                final_wells = found_wells.tolist()

            # 3. 분석 및 시각화
            res_img = img_rgb.copy()
            pos_cnt = 0
            
            for cx, cy, r in final_wells:
                if 0 <= cx < w and 0 <= cy < h:
                    # 인식/복원된 모든 Well은 노란색 테두리
                    cv2.circle(res_img, (cx, cy), r, (255, 255, 0), 1)
                    
                    # 중심부 녹색값 체크
                    roi_g = img_rgb[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2), 1]
                    avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
                    
                    if avg_g > threshold_g:
                        pos_cnt += 1
                        cv2.circle(res_img, (cx, cy), max(1, int(r*0.5)), (0, 255, 0), -1)

            st.image(res_img, use_container_width=True)
            
            total = len(final_wells)
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 Well (격자 포함)", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")
        else:
            st.warning("먼저 Well들이 인식되도록 설정을 조절하세요.")
