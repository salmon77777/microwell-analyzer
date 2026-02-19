import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="격자 복원형 분석기", layout="wide")
st.title("🧬 격자 복원형 Microwell 분석기")
st.info("신호가 없는(어두운) Well까지 격자 간격을 계산하여 자동으로 찾아냅니다.")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 인식 정밀도 (패턴 찾기)")
block_size = st.sidebar.slider("이진화 블록 크기", 3, 99, 31, step=2)
offset = st.sidebar.slider("이진화 보정치", 0, 50, 10)

st.sidebar.header("📏 2. Well 크기 필터")
min_area = st.sidebar.slider("Well 최소 면적", 1, 500, 50)
max_area = st.sidebar.slider("Well 최대 면적", 10, 2000, 800)

st.sidebar.header("🧪 3. 판정 및 분석")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
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
        
        # 1. 일차적 패턴 인식 (밝은 Well 찾기)
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
            
            # 2. 격자 복원 로직 (Grid Reconstruction)
            if grid_fix and len(found_wells) > 10:
                # 발견된 Well들의 좌표를 기반으로 격자 범위 추출
                all_x = found_wells[:, 0]
                all_y = found_wells[:, 1]
                
                # 평균 간격 계산 (X, Y축 각각)
                # 정렬 후 차이값의 중앙값으로 간격 추측
                dx = np.median(np.diff(np.unique(np.sort(all_x))))
                dy = np.median(np.diff(np.unique(np.sort(all_y))))
                
                # 실제 격자 좌표 생성 (발견된 Well의 최소~최대 범위 내)
                grid_x = np.arange(all_x.min(), all_x.max() + 1, dx)
                grid_y = np.arange(all_y.min(), all_y.max() + 1, dy)
                
                final_wells = []
                for gy in grid_y:
                    for gx in grid_x:
                        final_wells.append([int(gx), int(gy), int(np.mean(found_wells[:, 2]))])
            else:
                final_wells = found_wells

            # 3. 최종 분석 및 시각화
            res_img = img_rgb.copy()
            pos_cnt = 0
            
            for cx, cy, r in final_wells:
                # 좌표가 이미지 범위를 벗어나지 않게 처리
                if 0 <= cx < w and 0 <= cy < h:
                    # 노란색: 모든 Well (찾은 것 + 복원한 것)
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
            st.warning("먼저 밝은 Well들이 인식되도록 설정을 조절하세요.")
