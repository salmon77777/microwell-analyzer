import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="AI 패턴 인식 분석기", layout="wide")
st.title("🔬 Microwell 자동 분석 (노란색: 인식된 Well)")

# --- 사이드바: 인식 파라미터 ---
st.sidebar.header("⚙️ 인식 정밀도 조절")
block_size = st.sidebar.slider("적응형 이진화 블록 크기", 3, 99, 31, step=2)
offset = st.sidebar.slider("이진화 보정치", 0, 50, 10)

st.sidebar.header("📏 Well 크기 필터")
min_area = st.sidebar.slider("Well 최소 면적", 10, 1000, 100)
max_area = st.sidebar.slider("Well 최대 면적", 500, 5000, 1500)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 65)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 이미지 전처리 (이진화)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, offset
        )
        
        # 2. 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = img_rgb.copy()
        pos_cnt = 0
        valid_wells = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                peri = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
                
                if circularity > 0.4: # 원형도 기준을 살짝 완화
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    cx, cy, r = int(cx), int(cy), int(radius)
                    
                    # 가장자리 필터링
                    if (cx - r < 5) or (cx + r > w - 5) or (cy - r < 5) or (cy + r > h - 5):
                        continue
                    valid_wells.append((cx, cy, r))

        # 3. 결과 시각화
        if valid_wells:
            for cx, cy, r in valid_wells:
                # [중요] 모든 인식된 Well은 노란색(Yellow) 테두리로 표시
                cv2.circle(res_img, (cx, cy), r, (255, 255, 0), 1)
                
                # Green 채널 분석
                roi_g = img_rgb[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2), 1]
                avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
                
                # Positive 판정 시 초록색 점 추가
                if avg_g > threshold_g:
                    pos_cnt += 1
                    cv2.circle(res_img, (cx, cy), int(r*0.4), (0, 255, 0), -1) # 내부를 채운 초록색 원

            st.image(res_img, use_container_width=True, caption="노란색: 인식된 Well / 초록색 점: Positive 판정")
            
            # 통계 결과
            total = len(valid_wells)
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("탐지된 유효 Well", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")

            if ratio >= gmo_thresh:
                st.success(f"### 🧬 판정 결과: GMO Positive ({ratio:.1f}%)")
            else:
                st.error(f"### 🧬 판정 결과: Non-GMO ({ratio:.1f}%)")
        else:
            st.warning("분석 가능한 Well을 찾지 못했습니다. 사이드바 설정을 조절하세요.")
            if st.checkbox("시스템 인식 이미지(이진화) 확인"):
                st.image(thresh)
