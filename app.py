import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="AI 패턴 인식 분석기", layout="wide")
st.title("🔬 Microwell 패턴 기반 자동 분석기")

# --- 사이드바: 인식 파라미터 (좌표 입력 없음) ---
st.sidebar.header("⚙️ 인식 정밀도 조절")
st.sidebar.info("원이 아니라 '구멍 패턴'을 찾습니다. 인식이 안 되면 아래 값을 조절하세요.")

# 밝기 대비를 조절하여 구멍을 도드라지게 함
block_size = st.sidebar.slider("적응형 이진화 블록 크기", 3, 99, 31, step=2)
offset = st.sidebar.slider("이진화 보정치", 0, 50, 10)

st.sidebar.header("📏 Well 크기 필터")
min_area = st.sidebar.slider("Well 최소 면적", 10, 1000, 100)
max_area = st.sidebar.slider("Well 최대 면적", 500, 5000, 1500)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 65)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요. 패턴을 자동 분석합니다.", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 이미지 전처리: 밝기 대비 강조 (이진화)
        # 주변보다 밝은 Well 구멍들을 도드라지게 만듭니다.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, offset
        )
        
        # 2. 윤곽선(Contour) 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = img_rgb.copy()
        pos_cnt = 0
        valid_wells = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 설정한 면적 범위 내에 있는 것만 Well로 인정
            if min_area < area < max_area:
                # 원형도(Circularity) 체크: 너무 길쭉한 것은 제외
                peri = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
                
                if circularity > 0.5: # 0.5 이상이면 어느 정도 둥근 형태
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    cx, cy, r = int(cx), int(cy), int(radius)
                    
                    # [필터링] 사진 가장자리에 걸친 Well 제외
                    if (cx - r < 5) or (cx + r > w - 5) or (cy - r < 5) or (cy + r > h - 5):
                        continue
                        
                    valid_wells.append((cx, cy, r))

        # 3. 결과 시각화 및 GMO 분석
        if valid_wells:
            for cx, cy, r in valid_wells:
                # Well 중심부 Green 채널 분석
                roi_g = img_rgb[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3), 1]
                avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
                
                is_pos = avg_g > threshold_g
                if is_pos:
                    pos_cnt += 1
                
                color = (0, 255, 0) if is_pos else (255, 0, 0)
                cv2.circle(res_img, (cx, cy), r, color, 1)

            st.image(res_img, use_container_width=True)
            
            # 통계 결과
            total = len(valid_wells)
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("탐지된 전체 Well", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")

            if ratio >= gmo_thresh:
                st.success("🧬 판정 결과: **GMO Positive**")
            else:
                st.error("🧬 판정 결과: **Non-GMO**")
        else:
            st.warning("분석 가능한 Well 패턴을 찾지 못했습니다. 사이드바에서 'Well 최소 면적'을 줄여보세요.")
            # 디버깅용 이진화 이미지 (왜 못 찾는지 확인용)
            if st.checkbox("시스템 인식용 이미지 보기"):
                st.image(thresh, caption="이 이미지에서 하얀 점이 Well입니다. 점이 안 보이면 설정을 조절하세요.")
    else:
        st.error("이미지를 불러올 수 없습니다.")
