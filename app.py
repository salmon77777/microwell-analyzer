import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="초정밀 Well 분석기", layout="wide")
st.title("🔬 Microwell 초정밀 자동 분석기")

# --- 사이드바: 거리 제한 해제 ---
st.sidebar.header("⚙️ 인식 정밀도 조절")
st.sidebar.info("Well 사이 거리를 1로 설정하면 가장 빽빽한 격자도 찾아낼 수 있습니다.")

# 최소 거리 하한선을 1로 변경
min_dist = st.sidebar.slider("Well 사이 최소 거리", 1, 100, 5) # 기본값을 5로 대폭 낮춤
sensitivity = st.sidebar.slider("인식 민감도 (낮을수록 많이 찾음)", 1, 50, 20)

st.sidebar.header("📏 Well 크기 설정")
well_radius = st.sidebar.slider("Well 반지름 (픽셀)", 1, 100, 15)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 60)
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
        
        # 전처리: 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # [핵심] 적응형 이진화: 주변보다 조금이라도 어둡거나 밝으면 추출
        # blockSize는 Well 크기보다 커야 하므로 자동 계산 (홀수여야 함)
        bs = (well_radius * 2) + 1
        if bs % 2 == 0: bs += 1
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, max(3, bs), sensitivity
        )
        
        # 윤곽선 탐지
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = img_rgb.copy()
        valid_wells = []
        
        # 면적 필터 (사용자가 설정한 반지름 기준)
        target_area = np.pi * (well_radius ** 2)
        min_a, max_a = target_area * 0.2, target_area * 3.0

        # 중복 방지를 위한 좌표 저장
        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_a < area < max_a:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cx, cy = int(cx), int(cy)
                
                # [필터링] Well 사이 최소 거리 체크
                too_close = False
                for ox, oy in centers:
                    dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                    if dist < min_dist:
                        too_close = True
                        break
                
                if too_close: continue
                
                # 가장자리 잘린 Well 제외 (마진 2px)
                if (cx - r < 2) or (cx + r > w - 2) or (cy - r < 2) or (cy + r > h - 2):
                    continue
                
                centers.append((cx, cy))
                valid_wells.append((cx, cy, int(r)))

        # 결과 분석
        pos_cnt = 0
        if valid_wells:
            for cx, cy, r in valid_wells:
                # 노란색: 탐지된 모든 Well
                cv2.circle(res_img, (cx, cy), r, (255, 255, 0), 1)
                
                # 형광 분석
                roi = img_rgb[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2), 1]
                avg_g = np.mean(roi) if roi.size > 0 else 0
                
                if avg_g > threshold_g:
                    pos_cnt += 1
                    cv2.circle(res_img, (cx, cy), max(1, int(r*0.5)), (0, 255, 0), -1)

            st.image(res_img, use_container_width=True)
            
            total = len(valid_wells)
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("탐지된 전체 Well", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")
        else:
            st.warning("Well이 감지되지 않았습니다. 사이드바의 설정을 조절하세요.")
        
        # 왜 안 잡히는지 확인하기 위한 흑백 이미지 출력
        with st.expander("인식용 흑백 필터 (디버깅용)"):
            st.image(thresh, caption="여기에 하얀 점들이 생겨야 Well로 인식됩니다.")
    else:
        st.error("이미지를 읽을 수 없습니다.")
