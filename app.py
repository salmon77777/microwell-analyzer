import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정
st.set_config(page_title="AI Well Auto-Detector", layout="wide")
st.title("🤖 Microwell 완전 자동 분석기 (AI Detection)")

# --- 사이드바: 감도 조절 (좌표 입력 대신 감도를 조절합니다) ---
st.sidebar.header("⚙️ 분석 정밀도 설정")
st.sidebar.info("좌표 입력이 필요 없습니다. 원이 잘 안 잡히면 아래 슬라이더를 조절하세요.")

min_dist = st.sidebar.slider("Well 간 최소 거리", 10, 100, 30)
param1 = st.sidebar.slider("엣지 감지 강도", 10, 100, 35)
param2 = st.sidebar.slider("원 인식 감도 (낮을수록 많이 찾음)", 5, 50, 20)
min_r = st.sidebar.slider("Well 최소 반지름", 1, 50, 15)
max_r = st.sidebar.slider("Well 최대 반지름", 1, 100, 30)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 65)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하면 즉시 자동 분석합니다", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 로드
    f_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거 (인식률 향상)
        blurred = cv2.medianBlur(gray, 5)
        
        # [핵심] 허프 변환을 이용한 모든 원 자동 감지
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 1, 
            minDist=min_dist,
            param1=param1, 
            param2=param2, 
            minRadius=min_r, 
            maxRadius=max_r
        )
        
        res_img = img_rgb.copy()
        pos_cnt = 0
        valid_well_cnt = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # [필터링] 사진 가장자리에 걸친 잘린 원 제외
                # 원의 중심이 아니라 원의 테두리가 이미지 범위 안에 있어야 함
                margin = 5 # 약간의 여백
                if (cx - r < margin) or (cx + r > w - margin) or \
                   (cy - r < margin) or (cy + r > h - margin):
                    continue # 잘린 원은 무시
                
                valid_well_cnt += 1
                
                # 중심점의 Green 값 분석 (is_pos 판정)
                # 원 내부의 평균을 구하면 더 정확하지만, 속도를 위해 중심점 주변 추출
                roi = img_rgb[max(0, cy-2):cy+3, max(0, cx-2):cx+3, 1]
                avg_g = np.mean(roi)
                
                is_pos = avg_g > threshold_g
                if is_pos:
                    pos_cnt += 1
                
                # 시각화 (테두리 1px)
                color = (0, 255, 0) if is_pos else (255, 0, 0)
                cv2.circle(res_img, (cx, cy), r, color, 1)
                # 중심점 표시
                cv2.circle(res_img, (cx, cy), 2, (255, 255, 255), -1)

            st.image(res_img, use_container_width=True, caption="자동 탐지 결과 (Blue: Negative, Green: Positive)")
            
            # 결과 표시
            ratio = (pos_cnt / valid_well_cnt * 100) if valid_well_cnt > 0 else 0
            is_gmo = ratio >= gmo_thresh
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("탐지된 전체 Well", f"{valid_well_cnt}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")

            if is_gmo:
                st.success(f"### 🧬 최종 판정: GMO Positive")
            else:
                st.error(f"### 🧬 최종 판정: Non-GMO")
                
        else:
            st.warning("원을 찾지 못했습니다. 사이드바의 감도 설정을 조절해 보세요.")
    else:
        st.error("이미지 파일을 읽을 수 없습니다.")
else:
    st.info("💡 사진을 업로드하면 AI가 Well을 자동으로 찾아 분석을 시작합니다.")
