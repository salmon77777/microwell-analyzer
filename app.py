import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정 (오타 수정: set_config -> set_page_config)
st.set_page_config(page_title="AI Well Auto-Detector", layout="wide")
st.title("🤖 Microwell 완전 자동 분석기")

# --- 사이드바: 감도 조절 ---
st.sidebar.header("⚙️ 분석 정밀도 설정")
st.sidebar.info("좌표 입력 없이 자동으로 원을 찾습니다.")

# 원 인식 파라미터 조절
min_dist = st.sidebar.slider("Well 간 최소 거리", 10, 100, 25)
param1 = st.sidebar.slider("엣지 감지 강도", 10, 150, 50)
param2 = st.sidebar.slider("인식 민감도 (낮을수록 많이 찾음)", 5, 50, 20)
min_r = st.sidebar.slider("Well 최소 반지름", 1, 100, 10)
max_r = st.sidebar.slider("Well 최대 반지름", 5, 200, 30)

st.sidebar.header("🧪 판정 설정")
threshold_g = st.sidebar.slider("형광 임계값 (G)", 0, 255, 65)
gmo_thresh = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("사진을 업로드하세요 (잘린 Well 자동 제외)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 블러 처리를 통해 노이즈 제거 (원 인식률 향상)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # [핵심] 허프 변환 원 자동 탐지
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, dp=1.2, 
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
            # 인식된 원들을 정수형으로 변환
            circles = np.uint16(np.around(circles))
            
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # [필터링] 사진 가장자리에 걸쳐 잘린 원 무시
                # 원의 테두리가 이미지 경계를 벗어나면 제외
                if (cx - r < 5) or (cx + r > w - 5) or \
                   (cy - r < 5) or (cy + r > h - 5):
                    continue 
                
                valid_well_cnt += 1
                
                # Well 내부 Green 채널 강도 분석 (중심부 3x3 평균)
                # 인덱스 범위를 벗어나지 않도록 처리
                y_start, y_end = max(0, cy-1), min(h, cy+2)
                x_start, x_end = max(0, cx-1), min(w, cx+2)
                roi_g = img_rgb[y_start:y_end, x_start:x_end, 1]
                avg_g = np.mean(roi_g) if roi_g.size > 0 else 0
                
                is_pos = avg_g > threshold_g
                if is_pos:
                    pos_cnt += 1
                
                # 시각화 (테두리 두께 1px)
                color = (0, 255, 0) if is_pos else (255, 0, 0)
                cv2.circle(res_img, (cx, cy), r, color, 1)
            
            # 결과 이미지 출력
            st.image(res_img, use_container_width=True)
            
            # 통계 정보
            total = valid_well_cnt
            ratio = (pos_cnt / total * 100) if total > 0 else 0
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("탐지된 유효 Well", f"{total}개")
            c2.metric("Positive Well", f"{pos_cnt}개")
            c3.metric("GMO 신호율", f"{ratio:.1f}%")

            if ratio >= gmo_thresh:
                st.success("🧬 판정 결과: **GMO Positive**")
            else:
                st.error("🧬 판정 결과: **Non-GMO**")
        else:
            st.warning("원을 감지하지 못했습니다. 사이드바의 '민감도'나 '반지름' 설정을 조절해 보세요.")
    else:
        st.error("이미지를 불러올 수 없습니다.")
