import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="초정밀 격자 분석기", layout="wide")
st.title("🧪 회전 보정형 Microwell 격자 분석기")
st.info("사진의 기울기를 자동으로 감지하여 신호가 없는 빈 Well까지 정확히 추적합니다.")

# --- 사이드바: 정밀 튜닝 ---
st.sidebar.header("⚙️ 1. 인식 및 격자 설정")
well_radius = st.sidebar.slider("Well 표시 크기 (반지름)", 2, 20, 6)
min_brightness = st.sidebar.slider("인식 감도 (배경 제거)", 0, 255, 45)

st.sidebar.header("🧪 2. 판정 설정")
threshold_g = st.sidebar.slider("GMO 양성 판정 기준 (Green)", 0, 255, 75)

# 혹시 자동 계산이 미세하게 틀릴 경우를 대비한 수동 보정 도구
st.sidebar.header("🔄 3. 격자 미세 조정 (필요 시)")
offset_x = st.sidebar.slider("가로 위치 미세 조정", -50, 50, 0)
offset_y = st.sidebar.slider("세로 위치 미세 조정", -50, 50, 0)
manual_angle = st.sidebar.slider("기울기 미세 조정 (도)", -5.0, 5.0, 0.0, step=0.1)

uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석 속도 및 거리 계산 일관성을 위한 리사이징 (가로 1200px 기준)
    scale = 1200 / w
    tw, th = 1200, int(h * scale)
    img_small = cv2.resize(img_rgb, (tw, th))
    green_ch = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 1. 시드 포인트(밝은 Well) 추출
    local_max = cv2.dilate(blurred, np.ones((11, 11), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > min_brightness)
    yp, xp = np.where(peak_mask)
    
    if len(xp) > 30:
        # 2. [핵심] 기울기 및 간격 자동 추론
        pts = np.column_stack((xp, yp)).astype(np.float32)
        
        # 간격(Spacing) 계산
        def estimate_spacing(coords):
            c_sort = np.sort(coords)
            diffs = np.diff(c_sort)
            valid = diffs[(diffs > 8) & (diffs < 40)] # 예상 간격 범위
            return np.median(valid) if len(valid) > 0 else 15.0

        dx = estimate_spacing(xp)
        dy = estimate_spacing(yp)
        
        # 기울기(Angle) 계산: 근접한 점들 사이의 각도 평균
        angles = []
        for i in range(min(len(pts), 100)):
            dists = np.linalg.norm(pts - pts[i], axis=1)
            neighbors = pts[(dists > dx*0.8) & (dists < dx*1.2)]
            for n in neighbors:
                ang = np.degrees(np.arctan2(n[1] - pts[i][1], n[0] - pts[i][0]))
                # 0, 90, 180, 270도 근처의 각도만 수집
                ang = (ang + 45) % 90 - 45
                angles.append(ang)
        
        avg_angle = np.median(angles) + manual_angle
        
        # 3. 회전된 격자 생성 (Grid Generation)
        res_img = img_small.copy()
        pos_cnt = 0
        total_count = 0
        
        # 기준 원점 설정
        origin_x = np.median(xp) + offset_x
        origin_y = np.median(yp) + offset_y
        
        # 회전 행렬 정의
        cos_a = np.cos(np.radians(avg_angle))
        sin_a = np.sin(np.radians(avg_angle))
        
        # 이미지 전체를 덮도록 격자 범위 계산 (회전 고려)
        range_limit = int(max(tw, th) / min(dx, dy)) + 10
        for i in range(-range_limit, range_limit):
            for j in range(-range_limit, range_limit):
                # 로컬 좌표를 회전시켜 월드 좌표로 변환
                lx, ly = i * dx, j * dy
                cx = int(origin_x + lx * cos_a - ly * sin_a)
                cy = int(origin_y + lx * sin_a + ly * cos_a)
                
                if 5 <= cx < tw-5 and 5 <= cy < th-5:
                    total_count += 1
                    
                    # 모든 Well은 노란색 테두리
                    cv2.circle(res_img, (cx, cy), well_radius, (255, 255, 0), 1)
                    
                    # 형광 판정 (중심부 평균 밝기)
                    val = blurred[cy, cx]
                    if val > threshold_g:
                        pos_cnt += 1
                        # 양성은 내부에 초록색 점 표시
                        cv2.circle(res_img, (cx, cy), int(well_radius*0.6), (0, 255, 0), -1)

        st.image(res_img, use_container_width=True)
        
        # 결과 요약
        ratio = (pos_cnt / total_count * 100) if total_count > 0 else 0
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 Well (격자 복원)", f"{total_count}개")
        c2.metric("Positive Well", f"{pos_cnt}개")
        c3.metric("신호율", f"{ratio:.1f}%")
        
        st.caption(f"📏 분석 데이터: 간격({dx:.1f}px, {dy:.1f}px) / 기울기({avg_angle:.2f}도)")
    else:
        st.error("Well의 위치를 파악할 수 없습니다. 사이드바의 '인식 감도'를 낮춰보세요.")
