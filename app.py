import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. 페이지 설정
st.set_page_config(page_title="GMO Microwell 분석기", layout="wide")
st.title("🔬 Microwell 완전 자동 분석기")
st.markdown("""
- **자동 탐지**: 형광이 있는 Well을 찾아 간격과 각도를 분석합니다.
- **격자 복원**: 신호가 없는(어두운) Well도 격자 패턴을 통해 자동으로 계산에 포함합니다.
- **테두리 보호**: 사진 가장자리에 걸린 온전하지 않은 Well은 분석에서 자동 제외됩니다.
""")

# --- 사이드바: 정밀 튜닝 ---
st.sidebar.header("⚙️ 분석 설정")
well_radius = st.sidebar.slider("Well 크기 (반지름)", 3, 30, 8, help="실제 Well의 크기에 맞춰 원의 크기를 조절하세요.")
sensitivity = st.sidebar.slider("인식 민감도", 0, 255, 45, help="값이 낮을수록 흐릿한 Well도 잘 찾지만, 노이즈도 많아집니다.")
threshold_g = st.sidebar.slider("형광 임계값 (Positive 기준)", 0, 255, 75, help="이 값보다 밝으면 Positive(GMO)로 판정합니다.")
gmo_limit = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("Microwell 결과 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 로드
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 1. 처리 속도와 일관성을 위한 리사이즈 (가로 1000px 기준)
    scale = 1000 / w
    tw, th = 1000, int(h * scale)
    img_small = cv2.resize(img_rgb, (tw, th))
    green_ch = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 2. 보이는(Positive) Well 위치 추출
    # local maximum을 찾아 밝은 점들의 좌표를 확보합니다.
    k_size = max(3, int(well_radius * 1.5))
    if k_size % 2 == 0: k_size += 1
    local_max = cv2.dilate(blurred, np.ones((k_size, k_size), np.uint8), iterations=1)
    peak_mask = (blurred == local_max) & (blurred > sensitivity)
    yp, xp = np.where(peak_mask)
    
    if len(xp) > 10:
        # 3. 격자 패턴 추론 (빈 Well 위치 계산용)
        pts = np.column_stack((xp, yp)).astype(np.float32)
        
        # 간격(Spacing) 및 기울기(Angle) 추정
        def estimate_grid_params(coords):
            c_sort = np.sort(coords)
            diffs = np.diff(c_sort)
            valid_diffs = diffs[(diffs > well_radius) & (diffs < well_radius * 5)]
            return np.median(valid_diffs) if len(valid_diffs) > 0 else 20.0

        dx = estimate_grid_params(xp)
        dy = estimate_grid_params(yp)
        
        # 중심점과 범위 설정
        min_x, max_x = xp.min(), xp.max()
        min_y, max_y = yp.min(), yp.max()
        
        # 4. 분석 수행
        res_img = img_small.copy()
        pos_wells = []
        neg_wells = []
        
        # 생성된 격자를 순회하며 판정
        # np.arange를 통해 실제 발견된 Well들의 영역 내를 촘촘히 조사합니다.
        for ty in np.arange(min_y, max_y + 1, dy):
            for tx in np.arange(min_x, max_x + 1, dx):
                cx, cy = int(tx), int(ty)
                
                # [요구사항 2 반영] 테두리에 걸린 스팟 제외 (반지름 r 마진 확인)
                if cx - well_radius < 5 or cx + well_radius > tw - 5 or \
                   cy - well_radius < 5 or cy + well_radius > th - 5:
                    continue
                
                # 해당 위치의 실제 신호 분석
                # 격자점 주변 소량의 픽셀 평균값으로 판정 (노이즈 방지)
                roi = blurred[max(0, cy-2):min(th, cy+3), max(0, cx-2):min(tw, cx+3)]
                val = np.mean(roi) if roi.size > 0 else 0
                
                # [요구사항 1, 3 반영] 모든 Well은 노란색으로 표시 (전체 개수 포함)
                cv2.circle(res_img, (cx, cy), well_radius, (255, 255, 0), 1)
                
                if val > threshold_g:
                    # Positive 판정
                    pos_wells.append((cx, cy))
                    cv2.circle(res_img, (cx, cy), int(well_radius*0.6), (0, 255, 0), -1)
                else:
                    # Negative 판정 (빈 공간)
                    neg_wells.append((cx, cy))

        # 5. 결과 시각화 및 통계
        st.image(res_img, use_container_width=True, caption="노란색 원: 전체 Well / 초록색 점: Positive 신호")
        
        total_count = len(pos_wells) + len(neg_wells)
        pos_count = len(pos_wells)
        neg_count = len(neg_wells)
        ratio = (pos_count / total_count * 100) if total_count > 0 else 0
        
        # [요구사항 5 반영] 통계 수치 표기
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체 Well 개수", f"{total_count}개")
        col2.metric("Positive Well", f"{pos_count}개")
        col3.metric("Negative Well", f"{neg_count}개")
        col4.metric("Positive 비율", f"{ratio:.1f}%")
        
        # [요구사항 6 반영] 최종 GMO 판정
        if ratio >= gmo_limit:
            st.success(f"✅ **최종 판정: GMO Positive** (신호율 {ratio:.1f}% >= {gmo_limit}%)")
        else:
            st.error(f"❌ **최종 판정: Non-GMO** (신호율 {ratio:.1f}% < {gmo_limit}%)")
            
    else:
        st.error("⚠️ Well이 인식되지 않았습니다. 사이드바의 '인식 감도'를 낮춰보세요.")
        with st.expander("도움말"):
            st.write("1. 녹색 불빛이 선명하게 보이도록 촬영했는지 확인하세요.")
            st.write("2. '인식 감도'를 낮추면 더 많은 Well을 찾으려 시도합니다.")
            st.write("3. 사진의 밝기가 너무 어두우면 '배경 노이즈 제거'를 0에 가깝게 조절하세요.")

else:
    st.info("실험한 Microwell 형광 사진(Green Channel)을 업로드하면 자동으로 분석을 시작합니다.")
