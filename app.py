import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. 페이지 설정
st.set_page_config(page_title="GMO Microwell 분석 시스템", layout="wide")
st.title("🔬 Microwell 완전 자동 분석기 (자가 학습 격자형)")

# --- 사이드바: 정밀 제어 ---
st.sidebar.header("⚙️ 분석 파라미터")
well_r = st.sidebar.slider("Well 반지름 (크기 조절)", 2, 20, 6)
sensitivity = st.sidebar.slider("인식 감도 (배경 제거)", 10, 150, 50, help="높일수록 노이즈가 줄어듭니다.")
threshold_g = st.sidebar.slider("형광 임계값 (양성 기준)", 0, 255, 80, help="이 값보다 밝으면 Positive로 인식합니다.")
gmo_limit = st.sidebar.slider("GMO 판정 기준 (%)", 0, 100, 50)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("Microwell 형광 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 로드 및 전처리
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # 분석용 리사이징 (연산 속도와 노이즈 억제)
    target_w = 1200
    scale = target_w / w
    target_h = int(h * scale)
    img_small = cv2.resize(img_rgb, (target_w, target_h))
    
    # Green 채널 집중 분석 및 블러 처리
    green_ch = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)[:,:,1]
    blurred = cv2.GaussianBlur(green_ch, (5, 5), 0)
    
    # 1. 시드 포인트(확실한 형광 Well) 추출
    # 과다 인식을 막기 위해 dilate 기반의 확실한 정점만 찾습니다.
    k_size = max(3, well_r)
    kernel = np.ones((k_size, k_size), np.uint8)
    local_max = cv2.dilate(blurred, kernel, iterations=1)
    peak_mask = (blurred == local_max) & (blurred > sensitivity)
    yp, xp = np.where(peak_mask)
    
    if len(xp) > 20: # 최소 20개의 시드가 있어야 격자 분석 가능
        # 2. 격자 패턴 자동 학습 (Auto-Learning Grid)
        pts = np.column_stack((xp, yp)).astype(np.float32)
        
        # X, Y 축별 간격 추론
        def get_spacing(coords):
            c_sort = np.sort(coords)
            diffs = np.diff(c_sort)
            valid = diffs[(diffs > well_r) & (diffs < well_r * 6)]
            return np.median(valid) if len(valid) > 0 else 20.0

        dx = get_spacing(xp)
        dy = get_spacing(yp)
        
        # 3. 격자 확장 및 테두리 제외 분석
        res_img = img_small.copy()
        pos_wells = []
        neg_wells = []
        
        # 실제 발견된 점들의 범위를 기준으로 격자망 생성
        min_x, max_x = xp.min(), xp.max()
        min_y, max_y = yp.min(), yp.max()
        
        # 격자를 생성하며 모든 Well 위치 탐색
        for ty in np.arange(min_y, max_y + 1, dy):
            for tx in np.arange(min_x, max_x + 1, dx):
                cx, cy = int(tx), int(ty)
                
                # [요구사항 2] 테두리 5% 영역은 온전하지 않으므로 분석 제외
                if cx < target_w*0.02 or cx > target_w*0.98 or \
                   cy < target_h*0.02 or cy > target_h*0.98:
                    continue
                
                # [요구사항 3] 모든 Well 위치는 노란색 원으로 표시 (전체 카운팅)
                cv2.circle(res_img, (cx, cy), well_r, (255, 255, 0), 1)
                
                # 해당 지점의 형광 강도 확인
                val = blurred[cy, cx]
                if val > threshold_g:
                    pos_wells.append((cx, cy))
                    # Positive: 내부에 초록색 점 표시
                    cv2.circle(res_img, (cx, cy), int(well_r*0.6), (0, 255, 0), -1)
                else:
                    neg_wells.append((cx, cy))

        # 4. 결과 대시보드 출력
        st.image(res_img, use_container_width=True, caption="노란색 원: 자동 복원된 전체 Well / 초록색 점: 양성 판정")
        
        total = len(pos_wells) + len(neg_wells)
        pos = len(pos_wells)
        neg = len(neg_wells)
        ratio = (pos / total * 100) if total > 0 else 0
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체 Well (격자 복원)", f"{total}개")
        col2.metric("Positive Well", f"{pos}개")
        col3.metric("Negative Well", f"{neg}개")
        col4.metric("Positive 비율", f"{ratio:.1f}%")
        
        # [요구사항 6] 최종 GMO 판정
        if ratio >= gmo_limit:
            st.success(f"🧬 **최종 판정: GMO Positive** (신호율 {ratio:.1f}%)")
        else:
            st.error(f"🧬 **최종 판정: Non-GMO** (신호율 {ratio:.1f}%)")
            
    else:
        st.error("⚠️ Well을 탐지할 수 없습니다. '인식 감도'를 낮추거나 '반지름'을 확인해 주세요.")
