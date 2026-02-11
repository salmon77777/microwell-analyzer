import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_canvas import st_canvas

st.set_page_config(page_title="Custom Microwell Analyzer", layout="wide")
st.title("🔬 사용자 지정 Microwell 분석기")
st.write("이미지 위에 원을 드래그하여 그려주세요. 각 원 내부의 형광을 분석합니다.")

# 1. 설정 사이드바
st.sidebar.header("⚙️ 분석 설정")
threshold = st.sidebar.slider("형광 판정 임계값 (G값)", 0, 255, 120)
stroke_width = st.sidebar.slider("그리기 선 두께", 1, 5, 1)
realtime_update = st.sidebar.checkbox("실시간 업데이트", True)

# 2. 파일 업로드
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file:
    bg_image = Image.open(uploaded_file)
    w, h = bg_image.size
    # 화면에 맞게 이미지 크기 조정 (표시용)
    display_width = 800
    display_height = int(h * (display_width / w))
    
    st.subheader("🖍️ 마우스로 우물 위에 원을 그리세요")
    st.caption("왼쪽 도구 모음에서 'Circle' 아이콘(○)을 선택하고 드래그하세요.")

    # 3. 캔버스 도구 (사용자가 드래그해서 원을 그림)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",  # 원 내부 투명한 오렌지색
        stroke_width=stroke_width,
        stroke_color="blue", # 요청하신 파란색 선
        background_image=bg_image,
        update_streamlit=realtime_update,
        height=display_height,
        width=display_width,
        drawing_mode="circle",
        key="canvas",
    )

    # 4. 분석 로직
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            st.subheader(f"📊 분석 결과 (감지된 원: {len(objects)}개)")
            
            img_array = np.array(bg_image.convert("RGB"))
            pos_count = 0
            
            # 캔버스 좌표를 원본 이미지 좌표로 변환하기 위한 비율
            scale_x = w / display_width
            scale_y = h / display_height

            results_data = []

            for obj in objects:
                if obj["type"] == "circle":
                    # 원의 좌표 및 반지름 계산
                    left = obj["left"] * scale_x
                    top = obj["top"] * scale_y
                    radius = obj["radius"] * scale_x
                    
                    center_x = int(left + radius)
                    center_y = int(top + radius)
                    r = int(radius)

                    # 마스크 생성 및 평균값 추출
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask, (center_x, center_y), r, 255, -1)
                    mean_val = cv2.mean(img_array, mask=mask)
                    green_avg = mean_val[1] # Green 채널

                    is_positive = green_avg > threshold
                    if is_positive:
                        pos_count += 1
                    
                    results_data.append(is_positive)

            # 통계 표시
            total = len(objects)
            percent = (pos_count / total) * 100 if total > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("총 그린 우물", f"{total}개")
            col2.metric("Positive (형광)", f"{pos_count}개")
            col3.metric("형광 발현 비율", f"{percent:.1f}%")
            
            st.info("💡 팁: 원을 잘못 그렸다면 왼쪽 도구함의 쓰레기통 아이콘을 누르거나, 선택 모드(화살표)로 클릭 후 'Delete' 키를 누르세요.")
        else:
            st.warning("분석할 원을 하나 이상 그려주세요.")
