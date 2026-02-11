import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 앱 제목
st.title("🔬 형광 Microwell 분석기")

# 1. 사진 업로드
uploaded_file = st.file_uploader("Microwell 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 이미지 처리
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 2. 임계값 설정 슬라이더 (사용자가 직접 조절)
    threshold = st.slider("형광 감도(임계값) 설정", 0, 255, 100)
    
    # 3. 원형 감지 및 분석 (OpenCV 활용)
    # (여기에 원형 격자를 찾고 녹색 농도를 분석하는 수식이 들어갑니다)
    
    # 예시 결과 출력
    st.image(img_array, caption='분석 중인 이미지')
    
    # 4. 결과 리포트
    st.subheader("📊 분석 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 Well", "96개") # 예시 수치
    col2.metric("Positive", "45개")
    col3.metric("비율", "46.8%")

    # 5. 저장 버튼
    st.download_button("분석 사진 저장", data="이미지데이터", file_name="result.png")