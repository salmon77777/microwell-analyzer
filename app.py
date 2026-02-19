import streamlit as st
import cv2
import numpy as np

# ... (이전 코드 동일) ...

st.sidebar.header("🧬 5단계: GMO 판정 설정")
gmo_threshold_ratio = st.sidebar.slider("GMO 판정 기준 비율 (%)", 0, 100, 50)
gmo_label_on = st.sidebar.text_input("Positive 라벨명", "GMO Positive")
gmo_label_off = st.sidebar.text_input("Negative 라벨명", "Non-GMO")

# ... (이미지 분석 로직 동일) ...

        with tab2:
            st.image(display_img, use_container_width=True)
            
            total_wells = final_cols * final_rows
            pos_ratio = (pos_count / total_wells * 100) if total_wells > 0 else 0
            
            # --- GMO 판정 로직 추가 ---
            is_gmo = pos_ratio >= gmo_threshold_ratio
            
            st.markdown("---")
            st.subheader("🧬 최종 GMO 판정 결과")
            
            if is_gmo:
                st.success(f"### 🎉 판정 결과: {gmo_label_on}")
                st.balloons() # 축하 효과 (선택 사항)
            else:
                st.error(f"### ⚠️ 판정 결과: {gmo_label_off}")

            # 상세 지표 카드
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 우물 수", f"{total_wells}개")
            c2.metric("Positive 우물", f"{pos_count}개", delta=f"{pos_ratio:.1f}%", delta_color="normal")
            c3.metric("판정 기준", f"{gmo_threshold_ratio}% 이상")

            # 결과 리포트용 진행 바
            st.write("### 분석 진행도 및 비율")
            st.progress(pos_ratio / 100)
            
            # --- 분석 결과 요약 텍스트 ---
            st.info(f"""
            **분석 요약:** 총 **{total_wells}**개의 Microwell 중 **{pos_count}**개에서 형광 신호가 감지되었습니다.  
            현재 형광 발현율은 **{pos_ratio:.1f}%**이며, 이는 설정하신 GMO 기준인 **{gmo_threshold_ratio}%**를 
            {'초과하므로' if is_gmo else '하회하므로'} 최종적으로 **{gmo_label_on if is_gmo else gmo_label_off}** 샘플로 분류됩니다.
            """)

