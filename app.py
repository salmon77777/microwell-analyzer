"""Run: streamlit run app.py

UI is intentionally separated from measurement and export code. Upload all
all runtime .py files plus requirements.txt into the SAME GitHub directory.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import html
import io
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from wellscope_core import (
    APP_NAME, APP_SUBTITLE, VERSION, AnalysisError, Settings, BASES, CAL_COLUMNS, analyze, calibrate,
    decode_image, export_template, json_bytes, object_hash, read_profile, synthetic_demo,
)
from wellscope_report import (
    overlays, png_bytes, histogram_bytes, signal_plot_bytes, calibration_plot_bytes,
    panel_svg, report_html, csv_bytes, completed_summary, export_bundle, method_text, safe_id,
)

from wellscope_batch import read_model, prediction_settings, predict, FEATURES, feature_values
from wellscope_standards_ui import render_standards

DEFAULT_SCREENING_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "screening_model_DEFAULT.json"
)

def load_builtin_screening_model():
    if not DEFAULT_SCREENING_MODEL_PATH.exists():
        return None, None
    try:
        model = read_model(DEFAULT_SCREENING_MODEL_PATH.read_bytes())
        return model, None
    except Exception as exc:
        return None, str(exc)

st.set_page_config(page_title=APP_NAME, page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1580px}
[data-testid="stSidebar"]{border-right:1px solid #e3e9ee}
.ws-brand{background:#f7f9fb;border-radius:8px;padding:12px;font-family:Arial,Helvetica,sans-serif;color:#193348;font-size:38px;font-weight:750;letter-spacing:-1.2px;line-height:1.2}
.ws-sub{font-family:Arial,Helvetica,sans-serif;color:#64798b;font-size:16px;margin:8px 0 18px}
.ws-tag{font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#3c7685;background:#edf6f8;border:1px solid #d9eaee;border-radius:18px;padding:5px 12px;display:inline-block;margin-bottom:14px}
.ws-note{font-family:Arial,Helvetica,sans-serif;color:#5f7485;font-size:13px;line-height:1.6;padding:13px 16px;border-radius:8px;background:#f5f8fa;margin:8px 0 16px}
[data-testid="stMetric"]{border:1px solid #e0e8ed;padding:14px 16px;border-radius:10px;background:#f7f9fb}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *{font-size:13px;color:#526678 !important}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *{font-size:28px;color:#193348 !important;opacity:1 !important;-webkit-text-fill-color:#193348 !important}
div.stButton>button[kind="primary"]{border-radius:8px}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    lang=st.selectbox("언어 / Language",["한국어","English"],key="language")
EN=lang=="English"
def t(ko: str,en: str)->str:return en if EN else ko

def show_expected_error(exc: Exception) -> None:
    st.error(t("분석을 완료하지 못했습니다. 아래 원인과 조치 내용을 확인하세요.","Analysis could not be completed. Review the cause and guidance below."))
    message=str(exc)
    st.write(message)
    lower=message.lower()
    if any(word in lower for word in ("grid","geometry","lattice","pitch","candidate","corners")):
        st.info(t("격자 관련 오류: 먼저 원본 해상도와 분석 영역을 확인하세요. 형광이 적거나 없는 시료는 고정된 격자 템플릿 또는 알려진 행·열/모서리 좌표가 필요합니다. 신호 기준을 내려 결과를 억지로 맞추지 마세요.",
                  "Grid issue: check the native resolution and ROI. Sparse/negative samples need a registered template or known corner geometry; do not tune signal thresholds to force a desired result."))
    elif "calibration" in lower or "profile" in lower:
        st.info(t("보정/프로필 오류: 표준 시료와 미지 시료에 동일한 분석 프로필을 적용하고, 현재 프로필 ID와 CSV의 ID가 일치하는지 확인하세요.","Calibration/profile issue: use the same locked profile for standards and unknowns, and check profile IDs."))

st.markdown(f'<div class="ws-brand">{html.escape(APP_NAME)}</div><div class="ws-sub">{html.escape(APP_SUBTITLE)}</div>',unsafe_allow_html=True)
st.markdown(f'<div class="ws-tag">RESEARCH USE ONLY &nbsp; / &nbsp; v{VERSION}</div>',unsafe_allow_html=True)

with st.sidebar:
    workflow=st.selectbox(t("작업 선택","Workflow"),["single","standards"],
        format_func=lambda x:t("단일 시료 분석","Single-image analysis") if x=="single" else t("표준 이미지 일괄 분석","Standard-image series"),key="workflow")
if workflow=="standards":
    render_standards(EN)
    st.stop()

with st.sidebar:
    st.subheader(t("1. 이미지 입력","1. Image input"))
    uploaded=st.file_uploader(t("Microwell 형광 이미지","Microwell fluorescence image"),type=["png","jpg","jpeg","tif","tiff"],key="image_upload")
    demo=st.checkbox(t("합성 예제로 사용법 확인","Use synthetic demonstration"),value=False,key="demo_mode")
    st.caption(t("원본 PNG 또는 8/16-bit 단일 TIFF를 권장합니다. 업로드는 현재 Streamlit 서버로 전송됩니다. 공개 GitHub에는 실험 이미지를 올리지 마세요.",
                 "Prefer original PNG or single-plane 8/16-bit TIFF. Uploads go to the Streamlit server; do not commit private images to public GitHub."))

if uploaded is not None:
    raw_bytes=uploaded.getvalue();source_kind="uploaded";source_name=uploaded.name
elif demo:
    if "synthetic_bytes" not in st.session_state:st.session_state.synthetic_bytes=synthetic_demo()
    raw_bytes=st.session_state.synthetic_bytes;source_kind="synthetic";source_name="synthetic_demo.png"
else:
    st.markdown(t("### 이미지 한 장에서, 재현 가능한 분석 결과까지","### From one image to a reproducible analysis record"))
    st.write(t("왼쪽에서 이미지를 올리고 ‘분석 실행’을 누르세요. 처음에는 자동 탐색으로 격자를 확인하고, 논문 분석에는 대조군으로 확인한 고정 임계값을 저장해 사용합니다.",
               "Upload an image on the left and select Run analysis. Start with automatic exploration to inspect the grid, then use a control-checked fixed threshold for study measurements."))
    c1,c2,c3=st.columns(3)
    with c1:st.info(t("01  자동 격자 추정\n\n간격·기울기·완전한 측정 영역을 확인합니다.","01  Grid inference\n\nInspect spacing, orientation and measurable footprints."))
    with c2:st.info(t("02  신호 분석\n\n원래 픽셀에서 배경 보정 신호를 측정합니다.","02  Signal measurement\n\nMeasure background-corrected signal in native pixels."))
    with c3:st.info(t("03  결과 저장\n\n이미지·CSV·설정을 함께 저장합니다.","03  Export results\n\nSave images, CSV and settings together."))
    st.warning(t("양성 well 비율 ≠ GMO 함량. 보정표가 없는 경우 GMO 함량이나 GMO/non-GMO 판정은 출력하지 않습니다.","Positive-well fraction is not GMO content. No content estimate or GMO/non-GMO claim is generated without calibration."))
    st.stop()

input_hash=hashlib.sha256(raw_bytes).hexdigest()
try:
    if st.session_state.get("decoded_hash")!=input_hash:
        st.session_state.decoded_frame=decode_image(raw_bytes)
        st.session_state.decoded_hash=input_hash
    decoded=st.session_state.decoded_frame
except AnalysisError as exc:
    show_expected_error(exc);st.stop()
meta=decoded["metadata"];h,w=meta["height"],meta["width"]

with st.sidebar:
    sample_id=st.text_input(t("시료 ID (결과 화면 표시)","Sample ID (display label)"),value="MW-"+input_hash[:6].upper(),max_chars=80,key="sample_"+input_hash[:10])
    st.caption(f"{w} × {h} px · {meta['bit_depth']}-bit · {meta['color_mode']}")
    st.subheader(t("2. 형광 판정 기준","2. Fluorescence threshold"))
    mode=st.selectbox(t("분석 방식","Analysis mode"),["auto","fixed","profile","model"],
        format_func=lambda x:{"auto":t("자동 제안 · 탐색용","Automatic proposal · exploratory"),
                              "fixed":t("고정 임계값 · 대조군 확인 필요","Fixed threshold · controls required"),
                              "profile":t("저장된 분석 프로필","Saved analysis profile"),
                              "model":t("저장된 연구 선별 모델 · 권장","Saved study screening model")}[x],key="threshold_mode")
    fixed=60.0
    if mode=="fixed":
        fixed=st.number_input(t("고정 임계값 (배경 보정 후 원래 강도 단위)","Fixed threshold (background-corrected native units)"),value=60.0,step=1.0,format="%.3f",key="fixed_value")
        st.caption(t("이전 앱의 G 밝기 150과 같은 값이 아닙니다. 대조군·표준 시료로 새 측정법의 임계값을 확인하세요.","This is not equivalent to the old raw G-channel threshold. Check the new measurement definition with controls."))
    profile_upload=None;profile_obj=None
    if mode=="profile":
        profile_upload=st.file_uploader(t("analysis_profile.json 불러오기","Load analysis_profile.json"),type=["json"],key="profile_upload")
        if profile_upload is not None:
            try:profile_obj=read_profile(profile_upload.getvalue(),decoded)
            except AnalysisError as exc:show_expected_error(exc)
    screen_model=None
    if mode=="model":
        builtin_model,builtin_error=load_builtin_screening_model()
        source_options=["builtin","upload"] if builtin_model is not None else ["upload"]
        model_source=st.radio(
            t("선별 모델 소스","Screening model source"),
            source_options,
            index=0,
            format_func=lambda x:{
                "builtin":t("내장 연구 선별 모델 · 권장","Built-in study screening model · recommended"),
                "upload":t("다른 screening_model.json 불러오기","Load another screening_model.json")
            }[x],
            key="screening_model_source"
        )
        if builtin_error:
            st.warning(t("내장 모델을 읽을 수 없습니다: ","Unable to read built-in model: ")+builtin_error)

        if model_source=="builtin":
            screen_model=builtin_model
        else:
            model_upload=st.file_uploader(
                t("screening_model.json 불러오기","Load screening_model.json"),
                type=["json"],
                key="screening_model_upload"
            )
            if model_upload is not None:
                try:
                    screen_model=read_model(model_upload.getvalue())
                    st.session_state["active_screening_model"]=screen_model
                except AnalysisError as exc:
                    st.error(str(exc))
            else:
                screen_model=st.session_state.get("active_screening_model")

        if screen_model is None:
            st.info(t(
                "사용 가능한 선별 모델이 없습니다. 표준 일괄 분석에서 저장한 모델을 불러오세요.",
                "No screening model is available. Load a model exported from standard-series analysis."
            ))
        else:
            st.session_state["active_screening_model"]=screen_model
            st.success(t("연구 선별 모델을 불러왔습니다.","Study screening model loaded."))
            if model_source=="builtin":
                st.caption(t(
                    "내장 모델은 현재 개발용 모델입니다. 논문 최종 분석 전에는 최종 표준 데이터로 다시 생성한 모델로 교체하세요.",
                    "The built-in model is currently a development model. Replace it with the final locked model before manuscript analysis."
                ))
            st.caption(
                t("모델 ID: ","Model ID: ")+screen_model["model_id"]+
                f"\nTw = {screen_model['well_rule']['threshold_native']:.4g} | "
                f"Ts = {screen_model['sample_rule']['cutoff']:.4g}% | "
                f"study boundary = {screen_model['study_threshold_pct']:g}%"
            )
    st.subheader(t("3. 선택 설정","3. Optional settings"))
    advanced=st.checkbox(t("고급 설정 열기","Enable advanced settings"),value=False,key="advanced")

cfg=Settings(threshold_mode="fixed" if mode in ("fixed","profile") else "auto",fixed_threshold=float(fixed))
template_bytes=None;registered=False
if advanced:
    with st.sidebar:
        with st.expander(t("촬영 조건·측정법","Acquisition / measurement"),expanded=False):
            channel=st.selectbox(t("형광 채널","Signal channel"),["G","R","B"],key="channel")
            acquisition=st.text_input(t("촬영·반응 프로토콜 ID","Acquisition / assay protocol ID"),value="LAMP-FITC-v1",key="acquisition")
            st.caption(t("노출, 배율, gain, 반응 시간, DNA 투입/희석 조건을 같은 ID로 관리하세요. 앱이 이 조건을 자동 확인할 수는 없습니다.","Use the same ID for matching exposure, magnification, gain, reaction time and DNA input/dilution. The app cannot verify those conditions automatically."))
            inner=st.number_input(t("신호 원 반지름 / pitch","Signal radius / pitch"),min_value=.08,max_value=.35,value=.22,step=.01,key="inner_ratio")
            bg1=st.number_input(t("배경 안쪽 반지름 / pitch","Background inner radius / pitch"),min_value=.10,max_value=.47,value=.34,step=.01,key="bg1")
            bg2=st.number_input(t("배경 바깥 반지름 / pitch","Background outer radius / pitch"),min_value=.15,max_value=.49,value=.46,step=.01,key="bg2")
        cfg=replace(cfg,channel=channel,acquisition_id=acquisition,inner_ratio=float(inner),bg_inner_ratio=float(bg1),bg_outer_ratio=float(bg2))
        with st.expander(t("격자·분석 영역","Grid / region of interest"),expanded=False):
            geo=st.selectbox(t("격자 방식","Geometry method"),["auto","template","manual"],format_func=lambda x:{"auto":t("자동 추정","Automatic inference"),"template":t("정합된 격자 템플릿","Registered template"),"manual":t("알려진 행·열 + 네 모서리","Known rows/columns + corners")}[x],key="geometry_mode")
            cfg=replace(cfg,geometry_mode=geo)
            crop=st.checkbox(t("분석 영역 자르기","Limit analysis ROI"),value=False,key="use_roi")
            if crop:
                xr=st.slider(t("가로 범위 (pixel)","Horizontal range (pixel)"),0,w,(0,w),key="roi_x_"+input_hash[:8])
                yr=st.slider(t("세로 범위 (pixel)","Vertical range (pixel)"),0,h,(0,h),key="roi_y_"+input_hash[:8])
                cfg=replace(cfg,roi=(xr[0],yr[0],xr[1],yr[1]))
            if geo=="auto":
                override=st.checkbox(t("자동 격자 탐색값 직접 보정","Override automatic candidate settings"),value=False,key="geo_override")
                if override:
                    hint=st.number_input(t("예상 pitch (원래 pixel)","Expected pitch (native pixel)"),min_value=3.0,max_value=300.0,value=6.0,step=.1,key="pitch_hint")
                    floor=st.number_input(t("격자 후보 최소 신호 (원래 단위)","Geometry candidate floor (native units)"),min_value=0.0,value=15.0,step=1.0,key="floor")
                    cfg=replace(cfg,pitch_hint=float(hint),candidate_floor=float(floor))
            elif geo=="template":
                template=st.file_uploader(t("grid_template.json 불러오기","Load grid_template.json"),type=["json"],key="template_upload")
                if template is not None:
                    template_bytes=template.getvalue()
                    template_key=hashlib.sha256(template_bytes).hexdigest()[:8]
                    registered=st.checkbox(t("촬영 위치·배율·회전·좌표 정합이 같은 이미지입니다","I confirmed the same field of view, scale, orientation and pixel registration"),value=False,key="registered_"+input_hash[:8]+template_key)
            else:
                nr=st.number_input(t("행 (위→아래)","Rows (top to bottom)"),2,200,60,key="rows")
                nc=st.number_input(t("열 (왼쪽→오른쪽)","Columns (left to right)"),2,200,61,key="cols")
                st.caption(t("칩 외곽이 아니라 네 귀퉁이 well의 중심 좌표입니다. 상하좌우 순서를 확인하세요.","Enter the CENTER of each corner well, not the outside edge of the chip."))
                corners=[]
                for key,name,default in [("tl",t("좌상","Top left"),(5.,5.)),("tr",t("우상","Top right"),(float(w-6),5.)),("br",t("우하","Bottom right"),(float(w-6),float(h-6))),("bl",t("좌하","Bottom left"),(5.,float(h-6)))]:
                    c1,c2=st.columns(2)
                    x=c1.number_input(name+" X",0.0,float(w-1),default[0],step=.1,key=key+"x_"+input_hash[:8])
                    y=c2.number_input(name+" Y",0.0,float(h-1),default[1],step=.1,key=key+"y_"+input_hash[:8])
                    corners.append((float(x),float(y)))
                cfg=replace(cfg,manual_rows=int(nr),manual_cols=int(nc),manual_corners=tuple(corners))
        with st.expander(t("특정 well 제외 · 사유 기록","Exclude selected wells / audit reason"),expanded=False):
            ids=st.text_area(t("제외 ID: R001C001, R001C002 형식","Excluded IDs: R001C001, R001C002"),value="",key="exclude_ids")
            reason=st.text_input(t("제외 사유 (기록됨)","Exclusion reason (recorded)"),value="",key="exclude_reason")
            cfg=replace(cfg,excluded_ids=ids,exclusion_reason=reason)
            st.caption(t("분류 결과를 원하는 값으로 맞추기 위한 제외는 피하세요. 제외 규칙은 표준·미지 시료에 일관되게 적용해야 합니다.","Do not exclude wells to force a desired result. Apply prespecified rules consistently to standards and unknowns."))

profile_ready=(mode!="profile" or profile_obj is not None) and (mode!="model" or screen_model is not None)
if profile_obj is not None:
    cfg=replace(cfg,threshold_mode="fixed",fixed_threshold=float(profile_obj["threshold_native"]),
                channel=profile_obj["channel"],acquisition_id=profile_obj["acquisition_id"],
                inner_ratio=profile_obj["inner_ratio"],bg_inner_ratio=profile_obj["bg_inner_ratio"],bg_outer_ratio=profile_obj["bg_outer_ratio"])
    with st.sidebar:st.success(t("신호 측정 설정은 불러온 프로필로 고정됩니다.","Photometry settings are locked to the loaded profile."))
if screen_model is not None:
    cfg=prediction_settings(screen_model)
    with st.sidebar:st.info(t("측정법과 well 임계값은 모델로 고정됩니다. 고급 설정은 적용하지 않습니다.","The model locks photometry and well threshold; advanced overrides are ignored."))
with st.sidebar:
    run=st.button(t("분석 실행","Run analysis"),type="primary",width="stretch",disabled=not profile_ready,key="run_analysis")
    paper=st.checkbox(t("논문용 결과 보기 (영문)","Publication panel view (English)"),value=False,key="paper_view")
    st.caption(t("처음에는 기본 설정으로 실행하세요. 자동 분석 성공은 실험적 검증 완료를 뜻하지 않습니다.","Start with defaults. Successful software analysis is not experimental validation."))

signature=object_hash({"input":input_hash,"settings":asdict(cfg),"template":hashlib.sha256(template_bytes or b'').hexdigest(),"source_kind":source_kind})
if run:
    st.session_state.pop("result_signature", None)
    if cfg.geometry_mode=="template" and not registered:
        st.warning(t("템플릿의 좌표 정합을 확인하고 확인란을 선택한 뒤 실행하세요.","Confirm template registration before analysis."))
    else:
        try:
            with st.spinner(t("격자 추정 및 원래 픽셀 신호 측정 중…","Fitting grid and measuring native-pixel signals…")):
                result=analyze(raw_bytes,cfg,template_bytes,source_kind)
            st.session_state["result"]=result;st.session_state["result_signature"]=signature
            st.session_state.pop("bundle",None)
        except AnalysisError as exc:show_expected_error(exc)
        except Exception:
            # Do not reveal server internals to public users.
            traceback.print_exc()
            st.error(t("예기치 않은 오류가 발생했습니다. Streamlit의 Manage app 로그를 확인하세요. 파일 내용·개인정보를 지운 오류 메시지만 공유하세요.","Unexpected error. Inspect Manage app logs; redact private information before sharing them."))

if not profile_ready or st.session_state.get("result_signature")!=signature:
    if "result" in st.session_state:
        st.info(t("입력 또는 설정이 변경되었습니다. 이전 결과 대신 ‘분석 실행’으로 다시 계산하세요.","Input/settings changed. Run analysis again; stale results are not displayed."))
    st.image(decoded["display_rgb"],caption=t("입력 미리보기 · 아직 분석 결과가 아닙니다","Input preview · not an analysis result"),width="stretch",output_format="PNG")
    if mode=="profile" and not profile_ready:
        st.warning(t("저장된 analysis_profile.json을 먼저 올려주세요.","Upload a valid saved analysis profile first."))
    st.stop()

result=st.session_state["result"];summary=result["summary"]
result["summary"].pop("screening",None)
result["summary"]["features"]=feature_values(result)
if source_kind=="synthetic":st.warning(t("합성 예제입니다. 논문 실험 결과로 사용할 수 없습니다.","Synthetic example. Do not use as experimental evidence."))
reviewed=st.checkbox(t("격자 전체와 확대 영역을 확인했고, measurable well의 분모가 적절한지 검토했습니다","I reviewed the full/zoomed grid and the measurable-well denominator"),value=False,key="reviewed_"+signature[:16])

if screen_model is None:
    # Optional numerical calibration remains available without a screening model.
    with st.expander(t("GMO / 함량 보정 — 표준 시료 데이터를 확보한 뒤 사용","Content calibration — use after acquiring standard-sample data"),expanded=False):
        st.write(t("양성 well 비율만으로 GMO 함량을 정하지 않습니다. 동일 프로필로 분석한 표준 시료의 보정 CSV가 있어야 아래 기능을 사용할 수 있습니다. 파일명에 적힌 농도는 읽어서 정답으로 사용하지 않습니다.",
                   "Positive fraction alone does not determine GMO content. Supply a calibration CSV measured with the same profile. Concentrations in filenames are never used as ground truth."))
        cal_upload=st.file_uploader(t("보정 CSV","Calibration CSV"),type=["csv"],key="calibration_upload")
        b1,b2,b3=st.columns([2,1,1])
        basis=b1.selectbox(t("함량 단위의 근거","Quantity basis"),list(BASES),format_func=lambda x:{
            "calibrant_equivalent":t("표준 시료 상당 함량 (%)","Calibrant-equivalent content (%)"),
            "gm_mass_fraction":t("GM 질량분율 (%) · 이 기준으로 검증한 경우","GM mass fraction (%) · if validated"),
            "target_reference_copy_ratio":t("타깃/참조 유전자 copy 비율 (%)","Target/reference copy ratio (%)")}[x],key="quantity_basis")
        study=b2.number_input(t("연구상 분류 기준 (%)","Study threshold (%)"),0.0,100.0,3.0,step=.1,key="study_threshold")
        band=b3.number_input(t("재검토 구간 ± (%p)","Review band ± (pp)"),0.0,10.0,.3,step=.1,key="review_band")
        st.caption(t("±0.3%p는 사용자가 변경할 수 있는 연구 운영상 예시값이며, 신뢰구간이나 법적 기준이 아닙니다. 보정식의 측정불확도는 이 앱이 계산하지 않습니다.",
                     "±0.3 pp is a configurable operational example, not a confidence interval or legal rule. This app does not calculate calibration uncertainty."))
        confirm=st.checkbox(t("형광 임계값을 양성·음성 대조군으로 확인했고, 표준/미지 시료의 촬영·반응·DNA 투입/희석 조건과 프로필이 비교 가능하며, 보정과 독립 검증이 다름을 이해했습니다","I checked the fluorescence threshold against positive/negative controls, confirm comparable imaging, assay and DNA-input/dilution conditions, and understand that calibration is not independent validation"),value=False,key="calibration_confirm_"+signature[:12])
        blank=pd.DataFrame(columns=CAL_COLUMNS)
        st.download_button(t("빈 보정 CSV 양식","Blank calibration CSV"),csv_bytes(blank),file_name="calibration_template.csv",mime="text/csv",key="calibration_template_download")
        known=st.number_input(t("현재 시료가 표준일 때: 알고 있는 함량 (%)","For a known standard: assigned content (%)"),0.0,100.0,0.0,step=.1,key="known_content")
        standard_row=pd.DataFrame([{"sample_id":sample_id,"positive_fraction_pct":summary["positive_fraction_pct"],"known_content_pct":known,"profile_id":result["profile"]["profile_id"],"quantity_basis":basis}])
        st.download_button(t("현재 결과를 표준 1행 CSV로 저장","Save current result as one calibrant row"),csv_bytes(standard_row),file_name=safe_id(sample_id)+"_calibrant_row.csv",mime="text/csv",disabled=summary["threshold_mode"]!="fixed" or source_kind=="synthetic",key="save_cal_row")
        st.caption(t("표준 여러 개의 CSV에서 데이터 행을 모아 하나의 파일로 합칩니다. 헤더는 맨 위 한 번만 둡니다. 미지 시료에 아는 함량을 임의 입력해 보정표를 만들지 마세요.","Combine real standard rows into one CSV with a single header. Do not assign guessed content to unknown samples."))
    
    # Re-evaluate or reset calibration on every rerun: no stale model/result after upload removal.
    result["summary"]["calibration"]={"status":"not_loaded","estimated_content_pct":None,"decision":"Not evaluated"}
    if cal_upload is not None:
        try:
            result["summary"]["calibration"]=calibrate(result,cal_upload.getvalue(),basis,float(study),float(band),confirm)
        except AnalysisError as exc:
            st.warning(t("함량 추정 보류: ","Content estimation withheld: ")+str(exc))
else:
    result["summary"]["calibration"]={"status":"not_loaded","estimated_content_pct":None,"decision":"Not evaluated"}
    st.markdown("### "+t("연구 선별 판정","Study screening classification"))
    st.info(
        t(
            f"모델이 준비되었습니다. 개별 well 기준 Tw={screen_model['well_rule']['threshold_native']:.4g}, "
            f"시료 점수 기준 Ts={screen_model['sample_rule']['cutoff']:.4g}%, "
            f"연구 혼합비 경계={screen_model['study_threshold_pct']:g}% 입니다. "
            "아래 두 확인이 완료되어야 최종 시료 판정을 적용합니다.",
            f"Model ready. Well cutoff Tw={screen_model['well_rule']['threshold_native']:.4g}, "
            f"sample-score cutoff Ts={screen_model['sample_rule']['cutoff']:.4g}%, "
            f"study mixture boundary={screen_model['study_threshold_pct']:g}%. "
            "Complete both confirmations below to apply the sample-level classification."
        )
    )
    comparable=st.checkbox(
        t(
            "표준과 현재 시료의 촬영·반응·DNA 투입/희석 조건이 비교 가능함을 확인했습니다",
            "I confirm comparable acquisition, assay, DNA-input and dilution conditions"
        ),
        value=False,
        key="screen_conditions_"+signature[:10]+screen_model["model_id"]
    )
    if not reviewed or not comparable:
        st.warning(
            t(
                "선별 모델은 불러왔지만 아직 시료 판정은 적용되지 않았습니다. "
                "위의 격자 검토와 이 조건 확인을 모두 선택하세요.",
                "The screening model is loaded, but sample classification has not yet been applied. "
                "Complete both grid review and condition confirmation."
            )
        )
    try:
        if reviewed and comparable:
            screen=predict(result,screen_model,confirmed=True)
            result["summary"]["screening"]=screen
            result["summary"]["calibration"]={"status":screen["numeric_status"],
                "estimated_content_pct":screen["estimated_content_pct"],"decision":"Not evaluated",
                "quantity_basis":screen["quantity_basis"],"study_threshold_pct":screen["study_threshold_pct"]}
    except AnalysisError as exc:
        st.warning(str(exc))
summary=result["summary"];cal=summary["calibration"]

if paper:
    svg_text=panel_svg(result,sample_id,reviewed).decode("utf-8")
    components.html(
        """<html><head><style>
        html,body{margin:0;padding:0;background:white;overflow:hidden}
        .wrap{width:100%;background:white}
        .wrap svg{display:block;width:100% !important;height:auto !important}
        </style></head><body><div class="wrap">"""
        +svg_text+
        """</div></body></html>""",
        height=760,
        scrolling=False
    )
    st.caption(t("이 패널은 실제 계산 결과로 생성됩니다. 아래 SVG/HTML 내보내기를 사용하면 브라우저 사이드바 없이 저장할 수 있습니다.","This panel is generated from actual computed measurements. SVG/HTML exports exclude browser sidebars."))
else:
    c1,c2,c3,c4=st.columns(4)
    c1.metric(t("측정 가능 well","Measurable wells"),f"{summary['valid_wells']:,}")
    c2.metric(t("임계값 이상 well","Threshold-positive"),f"{summary['positive_wells']:,}")
    c3.metric(t("양성 well 비율","Positive fraction"),f"{summary['positive_fraction_pct']:.2f}%")
    est=cal.get("estimated_content_pct")
    if screen:
        if screen["decision"]=="At/above study threshold":
            class_value=t("≥3% 분류","≥3% class").replace("3%",f"{screen['study_threshold_pct']:g}%")
        elif screen["decision"]=="Below study threshold":
            class_value=t("<3% 분류","<3% class").replace("3%",f"{screen['study_threshold_pct']:g}%")
        else:
            class_value=screen["decision"]
        c4.metric(t("시료 분류","Sample classification"),class_value)
    else:
        c4.metric(t("추정 GMO 혼합비","Estimated GMO mixture"),f"{est:.2f}%" if est is not None else t("정량 보류","Not quantified"))
    status={"Not evaluated":t("함량 미판정 · 보정 데이터 필요","Content not evaluated · calibration required"),
            "Outside calibration range":t("보정 범위 밖 · 외삽하지 않음","Outside calibration range · no extrapolation"),
            "Review near study threshold":t("분류 기준 근처 · 재검토 필요","Near study threshold · review required"),
            "At/above study threshold":t("연구상 기준 이상","At/above study threshold"),
            "Below study threshold":t("연구상 기준 미만","Below study threshold")}[cal["decision"]]
    screen=summary.get("screening")
    if screen:
        d=screen["decision"]
        if d=="At/above study threshold":st.info(t("GMO 선별 양성 · ≥3% 연구 분류","GMO screen positive · >=3% study class").replace("3%",f"{screen['study_threshold_pct']:g}%"))
        elif d=="Below study threshold":st.info(t("3% 연구 기준 미만 · GMO 불검출이라는 뜻은 아닙니다","Below the 3% study screen · not proof of GMO absence").replace("3%",f"{screen['study_threshold_pct']:g}%"))
        else:st.warning(d)
        st.caption(f"Sample score {screen['score']:.5f} | Locked sample cutoff {screen['sample_score_cutoff']:.5f} | Well cutoff {screen['well_threshold_native']:.5f}")
        if screen["training_image_match"]:st.warning(t("모델 개발에 사용한 동일 이미지입니다. 이 결과는 학습 이미지 재분석이며 독립 검증이 아닙니다.","This image trained the model. This is training-image reanalysis, not independent validation."))
        else:st.caption(t("새 이미지에 대한 미검증 예측입니다. 모델은 아직 독립 검증 전입니다.","Unvalidated prediction on a new image. The model has not been independently validated."))
        if screen["numeric_status"]!="ready":
            st.caption(
                t("정확한 GMO 혼합비 수치 정량은 보류되었습니다. 선별 분류와 수치 정량은 서로 다른 결과입니다. 사유: ",
                  "Exact numerical GMO-mixture estimation is withheld. Screening classification and numerical quantification are different outputs. Reasons: ")
                +", ".join(screen["numeric_withheld_reasons"])
            )
    else:st.info(status)
    if est is not None:st.caption(t("단위 근거: ","Quantity basis: ")+cal["quantity_basis"]+t(". 기준 미만을 ‘GMO 불검출’ 또는 non-GMO 인증으로 해석하지 않습니다.",". Below threshold is not non-detection or non-GMO certification."))
    tabs=st.tabs([t("결과 이미지","Result image"),t("격자 확인","Grid review"),t("신호·품질 확인","Signal / QC"),t("well별 데이터","Per-well data")])
    imgs=overlays(result)
    with tabs[0]:
        c1,c2=st.columns(2)
        with c1:
            st.markdown(t("#### 입력 이미지","#### Input image"));st.image(imgs["raw"],width="stretch",output_format="PNG")
        with c2:
            st.markdown(t("#### 형광 분류","#### Fluorescence classification"));st.image(imgs["classification"],width="stretch",output_format="PNG")
        st.caption(t("노란 원: 임계값 이상 · 빨간 ×: 임계값 미만 · 회색 □: 제외. 확대는 표시용이며 원래 해상도를 증가시키지 않습니다.","Yellow circle: above threshold; red x: below threshold; grey square: excluded. Display enlargement adds no native resolution."))
    with tabs[1]:
        g=summary["geometry"]
        st.write(t("**추정/지정 격자:** ","**Inferred/defined lattice:** ")+f"{g['columns']} columns × {g['rows']} rows · pitch X {g['pitch_x_px']:.3f} px / Y {g['pitch_y_px']:.3f} px")
        st.caption(t("행×열은 격자 범위이고, 실제 분모는 완전한 측정 영역이 남은 well입니다. FL 영상만으로 충전 여부까지 확인한 값은 아닙니다.","Rows × columns describes the lattice extent. The denominator includes complete measurable footprints; filling is not established from fluorescence alone."))
        st.image(imgs["grid"],width="stretch",output_format="PNG")
        with st.expander(t("원래 픽셀의 확대 영역 확인","Inspect a zoomed native-pixel region"),expanded=False):
            zx=st.slider("Center X (px)",0,w-1,w//2,key="zoomx_"+input_hash[:8])
            zy=st.slider("Center Y (px)",0,h-1,h//2,key="zoomy_"+input_hash[:8])
            radius=max(12,int(g['pitch_px']*5));x0=max(0,zx-radius);x1=min(w,zx+radius);y0=max(0,zy-radius);y1=min(h,zy+radius)
            sx=imgs['grid'].shape[1]/w;sy=imgs['grid'].shape[0]/h
            patch=imgs['grid'][round(y0*sy):round(y1*sy),round(x0*sx):round(x1*sx)]
            st.image(patch,caption=f"Native coordinates X {x0}..{x1}; Y {y0}..{y1}",width="stretch",output_format="PNG")
    with tabs[2]:
        st.dataframe(pd.DataFrame([summary["features"]]),hide_index=True,width="stretch")
        st.image(histogram_bytes(result),width="stretch")
        st.image(signal_plot_bytes(result),width="stretch")
        if summary["threshold_mode"]=="auto":st.warning(t("자동 분리는 밝기 분포를 나눈 탐색 결과입니다. 음성/양성 대조군 없이 이 분리를 생물학적 양성·음성으로 확정할 수 없습니다.","The automatic split partitions intensity values. Without controls it is not a validated biological positive/negative decision."))
        st.dataframe(pd.DataFrame(summary["qc_flags"]),hide_index=True,width="stretch")
        st.caption(t("QC 주의 기준은 이 소프트웨어의 점검 기준이며 실험적으로 확립한 assay 합격 기준이 아닙니다.","QC cautions are software checks, not experimentally established assay acceptance criteria."))
        cal_plot=calibration_plot_bytes(result)
        if cal_plot:st.image(cal_plot,width="stretch")
    with tabs[3]:
        st.dataframe(result["table"],hide_index=True,width="stretch")
        st.caption(t("좌표는 0부터 시작하는 원래 픽셀 좌표이고, row/column은 1부터 시작하는 격자 인덱스입니다.","Coordinates are zero-based native pixel coordinates; row/column IDs are one-based lattice indices."))

st.divider()
st.subheader(t("분석 결과 내보내기","Export analysis results"))
st.caption(t("CSV는 측정 수치, SVG는 그림 패널, HTML은 브라우저용 보고서, ZIP은 이 파일들의 묶음입니다.","CSV: measured values; SVG: figure panel; HTML: browser report; ZIP: the complete bundle."))
st.caption(t("SVG는 벡터 글자와 실제 이미지가 들어 있는 편집용 패널입니다. PNG 확대나 300 dpi 표기만으로 부족한 원본 해상도가 복구되지는 않습니다.","SVG contains vector labels and actual raster evidence. Enlargement or a 300-dpi tag cannot restore missing source resolution."))
a,b,c=st.columns(3)
with a:st.download_button(t("논문용 패널 SVG","Publication panel SVG"),panel_svg(result,sample_id,reviewed),file_name=safe_id(sample_id)+"_panel.svg",mime="image/svg+xml",width="stretch",key="download_panel")
with b:st.download_button(t("결과 보고서 HTML","Analysis report HTML"),report_html(result,sample_id,reviewed),file_name=safe_id(sample_id)+"_report.html",mime="text/html",width="stretch",key="download_report")
with c:st.download_button(t("well별 수치 CSV","Per-well CSV"),csv_bytes(result["table"]),file_name=safe_id(sample_id)+"_per_well.csv",mime="text/csv",width="stretch",key="download_csv")
a,b,c=st.columns(3)
with a:st.download_button(t("신호 분석 프로필 JSON","Analysis profile JSON"),json_bytes(result["profile"]),file_name="analysis_profile.json",mime="application/json",width="stretch",key="download_profile")
with b:st.download_button(t("격자 템플릿 JSON","Geometry template JSON"),export_template(result),file_name="grid_template.json",mime="application/json",width="stretch",key="download_template")
with c:st.download_button(t("설정·품질 기록 JSON","Settings / QC record JSON"),json_bytes(completed_summary(result,sample_id,reviewed)),file_name=safe_id(sample_id)+"_summary.json",mime="application/json",width="stretch",key="download_summary")
st.caption(t("신호 프로필: 측정법·임계값을 재사용합니다. 격자 템플릿: 같은 촬영 좌표의 위치만 재사용하며, 새 이미지와 정합 확인이 필요합니다. 두 파일은 용도가 다릅니다.","Analysis profile reuses photometry and threshold. Geometry template reuses pixel positions and requires registration. These files have different purposes."))

bundle_key=object_hash({"analysis":signature,"sample_id":sample_id,"reviewed":reviewed,"calibration":summary["calibration"],"screening":summary.get("screening")})
if st.button(t("전체 결과 ZIP 준비","Prepare complete result ZIP"),key="prepare_bundle"):
    with st.spinner(t("내보내기 파일 생성 중…","Preparing export files…")):
        st.session_state["bundle"]=(bundle_key,export_bundle(result,sample_id,reviewed,raw_bytes))
if st.session_state.get("bundle",(None,))[0]==bundle_key:
    st.download_button(t("전체 결과 ZIP 저장","Download complete result ZIP"),st.session_state["bundle"][1],file_name=safe_id(sample_id)+"_analysis.zip",mime="application/zip",type="primary",key="download_bundle")
st.caption(t("자동 계산 완료 ≠ 분석법 검증 완료. 대조군, 독립 반복 시료, 수동/참조법 비교가 별도로 필요합니다.","Computation complete does not mean method validated. Controls, independent replicates and reference/manual comparisons remain necessary."))
