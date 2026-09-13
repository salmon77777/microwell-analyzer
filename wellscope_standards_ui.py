"""Streamlit standard-series page, separate from analysis code for testing."""
from __future__ import annotations
import hashlib, io
import pandas as pd
import streamlit as st
from wellscope_core import AnalysisError, Settings, object_hash, json_bytes
from wellscope_batch import (FEATURES, proposed_label, prepare_series, build_model,
    feature_diagnostics, predict, MAX_SERIES, MAX_SERIES_BYTES)
from wellscope_figure6 import chart_bytes, portrait_svg, series_bundle
from wellscope_report import csv_bytes, overlays


def render_standards(en:bool=False)->None:
    def t(ko,en_):return en_ if en else ko
    st.header(t('표준 이미지 일괄 분석','Standard-image series'))
    st.write(t('0% 대조 이미지에서 하나의 well 임계값을 정하고, 모든 표준에 똑같이 적용합니다. 다음 단계에서 시료 선별 기준을 저장합니다.',
               'Derive one well cutoff from zero-reference images and apply it to all standards. Then save a separate sample screening rule.'))
    st.caption(t('파일명은 표준 라벨의 제안에만 사용됩니다. 미지 시료를 예측할 때 파일명과 알고 있는 함량은 사용하지 않습니다. 한 번에 3–30장, 합계 100 MB까지.',
                 'Filenames only suggest labels here; they are never used for unknown predictions. Use 3–30 images, up to 100 MB total.'))
    files=st.file_uploader(t('표준 PNG/TIFF 파일을 함께 선택','Select standard PNG/TIFF images'),type=['png','jpg','jpeg','tif','tiff'],accept_multiple_files=True,key='standards_upload')
    if not files:
        st.info(t('먼저 0, 1, 3, 5, 10, 30, 50, 70, 90, 100% 이미지를 선택하세요. 반복 시료가 있으면 함께 올려도 됩니다.',
                  'Select the standard images. Independent replicate images can also be added.'))
        return
    if len(files)>MAX_SERIES or sum(f.size for f in files)>MAX_SERIES_BYTES:
        st.error(t('업로드 제한을 초과했습니다. 30장·합계 100 MB 이내로 줄여주세요.','Batch exceeds 30 images or 100 MB.'));return
    file_key=object_hash([{'name':f.name,'sha':hashlib.sha256(f.getvalue()).hexdigest()} for f in files])[:12]
    initial=pd.DataFrame([{'sample_id':f.name,'known_content_pct':proposed_label(f.name)} for f in files])
    table=st.data_editor(initial,disabled=['sample_id'],hide_index=True,key='standard_labels_'+file_key,
        column_config={'sample_id':st.column_config.TextColumn(t('파일 / 시료 ID','File / sample ID')),
                       'known_content_pct':st.column_config.NumberColumn(t('알고 있는 GMO/non-GMO 혼합비 (%)','Assigned GMO/non-GMO mixture (%)'),min_value=0.,max_value=100.,step=.1,required=True)})
    labels_confirmed=st.checkbox(t('표의 혼합비가 실제 표준 시료 정보와 일치함을 확인했습니다','I confirm the assigned mixture labels against the actual standard preparation'),key='labels_confirmed_'+file_key)
    with st.expander(t('공통 기준 설정 — 처음에는 기본값으로 시작','Shared-rule settings — start with defaults'),expanded=False):
        st.caption(t('99백분위는 탐색용 기본 규칙입니다. 0% 이미지가 여러 장이면 각 이미지의 백분위값 중앙값을 사용합니다. 1% 위양성률 보증이 아닙니다.',
                      'The 99th percentile is an exploratory default. With multiple zero images, use the median of their quantiles. It does not guarantee a 1% assay false-positive rate.'))
        basis_mode=st.selectbox(t('well 임계값 방법','Well cutoff method'),['zero','fixed'],format_func=lambda x:t('0% 대조군 백분위','Zero-reference quantile') if x=='zero' else t('사전 확정된 공통 값','Previously established fixed value'),key='batch_well_mode')
        q=st.number_input(t('0% 신호 백분위 (%)','Zero-reference percentile (%)'),90.,99.99,99.,step=.1,key='batch_quantile')
        fixed=None
        if basis_mode=='fixed':fixed=st.number_input(t('공통 well 임계값 (원래 강도 단위)','Fixed well cutoff (native units)'),value=60.,key='batch_fixed')
        acq=st.text_input(t('촬영·반응 프로토콜 ID','Acquisition / assay protocol ID'),value='LAMP-FITC-v1',key='batch_acq')
        chan=st.selectbox(t('형광 채널','Fluorescence channel'),['G','R','B'],key='batch_channel')
    signature=object_hash({'files':file_key,'labels':table.to_json(),'q':q,'fixed':fixed,'acq':acq,'channel':chan})
    if st.button(t('표준 이미지 분석 실행','Analyze standard series'),type='primary',disabled=not labels_confirmed,key='run_series'):
        st.session_state.pop('standard_result',None);st.session_state.pop('active_screening_model',None)
        try:
            mapping=dict(zip(table.sample_id,table.known_content_pct))
            items=[(f.name,f.getvalue(),float(mapping[f.name])) for f in files]
            bar=st.progress(0.)
            batch=prepare_series(items,Settings(channel=chan,acquisition_id=acq),quantile=q/100,fixed_threshold=fixed,
                progress=lambda i,n,name:bar.progress(i/n,text=f'{i}/{n}  {name}'))
            st.session_state['standard_result']=(signature,batch)
        except (AnalysisError,TypeError,ValueError) as exc:st.error(str(exc));return
    saved=st.session_state.get('standard_result')
    if not labels_confirmed or not saved or saved[0]!=signature:
        st.info(t('입력·라벨·조건을 확인한 후 분석을 실행하세요. 변경 후에는 이전 결과를 재사용하지 않습니다.','Run analysis after confirming inputs. Changed inputs invalidate the displayed series.'));return
    batch=saved[1];df=batch['table']
    st.success(t('공통 well 임계값: ','Shared well cutoff: ')+f"{batch['well_rule']['threshold_native']:.5f}"+t(' (원래 배경 보정 강도 단위)',' (native corrected-intensity units)'))
    st.subheader(t('이미지별 결과','Per-image results'))
    st.dataframe(df,hide_index=True,width='stretch')
    st.caption(t('automatic_* 열은 기존 이미지별 자동 분리 결과입니다. positive_fraction_pct는 이번 공통 임계값을 적용한 값입니다. 둘 다 GMO 혼합비 자체가 아닙니다.',
                 'automatic_* columns retain the old per-image split. positive_fraction_pct uses the shared cutoff. Neither is the GMO mixture percentage.'))
    with st.expander(t('모든 시료의 격자·분류 결과 확인','Review each image grid and classification'),expanded=False):
        selected=st.selectbox(t('확인할 시료','Review sample'),df.sample_id.tolist(),key='batch_preview')
        images=overlays(batch['results'][selected]);c1,c2=st.columns(2)
        c1.image(images['grid'],caption=t('격자: 중심 정합을 확인','Grid: check center alignment'),width='stretch')
        c2.image(images['classification'],caption=t('공통 well 임계값 적용','Shared well cutoff'),width='stretch')
    st.subheader(t('시료 선별 모델','Sample-level screening model'))
    feature=st.selectbox(t('시료를 비교할 지표','Sample-level feature'),list(FEATURES),format_func=lambda f:FEATURES[f][1 if en else 0],key='batch_feature')
    study=st.number_input(t('연구상 GMO 혼합비 경계 (%)','Assigned mixture decision boundary (%)'),min_value=.01,max_value=99.99,value=3.,step=.1,key='batch_study')
    quantity=st.selectbox(t('혼합비 정보의 근거','Quantity basis'),['calibrant_equivalent','gm_mass_fraction','target_reference_copy_ratio'],
        format_func=lambda x:{'calibrant_equivalent':t('제공한 표준의 혼합비 기준','Assigned calibrant mixture'),
                             'gm_mass_fraction':t('GM 질량분율 · 확인된 경우','GM mass fraction · if established'),
                             'target_reference_copy_ratio':t('타깃/참조 copy 비율 · 확인된 경우','Target/reference copy ratio · if established')}[x],key='batch_quantity')
    st.caption(t('기본 지표를 유지한 뒤 독립 시료로 검증하세요. 현재 10장에서 가장 잘 맞는 지표를 골랐다면 그 선택도 모델 개발에 포함되며, 검증 성능으로 보고하면 안 됩니다.',
                 'Prespecify the feature and validate independently. Selecting a feature on these images is part of model development, not validation.'))
    st.dataframe(feature_diagnostics(df,float(study)),hide_index=True,width='stretch')
    try:model=build_model(batch,feature,float(study),quantity)
    except AnalysisError as exc:st.warning(str(exc));return
    if model['status']=='ready':
        cut=model['sample_rule']['cutoff']
        st.info(t('표준들에서 학습한 시료 지표 기준: ','Sample score cutoff fitted on standards: ')+f'{cut:.5f}'+t(' — 3% 혼합비와 다른 단위입니다.',' — not the mixture percentage.'))
    else:st.error(t('현재 지표에서는 표준의 경계 전후 값이 겹칩니다. 3% 판정 모델을 저장하지 않습니다.','Training classes overlap. A 3% screening rule will not be saved.'))
    if model['numeric_calibration']['status']!='ready':
        st.warning(t('0–100% 수치 정량은 보류됩니다: ','Numerical mixture calibration withheld: ')+', '.join(model['numeric_calibration']['reasons']))
    cols=st.columns(2)
    cols[0].image(chart_bytes(batch,model,'response'),width='stretch')
    cols[1].image(chart_bytes(batch,model,'boundary'),width='stretch')
    rows=[]
    if model['status']=='ready':
        for name,a in batch['results'].items():
            p=predict(a,model,confirmed=True);rows.append({'sample_id':name,'score':p['score'],'study_screen':p['decision'],'evidence':'Training-image reanalysis'})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.warning(t('현재 이미지는 모델을 만드는 데도 사용됐습니다. 여기 보이는 분리는 독립 검증, 정확도, 민감도 또는 특이도가 아닙니다. 농도당 한 장이면 well 수를 반복 시료 수로 세지 않습니다.',
                 'These images trained the model. Their separation is not independent accuracy, sensitivity or specificity. Wells are not independent replicate samples.'))
    reviewed=st.checkbox(t('모든 표준의 격자와 동일 촬영·반응 조건을 확인했고, 이 모델이 아직 독립 검증 전임을 이해했습니다',
                           'I reviewed every grid and matching acquisition/assay conditions, and understand that the model is not independently validated'),
                           key='batch_review_'+signature[:12]+model['model_id'])
    a,b=st.columns(2)
    if a.button(t('이 모델을 단일 시료 분석에 사용','Use this model for single-image analysis'),disabled=not reviewed or model['status']!='ready',key='activate_model'):
        st.session_state['active_screening_model']=model
        st.success(t('저장됐습니다. 왼쪽 메뉴에서 단일 시료 분석 → 저장된 연구 선별 모델을 선택하세요.','Saved for this session. Choose Single-image analysis → Saved study screening model.'))
    b.download_button(t('screening_model.json 저장','Download screening_model.json'),json_bytes(model),file_name='screening_model.json',mime='application/json',disabled=not reviewed or model['status']!='ready',key='download_screen_model')
    st.subheader(t('분석 결과 내보내기','Export analysis results'))
    st.caption(t('CSV: 실제 수치 / SVG: A4 세로 Figure 6 / ZIP: 이미지·모델·수치를 함께 저장합니다. 공개 GitHub에는 결과 ZIP을 올리지 마세요.',
                 'CSV: measured values; SVG: A4 portrait Figure 6; ZIP: images, model and records. Keep study outputs out of public GitHub.'))
    a,b=st.columns(2)
    a.download_button(t('시리즈 결과 CSV','Series CSV'),csv_bytes(df),file_name='series_summary.csv',mime='text/csv',key='download_series_csv')
    b.download_button(t('A4 세로 Figure 6 SVG','A4 portrait Figure 6 SVG'),portrait_svg(batch,model),file_name='Figure6_A4_development.svg',mime='image/svg+xml',key='download_figure6')
    bundlekey=signature+model['model_id']
    if st.button(t('전체 결과 ZIP 준비','Prepare full series ZIP'),key='prepare_series_bundle'):
        with st.spinner(t('이미지와 수치 저장 중…','Preparing images and records…')):st.session_state['series_bundle']=(bundlekey,series_bundle(batch,model))
    if st.session_state.get('series_bundle',(None,))[0]==bundlekey:
        st.download_button(t('전체 결과 ZIP 저장','Download full series ZIP'),st.session_state['series_bundle'][1],file_name='WellScope_series_PRIVATE.zip',mime='application/zip',key='download_series_bundle')
