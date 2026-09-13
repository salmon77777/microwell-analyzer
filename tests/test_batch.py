"""Unit fixtures below are synthetic software tests, never experimental data."""
import copy, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from wellscope_core import Settings, analyze, synthetic_demo, json_bytes, AnalysisError, object_hash
from wellscope_batch import *
from wellscope_report import report_html, panel_svg, export_bundle
from wellscope_figure6 import portrait_svg, chart_bytes

@pytest.fixture(scope='module')
def batch():
    base=analyze(synthetic_demo(),Settings(threshold_mode='fixed',fixed_threshold=60))
    results={};rows=[]
    for i,(lev,amp) in enumerate([(0,0.),(1,5.),(3,10.),(5,20.)]):
        a=copy.deepcopy(base);a['summary']['source_kind']='uploaded'
        a['summary']['image']['raw_sha256']=str(i)*64
        a['table'].loc[a['table'].valid,'corrected_signal_native']+=amp
        a=rethreshold(a,60,'test_fixture');name=f'{lev}%.png';results[name]=a;s=a['summary']
        rows.append({'sample_id':name,'known_content_pct':float(lev),'image_sha256':s['image']['raw_sha256'],
            'valid_wells':s['valid_wells'],'positive_wells':s['positive_wells'],'excluded_wells':s['excluded_wells'],
            'pitch_px':s['geometry']['pitch_px'],'width':520,'height':520,
            'automatic_threshold_native':s['automatic_threshold_suggestion'],
            'automatic_positive_fraction_pct':s['positive_fraction_pct'],
            'shared_threshold_native':60.,**feature_values(a)})
    return {'results':results,'table':pd.DataFrame(rows),
        'well_rule':{'method':'test_fixture','threshold_native':60.,'operator':'>='},
        'settings':asdict(Settings(threshold_mode='fixed',fixed_threshold=60))}

@pytest.fixture
def model(batch):return build_model(batch,'median_corrected_signal',3.)

def test_filename_proposal_only():
    assert proposed_label('sample_3%(1).png')==3
    assert proposed_label('control_1234.png') is None
    assert proposed_label('1000%.png') is None

def test_feature_statistics(batch):
    a=batch['results']['0%.png'];f=feature_values(a)
    assert f['mean_corrected_signal']<=f['top10_mean_corrected_signal']

def test_rethreshold_does_not_mutate(batch):
    a=batch['results']['0%.png'];old=a['table'].classification.copy()
    b=rethreshold(a,1e10,'test')
    assert b['summary']['positive_wells']==0
    assert a['table'].classification.equals(old)

def test_all_above_cutoff_is_100(batch):
    a=rethreshold(batch['results']['0%.png'],-1e10,'test')
    assert a['summary']['positive_fraction_pct']==100.

def test_boundary_development_rule(batch,model):
    assert model['status']=='ready'
    assert model['independent_validation_n']==0
    a=batch['results']['3%.png'];p=predict(a,model,True)
    assert p['decision']=='At/above study threshold' and p['training_image_match']
    assert predict(batch['results']['1%.png'],model,True)['decision']=='Below study threshold'

def test_confirmation_required(batch,model):
    with pytest.raises(AnalysisError):predict(batch['results']['3%.png'],model)

def test_inference_does_not_use_label(batch,model):
    a=copy.deepcopy(batch['results']['1%.png']);a['summary']['sample_id']='100%.png'
    p=predict(a,model,True)
    assert p['decision']=='Below study threshold'

def test_same_bytes_renamed_still_training(batch,model):
    a=copy.deepcopy(batch['results']['3%.png']);a['summary']['sample_id']='blind.png'
    assert predict(a,model,True)['training_image_match']

def test_new_hash_marked_unvalidated(batch,model):
    a=copy.deepcopy(batch['results']['3%.png']);a['summary']['image']['raw_sha256']='a'*64
    assert predict(a,model,True)['evidence_status']=='new_image_unvalidated_prediction'

def test_unequal_bitdepth_blocked(batch,model):
    a=copy.deepcopy(batch['results']['3%.png']);a['summary']['image']['bit_depth']=16
    with pytest.raises(AnalysisError):predict(a,model,True)

def test_changed_threshold_blocked(batch,model):
    a=rethreshold(batch['results']['3%.png'],61,'test')
    with pytest.raises(AnalysisError):predict(a,model,True)

def test_dimension_tolerance(batch,model):
    a=copy.deepcopy(batch['results']['3%.png']);a['summary']['image']['width']=519
    predict(a,model,True)
    a['summary']['image']['width']=600
    with pytest.raises(AnalysisError):predict(a,model,True)

def test_saturation_withholds_numeric(batch):
    b=copy.deepcopy(batch);b['table']['saturated_wells_pct']=100
    m=build_model(b,'median_corrected_signal')
    assert m['numeric_calibration']['status']=='withheld'

def test_nonmonotonic_withholds_numeric(batch):
    b=copy.deepcopy(batch);b['table'].loc[0,'median_corrected_signal']=b['table'].loc[1,'median_corrected_signal']+1
    m=build_model(b,'median_corrected_signal')
    assert 'nonmonotonic_or_plateau_response' in m['numeric_calibration']['reasons']

def test_overlapping_no_classifier(batch):
    b=copy.deepcopy(batch);b['table'].loc[0,'median_corrected_signal']=1e6
    m=build_model(b,'median_corrected_signal')
    assert m['status']=='overlapping_training_classes'
    with pytest.raises(AnalysisError):predict(b['results']['3%.png'],m,True)

def test_model_integrity(model):
    loaded=read_model(json_bytes(model));assert loaded['model_id']==model['model_id']
    m=copy.deepcopy(model);m['sample_rule']['cutoff']=1
    with pytest.raises(AnalysisError):read_model(json_bytes(m))

def test_wrong_engine(model):
    m=copy.deepcopy(model);m['engine_sha256']='bad';m.pop('model_id');m['model_id']=object_hash(m)[:20]
    with pytest.raises(AnalysisError):read_model(json_bytes(m))

def test_duplicate_bytes_rejected():
    with pytest.raises(AnalysisError):prepare_series([('0%.png',b'a',0),('1%.png',b'a',1),('3%.png',b'b',3)])

def test_all_bad_labels_rejected():
    with pytest.raises(AnalysisError):prepare_series([('0%',b'a',0),('1%',b'b',float('nan')),('3%',b'c',3)])

def test_exports_training_note(batch,model):
    a=copy.deepcopy(batch['results']['3%.png']);a['summary']['screening']=predict(a,model,True)
    assert b'Training-image reanalysis' in report_html(a,'3%',True)
    assert b'TRAINING IMAGE' in panel_svg(a,'3%',True)
    assert b'Reproducible method text' not in report_html(a,'3%',True)

def test_portrait_export(batch,model):
    svg=portrait_svg(batch,model)
    assert b'width="210mm" height="297mm"' in svg
    assert b'not independent validation' in svg

@pytest.mark.parametrize('kind',['distribution','response','boundary','feature','saturation'])
def test_plot_outputs(batch,model,kind):
    assert chart_bytes(batch,model,kind).startswith(b'\x89PNG')


def test_stale_session_model_rejected(batch,model):
    m=copy.deepcopy(model);m['software_version']='old'
    with pytest.raises(AnalysisError):predict(batch['results']['3%.png'],m,True)
