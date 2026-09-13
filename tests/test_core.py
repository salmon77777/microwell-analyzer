"""Deterministic software tests; never substitute for assay validation."""
import io
import json
import sys
from pathlib import Path
from dataclasses import replace
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wellscope_core import *
from wellscope_report import *


def encode(a,fmt='PNG'):
    b=io.BytesIO();Image.fromarray(a).save(b,format=fmt);return b.getvalue()

@pytest.fixture(scope='module')
def demo_data():return synthetic_demo()

@pytest.fixture(scope='module')
def demo_result(demo_data):return analyze(demo_data,Settings(threshold_mode='fixed',fixed_threshold=60))


def test_decode_empty():
    with pytest.raises(AnalysisError):decode_image(b'')

def test_decode_invalid():
    with pytest.raises(AnalysisError):decode_image(b'not an image')

def test_uint16_preserved():
    a=(np.arange(1600).reshape(40,40)*30).astype(np.uint16)
    d=decode_image(encode(a,'TIFF'))
    assert d['metadata']['bit_depth']==16
    assert np.array_equal(a,d['rgb'][...,0])

def test_gray_preserved():
    a=np.arange(1600).reshape(40,40).astype(np.uint8)
    assert np.array_equal(decode_image(encode(a))['rgb'][...,1],a)

def test_alpha_not_counted_as_signal():
    a=np.zeros((40,40,4),np.uint8);a[...,1]=100;a[...,3]=255;a[0,:,3]=0
    d=decode_image(encode(a));assert not d['mask'][0].any();assert d['mask'][1:].all()

def test_multipage_rejected():
    b=io.BytesIO();im=Image.new('L',(30,30));im.save(b,format='TIFF',save_all=True,append_images=[im])
    with pytest.raises(AnalysisError):decode_image(b.getvalue())

def test_invalid_radius():
    with pytest.raises(AnalysisError):Settings(inner_ratio=.4).validate()

def test_invalid_threshold():
    with pytest.raises(AnalysisError):Settings(fixed_threshold=float('nan')).validate()

def test_demo_counts(demo_result):
    s=demo_result['summary'];assert s['geometry']['rows']==30;assert s['geometry']['columns']==30
    assert s['valid_wells']==900
    assert s['positive_wells']+s['negative_wells']==900
    assert s['calibration']['estimated_content_pct'] is None

def test_centers_unique(demo_result):
    c=demo_result['grid']['centers'];assert len(np.unique(c,axis=0))==len(c)

def test_raw_not_modified_by_annotations(demo_result):
    a=demo_result['frame']['rgb'].copy();b=demo_result['frame']['display_rgb'].copy()
    overlays(demo_result);assert np.array_equal(a,demo_result['frame']['rgb']);assert np.array_equal(b,demo_result['frame']['display_rgb'])

def test_template_same_result(demo_data,demo_result):
    template=export_template(demo_result)
    r=analyze(demo_data,Settings(geometry_mode='template',threshold_mode='fixed',fixed_threshold=60),template)
    assert r['summary']['positive_wells']==demo_result['summary']['positive_wells']
    assert r['summary']['valid_wells']==demo_result['summary']['valid_wells']

def test_template_negative_image(demo_result):
    a=np.zeros((520,520,3),np.uint8)
    r=analyze(encode(a),Settings(geometry_mode='template',threshold_mode='fixed',fixed_threshold=60),export_template(demo_result))
    assert r['summary']['positive_wells']==0 and r['summary']['negative_wells']==900

def test_template_size_mismatch(demo_result):
    with pytest.raises(AnalysisError):load_template(export_template(demo_result),decode_image(encode(np.zeros((100,100),np.uint8))))

def test_blank_image_auto_stops():
    with pytest.raises(AnalysisError):analyze(encode(np.zeros((100,100,3),np.uint8)),Settings())

def test_manual_grid_blank_known_geometry():
    cfg=Settings(geometry_mode='manual',manual_rows=8,manual_cols=8,
        manual_corners=((10.,10.),(90.,10.),(90.,90.),(10.,90.)),threshold_mode='fixed',fixed_threshold=10)
    r=analyze(encode(np.zeros((100,100,3),np.uint8)),cfg)
    assert r['summary']['valid_wells']==64;assert r['summary']['negative_wells']==64

def test_crossed_corners_rejected():
    cfg=Settings(geometry_mode='manual',manual_rows=8,manual_cols=8,
        manual_corners=((10.,10.),(90.,90.),(90.,10.),(10.,90.)))
    with pytest.raises(AnalysisError):analyze(encode(np.zeros((100,100,3),np.uint8)),cfg)

def test_manual_exclusion_recorded(demo_data,demo_result):
    cfg=Settings(threshold_mode='fixed',fixed_threshold=60,excluded_ids='R001C001',exclusion_reason='bubble by independent BF inspection')
    r=analyze(demo_data,cfg)
    assert r['summary']['valid_wells']==899
    assert r['table'].loc[r['table'].well_id=='R001C001','classification'].item()=='Excluded'

def test_exclusion_requires_reason(demo_data):
    with pytest.raises(AnalysisError):analyze(demo_data,Settings(excluded_ids='R001C001'))

def test_unknown_exclusion_rejected(demo_data):
    with pytest.raises(AnalysisError):analyze(demo_data,Settings(excluded_ids='R199C199',exclusion_reason='test'))

def test_roi_exclusion(demo_data):
    r=analyze(demo_data,Settings(roi=(40,40,450,450)))
    valid=r['table'].loc[lambda x:x.valid]
    assert valid.x_px.min()>40 and valid.x_px.max()<450

def test_profile_roundtrip(demo_result):
    p=read_profile(json_bytes(demo_result['profile']),demo_result['frame'])
    assert p['profile_id']==demo_result['profile']['profile_id']

def test_profile_tamper(demo_result):
    p=dict(demo_result['profile']);p['threshold_native']=444
    with pytest.raises(AnalysisError):read_profile(json_bytes(p),demo_result['frame'])

def test_otsu_constant():
    th,score=otsu_signal(np.ones(100)*5);assert th>5;assert score==0

def calibrants(r,xs=(0,50,95),ys=(0,5,10)):
    return pd.DataFrame([{'sample_id':f'fake_software_test_only_{i}',
          'positive_fraction_pct':x,'known_content_pct':y,'profile_id':r['profile']['profile_id'],
          'quantity_basis':'calibrant_equivalent'} for i,(x,y) in enumerate(zip(xs,ys))]).to_csv(index=False).encode()

def test_calibration_requires_confirmation(demo_result):
    with pytest.raises(AnalysisError):calibrate(demo_result,calibrants(demo_result),'calibrant_equivalent')

def test_calibration_internal_math(demo_result):
    c=calibrate(demo_result,calibrants(demo_result),'calibrant_equivalent',confirmed=True)
    assert c['status']=='ready';assert c['estimated_content_pct'] is not None
    assert c['uncertainty_interval'] is None

def test_calibration_wrong_profile(demo_result):
    b=calibrants(demo_result).replace(demo_result['profile']['profile_id'].encode(),b'wrong')
    with pytest.raises(AnalysisError):calibrate(demo_result,b,'calibrant_equivalent',confirmed=True)

def test_calibration_nonmonotonic(demo_result):
    with pytest.raises(AnalysisError):calibrate(demo_result,calibrants(demo_result,(0,50,40)),'calibrant_equivalent',confirmed=True)

def test_calibration_saturated(demo_result):
    with pytest.raises(AnalysisError):calibrate(demo_result,calibrants(demo_result,(0,100,100)),'calibrant_equivalent',confirmed=True)

def test_calibration_no_extrapolation(demo_result):
    c=calibrate(demo_result,calibrants(demo_result,(80,90,95)),'calibrant_equivalent',confirmed=True)
    assert c['status']=='out_of_range';assert c['estimated_content_pct'] is None

def test_auto_calibration_blocked(demo_data):
    r=analyze(demo_data,Settings())
    with pytest.raises(AnalysisError):calibrate(r,calibrants(r),'calibrant_equivalent',confirmed=True)

def test_demo_calibration_blocked(demo_data):
    r=analyze(demo_data,Settings(threshold_mode='fixed'),source_kind='synthetic')
    with pytest.raises(AnalysisError):calibrate(r,calibrants(r),'calibrant_equivalent',confirmed=True)

def test_report_escapes_html(demo_result):
    out=report_html(demo_result,'<script>alert(1)</script>').decode()
    assert '<script>alert(1)</script>' not in out and '&lt;script&gt;' in out

def test_bundle_complete(demo_result):
    data=export_bundle(demo_result,'Test',False)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert set(['per_well.csv','summary.json','analysis_panel.svg','analysis_profile.json','grid_template.json','methods.txt','report.html']).issubset(z.namelist())
        s=json.loads(z.read('summary.json'));assert s['positive_wells']==demo_result['summary']['positive_wells']

def test_csv_formula_protection():
    assert "'=test" in csv_bytes(pd.DataFrame({'x':['=test']})).decode('utf-8-sig')
