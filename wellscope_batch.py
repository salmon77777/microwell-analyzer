"""Standard-series analysis and locked, exploratory sample screening.

Two independent cutoffs:
  well cutoff: reference-zero signal quantile (or an operator-fixed value);
  sample cutoff: midpoint between nonoverlapping training classes in ONE feature.
No label or filename is read at prediction time. Calibration is not validation.
"""
from __future__ import annotations
from dataclasses import replace, asdict
from pathlib import Path
from typing import Callable
import copy, hashlib, io, json, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from wellscope_core import (AnalysisError, Settings, VERSION, ENGINE_SHA256,
    analyze, profile_for, object_hash, json_bytes, BASES)

MODEL_CODE_SHA256=hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
FEATURES={
 'positive_fraction_pct':('공통 임계값 이상 well 비율 (%)','Shared-threshold well fraction (%)'),
 'median_corrected_signal':('배경 보정 신호 중앙값','Median corrected signal'),
 'mean_corrected_signal':('배경 보정 신호 평균','Mean corrected signal'),
 'q90_corrected_signal':('배경 보정 신호 90백분위','90th percentile corrected signal'),
 'top10_mean_corrected_signal':('상위 10% well 신호 평균','Top-decile mean corrected signal'),
}
MAX_SERIES=30
MAX_SERIES_BYTES=100*1024*1024

def proposed_label(filename:str)->float|None:
    """A UI suggestion only. Caller must obtain explicit confirmation."""
    match=re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%',Path(filename).name)
    if not match:return None
    val=float(match.group(1))
    return val if 0<=val<=100 else None

def feature_values(result:dict)->dict:
    s=result['summary']; t=result['table']; x=t.loc[t.valid,'corrected_signal_native'].to_numpy(float)
    k=max(1,int(np.ceil(.1*len(x))))
    return {'positive_fraction_pct':float(s['positive_fraction_pct']),
       'median_corrected_signal':float(np.median(x)), 'mean_corrected_signal':float(np.mean(x)),
       'q90_corrected_signal':float(np.quantile(x,.9)),
       'top10_mean_corrected_signal':float(np.mean(np.sort(x)[-k:])),
       'integrated_corrected_signal':float(np.sum(x)),
       'mean_raw_inner_signal':float(t.loc[t.valid,'signal_mean_native'].mean()),
       'median_background':float(t.loc[t.valid,'background_median_native'].median()),
       'saturated_wells_pct':float(s['wells_with_saturated_pixels_pct'])}

def rethreshold(result:dict, threshold:float, origin:str)->dict:
    """Keep original geometry/pixels; replace only the well-level threshold."""
    if not np.isfinite(threshold):raise AnalysisError('The common well threshold must be finite.')
    a=copy.deepcopy(result); t=a['table']; s=a['summary']; good=t.valid
    t['threshold_native']=float(threshold)
    t.loc[good,'classification']=np.where(t.loc[good,'corrected_signal_native']>=threshold,'Positive','Negative')
    k=int((t.classification=='Positive').sum()); n=int(good.sum())
    cfg=replace(Settings(**s['settings']),threshold_mode='fixed',fixed_threshold=float(threshold))
    a['profile']=profile_for(a['frame'],cfg,float(threshold))
    s.update(settings=asdict(cfg),positive_wells=k,negative_wells=n-k,
        positive_fraction_pct=100.0*k/n,threshold_mode='fixed',threshold_native=float(threshold),
        profile_id=a['profile']['profile_id'],well_threshold_origin=origin)
    s['qc_flags']=[q for q in s['qc_flags'] if q['code'] not in ('EXPLORATORY_THRESHOLD','ALL_ONE_CLASS','WEAK_INTENSITY_SPLIT')]
    s['qc_flags'].append({'code':'CONTROL_RULE_NOT_VALIDATED','severity':'warning',
        'message':'A shared control-derived cutoff is reproducible, but this does not establish a validated assay false-positive rate.'})
    if k in (0,n):s['qc_flags'].append({'code':'ALL_ONE_CLASS','severity':'warning','message':'All measurable wells are in one class. Occupancy cannot establish a unique mixture percentage.'})
    s['features']=feature_values(a)
    return a

def prepare_series(items:list[tuple[str,bytes,float]], cfg:Settings|None=None, quantile:float=.99,
                   fixed_threshold:float|None=None, progress:Callable|None=None)->dict:
    if not 3<=len(items)<=MAX_SERIES:raise AnalysisError('Use 3 to 30 standard images per batch.')
    if sum(len(x[1]) for x in items)>MAX_SERIES_BYTES:raise AnalysisError('The batch limit is 100 MB. Use smaller batches or original cropped image planes.')
    if not .90<=quantile<=.9999:raise AnalysisError('Reference quantile must be between 0.90 and 0.9999.')
    cfg=cfg or Settings()
    if cfg.geometry_mode!='auto' or cfg.excluded_ids.strip():raise AnalysisError('Batch mode uses automatic geometry without sample-specific exclusions. Review all overlays; use individually audited analysis if a sample needs manual geometry.')
    names=[x[0] for x in items]
    if len(set(names))!=len(names):raise AnalysisError('Standard filenames must be unique.')
    labels=np.asarray([x[2] for x in items],float)
    if not np.isfinite(labels).all() or not ((labels>=0)&(labels<=100)).all():raise AnalysisError('Confirm a numeric known mixture percentage (0..100) for every standard.')
    if 0 not in labels and fixed_threshold is None:raise AnalysisError('A 0% reference image is required for the control-derived well cutoff.')
    hashes=[hashlib.sha256(b).hexdigest() for _,b,_ in items]
    if len(set(hashes))!=len(hashes):raise AnalysisError('Duplicate image bytes were uploaded. Do not count renamed copies as independent samples.')
    results={};zero_quantiles=[];auto=[];image_types=set()
    for i,(name,data,level) in enumerate(items):
        a=analyze(data,replace(cfg,threshold_mode='auto'),source_kind='uploaded')
        a['summary']['sample_id']=name
        results[name]=a
        m=a['summary']['image'];image_types.add((m['bit_depth'],m['color_mode']))
        auto.append((a['summary']['threshold_native'],a['summary']['positive_fraction_pct']))
        if float(level)==0:
            v=a['table'].loc[lambda t:t.valid,'corrected_signal_native'].to_numpy(float)
            zero_quantiles.append(float(np.quantile(v,quantile)))
        if progress:progress(i+1,len(items),name)
    if len(image_types)!=1:raise AnalysisError('All standards must have the same bit depth and color mode; intensities are not silently normalized.')
    # Equal image weighting: the median of per-zero-image quantiles, not pooled wells.
    threshold=float(fixed_threshold) if fixed_threshold is not None else float(np.median(zero_quantiles))
    origin='operator_fixed' if fixed_threshold is not None else 'median_of_zero_image_quantiles'
    rows=[]
    for i,(name,data,level) in enumerate(items):
        a=rethreshold(results[name],threshold,origin);results[name]=a;s=a['summary']
        rows.append({'sample_id':name,'known_content_pct':float(level),'image_sha256':hashes[i],
            'valid_wells':s['valid_wells'],'positive_wells':s['positive_wells'],'excluded_wells':s['excluded_wells'],
            'pitch_px':s['geometry']['pitch_px'], 'width':s['image']['width'],'height':s['image']['height'],
            'automatic_threshold_native':float(auto[i][0]),'automatic_positive_fraction_pct':float(auto[i][1]),
            'shared_threshold_native':threshold, **feature_values(a)})
    return {'results':results,'table':pd.DataFrame(rows).sort_values('known_content_pct',kind='stable').reset_index(drop=True),
         'well_rule':{'method':origin,'reference_content_pct':0.,'quantile':float(quantile) if fixed_threshold is None else None,
                      'zero_image_quantiles':zero_quantiles,'threshold_native':threshold,'operator':'>='},
         'settings':asdict(replace(cfg,threshold_mode='fixed',fixed_threshold=threshold))}

def feature_diagnostics(df:pd.DataFrame,study:float=3.0)->pd.DataFrame:
    rows=[]
    for f in FEATURES:
        groups=df.groupby('known_content_pct',sort=True)[f].mean();d=np.diff(groups.to_numpy())
        below=df.loc[df.known_content_pct<study,f];above=df.loc[df.known_content_pct>=study,f]
        gap=float(above.min()-below.max()) if len(below) and len(above) else None
        rho=spearmanr(df.known_content_pct,df[f]).statistic if df[f].nunique()>1 else None
        rows.append({'feature':f,'spearman_descriptive':float(rho) if rho is not None and np.isfinite(rho) else None,
          'descending_steps':int(np.sum(d< -1e-8)),'flat_steps':int(np.sum(np.abs(d)<=1e-8)),
          'training_class_gap':gap,'strictly_increasing_level_means':bool(np.all(d>1e-8))})
    return pd.DataFrame(rows)

def _contract(batch:dict)->dict:
    results=list(batch['results'].values());first=results[0]['summary'];cfg=batch['settings']
    pitches=[a['summary']['geometry']['pitch_px'] for a in results]
    sizes=[[a['summary']['image']['width'],a['summary']['image']['height']] for a in results]
    return {'bit_depth':first['image']['bit_depth'],'color_mode':first['image']['color_mode'],
      'channel':cfg['channel'],'inner_ratio':cfg['inner_ratio'],'bg_inner_ratio':cfg['bg_inner_ratio'],
      'bg_outer_ratio':cfg['bg_outer_ratio'],'acquisition_id':cfg['acquisition_id'],
      'signal_method':'native_disc_mean_minus_annulus_median','training_sizes':sizes,
      'size_tolerance_px':2,'pitch_range_px':[float(min(pitches)*.85),float(max(pitches)*1.15)],
      'tolerances_scope':'software compatibility only; matching exposure/assay/input conditions require operator confirmation'}

def build_model(batch:dict,feature:str='positive_fraction_pct',study:float=3.,quantity_basis:str='calibrant_equivalent')->dict:
    if feature not in FEATURES:raise AnalysisError('Unknown sample-level feature.')
    if not np.isfinite(study) or not 0<study<100:raise AnalysisError('Study threshold must lie inside 0..100%.')
    if quantity_basis not in BASES:raise AnalysisError('Unknown quantity basis.')
    df=batch['table'];below=df.loc[df.known_content_pct<study,feature];above=df.loc[df.known_content_pct>=study,feature]
    if len(below)<1 or len(above)<1 or not np.isclose(df.known_content_pct,study,rtol=0,atol=1e-8).any():
        raise AnalysisError('Include at least one confirmed boundary standard, one below it, and one above it.')
    if not (df.known_content_pct>study).any():raise AnalysisError('Include a standard strictly above the study boundary.')
    high_below=float(below.max());low_above=float(above.min());gap=low_above-high_below
    ready=gap>1e-8
    cutoff=(high_below+low_above)/2 if ready else None
    levels=df.groupby('known_content_pct',sort=True)[feature].agg(['mean','count']).reset_index()
    xs=levels['mean'].to_numpy(float); ys=levels.known_content_pct.to_numpy(float)
    problems=[]
    if not np.all(np.diff(xs)>1e-8):problems.append('nonmonotonic_or_plateau_response')
    # Deliberately conservative numerical-calibration gate, not an assay standard.
    if (df.saturated_wells_pct>1).any():problems.append('saturated_well_interiors_gt_1pct_software_guard')
    quant_ready=not problems
    records=[]
    for row in df.to_dict('records'):
        records.append({k:(int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else v) for k,v in row.items()})
    model={'schema':'wellscope.screening.v1','software_version':VERSION,'engine_sha256':ENGINE_SHA256,
        'model_code_sha256':MODEL_CODE_SHA256,'status':'ready' if ready else 'overlapping_training_classes',
        'feature':feature,'study_threshold_pct':float(study),'quantity_basis':quantity_basis,
        'well_rule':copy.deepcopy(batch['well_rule']),'settings':batch['settings'],'contract':_contract(batch),
        'sample_rule':{'method':'midpoint_of_separated_training_extrema','operator':'>=',
             'cutoff':cutoff,'maximum_below_class_score':high_below,'minimum_at_or_above_score':low_above,
             'training_gap':gap,'feature_unit':'percent_threshold_positive_wells' if feature=='positive_fraction_pct' else 'native_corrected_intensity'},
        'numeric_calibration':{'status':'ready' if quant_ready else 'withheld','reasons':problems,
             'method':'strict_monotonic_piecewise_linear_no_extrapolation','uncertainty_interval':None,
             'x':xs.tolist(),'y':ys.tolist()},
        'reference_rows':records,'training_image_hashes':df.image_sha256.tolist(),
        'score_range':[float(df[feature].min()),float(df[feature].max())],
        'evidence_status':'model_development_only','independent_validation_n':0,
        'sample_counts_by_level':{str(float(y)):int(n) for y,n in zip(levels.known_content_pct,levels['count'])},
        'notes':['A screen-negative result is below a study decision threshold, not proof of absence or non-GMO certification.',
                 'The boundary and well cutoffs are different quantities. No mixture percentage is inferred from a filename.',
                 'Apparent agreement on calibration images is not sensitivity, specificity or independent accuracy.',
                 'Automatic grid extent and geometry must be reviewed; low resolution, saturation and artifacts remain limitations.']}
    model['model_id']=object_hash(model)[:20]
    return model

def read_model(data:bytes)->dict:
    try:
        if len(data)>3_000_000:raise ValueError('Model file too large')
        m=json.loads(data);mid=m.pop('model_id')
        if m.get('schema')!='wellscope.screening.v1' or m.get('software_version')!=VERSION or m.get('engine_sha256')!=ENGINE_SHA256 or m.get('model_code_sha256')!=MODEL_CODE_SHA256:
            raise AnalysisError('Model version/measurement engine mismatch. Rebuild the model with the current version.')
        if object_hash(m)[:20]!=mid:raise AnalysisError('Model checksum mismatch. Upload an unedited exported model.')
        m['model_id']=mid
        if m['feature'] not in FEATURES:raise ValueError()
        Settings(**m['settings']).validate()
        return m
    except AnalysisError:raise
    except Exception as exc:raise AnalysisError('Invalid screening model JSON.') from exc

def prediction_settings(model:dict)->Settings:
    # Do not use image labels or store image-specific brightness adjustments.
    return Settings(**model['settings'])

def predict(result:dict,model:dict,confirmed:bool=False,review_half_width:float=0.)->dict:
    if model.get('software_version')!=VERSION or model.get('engine_sha256')!=ENGINE_SHA256 or model.get('model_code_sha256')!=MODEL_CODE_SHA256:
        raise AnalysisError('The saved session model uses an older analysis engine. Reload or rebuild it with this version.')
    if not confirmed:raise AnalysisError('Confirm comparable acquisition/assay/input conditions and grid review before applying this research screening model.')
    if result['summary']['source_kind']=='synthetic':raise AnalysisError('Study screening cannot be applied to the synthetic demonstration.')
    if model['status']!='ready':raise AnalysisError('Training classes overlap in the selected feature; no sample cutoff has been fitted.')
    if not np.isfinite(review_half_width) or review_half_width<0:raise AnalysisError('Review half-width must be nonnegative and finite.')
    s=result['summary'];c=model['contract'];cfg=s['settings'];meta=s['image']
    for key in ['channel','inner_ratio','bg_inner_ratio','bg_outer_ratio','acquisition_id']:
        if cfg[key]!=c[key]:raise AnalysisError('Measurement settings differ from the locked screening model.')
    if meta['bit_depth']!=c['bit_depth'] or meta['color_mode']!=c['color_mode']:raise AnalysisError('Image bit depth or color mode differs from the model.')
    size=np.array([meta['width'],meta['height']]);sizes=np.asarray(c['training_sizes'])
    if not np.any(np.max(np.abs(sizes-size),axis=1)<=c['size_tolerance_px']):raise AnalysisError('Image dimensions are incompatible with the model. Do not resize to force matching; rebuild using equivalent native acquisition.')
    if not c['pitch_range_px'][0]<=s['geometry']['pitch_px']<=c['pitch_range_px'][1]:raise AnalysisError('Native pixel spacing differs from the model. Check acquisition scale.')
    if s['threshold_mode']!='fixed' or not np.isclose(s['threshold_native'],model['well_rule']['threshold_native'],rtol=0,atol=1e-8):raise AnalysisError('Use the model-locked well cutoff for prediction.')
    f=feature_values(result);score=float(f[model['feature']]);cut=float(model['sample_rule']['cutoff'])
    match=meta['raw_sha256'] in model['training_image_hashes']
    inside=model['score_range'][0]-1e-8<=score<=model['score_range'][1]+1e-8
    decision='At/above study threshold' if score>=cut else 'Below study threshold'
    if review_half_width>0 and abs(score-cut)<=review_half_width:decision='Review near study threshold'
    if not inside:decision='Outside calibration range'
    cal=model['numeric_calibration'];estimate=None
    if cal['status']=='ready' and cal['x'][0]<=score<=cal['x'][-1]:estimate=float(np.interp(score,cal['x'],cal['y']))
    out={'model_id':model['model_id'],'feature':model['feature'],'score':score,'sample_score_cutoff':cut,
        'study_threshold_pct':model['study_threshold_pct'],'well_threshold_native':s['threshold_native'],
        'decision':decision,'training_image_match':match,
        'evidence_status':'training_image_reanalysis' if match else 'new_image_unvalidated_prediction',
        'estimated_content_pct':estimate,'quantity_basis':model['quantity_basis'],
        'numeric_status':cal['status'],'numeric_withheld_reasons':cal['reasons'],
        'review_half_width_score_units':float(review_half_width),'review_band_is_confidence_interval':False,
        'saturated_wells_pct':f['saturated_wells_pct'],'uncertainty_interval':None}
    return out
