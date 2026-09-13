"""Real-data A4 portrait figure and series exports. No synthetic performance data."""
from __future__ import annotations
import base64, html, io, json, zipfile
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from wellscope_core import VERSION, json_bytes
from wellscope_batch import FEATURES, predict, feature_diagnostics
from wellscope_report import overlays, png_bytes, csv_bytes, report_html


def chart_bytes(batch:dict,model:dict,kind:str,fmt:str='png')->bytes:
    df=batch['table']; feature=model['feature']
    fig=Figure(figsize=(4.2,2.65),layout='constrained');ax=fig.subplots()
    labels=[f'{x:g}' for x in df.known_content_pct]
    if kind=='distribution':
        values=[batch['results'][name]['table'].loc[lambda t:t.valid,'corrected_signal_native'].to_numpy() for name in df.sample_id]
        ax.boxplot(values,positions=np.arange(len(values)),widths=.55,whis=(10,90),showfliers=False)
        ax.set_xticks(np.arange(len(labels)),labels);ax.set_ylabel('Corrected signal (native units)')
        ax.axhline(batch['well_rule']['threshold_native'],ls='--',lw=1,label='Common well cutoff')
        ax.legend(fontsize=6.5,frameon=False)
    elif kind=='response':
        ax.plot(df.known_content_pct,df.automatic_positive_fraction_pct,'o--',ms=3,lw=1,label='Image-specific automatic cutoff')
        ax.plot(df.known_content_pct,df.positive_fraction_pct,'s-',ms=3,lw=1,label='Shared reference-derived cutoff')
        ax.set_ylim(-4,105);ax.set_ylabel('Threshold-positive wells (%)');ax.legend(fontsize=6.4,frameon=False,loc='lower right')
    elif kind=='boundary':
        d=df[df.known_content_pct<=model['study_threshold_pct']]
        ax.scatter(d.known_content_pct,d[feature],s=35)
        for row in d.itertuples():ax.annotate(f'{getattr(row,feature):.2f}',(row.known_content_pct,getattr(row,feature)),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        cut=model['sample_rule']['cutoff']
        if cut is not None:ax.axhline(cut,ls='--',lw=1,label=f'Sample score cutoff = {cut:.3f}')
        ax.set_xticks(sorted(d.known_content_pct.unique()))
        bottom=min(0,float(d[feature].min())*.9);top=float(d[feature].max())*1.35+.01
        ax.set_ylim(bottom,top);ax.set_ylabel(FEATURES[feature][1]);ax.legend(fontsize=6.8,frameon=False,loc='upper left')
    elif kind=='feature':
        ax.plot(df.known_content_pct,df[feature],'o-',ms=4,lw=1)
        if model['sample_rule']['cutoff'] is not None:ax.axhline(model['sample_rule']['cutoff'],ls='--',lw=1,label='Sample screen cutoff')
        ax.set_ylabel(FEATURES[feature][1]);ax.legend(fontsize=7,frameon=False)
    elif kind=='saturation':
        ax.scatter(df.known_content_pct,df.saturated_wells_pct,s=28)
        ax.set_ylim(-3,105);ax.set_ylabel('Wells with saturated inner pixels (%)')
    else:raise ValueError(kind)
    ax.set_xlabel('Assigned GMO/non-GMO mixture (%)',fontsize=8)
    ax.tick_params(labelsize=7);ax.yaxis.label.set_size(8);ax.spines[['top','right']].set_visible(False)
    out=io.BytesIO();fig.savefig(out,format=fmt,dpi=240);return out.getvalue()


def _uri(data:bytes,mime:str='image/png')->str:
    return 'data:'+mime+';base64,'+base64.b64encode(data).decode()


def portrait_svg(batch:dict,model:dict)->bytes:
    """210 x 297 mm, real-data panels. Training status cannot be hidden."""
    df=batch['table'];esc=lambda x:html.escape(str(x));parts=[]
    def text(x,y,s,size=24,bold=False,color='#1b2936'):
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{esc(s)}</text>')
    def rect(x,y,w,h,fill='white',stroke='#d5dce2'):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="7"/>')
    def image(x,y,w,h,b):parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="{_uri(b)}"/>')
    def title(letter,label,x,y):text(x,y,letter,40,True);text(x+55,y-2,label,29,True)
    def select(level):return df.iloc[int(np.argmin(abs(df.known_content_pct-level)))]['sample_id']
    study=float(model['study_threshold_pct'])
    below_levels=sorted(df.loc[(df.known_content_pct<study)&(df.known_content_pct>0),'known_content_pct'].unique())
    low=float(below_levels[-1]) if below_levels else float(df.known_content_pct.min())
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="210mm" height="297mm" viewBox="0 0 2100 2970">')
    rect(0,0,2100,2970,stroke='white')
    text(110,100,'Figure 6 | App-assisted microwell image analysis',41,True)
    text(110,146,'MODEL-DEVELOPMENT EXAMPLE · shared-cutoff screening, not independent validation',22)
    text(110,183,f'{len(df)} supplied images; {df.known_content_pct.nunique()} assigned mixture levels. No assay accuracy claim.',21)
    # A. Four key samples, native values and exact assigned labels.
    title('A','Representative input images',110,260)
    for i,lev in enumerate([float(df.known_content_pct.min()),low,study,float(df.known_content_pct.max())]):
        name=select(lev);a=batch['results'][name];actual=float(df.set_index('sample_id').loc[name,'known_content_pct']);x=110+i*480
        text(x+180,300,f'{actual:g}%',28,True);image(x,320,450,440,png_bytes(a['frame']['display_rgb']))
    text(110,795,'Original PNG values; common display mapping; no intensity enhancement or physical scale inferred.',20)
    # B/C
    title('B',f'Well-level analysis ({study:g}% image)',110,880)
    name=select(study);a=batch['results'][name];imgs=overlays(a);h,w=a['frame']['display_rgb'].shape[:2]
    x0,y0,x1,y1=round(w*.26),round(h*.27),round(w*.46),round(h*.47)
    for i,(key,label) in enumerate([('raw','Input crop'),('grid','Grid'),('classification','Classified')]):
        arr=imgs[key];sx=arr.shape[1]/w;sy=arr.shape[0]/h
        crop=arr[round(y0*sy):round(y1*sy),round(x0*sx):round(x1*sx)];x=110+i*298
        text(x,925,label,21,True);image(x,948,275,285,png_bytes(crop))
    text(110,1273,'Same native-coordinate crop in all three images.',20)
    text(110,1303,'Cyan +: grid; yellow circle: above well cutoff;',20)
    text(110,1333,'red x: below well cutoff; grey square: excluded.',20)
    text(110,1373,f'N = {a["summary"]["valid_wells"]:,} measurable wells (not independent samples).',20)
    title('C','Well-signal distributions',1090,880)
    image(1050,920,960,475,chart_bytes(batch,model,'distribution'))
    text(1090,1425,'Boxes: within-image Q1/median/Q3; whiskers: P10–P90.',19)
    text(1090,1454,'Well-level distributions are not biological replicates.',19)
    # D/E
    title('D','Full-series response',110,1560)
    image(85,1600,940,575,chart_bytes(batch,model,'response'))
    text(110,2210,'Connected points are guides, not fitted quantitative curves.',19)
    title('E','Low-level screening boundary',1090,1560)
    image(1055,1600,960,575,chart_bytes(batch,model,'boundary'))
    text(1090,2210,'The displayed images also trained this rule.',19)
    # F numerical cards, not fabricated screen captures.
    title('F','Computed result cards · selected development images',110,2310)
    for i,lev in enumerate([low,study]):
        name=select(lev);a=batch['results'][name];x=110+i*960;rect(x,2345,915,375,fill='#f5f7f9')
        text(x+25,2390,f'Assigned mixture: {lev:g}%  |  WellScope LAMP v{VERSION}',23,True)
        score=float(df.set_index('sample_id').loc[name,model['feature']]);cut=model['sample_rule']['cutoff']
        text(x+25,2433,FEATURES[model['feature']][1],21)
        text(x+25,2498,f'{score:.3f}',49,True)
        label=f'Screen positive: >={study:g}% study class' if cut is not None and score>=cut else f'Screen negative: <{study:g}% study class' if cut is not None else 'Screen not evaluated'
        text(x+25,2550,label,27,True)
        text(x+25,2600,'Mixture estimate: '+('withheld (nonunique / saturated response)' if model['numeric_calibration']['status']!='ready' else 'see calibrated individual result'),20)
        text(x+25,2640,'Training-image reanalysis; not an independent validation.',20)
        text(x+25,2680,'Below the study cutoff does not mean GMO-free.',20)
    cut=model['sample_rule']['cutoff'];sct=f'{cut:.5f}' if cut is not None else 'not fitted'
    text(110,2780,f'Well cutoff = {batch["well_rule"]["threshold_native"]:.5f} native units; sample score cutoff = {sct}.',22)
    text(110,2820,f'These are different thresholds. {study:g}% is the assigned mixture boundary, not the well-positive fraction.',20)
    text(110,2860,'Development images per level: '+', '.join(f'{float(k):g}% (n={v})' for k,v in model['sample_counts_by_level'].items()),19)
    text(110,2905,'A4 portrait · actual measurements · replace panel E with independent validation when acquired.',20)
    parts.append('</svg>');return ''.join(parts).encode()


def series_bundle(batch:dict,model:dict)->bytes:
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('screening_model.json',json_bytes(model))
        z.writestr('series_summary.csv',csv_bytes(batch['table']))
        z.writestr('feature_diagnostics.csv',csv_bytes(feature_diagnostics(batch['table'],model['study_threshold_pct'])))
        z.writestr('Figure6_A4_development.svg',portrait_svg(batch,model))
        for kind in ['distribution','response','boundary','feature','saturation']:
            z.writestr(f'charts/{kind}.png',chart_bytes(batch,model,kind));z.writestr(f'charts/{kind}.svg',chart_bytes(batch,model,kind,'svg'))
        for i,(name,result) in enumerate(batch['results'].items(),1):
            prefix=f'images/{i:02d}'
            z.writestr(prefix+'_per_well.csv',csv_bytes(result['table']))
            if model['status']=='ready':result['summary']['screening']=predict(result,model,confirmed=True)
            for kind,array in overlays(result).items():z.writestr(prefix+'_'+kind+'.png',png_bytes(array))
            z.writestr(prefix+'_report.html',report_html(result,name,False))
        z.writestr('README_PRIVATE.txt','Private study output: do not upload this archive to a public GitHub repository.\nThe model and plotted samples are development data, not independent validation.\nDo not treat image well counts as biological sample size. No physical scale was supplied.\n')
    return out.getvalue()
