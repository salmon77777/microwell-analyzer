"""Optional command-line entrypoint; no Streamlit server required.
Example (exploration): python analyze_image.py image.png --output-dir output
A fixed threshold must first be chosen using experimental controls.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from wellscope_core import AnalysisError,Settings,analyze,decode_image,read_profile
from wellscope_report import export_bundle,safe_id

def main():
    parser=argparse.ArgumentParser(description='WellScope LAMP native-pixel image analysis')
    parser.add_argument('image',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('wellscope_output'))
    parser.add_argument('--sample-id',default='sample')
    parser.add_argument('--threshold',type=float,help='Fixed, control-checked threshold in corrected native intensity units')
    parser.add_argument('--profile',type=Path,help='Saved analysis_profile.json')
    parser.add_argument('--template',type=Path,help='Registered grid_template.json (same field and pixel coordinates)')
    args=parser.parse_args()
    try:
        data=args.image.read_bytes();cfg=Settings()
        if args.threshold is not None:cfg=replace(cfg,threshold_mode='fixed',fixed_threshold=args.threshold)
        if args.profile:
            p=read_profile(args.profile.read_bytes(),decode_image(data))
            cfg=replace(cfg,threshold_mode='fixed',fixed_threshold=p['threshold_native'],channel=p['channel'],
                        acquisition_id=p['acquisition_id'],inner_ratio=p['inner_ratio'],bg_inner_ratio=p['bg_inner_ratio'],bg_outer_ratio=p['bg_outer_ratio'])
        template=args.template.read_bytes() if args.template else None
        if template:cfg=replace(cfg,geometry_mode='template')
        result=analyze(data,cfg,template)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        target=args.output_dir/(safe_id(args.sample_id)+'_analysis.zip')
        target.write_bytes(export_bundle(result,args.sample_id,False,data))
        print('Saved:',target.resolve())
        print('Exploratory signal fraction:',result['summary']['positive_fraction_pct'],'%' if cfg.threshold_mode=='auto' else '(fixed threshold)')
        print('GMO content: not inferred without a user-supplied calibration.')
    except (OSError,AnalysisError) as exc:parser.exit(1,f'Error: {exc}\n')
if __name__=='__main__':main()
