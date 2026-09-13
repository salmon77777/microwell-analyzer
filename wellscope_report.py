"""Publication-oriented exports using real measurements only; no generated data."""
from __future__ import annotations
import base64
import copy
import hashlib
import html
import importlib.metadata
import io
import json
import platform
import re
import textwrap
import zipfile
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from wellscope_core import APP_NAME, APP_SUBTITLE, VERSION, export_template, json_bytes


def png_bytes(a: np.ndarray) -> bytes:
    buffer=io.BytesIO();Image.fromarray(np.asarray(a,dtype=np.uint8)).save(buffer,format="PNG")
    return buffer.getvalue()


def _data_uri(b: bytes, mime: str="image/png") -> str:
    return f"data:{mime};base64,"+base64.b64encode(b).decode("ascii")


def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+","_",text.strip()).strip("._")[:70] or "sample"


def csv_bytes(df: pd.DataFrame) -> bytes:
    clean=df.copy()
    # Prevent formula interpretation when text-containing CSV is opened in Excel.
    for name in clean.select_dtypes(include=["object","string"]).columns:
        clean[name]=clean[name].map(lambda x: "'"+x if isinstance(x,str) and x.startswith(("=","+","-","@")) else x)
    return clean.to_csv(index=False,float_format="%.8g",na_rep="").encode("utf-8-sig")


def overlays(result: dict, max_side: int=1400) -> dict[str,np.ndarray]:
    """Annotation is drawn on copies only; original measured array is never edited."""
    native=result["frame"]["display_rgb"]
    h,w=native.shape[:2];scale=min(3.0,max_side/max(h,w))
    # Enlarging display does not create analytical resolution.
    dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
    raw=cv2.resize(native,(dw,dh),interpolation=cv2.INTER_NEAREST if scale>1 else cv2.INTER_AREA)
    grid=raw.copy();classified=raw.copy()
    pitch=result["grid"]["details"]["pitch_px"]*scale
    rad=max(2,round(pitch*0.29));thin=max(1,round(scale*0.42))
    for row in result["table"].itertuples(index=False):
        x,y=round(row.x_px*scale),round(row.y_px*scale)
        if x<0 or y<0 or x>=dw or y>=dh:continue
        if row.valid:
            cv2.drawMarker(grid,(x,y),(0,218,231),cv2.MARKER_CROSS,max(3,round(rad*1.5)),thin)
            if row.classification=="Positive":
                cv2.circle(classified,(x,y),rad,(255,211,61),thin)
            else:
                cv2.drawMarker(classified,(x,y),(255,87,87),cv2.MARKER_TILTED_CROSS,max(3,rad*2),thin)
        else:
            cv2.rectangle(grid,(x-2,y-2),(x+2,y+2),(157,166,178),1)
            cv2.rectangle(classified,(x-2,y-2),(x+2,y+2),(157,166,178),1)
    return {"raw":raw,"grid":grid,"classification":classified}


def histogram_bytes(result: dict, fmt: str="png") -> bytes:
    values=result["table"].loc[lambda d:d.valid,"corrected_signal_native"].to_numpy()
    figure=Figure(figsize=(6.8,3.5),layout="constrained");ax=figure.subplots()
    ax.hist(values,bins=min(80,max(12,int(np.sqrt(len(values))))))
    threshold=result["summary"]["threshold_native"]
    ax.axvline(threshold,linestyle="--",label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Background-corrected signal (native intensity units)");ax.set_ylabel("Well count")
    ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,fontsize=8)
    ax.set_title("Intensity distribution",loc="left",fontsize=12)
    if result["summary"]["threshold_mode"]=="auto":
        figure.suptitle("Exploratory threshold; control validation required",fontsize=9)
    out=io.BytesIO();figure.savefig(out,format=fmt,dpi=240);return out.getvalue()


def signal_plot_bytes(result: dict, fmt: str="png") -> bytes:
    tab=result["table"].loc[lambda d:d.valid]
    fig=Figure(figsize=(6.8,3.5),layout="constrained");ax=fig.subplots()
    ax.scatter(np.arange(1,len(tab)+1),tab.corrected_signal_native,s=4,alpha=.7)
    ax.axhline(result["summary"]["threshold_native"],linestyle="--",label="Fixed threshold" if result["summary"]["threshold_mode"]=="fixed" else "Exploratory threshold")
    ax.set_xlabel("Measurable well index");ax.set_ylabel("Corrected signal (native units)")
    ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,fontsize=8)
    out=io.BytesIO();fig.savefig(out,format=fmt,dpi=240);return out.getvalue()


def calibration_plot_bytes(result: dict, fmt: str="png") -> bytes | None:
    cal=result["summary"]["calibration"]
    if not cal.get("reference_rows"):return None
    rows=pd.DataFrame(cal["reference_rows"]);means=pd.DataFrame(cal["level_means"])
    fig=Figure(figsize=(6.8,3.5),layout="constrained");ax=fig.subplots()
    ax.scatter(rows.known_content_pct,rows.positive_fraction_pct,s=16,label="Calibration replicates")
    ax.plot(means.known_content_pct,means.positive_fraction_pct,label="Piecewise linear calibration")
    ax.axvline(cal["study_threshold_pct"],linestyle="--",label="Study threshold")
    ax.set_xlabel("Known calibrant content (%)");ax.set_ylabel("Positive-well fraction (%)")
    ax.set_title("Calibration, not independent validation",loc="left",fontsize=11)
    ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,fontsize=8)
    out=io.BytesIO();fig.savefig(out,format=fmt,dpi=240);return out.getvalue()


def completed_summary(result: dict, sample_id: str, reviewed: bool) -> dict:
    s=copy.deepcopy(result["summary"]);s["sample_id"]=sample_id[:100];s["operator_grid_reviewed"]=bool(reviewed)
    versions={}
    for dist in ("streamlit","numpy","scipy","pandas","Pillow","opencv-python-headless","matplotlib"):
        try:versions[dist]=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:versions[dist]="not installed under this package name"
    s["environment"]={"python":platform.python_version(),"packages":versions}
    return s


def method_text(result: dict, sample_id: str, reviewed: bool) -> str:
    s=result["summary"];g=s["geometry"];cfg=s["settings"]
    auto="An image-specific Otsu threshold was used for exploratory inspection only." if s["threshold_mode"]=="auto" else "A fixed fluorescence threshold was applied. Its experimental validation must be reported separately."
    text=(f"Microwell images were analyzed using {APP_NAME} v{VERSION}. "
          f"The {s['image']['bit_depth']}-bit image ({s['image']['width']} x {s['image']['height']} pixels) "
          f"was measured in its native pixel coordinates using channel {cfg['channel']}. "
          f"Geometry was obtained using {g['geometry_source']}. The median grid spacings were "
          f"{g['pitch_x_px']:.4f} and {g['pitch_y_px']:.4f} pixels. "
          f"Signal was the native-pixel mean within a disc of radius {cfg['inner_ratio']:.3f} times the smaller median pitch, "
          f"minus the median in a concentric annulus spanning {cfg['bg_inner_ratio']:.3f}-"
          f"{cfg['bg_outer_ratio']:.3f} times that pitch. Pixel intensities were not rescaled for measurement. "
          f"Incomplete measurement footprints, pixels outside the ROI, non-opaque footprints and explicitly excluded wells "
          f"were excluded rather than counted as negatives. The threshold was {s['threshold_native']:.8g} native units "
          f"with the operator >=. {auto} Of {s['fitted_positions']} fitted positions, "
          f"{s['valid_wells']} were measurable and {s['positive_wells']} exceeded the threshold "
          f"({s['positive_fraction_pct']:.6g}%). This fraction is not itself GMO content. "
          f"Profile ID: {s['profile_id']}. Operator grid review recorded: {reviewed}. "
          "These software checks do not establish filling quality, analytical sensitivity, specificity, or regulatory compliance. ")
    cal=s["calibration"]
    if cal.get("estimated_content_pct") is None:
        text+="No calibrated content estimate was reported."
    else:
        text+=(f"The user-supplied calibration {cal['calibration_id']} was applied using piecewise linear interpolation "
               f"within its observed range; the quantity basis was {cal['quantity_basis']}. "
               f"Estimated content was {cal['estimated_content_pct']:.6g}%, with a study threshold of "
               f"{cal['study_threshold_pct']:.6g}%. The operational review band was +/-{cal['review_band_percentage_points']:.6g} "
               "percentage points and is not a confidence interval. No measurement-uncertainty interval was calculated. "
               "Independent validation is required; a below-threshold result is not a non-GMO certification.")
    if s["source_kind"]=="synthetic":text="SYNTHETIC DEMONSTRATION — NOT EXPERIMENTAL DATA. "+text
    return text



def decision_text(result: dict) -> str:
    screen=result["summary"].get("screening")
    if screen:
        threshold=screen["study_threshold_pct"]
        if screen["decision"]=="At/above study threshold":return f"GMO screen positive | >= {threshold:g}% study class"
        if screen["decision"]=="Below study threshold":return f"Below {threshold:g}% study screen | not GMO-free certification"
        return screen["decision"]
    return result["summary"]["calibration"].get("decision","Not evaluated")

def panel_svg(result: dict, sample_id: str, reviewed: bool=False) -> bytes:
    """Editable vector labels with embedded raster evidence, independent of browser size."""
    s=result["summary"];images=overlays(result,max_side=1200);esc=html.escape
    title=esc(sample_id[:65]);cal=s["calibration"];est=cal.get("estimated_content_pct")
    content=f"{est:.2f}%" if est is not None else "Not quantified"
    decision=decision_text(result)
    badge="SYNTHETIC DEMONSTRATION" if s["source_kind"]=="synthetic" else ("FIXED THRESHOLD" if s["threshold_mode"]=="fixed" else "EXPLORATORY ANALYSIS")
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1140" viewBox="0 0 1800 1140">',
         '<rect width="1800" height="1140" fill="white"/>',
         '<g font-family="Arial, Helvetica, sans-serif" fill="#172d43">',
         f'<text x="45" y="61" font-size="32" font-weight="700">{esc(APP_NAME)}</text>',
         f'<text x="45" y="97" font-size="19" fill="#617184">{esc(APP_SUBTITLE)}</text>',
         f'<text x="1755" y="58" text-anchor="end" font-size="17" fill="#536879">{badge}</text>',
         f'<text x="1755" y="89" text-anchor="end" font-size="18">Sample: {title}</text>',
         '<line x1="45" y1="121" x2="1755" y2="121" stroke="#dce4e9"/>']
    for i,(key,labeltext) in enumerate([("raw","A  Input image"),("grid","B  Fitted grid"),("classification","C  Well classification")]):
        x=45+i*578
        svg.append(f'<text x="{x}" y="161" font-size="24" font-weight="700">{labeltext}</text>')
        uri=_data_uri(png_bytes(images[key]))
        svg.append(f'<rect x="{x}" y="180" width="554" height="548" fill="#05090b" rx="5"/>')
        svg.append(f'<image x="{x}" y="180" width="554" height="548" preserveAspectRatio="xMidYMid meet" href="{uri}"/>')
        legend=[f"Native input: {s['image']['width']} x {s['image']['height']} px", "Cyan +: measured; grey square: excluded", "Yellow circle: positive; red x: negative"][i]
        svg.append(f'<text x="{x}" y="759" font-size="17" fill="#617184">{esc(legend)}</text>')
    labels=[("Measurable wells",f"{s['valid_wells']:,}"),("Threshold-positive",f"{s['positive_wells']:,}"),
            ("Positive fraction",f"{s['positive_fraction_pct']:.2f}%"),("Estimated mixture",content)]
    for i,(labeltext,value) in enumerate(labels):
        x=45+i*435
        svg.append(f'<rect x="{x}" y="799" width="405" height="108" rx="9" fill="#f4f7f9"/>')
        svg.append(f'<text x="{x+20}" y="832" font-size="17" fill="#617184">{labeltext}</text>')
        svg.append(f'<text x="{x+20}" y="880" font-size="32" font-weight="700">{value}</text>')
    svg.append(f'<text x="45" y="950" font-size="24" font-weight="700">Interpretation: {esc(decision)}</text>')
    review="Operator grid review recorded" if reviewed else "Operator grid review not recorded"
    mode="fixed" if s['threshold_mode']=='fixed' else "exploratory automatic"
    lines=[f"Threshold {s['threshold_native']:.2f} native units ({mode}) | Excluded positions: {s['excluded_wells']} | {review}",
           f"Profile {s['profile_id']} | v{VERSION} | Grid: {s['geometry']['columns']} columns x {s['geometry']['rows']} rows (inferred/defined extent)",
           "Positive-well fraction is not GMO content. Software checks do not establish assay validity. Display enlargement adds no resolution."]
    if est is not None:
        lines[-1]=f"Basis: {cal['quantity_basis']} | Study threshold: {cal['study_threshold_pct']:g}% | Review band is not a confidence interval; not non-GMO certification."
    for i,line in enumerate(lines):svg.append(f'<text x="45" y="{993+i*32}" font-size="17" fill="#617184">{esc(line)}</text>')
    if s.get("screening"):
        sc=s["screening"]
        evidence="TRAINING IMAGE - NOT INDEPENDENT VALIDATION" if sc["training_image_match"] else "NEW IMAGE - MODEL NOT INDEPENDENTLY VALIDATED"
        svg.append(f'<rect x="40" y="1040" width="1730" height="42" fill="white"/>')
        svg.append(f'<text x="45" y="1068" font-size="17" fill="#617184">{esc(evidence)} | score {sc["score"]:.5f} | cutoff {sc["sample_score_cutoff"]:.5f}</text>')
    cautions=[q["code"] for q in s["qc_flags"] if q["severity"]=="warning"]
    qc_line="QC notes: "+("; ".join(cautions[:4]) if cautions else "See full report; not an assay-validation certificate")
    svg.append(f'<text x="45" y="1093" font-size="15" fill="#617184">{esc(qc_line)}</text>')
    svg.append('</g></svg>');return ''.join(svg).encode('utf-8')


def report_html(result: dict, sample_id: str, reviewed: bool=False) -> bytes:
    esc=html.escape;s=result["summary"];imgs=overlays(result);cal=s["calibration"]
    est=cal.get("estimated_content_pct");content=f"{est:.2f}%" if est is not None else "Not quantified"
    state="Synthetic demonstration" if s["source_kind"]=="synthetic" else ("Fixed-threshold analysis" if s["threshold_mode"]=="fixed" else "Exploratory analysis")
    metrics=[("Measurable wells",f"{s['valid_wells']:,}"),("Threshold-positive wells",f"{s['positive_wells']:,}"),
             ("Positive-well fraction",f"{s['positive_fraction_pct']:.2f}%"),("Estimated mixture",content)]
    metric_html=''.join(f'<div class="metric"><span>{a}</span><strong>{b}</strong></div>' for a,b in metrics)
    cards=''.join(f'<section class="imagecard"><h2>{heading}</h2><img src="{_data_uri(png_bytes(imgs[key]))}"><p>{note}</p></section>' for key,heading,note in [
        ("raw","A / Input image",f"{s['image']['width']} x {s['image']['height']} native pixels; display enlarged only."),
        ("grid","B / Fitted grid","Cyan +: measurable positions; grey square: excluded."),
        ("classification","C / Well classification","Yellow circle: threshold-positive; red x: negative; grey square: excluded.")])
    flags=''.join(f'<li><b>{esc(q["code"])}</b> — {esc(q["message"])}</li>' for q in s['qc_flags'])
    screen=s.get("screening")
    screening_note=""
    if screen:
        evidence="Training-image reanalysis; not independent validation" if screen["training_image_match"] else "New-image prediction; model not independently validated"
        screening_note=f'<p class="small"><b>{esc(evidence)}</b><br>Sample score: {screen["score"]:.5f}; sample cutoff: {screen["sample_score_cutoff"]:.5f}; model: {screen["model_id"]}</p>'
    config=json.dumps(completed_summary(result,sample_id,reviewed),ensure_ascii=False,indent=2,allow_nan=False)
    out=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(APP_NAME)} | {esc(sample_id)}</title>
<style>
*{{box-sizing:border-box}}body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#eef3f6;color:#182f45}}main{{max-width:1460px;margin:28px auto;background:white;padding:36px 42px;border-radius:16px}}header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #dce5ea;padding-bottom:24px}}h1{{font-size:32px;margin:0;letter-spacing:-.7px}}.sub{{color:#637589;margin:8px 0 0}}.meta{{text-align:right;font-size:14px;line-height:1.8}}.badge{{color:#276473;background:#edf7f8;border-radius:20px;padding:7px 14px;display:inline-block;font-size:13px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:26px 0}}.metric{{padding:20px;background:#f5f8fa;border:1px solid #e0e8ed;border-radius:10px}}.metric span{{font-size:13px;color:#607589;display:block}}.metric strong{{display:block;font-size:29px;margin-top:10px;letter-spacing:-.5px}}.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}h2{{font-size:18px}}.imagecard img{{width:100%;aspect-ratio:1;object-fit:contain;background:#05090b;border-radius:6px}}.imagecard p{{font-size:12px;color:#637589;line-height:1.5;min-height:36px}}.decision{{border:1px solid #dfe7ec;background:#f9fbfc;border-left:4px solid #487f8e;border-radius:8px;padding:16px 20px;margin:22px 0;line-height:1.7}}.decision strong{{font-size:20px}}.small{{font-size:12px;color:#637589;line-height:1.6}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.chart{{width:100%}}li{{font-size:13px;margin:8px 0;line-height:1.5}}details{{margin-top:18px;border-top:1px solid #dfe7ec;padding-top:16px}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;padding:18px;font-size:12px}}.method{{line-height:1.8;font-size:14px}}@media(max-width:800px){{main{{padding:20px}}.metrics,.images,.grid2{{grid-template-columns:1fr}}header{{display:block}}.meta{{text-align:left;margin-top:15px}}}}@media print{{body{{background:white}}main{{margin:0;padding:10px;max-width:none}}details{{break-inside:avoid}}.images,.metrics{{break-inside:avoid}}}}
</style><main><header><div><h1>{esc(APP_NAME)}</h1><p class="sub">{esc(APP_SUBTITLE)}</p></div><div class="meta"><div class="badge">{esc(state)}</div><br>Sample: <b>{esc(sample_id)}</b><br>Profile: {s['profile_id']} &nbsp; / &nbsp; v{VERSION}</div></header>
<div class="metrics">{metric_html}</div><div class="images">{cards}</div>
<div class="decision"><strong>{esc(decision_text(result))}</strong><br>Positive-well fraction and GMO content are different quantities. A below-threshold result is not a non-GMO certification.</div>
<p class="small">Native threshold: {s['threshold_native']:.4g} | Negative wells: {s['negative_wells']:,} | Excluded positions: {s['excluded_wells']:,} | Operator grid review: {'recorded' if reviewed else 'not recorded'}<br>Software geometric checks are not evidence of filling, amplification validity or analytical accuracy.</p>
{screening_note}<div class="grid2"><section><h2>Signal distribution</h2><img class="chart" src="{_data_uri(histogram_bytes(result))}"></section><section><h2>Quality-control notes</h2><ul>{flags}</ul></section></div>

<details><summary>Full analysis record / JSON</summary><pre>{esc(config)}</pre></details>
<p class="small">Data source: {esc(s['source_kind'])} | Image SHA-256: {s['image']['raw_sha256']}<br>No calibration coefficients, confidence intervals or experimental validation statistics are fabricated by this report.</p></main></html>'''
    return out.encode('utf-8')


def export_bundle(result: dict, sample_id: str, reviewed: bool, original_data: bytes | None=None) -> bytes:
    images=overlays(result);s=completed_summary(result,sample_id,reviewed)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('report.html',report_html(result,sample_id,reviewed))
        z.writestr('analysis_panel.svg',panel_svg(result,sample_id,reviewed))
        z.writestr('per_well.csv',csv_bytes(result['table']))
        z.writestr('summary.json',json_bytes(s))
        row={key:s[key] for key in ("sample_id","source_kind","software_version","valid_wells","positive_wells","negative_wells","excluded_wells","positive_fraction_pct","threshold_native","threshold_mode","profile_id")}
        row.update(image_sha256=s["image"]["raw_sha256"],estimated_content_pct=s["calibration"].get("estimated_content_pct"),quantity_basis=s["calibration"].get("quantity_basis"),decision=s["calibration"].get("decision"),operator_grid_reviewed=bool(reviewed))
        screen=s.get('screening')
        if screen:
            row.update(screening_decision=screen['decision'],sample_score=screen['score'],sample_score_cutoff=screen['sample_score_cutoff'],training_image_match=screen['training_image_match'],model_id=screen['model_id'])
        z.writestr('sample_summary.csv',csv_bytes(pd.DataFrame([row])))
        z.writestr('analysis_profile.json',json_bytes(result['profile']))
        z.writestr('grid_template.json',export_template(result))
        # Retain settings/QC records, without an unsolicited manuscript-writing paragraph.
        for name,array in images.items():z.writestr(name+'_display.png',png_bytes(array))
        z.writestr('signal_histogram.png',histogram_bytes(result))
        z.writestr('signal_histogram.svg',histogram_bytes(result,'svg'))
        z.writestr('signal_by_well.png',signal_plot_bytes(result))
        cal=calibration_plot_bytes(result)
        if cal is not None:z.writestr('calibration_not_validation.png',cal)
        if original_data is not None:
            fmt=result['summary']['image']['source_format'].lower();ext={'jpeg':'jpg','tiff':'tif'}.get(fmt,fmt)
            z.writestr('input_original.'+safe_id(ext),original_data)
        z.writestr('READ_ME.txt',
            'PRIVATE ANALYSIS OUTPUT — do not upload this ZIP to a public GitHub repository.\n'
            'Open report.html in a browser. SVG has editable vector labels with embedded raster panels.\n'
            'display.png images are annotated/enlarged visualization only; measurements used native pixels.\n'
            'Valid/measurable means the selected image footprint was measurable, not that a well was filled or amplification was valid.\n'
            'A fixed threshold is not automatically an experimentally validated threshold.\n'
            'GMO-content estimation requires user calibration and an independently validated assay.\n')
    return out.getvalue()
