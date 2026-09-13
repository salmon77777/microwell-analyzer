"""WellScope LAMP 1.1.0 — reproducible, native-pixel microwell measurements.

This module intentionally does not infer GMO content from a filename or from a
positive-well percentage alone. It has no network, Streamlit, or AI dependency.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import center_of_mass, gaussian_filter, label, maximum_filter
from scipy.spatial import cKDTree

APP_NAME = "WellScope LAMP"
APP_SUBTITLE = "Microwell fluorescence analysis"
VERSION = "1.1.0"
ENGINE_SHA256 = hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_WELLS = 40_000
CAL_COLUMNS = ["sample_id", "positive_fraction_pct", "known_content_pct", "profile_id", "quantity_basis"]
BASES = ("calibrant_equivalent", "gm_mass_fraction", "target_reference_copy_ratio")


class AnalysisError(ValueError):
    """An expected, user-actionable input/analysis problem."""


@dataclass(frozen=True)
class Settings:
    channel: str = "G"
    threshold_mode: str = "auto"
    fixed_threshold: float = 60.0
    inner_ratio: float = 0.22
    bg_inner_ratio: float = 0.34
    bg_outer_ratio: float = 0.46
    geometry_mode: str = "auto"
    roi: tuple[int, int, int, int] | None = None  # inclusive x0,y0; exclusive x1,y1
    candidate_floor: float | None = None
    pitch_hint: float | None = None
    manual_rows: int = 0
    manual_cols: int = 0
    # Corner centers in order TL, TR, BR, BL (not chip outer corners).
    manual_corners: tuple[tuple[float, float], ...] = ()
    excluded_ids: str = ""
    exclusion_reason: str = ""
    acquisition_id: str = "LAMP-FITC-v1"

    def validate(self) -> None:
        if self.channel not in ("R", "G", "B"):
            raise AnalysisError("Signal channel must be R, G or B.")
        if self.threshold_mode not in ("auto", "fixed"):
            raise AnalysisError("Threshold mode must be auto or fixed.")
        if not np.isfinite(self.fixed_threshold):
            raise AnalysisError("The fluorescence threshold must be finite.")
        if not (0.08 <= self.inner_ratio < self.bg_inner_ratio < self.bg_outer_ratio < 0.50):
            raise AnalysisError("Require 0.08 <= inner radius < background inner < background outer < 0.50 pitch.")
        if self.geometry_mode not in ("auto", "manual", "template"):
            raise AnalysisError("Unknown geometry mode.")
        if not self.acquisition_id.strip():
            raise AnalysisError("Enter an acquisition / assay protocol ID.")
        if self.pitch_hint is not None and (not np.isfinite(self.pitch_hint) or self.pitch_hint < 3):
            raise AnalysisError("Pitch hint must be at least 3 native pixels.")
        if self.candidate_floor is not None and (not np.isfinite(self.candidate_floor) or self.candidate_floor < 0):
            raise AnalysisError("Candidate floor must be finite and nonnegative.")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def object_hash(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def _orientation(a: np.ndarray, code: int) -> np.ndarray:
    if code == 2: a = np.fliplr(a)
    elif code == 3: a = np.rot90(a, 2)
    elif code == 4: a = np.flipud(a)
    elif code == 5: a = np.swapaxes(a, 0, 1)
    elif code == 6: a = np.rot90(a, 3)
    elif code == 7: a = np.flip(np.swapaxes(a, 0, 1), axis=(0, 1))
    elif code == 8: a = np.rot90(a, 1)
    return np.ascontiguousarray(a)


def decode_image(data: bytes) -> dict[str, Any]:
    if not data or len(data) > MAX_FILE_BYTES:
        raise AnalysisError("Choose a nonempty image file no larger than 30 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS or min(im.size) < 24:
                raise AnalysisError("Image size must be at least 24 pixels per side and at most 16 million pixels.")
            if getattr(im, "n_frames", 1) != 1:
                raise AnalysisError("Multipage TIFF is not supported. Export one plane without changing its pixel values.")
            orientation = int(im.getexif().get(274, 1))
            source_format = str(im.format)
        a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError("The file could not be decoded as a supported image.") from exc
    if a is None:
        raise AnalysisError("OpenCV could not decode this image. Try uncompressed TIFF or PNG.")
    if a.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise AnalysisError("Use an 8-bit or 16-bit unsigned image. Floating-point/signed TIFF is not converted silently.")
    a = _orientation(a, orientation)
    top = int(np.iinfo(a.dtype).max)
    if a.ndim == 2:
        rgb = np.repeat(a[..., None], 3, axis=2)
        alpha_valid = np.ones(a.shape, bool)
        color_mode = "grayscale"
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        rgb = np.ascontiguousarray(a[..., :3][..., ::-1])
        alpha_valid = a[..., 3] >= top * 0.99 if a.shape[2] == 4 else np.ones(a.shape[:2], bool)
        color_mode = "RGB"
    else:
        raise AnalysisError("Unsupported image channel layout.")
    display = np.rint(rgb.astype(np.float32) * (255.0 / top)).clip(0, 255).astype(np.uint8)
    display[~alpha_valid] = 0
    h, w = rgb.shape[:2]
    return {"rgb": rgb, "display_rgb": display, "mask": alpha_valid,
            "metadata": {"width": w, "height": h, "dtype": str(rgb.dtype),
                         "bit_depth": int(rgb.dtype.itemsize * 8), "color_mode": color_mode,
                         "source_format": source_format, "exif_orientation_applied": orientation,
                         "raw_sha256": hashlib.sha256(data).hexdigest(),
                         "display_mapping": f"linear 0..{top} to 0..255; analysis remains native"}}


def _roi_mask(frame: dict, cfg: Settings) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame["rgb"].shape[:2]
    roi = cfg.roi or (0, 0, w, h)
    x0, y0, x1, y1 = map(int, roi)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise AnalysisError("ROI must lie inside the image, with left < right and top < bottom.")
    mask = frame["mask"].copy()
    mask[:y0] = False; mask[y1:] = False
    mask[:, :x0] = False; mask[:, x1:] = False
    return mask, (x0, y0, x1, y1)


def _candidates(plane: np.ndarray, valid: np.ndarray, cfg: Settings) -> tuple[np.ndarray, dict]:
    # Filtering is used only to locate geometry. Photometry uses the original plane.
    smooth = gaussian_filter(plane.astype(np.float32), 0.45)
    values = smooth[valid]
    if len(values) < 100 or float(np.ptp(values)) <= 0:
        raise AnalysisError("No spatial signal is available for automatic grid fitting. Use a registered grid template or manual corners.")
    baseline = float(np.percentile(values, 20))
    high = float(np.percentile(values, 99.5))
    floor = float(cfg.candidate_floor) if cfg.candidate_floor is not None else baseline + max((high-baseline)*0.07, 0.1)
    neighborhood = max(3, int(cfg.pitch_hint * 0.45) | 1) if cfg.pitch_hint else 3
    peak = (smooth == maximum_filter(smooth, size=neighborhood)) & (smooth > floor) & valid
    labs, n = label(peak)
    if n < 20:
        raise AnalysisError("Too few geometry candidates. A dim/negative sample needs a registered grid template; do not lower thresholds just to force a result.")
    if n > 80_000:
        raise AnalysisError("Too many candidate peaks. Crop to the array or specify a pitch hint in Advanced settings.")
    yx = np.asarray(center_of_mass(smooth, labs, np.arange(1, n+1)))
    pts = yx[:, ::-1]
    # Refine to a weighted 3x3 centroid without resampling the original image.
    refined = []
    h, w = plane.shape
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        x0,x1=max(0,ix-1),min(w,ix+2); y0,y1=max(0,iy-1),min(h,iy+2)
        patch=plane[y0:y1,x0:x1].astype(np.float64)
        weights=np.maximum(patch-patch.min(),0)
        if weights.sum()>0:
            yy,xx=np.mgrid[y0:y1,x0:x1]
            x=float((xx*weights).sum()/weights.sum()); y=float((yy*weights).sum()/weights.sum())
        refined.append((x,y))
    pts=np.asarray(refined)
    tree=cKDTree(pts); distances,_=tree.query(pts,k=2)
    pitch=float(np.median(distances[:,1]))
    if cfg.pitch_hint is not None:
        pitch=float(cfg.pitch_hint)
    if pitch < 3:
        raise AnalysisError("Estimated pitch is below 3 native pixels. Use a higher-resolution original or a checked pitch hint.")
    # Suppress duplicate maxima only after measuring a spacing estimate.
    intens=smooth[np.clip(np.rint(pts[:,1]).astype(int),0,h-1),np.clip(np.rint(pts[:,0]).astype(int),0,w-1)]
    suppressed=np.zeros(len(pts),bool); kept=[]
    for i in np.argsort(-intens, kind="stable"):
        if suppressed[i]: continue
        kept.append(int(i)); suppressed[tree.query_ball_point(pts[i],pitch*0.38)] = True
    pts=pts[np.sort(kept)]
    if len(pts)<20:
        raise AnalysisError("Too few independent candidates remain. Check the pitch hint and ROI.")
    return pts, {"candidate_floor_native": floor, "candidate_count": int(len(pts)), "initial_pitch_px": pitch}


def _design(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(c),c,r,c*c,c*r,r*r))


def automatic_grid(plane: np.ndarray, valid: np.ndarray, cfg: Settings) -> dict:
    pts, details = _candidates(plane,valid,cfg)
    tree=cKDTree(pts); distances,inds=tree.query(pts,k=min(9,len(pts)))
    rough=details["initial_pitch_px"]
    vectors=(pts[inds[:,1:]]-pts[:,None,:]).reshape(-1,2)
    lengths=np.linalg.norm(vectors,axis=1)
    vectors=vectors[(lengths>0.65*rough)&(lengths<1.35*rough)]
    if len(vectors)<16:
        raise AnalysisError("A regular lattice could not be found. Check the crop/pitch or use Manual geometry.")
    angles=np.arctan2(vectors[:,1],vectors[:,0])
    theta=float(np.angle(np.mean(np.exp(4j*angles)))/4)
    unitx=np.array([np.cos(theta),np.sin(theta)]); unity=np.array([-np.sin(theta),np.cos(theta)])
    along=vectors@unitx; across=vectors@unity
    right=vectors[(along>0)&(np.abs(across)<along*0.55)]
    down=vectors[(across>0)&(np.abs(along)<across*0.55)]
    if len(right)<8 or len(down)<8:
        raise AnalysisError("Both grid directions must be visible. Rotate/crop the original or use a template.")
    vx=np.median(right,axis=0); vy=np.median(down,axis=0)
    if abs(np.linalg.det(np.array([vx,vy]))) < rough*rough*0.5:
        raise AnalysisError("Unstable grid axes. Use manual corners and known row/column counts.")
    # Traverse *observed* neighbors only; never grow a virtual lattice indefinitely.
    steps=[(1,0,vx),(-1,0,-vx),(0,1,vy),(0,-1,-vy)]
    remaining=set(range(len(pts))); best={}
    order=np.argsort(np.linalg.norm(pts-np.median(pts,axis=0),axis=1))
    for seed in order:
        seed=int(seed)
        if seed not in remaining: continue
        remaining.remove(seed); queue=deque([seed]); component={seed:(0,0)}; occupied={(0,0):seed}
        while queue:
            i=queue.popleft(); c,r=component[i]
            for dc,dr,step in steps:
                dist,j=tree.query(pts[i]+step); j=int(j); ij=(c+dc,r+dr)
                if dist>rough*0.35 or j==i or j not in remaining or ij in occupied: continue
                component[j]=ij; occupied[ij]=j; remaining.remove(j); queue.append(j)
        if len(component)>len(best): best=component
        if len(best)>len(pts)*0.95: break
    support=len(best)/len(pts)
    if len(best)<25 or support<0.30:
        raise AnalysisError("Automatic grid is unreliable for this sparse/irregular image. Load a registered geometry template or enter the known grid corners.")
    cr=np.asarray(list(best.values()),float); observed=pts[list(best)]
    cr-=cr.min(axis=0); c,r=cr.T
    cols=int(c.max())+1; rows=int(r.max())+1
    if min(rows,cols)<4 or rows*cols>MAX_WELLS:
        raise AnalysisError("Grid dimensions are implausible or exceed 40,000 positions.")
    design=_design(c,r); coeff=np.linalg.lstsq(design,observed,rcond=None)[0]
    for _ in range(6):
        residual=np.linalg.norm(design@coeff-observed,axis=1)
        good=residual<rough*0.30
        if good.sum()<max(20,int(0.50*len(good))):
            raise AnalysisError("Grid fit has large residuals. Use manual geometry or a higher-resolution image.")
        coeff=np.linalg.lstsq(design[good],observed[good],rcond=None)[0]
    cc,rr=np.meshgrid(np.arange(cols),np.arange(rows)); cc=cc.ravel();rr=rr.ravel()
    predicted=_design(cc,rr)@coeff
    # Smooth residual field corrects small local curvature for every position,
    # including dim positions; not intensity-dependent snapping during classification.
    rg=np.zeros((rows,cols,2),float); weights=np.zeros((rows,cols),float)
    delta=observed-design@coeff
    for (ci,ri), dv,ok in zip(cr.astype(int),delta,good):
        if ok: rg[ri,ci]=dv;weights[ri,ci]=1
    den=gaussian_filter(weights,2.0,mode="nearest")
    field=np.zeros_like(rg)
    for k in range(2):
        field[...,k]=gaussian_filter(rg[...,k],2.0,mode="nearest")/np.maximum(den,1e-8)
    predicted+=field.reshape(-1,2)
    ix=(cr[:,1].astype(int)*cols+cr[:,0].astype(int))
    residual=np.linalg.norm(predicted[ix]-observed,axis=1)
    dups,_=cKDTree(predicted).query(predicted,k=2)
    if np.min(dups[:,1])<rough*0.45:
        raise AnalysisError("Grid points overlap. Automatic fit rejected; use manual geometry.")
    arr=predicted.reshape(rows,cols,2)
    dx=np.linalg.norm(np.diff(arr,axis=1),axis=2)
    dy=np.linalg.norm(np.diff(arr,axis=0),axis=2)
    p_x=float(np.median(dx));p_y=float(np.median(dy)); pitch=min(p_x,p_y)
    density=len(best)/(rows*cols)
    details.update({"geometry_source":"auto_quadratic_smoothed_lattice", "rows":rows,"columns":cols,
                    "pitch_x_px":p_x,"pitch_y_px":p_y,"pitch_px":pitch,
                    "angle_deg":math.degrees(theta),"component_support_fraction":support,
                    "observed_geometry_fraction":density,
                    "fit_residual_median_px":float(np.median(residual)),
                    "fit_residual_p95_px":float(np.percentile(residual,95)),
                    "pitch_cv_pct":float(np.std(np.r_[dx.ravel(),dy.ravel()])/np.mean(np.r_[dx.ravel(),dy.ravel()])*100)})
    if density<0.25 or details["fit_residual_p95_px"]>rough*0.32:
        raise AnalysisError("Grid geometry failed the internal sanity check; use a registered template.")
    return {"centers":predicted,"c":cc+1,"r":rr+1,"details":details}


def manual_grid(cfg: Settings, frame: dict) -> dict:
    rows=int(cfg.manual_rows); cols=int(cfg.manual_cols)
    corners=np.asarray(cfg.manual_corners,dtype=float)
    if min(rows,cols)<2 or rows*cols>MAX_WELLS or corners.shape!=(4,2) or not np.all(np.isfinite(corners)):
        raise AnalysisError("Manual grid requires 2+ rows/columns (maximum 40,000 wells) and four finite corner centers.")
    h,w=frame["rgb"].shape[:2]
    if (corners[:,0]<0).any() or (corners[:,0]>w-1).any() or (corners[:,1]<0).any() or (corners[:,1]>h-1).any():
        raise AnalysisError("All four corner centers must be inside the image.")
    if not cv2.isContourConvex(corners.astype(np.float32)) or cv2.contourArea(corners.astype(np.float32))<16:
        raise AnalysisError("Corners must form a convex TL-TR-BR-BL quadrilateral, without crossing.")
    src=np.array([[0,0],[cols-1,0],[cols-1,rows-1],[0,rows-1]],np.float32)
    matrix=cv2.getPerspectiveTransform(src,corners.astype(np.float32))
    cc,rr=np.meshgrid(np.arange(cols),np.arange(rows))
    centers=cv2.perspectiveTransform(np.c_[cc.ravel(),rr.ravel()].astype(np.float32)[None],matrix)[0].astype(float)
    arr=centers.reshape(rows,cols,2)
    dx=np.linalg.norm(np.diff(arr,axis=1),axis=2);dy=np.linalg.norm(np.diff(arr,axis=0),axis=2)
    px=float(np.median(dx));py=float(np.median(dy));p=min(px,py)
    if p<3 or np.min(np.r_[dx.ravel(),dy.ravel()])<2:
        raise AnalysisError("Manual grid pitch is too small for this image.")
    return {"centers":centers,"c":cc.ravel()+1,"r":rr.ravel()+1,
            "details":{"geometry_source":"manual_projective","rows":rows,"columns":cols,
                       "pitch_px":p,"pitch_x_px":px,"pitch_y_px":py,
                       "fit_residual_median_px":None,"fit_residual_p95_px":None,
                       "pitch_cv_pct":float(np.std(np.r_[dx.ravel(),dy.ravel()])/np.mean(np.r_[dx.ravel(),dy.ravel()])*100)}}


def load_template(data: bytes, frame: dict) -> dict:
    try:
        if len(data)>8_000_000: raise ValueError()
        obj=json.loads(data)
        if obj.get("schema")!="wellscope.geometry.v1": raise ValueError()
        meta=frame["metadata"]
        if obj["image_size"] != [meta["width"],meta["height"]]:
            raise AnalysisError("Grid template size does not match this image. It is never silently resized.")
        positions=np.asarray(obj["positions"],float)
        if positions.ndim!=2 or positions.shape[1]!=4 or not 4<=len(positions)<=MAX_WELLS or not np.all(np.isfinite(positions)):
            raise ValueError()
        cr=positions[:,:2]
        if np.any(cr<1) or not np.all(cr==np.rint(cr)) or len(np.unique(cr,axis=0))!=len(cr): raise ValueError()
        centers=positions[:,2:]
        d,_=cKDTree(centers).query(centers,k=2)
        p=float(obj["pitch_px"])
        if not np.isfinite(p) or p<3 or np.min(d[:,1])<0.45*p: raise ValueError()
        detail={"geometry_source":"registered_template","template_id":hashlib.sha256(data).hexdigest()[:16],
                "pitch_px":p,"pitch_x_px":p,"pitch_y_px":p,
                "rows":int(cr[:,1].max()),"columns":int(cr[:,0].max()),
                "fit_residual_median_px":None,"fit_residual_p95_px":None,"pitch_cv_pct":None}
        return {"centers":centers,"c":cr[:,0].astype(int),"r":cr[:,1].astype(int),"details":detail}
    except AnalysisError: raise
    except Exception as exc:
        raise AnalysisError("Invalid grid-template JSON. Use a template exported by this app.") from exc


def parse_exclusions(text: str) -> set[str]:
    if not text.strip():return set()
    tokens=[s.strip().upper() for s in re.split(r"[,;\s]+",text.strip()) if s.strip()]
    if any(re.fullmatch(r"R\d{3}C\d{3}",s) is None for s in tokens):
        raise AnalysisError("Excluded well IDs must look like R001C001, separated by commas/spaces.")
    return set(tokens)


def measure(plane: np.ndarray, valid_mask: np.ndarray, geometry: dict, cfg: Settings, top: int) -> pd.DataFrame:
    pitch=float(geometry["details"]["pitch_px"])
    ri=pitch*cfg.inner_ratio;rb1=pitch*cfg.bg_inner_ratio;rb2=pitch*cfg.bg_outer_ratio
    h,w=plane.shape;excluded=parse_exclusions(cfg.excluded_ids)
    if excluded and not cfg.exclusion_reason.strip():
        raise AnalysisError("Provide a recorded reason for manual well exclusion.")
    rows=[]
    for c,r,(x,y) in zip(geometry["c"],geometry["r"],geometry["centers"]):
        wid=f"R{int(r):03d}C{int(c):03d}"
        entry={"well_id":wid,"row":int(r),"column":int(c),"x_px":float(x),"y_px":float(y),
               "valid":False,"exclusion_reason":"", "signal_mean_native":np.nan,
               "background_median_native":np.nan,"corrected_signal_native":np.nan,
               "inner_pixel_count":0,"saturated_inner_pixel_fraction":np.nan}
        if x-rb2<0 or y-rb2<0 or x+rb2>w-1 or y+rb2>h-1:
            entry["exclusion_reason"]="incomplete_measurement_footprint";rows.append(entry);continue
        x0=int(np.floor(x-rb2));x1=int(np.ceil(x+rb2))+1;y0=int(np.floor(y-rb2));y1=int(np.ceil(y+rb2))+1
        yy,xx=np.mgrid[y0:y1,x0:x1];distance=(xx-x)**2+(yy-y)**2
        inner=distance<=ri*ri;bg=(distance>=rb1*rb1)&(distance<=rb2*rb2)
        mask=valid_mask[y0:y1,x0:x1]
        if not np.all(mask[inner|bg]):
            entry["exclusion_reason"]="ROI_or_transparency";rows.append(entry);continue
        ni=int(inner.sum());nb=int(bg.sum());entry["inner_pixel_count"]=ni
        if ni<2 or nb<4:
            entry["exclusion_reason"]="insufficient_native_pixels";rows.append(entry);continue
        patch=plane[y0:y1,x0:x1]
        signal=float(np.mean(patch[inner]));background=float(np.median(patch[bg]))
        entry.update(signal_mean_native=signal,background_median_native=background,
                     corrected_signal_native=signal-background,
                     saturated_inner_pixel_fraction=float(np.mean(patch[inner]>=top)))
        if wid in excluded:entry["exclusion_reason"]="manual: "+cfg.exclusion_reason.strip()
        else:entry["valid"]=True
        rows.append(entry)
    frame=pd.DataFrame(rows)
    unknown=excluded-set(frame.well_id)
    if unknown:raise AnalysisError("Unknown exclusion IDs: "+", ".join(sorted(unknown)[:6]))
    if int(frame.valid.sum())<4:
        raise AnalysisError("Fewer than four complete measurable wells remain. Check the ROI/corners/image resolution.")
    return frame


def otsu_signal(values: np.ndarray) -> tuple[float,float]:
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    lo=float(a.min());hi=float(a.max())
    if hi-lo<1e-8:return hi+max(1e-6,abs(hi)*1e-6),0.0
    hist,edges=np.histogram(a,bins=256,range=(lo,hi));p=hist/hist.sum();centers=(edges[:-1]+edges[1:])/2
    omega=np.cumsum(p);mu=np.cumsum(p*centers);total=mu[-1]
    between=(total*omega-mu)**2/np.maximum(omega*(1-omega),1e-20)
    between[(omega<=0)|(omega>=1)]=0
    index=int(np.argmax(between));variance=float(np.var(a))
    return float(edges[index+1]),float(between[index]/max(variance,1e-20))


def profile_for(frame: dict, cfg: Settings, threshold: float) -> dict:
    meta=frame["metadata"]
    obj={"schema":"wellscope.profile.v1","software_version":VERSION,"engine_sha256":ENGINE_SHA256,
         "acquisition_id":cfg.acquisition_id.strip(),"channel":cfg.channel,
         "image_size":[meta["width"],meta["height"]],"bit_depth":meta["bit_depth"],
         "color_mode":meta["color_mode"],"signal_method":"native_disc_mean_minus_annulus_median",
         "inner_ratio":cfg.inner_ratio,"bg_inner_ratio":cfg.bg_inner_ratio,"bg_outer_ratio":cfg.bg_outer_ratio,
         "threshold_native":float(threshold),"threshold_operator":">="}
    obj["profile_id"]=object_hash(obj)[:16]
    return obj


def read_profile(data: bytes, frame: dict) -> dict:
    try:
        if len(data)>100_000:raise ValueError()
        p=json.loads(data)
        if p.get("schema")!="wellscope.profile.v1" or p.get("software_version")!=VERSION or p.get("engine_sha256")!=ENGINE_SHA256:raise ValueError()
        pid=p.pop("profile_id")
        if object_hash(p)[:16]!=pid:raise AnalysisError("The analysis profile has been edited or corrupted.")
        p["profile_id"]=pid;m=frame["metadata"]
        if p["image_size"]!=[m["width"],m["height"]] or p["bit_depth"]!=m["bit_depth"] or p["color_mode"]!=m["color_mode"]:
            raise AnalysisError("The saved profile does not match image size, bit depth or color mode.")
        return p
    except AnalysisError:raise
    except Exception as exc:raise AnalysisError("Invalid analysis-profile JSON.") from exc


def analyze(data: bytes, cfg: Settings, template_bytes: bytes | None = None, source_kind: str="uploaded") -> dict:
    cfg.validate();frame=decode_image(data);mask,roi=_roi_mask(frame,cfg)
    channel_index={"R":0,"G":1,"B":2}[cfg.channel]
    plane=frame["rgb"][...,channel_index].astype(np.float32)
    if cfg.geometry_mode=="template":
        if template_bytes is None:raise AnalysisError("Upload a registered grid template.")
        grid=load_template(template_bytes,frame)
    elif cfg.geometry_mode=="manual":grid=manual_grid(cfg,frame)
    else:grid=automatic_grid(plane,mask,cfg)
    top=int(np.iinfo(frame["rgb"].dtype).max)
    table=measure(plane,mask,grid,cfg,top)
    vals=table.loc[table.valid,"corrected_signal_native"].to_numpy()
    suggested,separation=otsu_signal(vals)
    threshold=suggested if cfg.threshold_mode=="auto" else float(cfg.fixed_threshold)
    table["threshold_native"]=threshold
    table["classification"]="Excluded"
    table.loc[table.valid,"classification"]=np.where(vals>=threshold,"Positive","Negative")
    n=int(table.valid.sum());k=int((table.classification=="Positive").sum());negative=n-k
    flags=[]
    def flag(code: str, severity: str, message: str):flags.append({"code":code,"severity":severity,"message":message})
    if cfg.threshold_mode=="auto":flag("EXPLORATORY_THRESHOLD","warning","Automatic threshold is exploratory; it is not validated by positive/negative controls.")
    if grid["details"]["pitch_px"]<8:flag("LOW_NATIVE_RESOLUTION","warning","Pitch is below 8 native pixels. This is a software caution, not a validated acceptance limit; verify using the original higher-resolution image.")
    if cfg.geometry_mode=="auto":
        flag("INFERRED_GEOMETRY","info","Grid extent is inferred from visible spots. Completely dark outer rows cannot be recovered reliably; a structural image/registered template or known geometry is needed for such samples.")
        if grid["details"].get("component_support_fraction",1)<0.9:flag("PARTIAL_GRID_SUPPORT","warning","Some observed spots are outside the main fitted lattice; inspect the full grid.")
    if cfg.geometry_mode=="template":flag("REGISTRATION_REQUIRED","warning","Template geometry is fixed in pixel coordinates. Confirm identical field of view, orientation and registration.")
    sat=float((table.loc[table.valid,"saturated_inner_pixel_fraction"]>0).mean())
    if sat>0.01:flag("SATURATED_PIXELS","warning","Some well interiors contain saturated pixels. Check exposure; saturated wells have not been silently discarded.")
    if n/len(table)<0.8:flag("MANY_EXCLUDED_WELLS","warning","More than 20% of fitted positions were excluded by footprint/ROI/manual rules. Review geometry.")
    if separation<0.5 and cfg.threshold_mode=="auto":flag("WEAK_INTENSITY_SPLIT","warning","The intensity distribution is poorly separated; do not interpret its automatic split as assay positivity.")
    if k==0 or k==n:flag("ALL_ONE_CLASS","warning","All measured wells fall in one class. Verify controls and avoid unvalidated occupancy-based quantification.")
    if source_kind=="synthetic":flag("SYNTHETIC_DEMO","warning","Synthetic interface demonstration; not experimental data.")
    profile=profile_for(frame,cfg,threshold)
    summary={"software":APP_NAME,"software_version":VERSION,"engine_sha256":ENGINE_SHA256,
             "created_utc":datetime.now(timezone.utc).isoformat(),"source_kind":source_kind,
             "image":frame["metadata"],"settings":asdict(cfg),"roi_used":list(roi),
             "geometry":grid["details"],"fitted_positions":len(table),"valid_wells":n,
             "positive_wells":k,"negative_wells":negative,"excluded_wells":int(len(table)-n),
             "positive_fraction_pct":float(100*k/n),"threshold_native":float(threshold),
             "threshold_mode":cfg.threshold_mode,"automatic_threshold_suggestion":suggested,
             "otsu_separability":separation,"wells_with_saturated_pixels_pct":sat*100,
             "profile_id":profile["profile_id"],"qc_flags":flags,
             "qc_scope":"software geometry/measurement checks only; not filling, chemistry, specificity or regulatory validation",
             "calibration":{"status":"not_loaded","estimated_content_pct":None,"decision":"Not evaluated"}}
    return {"frame":frame,"table":table,"grid":grid,"summary":summary,"profile":profile}


def export_template(result: dict) -> bytes:
    g=result["grid"];m=result["summary"]["image"]
    obj={"schema":"wellscope.geometry.v1","software_version":VERSION,
         "source_image_sha256":m["raw_sha256"],"image_size":[m["width"],m["height"]],
         "pitch_px":g["details"]["pitch_px"],
         "positions":np.column_stack((g["c"],g["r"],g["centers"])).tolist(),
         "note":"Reusable only for registered images of exactly the same geometry; manual exclusions are not included."}
    return json_bytes(obj)


def calibrate(result: dict, csv_data: bytes, quantity_basis: str, study_threshold: float=3.0,
              review_band_pp: float=0.3, confirmed: bool=False) -> dict:
    """Piecewise linear calibration with no extrapolation and no invented CI.

    review_band_pp is a user-selected operational band, NOT a confidence interval.
    Input is percentage units (0..100), not fraction (0..1).
    """
    if result["summary"]["threshold_mode"]!="fixed":
        raise AnalysisError("Content calibration requires a fixed, control-checked fluorescence threshold or saved profile.")
    if result["summary"]["source_kind"]=="synthetic":
        raise AnalysisError("GMO-content interpretation is disabled for the synthetic demonstration.")
    if not confirmed:
        raise AnalysisError("Confirm comparable acquisition, total DNA/input normalization and independent validation before enabling interpretation.")
    if quantity_basis not in BASES or not np.isfinite(study_threshold) or not 0<=study_threshold<=100:
        raise AnalysisError("Invalid quantity basis or study threshold.")
    if not np.isfinite(review_band_pp) or review_band_pp<0 or review_band_pp>100:
        raise AnalysisError("Review band must be 0..100 percentage points.")
    try:
        if len(csv_data)>2_000_000:raise ValueError()
        df=pd.read_csv(io.BytesIO(csv_data),dtype={"profile_id":str,"sample_id":str,"quantity_basis":str})
        if not set(CAL_COLUMNS).issubset(df.columns) or len(df)<3 or len(df)>10_000:raise ValueError()
        if df[CAL_COLUMNS].isna().any().any():raise ValueError()
        x=pd.to_numeric(df.positive_fraction_pct,errors="raise");y=pd.to_numeric(df.known_content_pct,errors="raise")
        if not (np.isfinite(x).all() and np.isfinite(y).all() and x.between(0,100).all() and y.between(0,100).all()):raise ValueError()
        if not (df.profile_id==result["profile"]["profile_id"]).all():
            raise AnalysisError("Calibration profile_id does not match this measurement. Reanalyze standards with the same locked profile.")
        if not (df.quantity_basis==quantity_basis).all():
            raise AnalysisError("Calibration quantity_basis does not match the selected basis.")
        df=df.assign(positive_fraction_pct=x,known_content_pct=y)
        levels=df.groupby("known_content_pct",sort=True).positive_fraction_pct.agg(["mean","std","count"]).reset_index()
        if len(levels)<3:raise AnalysisError("Use at least three distinct calibration content levels; more and independent replicates are recommended.")
        xs=levels["mean"].to_numpy();ys=levels.known_content_pct.to_numpy()
        if np.any(np.diff(xs)<=1e-8):
            raise AnalysisError("The mean positive fraction must rise strictly with content. Nonmonotonic/saturated standards cannot be inverted reliably.")
        if not (ys.min()<study_threshold<ys.max()):
            raise AnalysisError("Calibration must include levels below and above the study threshold.")
        signal=result["summary"]["positive_fraction_pct"]
        out={"status":"ready","calibration_id":hashlib.sha256(csv_data).hexdigest()[:16],
             "model":"piecewise_linear_interpolation_no_extrapolation","quantity_basis":quantity_basis,
             "study_threshold_pct":float(study_threshold),"review_band_percentage_points":float(review_band_pp),
             "review_band_is_confidence_interval":False,"uncertainty_interval":None,
             "calibration_levels":int(len(ys)),"calibration_rows":len(df),
             "positive_fraction_range_pct":[float(xs.min()),float(xs.max())],
             "known_content_range_pct":[float(ys.min()),float(ys.max())],
             "estimated_content_pct":None,"decision":"Outside calibration range",
             "reference_rows":df[CAL_COLUMNS].to_dict(orient="records"),
             "level_means":[{"positive_fraction_pct":float(a),"known_content_pct":float(b)} for a,b in zip(xs,ys)]}
        if not xs.min()<=signal<=xs.max():
            out["status"]="out_of_range";return out
        estimate=float(np.interp(signal,xs,ys));out["estimated_content_pct"]=estimate
        if abs(estimate-study_threshold)<=review_band_pp and review_band_pp>0:
            out["decision"]="Review near study threshold"
        elif estimate>=study_threshold:out["decision"]="At/above study threshold"
        else:out["decision"]="Below study threshold"
        return out
    except AnalysisError:raise
    except Exception as exc:
        raise AnalysisError("Invalid calibration CSV. Use the supplied headers; all rows need finite numeric percentages, profile ID and quantity basis.") from exc


def synthetic_demo() -> bytes:
    """Deterministic UI demonstration, not a study sample or calibration."""
    rng=np.random.default_rng(42);h=w=520
    yy,xx=np.mgrid[:h,:w];green=np.full((h,w),5.0)
    for r in range(30):
        for c in range(30):
            x=28+c*15.4+0.018*r;y=28+r*15.4-0.035*c
            amp=float(rng.uniform(150,220) if rng.random()<0.36 else rng.uniform(20,35))
            green+=amp*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.6**2))
    green+=rng.normal(0,.6,green.shape)
    rgb=np.zeros((h,w,3),np.uint8);rgb[...,1]=green.clip(0,255).astype(np.uint8)
    out=io.BytesIO();Image.fromarray(rgb).save(out,format="PNG");return out.getvalue()
