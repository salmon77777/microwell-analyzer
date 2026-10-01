"""
WellScope LAMP v1.2.1-rc1
Robust-normalized research screening.

Depends on the user's unchanged wellscope_core.py v1.2.0.
This is a review build, not an experimentally validated assay.

Run through app.py:
    from wellscope_robust_app import main
    main()
"""
from __future__ import annotations

import base64
import copy
import hashlib
import html
import io
import json
import math
import re
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial import cKDTree

import wellscope_core as core


VERSION = "1.2.1-rc1"
SCHEMA = "wellscope.screening.v2"
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "robust_screening_DEFAULT.json"
MODULE_SHA256 = hashlib.sha256(
    Path(__file__).read_text(encoding="utf-8").encode("utf-8")
).hexdigest()

MAD_SCALE = 1.4826
MAX_IMAGES = 250
MAX_TOTAL_BYTES = 240 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# These are software safeguards, not validated assay acceptance limits.
DEFAULT_CONFIG = {
    "channel": "G",
    "inner_ratio": 0.22,
    "bg_inner_ratio": 0.34,
    "bg_outer_ratio": 0.46,
    "acquisition_id": "LAMP-FITC-HR-v1",
    "allow_global_fallback": True,
    "min_measurable_wells": 30,
    "min_measurable_fraction": 0.50,
    "max_mean_inner_clipping": 0.20,
    "background_outlier_z": 6.0,
    "annulus_clipping_flag": 0.05,
}

EVIDENCE_NOTE = (
    "Research prototype. A predicted >=3% study class is not a measured "
    "GMO percentage. A below-threshold result is not proof of GMO absence. "
    "Internal cross-validation after method exploration is not external "
    "validation."
)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(
        clean_json(value), ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False
    ).encode("utf-8")


def object_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def csv_bytes(table):
    out = table.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(
            lambda v: "'" + v
            if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
            else v
        )
    return out.to_csv(
        index=False, float_format="%.10g", na_rep=""
    ).encode("utf-8-sig")


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return text[:80] or "sample"


def fmt(value, digits=3):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG):
        raise core.AnalysisError("측정 설정 형식이 이 버전과 다릅니다.")
    if not isinstance(config["allow_global_fallback"], bool):
        raise core.AnalysisError("Fallback 설정은 true/false여야 합니다.")
    if not str(config["acquisition_id"]).strip():
        raise core.AnalysisError("촬영·반응 protocol ID를 입력하세요.")

    for key in (
        "inner_ratio", "bg_inner_ratio", "bg_outer_ratio",
        "min_measurable_wells", "min_measurable_fraction",
        "max_mean_inner_clipping", "background_outlier_z",
        "annulus_clipping_flag"
    ):
        if not np.isfinite(float(config[key])):
            raise core.AnalysisError(f"유효하지 않은 설정: {key}")

    if int(config["min_measurable_wells"]) < 4:
        raise core.AnalysisError("최소 측정 well 수는 4 이상이어야 합니다.")
    for key in (
        "min_measurable_fraction",
        "max_mean_inner_clipping",
        "annulus_clipping_flag"
    ):
        if not 0 < float(config[key]) <= 1:
            raise core.AnalysisError(f"{key}는 0 초과 1 이하여야 합니다.")
    if float(config["background_outlier_z"]) <= 0:
        raise core.AnalysisError("QC outlier 기준은 양수여야 합니다.")

    make_settings(config).validate()


def make_settings(config, excluded_ids="", reason=""):
    return core.Settings(
        channel=config["channel"],
        threshold_mode="fixed",
        fixed_threshold=0.0,
        inner_ratio=float(config["inner_ratio"]),
        bg_inner_ratio=float(config["bg_inner_ratio"]),
        bg_outer_ratio=float(config["bg_outer_ratio"]),
        acquisition_id=str(config["acquisition_id"]),
        excluded_ids=excluded_ids,
        exclusion_reason=reason,
    )


def robust_values(values):
    """No arbitrary epsilon or silent replacement when MAD is zero."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise core.AnalysisError("정규화할 유효한 신호가 없습니다.")
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = MAD_SCALE * mad
    if not np.isfinite(scale) or scale <= 0:
        return median, mad, scale, None
    return median, mad, scale, (x - median) / scale


def unpack_images(uploads):
    """Read images/ZIPs in memory. Never extract archive paths to disk."""
    images = {}
    total = 0

    def add(name, data):
        nonlocal total
        if name in images:
            raise core.AnalysisError(f"중복 파일 경로: {name}")
        if not data or len(data) > core.MAX_FILE_BYTES:
            raise core.AnalysisError(f"파일 크기 제한 초과: {name}")
        total += len(data)
        if total > MAX_TOTAL_BYTES or len(images) >= MAX_IMAGES:
            raise core.AnalysisError(
                "한 번에 최대 250장, 이미지 파일 합계 240 MB까지 처리합니다."
            )
        images[name] = data

    for uploaded in uploads:
        name = uploaded.name
        data = uploaded.getvalue()
        if Path(name).suffix.lower() == ".zip":
            if len(data) > MAX_TOTAL_BYTES:
                raise core.AnalysisError("ZIP 업로드 크기 제한을 초과했습니다.")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [
                    x for x in archive.infolist()
                    if not x.is_dir()
                    and Path(x.filename).suffix.lower() in IMAGE_SUFFIXES
                    and not x.filename.startswith("__MACOSX/")
                ]
                if len(members) + len(images) > MAX_IMAGES:
                    raise core.AnalysisError("ZIP 안의 이미지가 너무 많습니다.")
                if sum(x.file_size for x in members) + total > MAX_TOTAL_BYTES:
                    raise core.AnalysisError("ZIP 내부 파일 합계가 너무 큽니다.")
                for member in members:
                    path = Path(member.filename.replace("\\", "/"))
                    if path.is_absolute() or ".." in path.parts:
                        raise core.AnalysisError("허용되지 않는 ZIP 내부 경로입니다.")
                    if member.file_size > core.MAX_FILE_BYTES:
                        raise core.AnalysisError("ZIP 내부 이미지가 너무 큽니다.")
                    if member.flag_bits & 1:
                        raise core.AnalysisError("암호화 ZIP은 지원하지 않습니다.")
                    add(
                        Path(name).stem + "/" + path.as_posix(),
                        archive.read(member)
                    )
        elif Path(name).suffix.lower() in IMAGE_SUFFIXES:
            add(name, data)

    if not images:
        raise core.AnalysisError("PNG/JPEG/단일 TIFF 이미지를 찾지 못했습니다.")
    return images


def read_exclusions(upload, images):
    """Explicit image-specific exclusions, recorded separately from the model."""
    if upload is None:
        return {}
    table = pd.read_csv(
        io.BytesIO(upload.getvalue()), dtype=str, keep_default_na=False
    )
    required = {"image_id", "well_id", "reason"}
    if not required.issubset(table.columns):
        raise core.AnalysisError(
            "제외 CSV에는 image_id,well_id,reason 열이 필요합니다."
        )
    if table[list(required)].apply(
        lambda s: s.str.strip().eq("")
    ).any().any():
        raise core.AnalysisError("제외 CSV에 빈 항목이 있습니다.")
    if table.duplicated(["image_id", "well_id"]).any():
        raise core.AnalysisError("제외 CSV의 well ID가 중복됩니다.")

    result = {}
    for image_id, part in table.groupby("image_id", sort=False):
        if image_id not in images:
            raise core.AnalysisError(f"제외 CSV의 이미지가 없습니다: {image_id}")
        ids = " ".join(part.well_id.str.strip())
        core.parse_exclusions(ids)
        reasons = {
            row.well_id.strip().upper(): row.reason.strip()
            for row in part.itertuples()
        }
        result[image_id] = {
            "ids": ids,
            "reason": "Audited exclusion CSV",
            "reasons": reasons,
        }
    return result


def global_lattice_fallback(plane, valid, config):
    """
    Conservative affine fit across disconnected observed candidates.
    Does not infer invisible rows beyond the observed coordinate extent.
    This fallback still requires operator geometry review.
    """
    settings = make_settings(config)
    points, details = core._candidates(plane, valid, settings)
    if not details.get("scale_aware_pitch_used"):
        raise core.AnalysisError(
            "Fallback에는 신뢰 가능한 전체 주기 추정이 필요합니다."
        )
    rough = float(details["initial_pitch_px"])
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=min(12, len(points)))
    vectors = (points[neighbors[:, 1:]] - points[:, None, :]).reshape(-1, 2)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[(lengths > 0.7 * rough) & (lengths < 1.3 * rough)]
    if len(vectors) < 40:
        raise core.AnalysisError("Fallback 방향 추정에 필요한 후보가 부족합니다.")

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    theta = float(np.angle(np.mean(np.exp(4j * angles))) / 4)
    ux = np.array([np.cos(theta), np.sin(theta)])
    uy = np.array([-np.sin(theta), np.cos(theta)])
    a, b = vectors @ ux, vectors @ uy
    right = vectors[(a > 0) & (np.abs(b) < 0.4 * a)]
    down = vectors[(b > 0) & (np.abs(a) < 0.4 * b)]
    if min(len(right), len(down)) < 12:
        raise core.AnalysisError("Fallback의 두 격자 방향이 불명확합니다.")

    basis = np.column_stack((np.median(right, axis=0), np.median(down, axis=0)))
    if abs(np.linalg.det(basis)) < 0.6 * rough * rough:
        raise core.AnalysisError("Fallback 격자 축이 불안정합니다.")

    uv = points @ np.linalg.inv(basis).T
    phase_vector = np.mean(np.exp(2j * np.pi * uv), axis=0)
    if np.min(np.abs(phase_vector)) < 0.20:
        raise core.AnalysisError("Fallback 격자 위상 신뢰도가 낮습니다.")
    phase = np.angle(phase_vector) / (2 * np.pi)
    indices = np.rint(uv - phase).astype(int)
    residual = np.linalg.norm(
        points - (indices + phase) @ basis.T, axis=1
    )

    selected = {}
    for i in np.argsort(residual):
        if residual[i] <= 0.28 * rough:
            selected.setdefault(tuple(indices[i]), int(i))
    if len(selected) < 40:
        raise core.AnalysisError("Fallback에서 일관된 격자점이 부족합니다.")

    chosen = np.array(list(selected.values()), dtype=int)
    cr = indices[chosen].astype(float)
    observed = points[chosen]
    design = np.column_stack((np.ones(len(cr)), cr))
    keep = np.ones(len(cr), dtype=bool)
    for _ in range(5):
        coeff = np.linalg.lstsq(design[keep], observed[keep], rcond=None)[0]
        errors = np.linalg.norm(design @ coeff - observed, axis=1)
        new_keep = errors <= 0.23 * rough
        if new_keep.sum() < max(30, int(0.5 * len(cr))):
            raise core.AnalysisError("Fallback residual 검사에 실패했습니다.")
        keep = new_keep

    coeff = np.linalg.lstsq(design[keep], observed[keep], rcond=None)[0]
    errors = np.linalg.norm(design[keep] @ coeff - observed[keep], axis=1)
    good_cr = cr[keep]
    lower = good_cr.min(axis=0).astype(int)
    upper = good_cr.max(axis=0).astype(int)
    cols, rows = (upper - lower + 1).astype(int)

    if min(rows, cols) < 5 or rows * cols > core.MAX_WELLS:
        raise core.AnalysisError("Fallback 격자 크기가 허용 범위 밖입니다.")
    density = float(keep.sum() / (rows * cols))
    if density < 0.40 or np.percentile(errors, 95) > 0.23 * rough:
        raise core.AnalysisError("Fallback 관찰 지지율 또는 residual이 부족합니다.")

    midpoint = (lower + upper) / 2
    for xside in (False, True):
        for yside in (False, True):
            quadrant = (
                ((good_cr[:, 0] >= midpoint[0]) == xside)
                & ((good_cr[:, 1] >= midpoint[1]) == yside)
            )
            if quadrant.sum() < 5:
                raise core.AnalysisError("Fallback 지지점이 한쪽에 편중돼 있습니다.")

    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.column_stack((cc.ravel(), rr.ravel())) + lower
    centers = np.column_stack((np.ones(len(coords)), coords)) @ coeff
    spacing, _ = cKDTree(centers).query(centers, k=2)
    if np.min(spacing[:, 1]) < 0.5 * rough:
        raise core.AnalysisError("Fallback 격자점이 겹칩니다.")

    grid_array = centers.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid_array, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid_array, axis=0), axis=2)
    px, py = float(np.median(dx)), float(np.median(dy))
    if not (0.75 * rough <= px <= 1.25 * rough
            and 0.75 * rough <= py <= 1.25 * rough):
        raise core.AnalysisError("Fallback의 최종 pitch가 초기 추정과 다릅니다.")

    all_spacings = np.r_[dx.ravel(), dy.ravel()]
    details.update({
        "geometry_source": "global_affine_fallback_review_required",
        "rows": int(rows),
        "columns": int(cols),
        "pitch_x_px": px,
        "pitch_y_px": py,
        "pitch_px": min(px, py),
        "pitch_cv_pct": float(100 * all_spacings.std() / all_spacings.mean()),
        "component_support_fraction": float(keep.sum() / len(points)),
        "observed_geometry_fraction": density,
        "fit_residual_median_px": float(np.median(errors)),
        "fit_residual_p95_px": float(np.percentile(errors, 95)),
        "angle_deg": float(np.degrees(theta)),
    })
    return {
        "centers": centers,
        "c": cc.ravel() + 1,
        "r": rr.ravel() + 1,
        "details": details,
    }


def add_qc_diagnostics(table, plane, geometry, config, top):
    """
    Suspect flags only. Fluorescence-based flags do NOT establish bubbles.
    No automatic exclusion by brightness, saturation or suspect flag.
    """
    table = table.copy()
    table["background_mad_native"] = np.nan
    table["annulus_saturated_fraction"] = np.nan
    table["large_saturated_object_overlap"] = 0.0
    table["artifact_suspect"] = False
    table["artifact_reason"] = ""

    pitch = float(geometry["details"]["pitch_px"])
    r0 = pitch * config["inner_ratio"]
    r1 = pitch * config["bg_inner_ratio"]
    r2 = pitch * config["bg_outer_ratio"]

    saturated = (plane >= top).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, 8)
    large_ids = []
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= pitch * pitch and max(width, height) >= 2 * pitch:
            large_ids.append(i)
    large_mask = np.isin(labels, large_ids) if large_ids else None

    for index, row in table.loc[table.valid].iterrows():
        x, y = float(row.x_px), float(row.y_px)
        x0, x1 = int(np.floor(x-r2)), int(np.ceil(x+r2))+1
        y0, y1 = int(np.floor(y-r2)), int(np.ceil(y+r2))+1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = (xx-x)**2 + (yy-y)**2
        annulus = (distance >= r1*r1) & (distance <= r2*r2)
        footprint = (distance <= r0*r0) | annulus
        values = plane[y0:y1, x0:x1][annulus].astype(float)
        med = float(np.median(values))
        table.at[index, "background_mad_native"] = float(
            np.median(np.abs(values-med))
        )
        table.at[index, "annulus_saturated_fraction"] = float(
            np.mean(values >= top)
        )
        if large_mask is not None:
            table.at[index, "large_saturated_object_overlap"] = float(
                np.mean(large_mask[y0:y1, x0:x1][footprint])
            )

    valid_index = table.index[table.valid]
    reasons = {i: [] for i in valid_index}
    for column, label_text in (
        ("background_median_native", "unusually_high_annulus"),
        ("background_mad_native", "heterogeneous_annulus"),
    ):
        values = table.loc[valid_index, column].to_numpy(float)
        med, mad, scale, z = robust_values(values)
        if z is not None:
            for i in valid_index[z >= config["background_outlier_z"]]:
                reasons[i].append(label_text)

    for i in valid_index:
        if table.at[i, "annulus_saturated_fraction"] >= config["annulus_clipping_flag"]:
            reasons[i].append("annulus_saturation")
        if table.at[i, "large_saturated_object_overlap"] >= 0.10:
            reasons[i].append("large_clipped_structure")
        table.at[i, "artifact_reason"] = ";".join(reasons[i])
        table.at[i, "artifact_suspect"] = bool(reasons[i])
    return table


def analyze_image(image_id, data, config, exclusion=None):
    validate_config(config)
    exclusion = exclusion or {"ids": "", "reason": "", "reasons": {}}
    settings = make_settings(config, exclusion["ids"], exclusion["reason"])
    frame = core.decode_image(data)
    mask, roi = core._roi_mask(frame, settings)
    plane = frame["rgb"][..., {"R": 0, "G": 1, "B": 2}[config["channel"]]].astype(np.float32)

    primary_error = None
    try:
        geometry = core.automatic_grid(plane, mask, settings)
    except core.AnalysisError as exc:
        primary_error = str(exc)
        if not config["allow_global_fallback"]:
            raise
        geometry = global_lattice_fallback(plane, mask, config)

    top = int(np.iinfo(frame["rgb"].dtype).max)
    table = core.measure(plane, mask, geometry, settings, top)

    for index, row in table.iterrows():
        if row.well_id in exclusion["reasons"]:
            table.at[index, "exclusion_reason"] = (
                "manual: " + exclusion["reasons"][row.well_id]
            )

    table = add_qc_diagnostics(table, plane, geometry, config, top)
    values = table.loc[table.valid, "corrected_signal_native"].to_numpy(float)
    median, mad, scale, z = robust_values(values)
    table["robust_z"] = np.nan
    if z is not None:
        table.loc[table.valid, "robust_z"] = z

    n = int(table.valid.sum())
    mean_clip = float(
        table.loc[table.valid, "saturated_inner_pixel_fraction"].mean()
    )
    hold = []
    if n < config["min_measurable_wells"]:
        hold.append("too_few_measurable_wells")
    if n / len(table) < config["min_measurable_fraction"]:
        hold.append("too_many_incomplete_measurements")
    if z is None:
        hold.append("MAD_zero_or_invalid")
    if mean_clip > config["max_mean_inner_clipping"]:
        hold.append("excessive_center_clipping")

    pixel_hash = hashlib.sha256(
        str(frame["rgb"].shape).encode()
        + str(frame["rgb"].dtype).encode()
        + frame["rgb"].tobytes()
        + frame["mask"].tobytes()
    ).hexdigest()

    summary = {
        "image_id": image_id,
        "status": "qc_hold" if hold else "ready",
        "hold_reasons": ";".join(hold),
        "N": n,
        "excluded_wells": int(len(table)-n),
        "fitted_positions": int(len(table)),
        "median_corrected_signal": median,
        "MAD": mad,
        "robust_scale": scale,
        "normalization_ready": z is not None,
        "mean_inner_clipping_fraction": mean_clip,
        "wells_with_center_clipping_pct": float(
            100 * (table.loc[table.valid, "saturated_inner_pixel_fraction"] > 0).mean()
        ),
        "artifact_suspect_wells": int(
            table.loc[table.valid, "artifact_suspect"].sum()
        ),
        "raw_sha256": frame["metadata"]["raw_sha256"],
        "pixel_sha256": pixel_hash,
        "width": frame["metadata"]["width"],
        "height": frame["metadata"]["height"],
        "bit_depth": frame["metadata"]["bit_depth"],
        "color_mode": frame["metadata"]["color_mode"],
        "geometry": geometry["details"],
        "primary_geometry_error": primary_error,
        "roi_used": list(roi),
        "config": copy.deepcopy(config),
        "manual_exclusion_record": exclusion,
        "software_version": VERSION,
        "core_version": core.VERSION,
        "core_sha256": core.ENGINE_SHA256,
        "robust_module_sha256": MODULE_SHA256,
        "qc_scope": (
            "Suspect flags are not confirmed bubbles or filling defects. "
            "Suspects are retained; only explicit/footprint exclusions affect N."
        ),
    }
    return {"summary": summary, "wells": table}


def score_image(record, zw):
    table = record["wells"].copy()
    summary = copy.deepcopy(record["summary"])
    table["threshold_z"] = zw
    table["classification"] = np.where(table.valid, "Not called", "Excluded")
    summary.update(K_Z=None, qZ_pct=None, threshold_z=zw,
                   effective_threshold_native=None)

    if zw is not None and summary["normalization_ready"]:
        good = table.valid & np.isfinite(table.robust_z)
        high = good & (table.robust_z >= float(zw))
        table.loc[good, "classification"] = np.where(
            high.loc[good], "High signal", "Below well cutoff"
        )
        k = int(high.sum())
        summary.update(
            K_Z=k,
            qZ_pct=100.0*k/int(good.sum()),
            effective_threshold_native=(
                summary["median_corrected_signal"]
                + float(zw)*summary["robust_scale"]
            ),
        )
    return {"summary": summary, "wells": table}


def run_images(images, config, exclusions):
    records = {}
    failures = []
    seen_pixels = {}
    bar = st.progress(0.0)

    for index, (name, data) in enumerate(images.items(), 1):
        try:
            record = analyze_image(name, data, config, exclusions.get(name))
            ph = record["summary"]["pixel_sha256"]
            if ph in seen_pixels:
                raise core.AnalysisError(
                    "동일 픽셀 이미지가 중복됩니다: " + seen_pixels[ph]
                )
            seen_pixels[ph] = name
            records[name] = record
        except Exception as exc:
            failures.append({
                "image_id": name,
                "status": "analysis_failed",
                "reason": str(exc),
            })
        bar.progress(index/len(images), text=f"{index}/{len(images)}  {name}")
    return records, pd.DataFrame(failures)


def summary_table(records, zw=None):
    rows = []
    for record in records.values():
        s = score_image(record, zw)["summary"]
        g = s["geometry"]
        rows.append({
            key: s.get(key) for key in (
                "image_id", "status", "hold_reasons", "N", "K_Z", "qZ_pct",
                "excluded_wells", "median_corrected_signal", "MAD",
                "robust_scale", "threshold_z", "effective_threshold_native",
                "artifact_suspect_wells", "mean_inner_clipping_fraction",
                "wells_with_center_clipping_pct", "raw_sha256", "pixel_sha256"
            )
        })
        rows[-1].update(
            rows=g["rows"], columns=g["columns"],
            pitch_x_px=g["pitch_x_px"], pitch_y_px=g["pitch_y_px"],
            pitch_cv_pct=g.get("pitch_cv_pct"),
            geometry_source=g["geometry_source"],
        )
    return pd.DataFrame(rows)


def validate_manifest(manifest, names, require_cv=False):
    table = manifest.copy()
    required = ["image_id", "mixture_pct", "specimen_id", "cv_group"]
    if not set(required).issubset(table.columns):
        raise core.AnalysisError("표준 메타데이터 열이 부족합니다.")
    if table.image_id.duplicated().any() or set(table.image_id) != set(names):
        raise core.AnalysisError("메타데이터의 이미지 목록이 업로드와 다릅니다.")
    table["mixture_pct"] = pd.to_numeric(table.mixture_pct, errors="coerce")
    if (
        table.mixture_pct.isna().any()
        or not np.isfinite(table.mixture_pct).all()
        or not table.mixture_pct.between(0, 100).all()
    ):
        raise core.AnalysisError("모든 이미지의 실제 혼합비를 확인하세요.")

    for key in ("specimen_id", "cv_group"):
        table[key] = table[key].fillna("").astype(str).str.strip()
    if table.specimen_id.eq("").any():
        raise core.AnalysisError("모든 이미지에 specimen_id를 입력하세요.")
    if require_cv and table.cv_group.eq("").any():
        raise core.AnalysisError("내부 검증에는 모든 cv_group이 필요합니다.")

    for specimen, part in table.groupby("specimen_id"):
        if part.mixture_pct.nunique() != 1 or part.cv_group.nunique() != 1:
            raise core.AnalysisError(
                f"{specimen}: 한 specimen 안의 혼합비/cv_group이 다릅니다."
            )
    return table


def check_compatible(record, model):
    s = record["summary"]
    c = model["contract"]
    if s["config"] != model["config"]:
        raise core.AnalysisError("측정/QC 설정이 저장 모델과 다릅니다.")
    if s["bit_depth"] != c["bit_depth"] or s["color_mode"] != c["color_mode"]:
        raise core.AnalysisError("영상 bit depth 또는 채널 형식이 다릅니다.")
    size = np.array([s["width"], s["height"]])
    sizes = np.asarray(c["image_sizes"])
    if not np.any(np.max(np.abs(sizes-size), axis=1) <= 2):
        raise core.AnalysisError("영상 크기가 모델의 촬영 조건과 다릅니다.")
    p = s["geometry"]["pitch_px"]
    if not c["pitch_range"][0] <= p <= c["pitch_range"][1]:
        raise core.AnalysisError("영상의 native pitch가 모델 범위 밖입니다.")
    if s["status"] != "ready":
        raise core.AnalysisError("QC 판정 보류: " + s["hold_reasons"])


def fit_model(records, manifest, config, quantile, expected_views, study=3.0):
    validate_config(config)
    if not 0.90 <= quantile <= 0.9999:
        raise core.AnalysisError("대조군 quantile 범위를 확인하세요.")
    if int(expected_views) < 1:
        raise core.AnalysisError("Specimen당 이미지 수는 1 이상이어야 합니다.")
    if not np.isfinite(study) or not 0 < study < 100:
        raise core.AnalysisError("연구 혼합비 경계를 확인하세요.")

    meta = validate_manifest(manifest, manifest.image_id.tolist())
    missing = set(meta.image_id) - set(records)
    if missing:
        raise core.AnalysisError(
            "분석 실패 이미지가 있어 모델을 만들지 않습니다: "
            + ", ".join(sorted(missing)[:5])
        )
    for name in meta.image_id:
        if records[name]["summary"]["status"] != "ready":
            raise core.AnalysisError(f"QC 보류 이미지를 먼저 검토하세요: {name}")

    counts = meta.groupby("specimen_id").size()
    if not (counts == int(expected_views)).all():
        raise core.AnalysisError(
            "각 specimen의 이미지 수가 사전에 정한 expected_views와 다릅니다."
        )
    levels = meta.mixture_pct.to_numpy(float)
    if (
        not np.any(levels == 0)
        or not np.any(levels < study)
        or not np.any(levels == study)
        or not np.any(levels > study)
    ):
        raise core.AnalysisError("0%, 경계 미만, 정확한 경계, 경계 초과 표준이 필요합니다.")

    # Equal weighting: image quantiles -> specimen median -> control median.
    zero_rows = []
    zero = meta[meta.mixture_pct == 0]
    for specimen, part in zero.groupby("specimen_id", sort=True):
        image_quantiles = []
        for name in part.image_id:
            tab = records[name]["wells"]
            z = tab.loc[tab.valid, "robust_z"].to_numpy(float)
            image_quantiles.append(float(np.quantile(z, quantile, method="linear")))
        zero_rows.append({
            "specimen_id": specimen,
            "image_quantiles": image_quantiles,
            "specimen_quantile": float(np.median(image_quantiles)),
        })
    zw = float(np.median([x["specimen_quantile"] for x in zero_rows]))
    if not np.isfinite(zw) or zw <= 0:
        raise core.AnalysisError("대조군으로 유효한 양의 Zw를 만들지 못했습니다.")

    unit_rows = []
    for specimen, part in meta.groupby("specimen_id", sort=True):
        scores = [
            score_image(records[name], zw)["summary"]["qZ_pct"]
            for name in part.image_id
        ]
        unit_rows.append({
            "specimen_id": specimen,
            "mixture_pct": float(part.mixture_pct.iloc[0]),
            "cv_group": part.cv_group.iloc[0],
            "image_count": len(scores),
            "qZ_pct": float(np.median(scores)),
        })
    units = pd.DataFrame(unit_rows)
    below = units.loc[units.mixture_pct < study, "qZ_pct"]
    above = units.loc[units.mixture_pct >= study, "qZ_pct"]
    max_below, min_above = float(below.max()), float(above.min())
    gap = min_above - max_below
    cutoff = (max_below+min_above)/2 if gap > 1e-9 else None

    selected = [records[name]["summary"] for name in meta.image_id]
    types = {(s["bit_depth"], s["color_mode"]) for s in selected}
    if len(types) != 1:
        raise core.AnalysisError("표준의 bit depth/색 형식을 통일하세요.")
    pitches = [s["geometry"]["pitch_px"] for s in selected]
    if max(pitches)/min(pitches) > 1.15:
        raise core.AnalysisError("촬영 scale이 다른 표준을 한 모델에 섞지 마세요.")

    model = {
        "schema": SCHEMA,
        "software_version": VERSION,
        "core_sha256": core.ENGINE_SHA256,
        "robust_module_sha256": MODULE_SHA256,
        "status": "ready" if cutoff is not None else "not_separable",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": copy.deepcopy(config),
        "normalization": {
            "method": "within_image_median_MAD",
            "MAD_scale": MAD_SCALE,
            "population": "all_measurable_nonmanually_excluded_wells",
            "zero_MAD_policy": "indeterminate",
        },
        "well_rule": {
            "method": "zero_control_quantile_equal_specimen_weight",
            "quantile": quantile,
            "quantile_method": "linear",
            "threshold_z": zw,
            "operator": ">=",
            "zero_controls": zero_rows,
        },
        "sample_rule": {
            "feature": "robust_high_signal_fraction_pct",
            "aggregation": "median_of_image_qZ",
            "expected_images_per_specimen": int(expected_views),
            "cutoff": cutoff,
            "operator": ">=",
            "maximum_below_score": max_below,
            "minimum_at_or_above_score": min_above,
            "training_gap": gap,
        },
        "study_boundary_pct": float(study),
        "contract": {
            "bit_depth": selected[0]["bit_depth"],
            "color_mode": selected[0]["color_mode"],
            "image_sizes": sorted({(s["width"], s["height"]) for s in selected}),
            "pitch_range": [0.85*min(pitches), 1.15*max(pitches)],
        },
        "training_pixel_hashes": [s["pixel_sha256"] for s in selected],
        "training_raw_hashes": [s["raw_sha256"] for s in selected],
        "training_manifest": meta.to_dict("records"),
        "training_units": units.to_dict("records"),
        "score_range": [float(units.qZ_pct.min()), float(units.qZ_pct.max())],
        "numeric_content_estimation": "not_supported_by_this_model",
        "evidence_status": "development_only",
        "note": EVIDENCE_NOTE,
    }
    model["model_id"] = object_hash(model)[:24]
    return model, units


def read_model(data):
    if len(data) > 8_000_000:
        raise core.AnalysisError("모델 JSON이 너무 큽니다.")
    try:
        model = json.loads(data)
        mid = model.pop("model_id")
        if (
            model.get("schema") != SCHEMA
            or model.get("software_version") != VERSION
            or model.get("core_sha256") != core.ENGINE_SHA256
            or model.get("robust_module_sha256") != MODULE_SHA256
        ):
            raise core.AnalysisError(
                "구버전 또는 다른 코드의 모델입니다. 이 robust 버전에서 새로 만드세요."
            )
        if object_hash(model)[:24] != mid:
            raise core.AnalysisError("모델 checksum이 다릅니다. 원본 JSON을 사용하세요.")
        model["model_id"] = mid
        validate_config(model["config"])
        if model["status"] != "ready":
            raise core.AnalysisError("분리되지 않아 잠그지 못한 모델입니다.")
        if model["normalization"]["MAD_scale"] != MAD_SCALE:
            raise core.AnalysisError("MAD scale이 다릅니다.")
        zw = float(model["well_rule"]["threshold_z"])
        ts = float(model["sample_rule"]["cutoff"])
        if not np.isfinite([zw, ts]).all() or zw <= 0 or ts <= 0:
            raise core.AnalysisError("모델 기준값이 유효하지 않습니다.")
        return model
    except core.AnalysisError:
        raise
    except Exception as exc:
        raise core.AnalysisError("유효한 robust screening JSON이 아닙니다.") from exc


def predict_specimen(records, names, model):
    if model["status"] != "ready":
        raise core.AnalysisError("MODEL NOT SEPARABLE")
    expected = model["sample_rule"]["expected_images_per_specimen"]
    if len(names) != expected:
        raise core.AnalysisError(
            f"이 모델은 specimen당 {expected}장의 이미지를 요구합니다. "
            "서로 다른 독립 시료를 한 specimen으로 합치지 마세요."
        )

    zw = model["well_rule"]["threshold_z"]
    ts = model["sample_rule"]["cutoff"]
    scores = []
    hashes = []
    for name in names:
        if name not in records:
            raise core.AnalysisError(f"분석 실패 이미지가 포함됩니다: {name}")
        record = records[name]
        check_compatible(record, model)
        scores.append(score_image(record, zw)["summary"]["qZ_pct"])
        hashes.append(record["summary"]["pixel_sha256"])
    if len(set(hashes)) != len(hashes):
        raise core.AnalysisError("동일 이미지 중복입니다.")

    score = float(np.median(scores))
    return {
        "model_id": model["model_id"],
        "qZ_pct": score,
        "threshold_z": float(zw),
        "sample_cutoff": float(ts),
        "decision_ratio": score/ts,
        "decision": "At/above study threshold" if score >= ts else "Below study threshold",
        "predicted_positive": bool(score >= ts),
        "study_boundary_pct": model["study_boundary_pct"],
        "image_count": len(names),
        "training_image_match": any(
            h in model["training_pixel_hashes"] for h in hashes
        ),
        "outside_training_score_range": not (
            model["score_range"][0] <= score <= model["score_range"][1]
        ),
        "estimated_GMO_content_pct": None,
        "note": EVIDENCE_NOTE,
    }


def internal_cv(records, manifest, config, quantile, views, study):
    meta = validate_manifest(manifest, manifest.image_id.tolist(), require_cv=True)
    groups = sorted(meta.cv_group.unique())
    if len(groups) < 3:
        raise core.AnalysisError("내부 검증에는 독립성을 반영한 cv_group 3개 이상이 필요합니다.")

    rows, folds = [], []
    for heldout in groups:
        train = meta[meta.cv_group != heldout].copy()
        test = meta[meta.cv_group == heldout].copy()
        model = None
        error = None
        try:
            model, _ = fit_model(records, train, config, quantile, views, study)
            if model["status"] != "ready":
                raise core.AnalysisError("Training classes overlap")
        except Exception as exc:
            error = str(exc)
        folds.append({
            "heldout_group": heldout,
            "status": "ready" if error is None else "not_fitted",
            "Zw": model["well_rule"]["threshold_z"] if model else None,
            "Ts": model["sample_rule"]["cutoff"] if model else None,
            "reason": error,
        })

        for specimen, part in test.groupby("specimen_id", sort=True):
            row = {
                "specimen_id": specimen,
                "mixture_pct": float(part.mixture_pct.iloc[0]),
                "heldout_group": heldout,
                "actual_positive": bool(part.mixture_pct.iloc[0] >= study),
                "predicted_positive": None,
                "decision": "Indeterminate",
                "qZ_pct": None,
                "Zw": None,
                "Ts": None,
                "decision_ratio": None,
                "reason": error,
            }
            if error is None:
                try:
                    p = predict_specimen(records, part.image_id.tolist(), model)
                    row.update(
                        predicted_positive=p["predicted_positive"],
                        decision=p["decision"],
                        qZ_pct=p["qZ_pct"],
                        Zw=p["threshold_z"],
                        Ts=p["sample_cutoff"],
                        decision_ratio=p["decision_ratio"],
                        reason=None,
                    )
                except Exception as exc:
                    row["reason"] = str(exc)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(folds)


def classification_summary(table):
    counts = {"TN": 0, "FP": 0, "FN": 0, "TP": 0, "unclassified": 0}
    for row in table.itertuples():
        pred = row.predicted_positive
        if pd.isna(pred):
            counts["unclassified"] += 1
        elif bool(row.actual_positive):
            counts["TP" if bool(pred) else "FN"] += 1
        else:
            counts["FP" if bool(pred) else "TN"] += 1

    called = counts["TP"]+counts["TN"]+counts["FP"]+counts["FN"]
    total = len(table)
    ratio = lambda a, b: None if b == 0 else a/b
    counts.update(
        attempted_specimens=total,
        called_specimens=called,
        call_rate=ratio(called, total),
        accuracy_among_called=ratio(counts["TP"]+counts["TN"], called),
        correct_fraction_of_all_attempted=ratio(counts["TP"]+counts["TN"], total),
        sensitivity_among_called=ratio(counts["TP"], counts["TP"]+counts["FN"]),
        specificity_among_called=ratio(counts["TN"], counts["TN"]+counts["FP"]),
        interpretation="Descriptive internal CV; no independence-based confidence interval fabricated",
    )
    return counts


def png_bytes(array):
    out = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(out, format="PNG")
    return out.getvalue()


def display_images(data, record, zw=None):
    frame = core.decode_image(data)
    source = frame["display_rgb"]
    h, w = source.shape[:2]
    scale = min(1.0, 1100/max(h, w))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    raw = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    grid, calls, qc = raw.copy(), raw.copy(), raw.copy()
    scored = score_image(record, zw)
    radius = max(2, round(record["summary"]["geometry"]["pitch_px"]*scale*0.28))

    for row in scored["wells"].itertuples():
        x, y = round(row.x_px*scale), round(row.y_px*scale)
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            continue
        if not row.valid:
            for arr in (grid, calls, qc):
                cv2.rectangle(arr, (x-2, y-2), (x+2, y+2), (155, 155, 155), 1)
            continue
        cv2.drawMarker(grid, (x, y), (0, 218, 231), cv2.MARKER_CROSS, radius*2, 1)
        if row.classification == "High signal":
            cv2.circle(calls, (x, y), radius, (255, 211, 61), 1)
        elif row.classification == "Below well cutoff":
            cv2.drawMarker(calls, (x, y), (255, 87, 87),
                           cv2.MARKER_TILTED_CROSS, radius*2, 1)
        if row.artifact_suspect:
            cv2.rectangle(qc, (x-radius, y-radius), (x+radius, y+radius),
                          (220, 90, 220), 1)
    return {"raw": raw, "grid": grid, "calls": calls, "QC_suspects": qc}


def signal_figure(record, zw=None, standardized=True):
    table = record["wells"]
    col = "robust_z" if standardized else "corrected_signal_native"
    x = table.loc[table.valid, col].to_numpy(float)
    x = x[np.isfinite(x)]
    fig = Figure(figsize=(6, 3.4), layout="constrained")
    ax = fig.subplots()
    if len(x):
        ax.hist(x, bins=60, weights=np.full(len(x), 100/len(x)))
    if standardized and zw is not None:
        ax.axvline(zw, linestyle="--", linewidth=1, label=f"Locked Zw = {zw:.4g}")
        ax.legend(frameon=False)
    ax.set_xlabel("Robust within-image Z" if standardized else "Corrected native signal")
    ax.set_ylabel("Wells (%)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def figure_bytes(fig, kind="png"):
    out = io.BytesIO()
    fig.savefig(out, format=kind, dpi=240)
    return out.getvalue()


def report_html(record, data, zw, decision=None, reviewed=False):
    s = score_image(record, zw)["summary"]
    images = display_images(data, record, zw)
    escape = html.escape

    def uri(b):
        return "data:image/png;base64," + base64.b64encode(b).decode()

    cards = "".join(
        f'<section><h3>{escape(label)}</h3>'
        f'<img src="{uri(png_bytes(arr))}"></section>'
        for label, arr in images.items()
    )
    if decision is None:
        label = "Screening not applied: model or review pending"
    elif decision["predicted_positive"]:
        label = f"GMO screening positive | >={decision['study_boundary_pct']:g}% study class"
    else:
        label = f"Below {decision['study_boundary_pct']:g}% study threshold"
    evidence = ""
    if decision:
        evidence = (
            "TRAINING-IMAGE REANALYSIS — NOT INDEPENDENT VALIDATION"
            if decision["training_image_match"]
            else "NEW SPECIMEN — PERFORMANCE NOT ESTABLISHED BY THIS OUTPUT"
        )

    details = {
        "image_summary": s,
        "specimen_decision": decision,
        "operator_reviewed": reviewed,
    }
    chart = figure_bytes(signal_figure(record, zw, True))
    return (
        '<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;'
        'padding:20px;color:#193348;background:white}'
        '.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}'
        'img{width:100%;height:auto}section{border:1px solid #dde3e7;padding:12px}'
        '.decision{padding:18px;border:2px solid #426979;font-size:22px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}'
        '@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>'
        f'<h1>WellScope LAMP {escape(VERSION)}</h1>'
        f'<h2>{escape(s["image_id"])}</h2>'
        f'<div class="decision">{escape(label)}</div>'
        f'<p><b>{escape(evidence)}</b></p>'
        f'<p>N={s["N"]:,}; KZ={fmt(s["K_Z"],0)}; '
        f'image qZ={fmt(s["qZ_pct"])}%; Zw={fmt(zw)}.</p>'
        '<p>For multiple views, the specimen decision uses the median of image qZ.</p>'
        '<p>Exact GMO content: not quantified. '
        'QC suspect markings are not confirmed bubbles and were not automatically excluded.</p>'
        f'<div class="grid">{cards}</div>'
        f'<h3>Within-image signal distribution</h3><img src="{uri(chart)}">'
        '<p>Cyan: grid; yellow: high signal; red: below Zw; grey: excluded; '
        'magenta: QC suspect. Measurements use original pixels.</p>'
        f'<details><summary>Analysis record</summary><pre>'
        f'{escape(json_bytes(details).decode())}</pre></details>'
        f'<p>{escape(EVIDENCE_NOTE)}</p>'
    ).encode("utf-8")


def result_zip(records, failures, model=None, manifest=None, cv=None, folds=None):
    out = io.BytesIO()
    zw = model["well_rule"]["threshold_z"] if model else None
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("image_summary.csv", csv_bytes(summary_table(records, zw)))
        archive.writestr("failed_images.csv", csv_bytes(failures))
        if manifest is not None:
            archive.writestr("image_manifest.csv", csv_bytes(manifest))
        if model is not None:
            archive.writestr(
                "screening_model.json" if model["status"] == "ready"
                else "NOT_LOCKED_model_diagnostics.json",
                json_bytes(model)
            )
        for i, (name, record) in enumerate(records.items(), 1):
            scored = score_image(record, zw)
            prefix = f"images/{i:03d}_{safe_name(name)}"
            archive.writestr(prefix+"_wells.csv", csv_bytes(scored["wells"]))
            archive.writestr(prefix+"_summary.json", json_bytes(scored["summary"]))
        if cv is not None:
            archive.writestr("internal_cv_predictions.csv", csv_bytes(cv))
            archive.writestr("internal_cv_folds.csv", csv_bytes(folds))
            archive.writestr(
                "internal_cv_descriptive_metrics.json",
                json_bytes(classification_summary(cv))
            )
        archive.writestr(
            "READ_ME.txt",
            "PRIVATE RESEARCH DATA. Do not upload this result ZIP to a public repository.\n"
            + EVIDENCE_NOTE
            + "\nGeometry failures and indeterminate predictions must not be silently removed.\n"
            "Per-well CSV rows are not independent experimental replicates.\n"
        )
    return out.getvalue()


def code_zip():
    """Package only explicitly listed source files, never study images/models."""
    out = io.BytesIO()
    files = [
        "app.py", "wellscope_robust_app.py", "wellscope_core.py",
        "wellscope_batch.py", "wellscope_report.py",
        "wellscope_standards_ui.py", "wellscope_figure6.py",
        "requirements.txt", ".streamlit/config.toml",
    ]
    hashes = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = ROOT/name
            if not path.is_file():
                continue
            data = path.read_bytes()
            if path.suffix == ".py":
                compile(data.decode("utf-8-sig"), name, "exec")
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("SOURCE_SHA256.json", json_bytes(hashes))
        archive.writestr(
            "BUILD_NOTE.txt",
            "Syntax compilation only at ZIP generation. "
            "This does not establish UI correctness, grid accuracy or assay performance.\n"
            "Application: 1.2.1-rc1; original native-pixel engine retained.\n"
        )
    return out.getvalue()


def arithmetic_self_check():
    x = np.r_[np.linspace(10, 30, 100), 75.0, 95.0]
    original = x.copy()
    m, mad, scale, z = robust_values(x)
    _, _, _, z2 = robust_values(3*x+17)
    _, _, _, flat = robust_values(np.ones(50))
    checks = {
        "positive_affine_invariance": bool(np.allclose(z, z2, atol=1e-10)),
        "zero_MAD_is_indeterminate": flat is None,
        "input_array_unchanged": bool(np.array_equal(x, original)),
        "MAD_scale_formula": bool(np.isclose(scale, MAD_SCALE*mad)),
        "well_fraction_bounds": bool(0 <= 100*np.mean(z >= 2) <= 100),
    }
    return checks


def config_ui(model=None):
    if model is not None:
        st.sidebar.info("측정·QC 설정은 불러온 모델로 고정됩니다.")
        return copy.deepcopy(model["config"])

    config = copy.deepcopy(DEFAULT_CONFIG)
    with st.sidebar.expander("측정 조건 · 개발용 설정", expanded=False):
        config["acquisition_id"] = st.text_input(
            "촬영·반응 protocol ID", value=config["acquisition_id"],
            key="r121_acquisition"
        )
        config["channel"] = st.selectbox(
            "측정 채널", ["G", "R", "B"], key="r121_channel"
        )
        config["allow_global_fallback"] = st.checkbox(
            "격자 연결 실패 시 보수적 fallback 시도",
            value=True, key="r121_fallback"
        )
        st.caption(
            "반지름 비율은 0.22 / 0.34 / 0.46 pitch입니다. "
            "QC 의심 표시는 자동 제외가 아닙니다. "
            "평균 중심 pixel 포화율 >20%는 이 검토본의 판정 보류 기준이며, "
            "실험적으로 검증된 허용치가 아닙니다."
        )
    return config


def model_ui():
    choice = st.sidebar.radio(
        "모델", ["세션/기본 모델", "JSON 업로드", "모델 없이 신호 확인"],
        key="r121_model_choice"
    )
    if choice == "모델 없이 신호 확인":
        return None
    if choice == "JSON 업로드":
        uploaded = st.sidebar.file_uploader(
            "Robust screening model JSON", type=["json"],
            key="r121_model_upload"
        )
        if uploaded is None:
            st.sidebar.info("이 버전에서 만든 모델을 선택하세요.")
            return None
        model = read_model(uploaded.getvalue())
        st.session_state["r121_active_model"] = model
        return model

    active = st.session_state.get("r121_active_model")
    if active is not None:
        return read_model(json_bytes(active))
    if DEFAULT_MODEL.is_file():
        return read_model(DEFAULT_MODEL.read_bytes())
    st.sidebar.info("아직 robust 모델이 없습니다. 개발 분석에서 새로 만드세요.")
    return None


def render_record(records, images, zw, decision=None, reviewed=False):
    name = st.selectbox("확인할 이미지", list(records), key="r121_preview_image")
    record = records[name]
    s = score_image(record, zw)["summary"]
    a, b, c, d = st.columns(4)
    a.metric("측정 가능 well", f"{s['N']:,}")
    b.metric("고신호 well KZ", fmt(s["K_Z"], 0))
    c.metric("이미지 점수 qZ (%)", fmt(s["qZ_pct"]))
    d.metric("QC 의심 well · 유지됨", str(s["artifact_suspect_wells"]))

    if s["status"] != "ready":
        st.error("판정 보류: " + s["hold_reasons"])
    g = s["geometry"]
    st.caption(
        f"{g['columns']} × {g['rows']} | "
        f"pitch {g['pitch_x_px']:.3f}/{g['pitch_y_px']:.3f} px | "
        f"{g['geometry_source']} | "
        f"M={fmt(s['median_corrected_signal'])}, MAD={fmt(s['MAD'])}"
    )

    tabs = st.tabs(["격자·분류", "신호 분포", "well 데이터", "기록·보고서"])
    with tabs[0]:
        rendered = display_images(images[name], record, zw)
        cols = st.columns(2)
        for i, (label_text, array) in enumerate(rendered.items()):
            cols[i % 2].image(array, caption=label_text, width="stretch")
        st.caption(
            "QC_suspects의 자주색 표시는 의심 위치이며 기포 확정이 아닙니다. "
            "그 위치는 자동으로 분모에서 제외하지 않았습니다."
        )
    with tabs[1]:
        st.pyplot(signal_figure(record, zw, True))
        st.pyplot(signal_figure(record, zw, False))
        st.caption(
            "히스토그램은 해당 이미지의 분포입니다. "
            "조건 비교용 최종 Figure에서는 동일 bin/축을 사용하세요."
        )
    with tabs[2]:
        scored = score_image(record, zw)
        st.dataframe(scored["wells"], hide_index=True)
        st.download_button(
            "well CSV 저장", csv_bytes(scored["wells"]),
            file_name=safe_name(name)+"_wells.csv", mime="text/csv",
            key="r121_one_well_csv"
        )
    with tabs[3]:
        st.json(clean_json(s))
        if st.button("HTML 보고서 생성", key="r121_make_html"):
            report = report_html(record, images[name], zw, decision, reviewed)
            st.session_state["r121_report"] = (
                object_hash({"s": s, "d": decision, "review": reviewed}), report
            )
        report_key = object_hash({"s": s, "d": decision, "review": reviewed})
        saved = st.session_state.get("r121_report")
        if saved and saved[0] == report_key:
            st.download_button(
                "HTML 보고서 저장", saved[1],
                file_name=safe_name(name)+"_report.html",
                mime="text/html", key="r121_html_download"
            )
            components.html(saved[1].decode("utf-8"), height=900, scrolling=True)


def main():
    st.set_page_config(
        page_title="WellScope LAMP · Robust", page_icon="🔬", layout="wide"
    )
    if core.VERSION != "1.2.0":
        st.error("이 추가 모듈은 올려주신 wellscope_core.py v1.2.0을 기준으로 작성됐습니다.")
        st.stop()

    st.title("WellScope LAMP")
    st.caption(
        f"ROBUST RESEARCH BUILD {VERSION} | Native engine {core.VERSION} | "
        "미검증 검토본"
    )
    st.info(
        "qZ는 상대 고신호 well의 비율이며 GMO 함량이 아닙니다. "
        "이상 영역은 기본적으로 표시만 하며, 판정 결과에 맞춰 삭제하지 않습니다."
    )

    with st.sidebar.expander("코드·계산 점검", expanded=False):
        st.code(
            f"App {VERSION}\nCore {core.VERSION}\n"
            f"Core SHA {core.ENGINE_SHA256[:16]}\n"
            f"Robust SHA {MODULE_SHA256[:16]}"
        )
        if st.button("계산 self-check 실행", key="r121_selfcheck"):
            checks = arithmetic_self_check()
            st.write(checks)
            if all(checks.values()):
                st.success("계산 self-check 통과. 영상/실험 검증을 뜻하지 않습니다.")
            else:
                st.error("Self-check 실패. 분석을 진행하지 마세요.")
        if st.button("현재 코드 ZIP 준비", key="r121_code_prepare"):
            st.session_state["r121_code_zip"] = code_zip()
        if "r121_code_zip" in st.session_state:
            st.download_button(
                "수정본 코드 ZIP 저장",
                st.session_state["r121_code_zip"],
                file_name="WellScope_LAMP_1.2.1_rc1_SOURCE.zip",
                mime="application/zip", key="r121_code_download"
            )

    workflow = st.sidebar.radio(
        "작업", ["시료 분석", "개발 모델 / 내부 검증"], key="r121_workflow"
    )
    model = None
    if workflow == "시료 분석":
        model = model_ui()
    config = config_ui(model)
    validate_config(config)

    uploaded = st.file_uploader(
        "원본 PNG/JPEG/단일 TIFF 또는 이미지 ZIP",
        type=["png", "jpg", "jpeg", "tif", "tiff", "zip"],
        accept_multiple_files=True, key="r121_images"
    )
    if not uploaded:
        st.write("원본 이미지를 선택하세요. 기존 v1.1/v1.2 absolute 모델은 사용하지 않습니다.")
        return

    images = unpack_images(uploaded)
    file_signature = object_hash([
        [name, hashlib.sha256(data).hexdigest()] for name, data in images.items()
    ])

    with st.expander("제외 기록 CSV · 선택 사항", expanded=False):
        st.caption(
            "열: image_id,well_id,reason. "
            "독립적인 QC 근거와 사전에 정한 규칙이 있을 때만 사용하세요. "
            "이미지 ID는 ZIP 경로를 포함한 현재 목록과 같아야 합니다."
        )
        st.write(list(images))
        exclusion_upload = st.file_uploader(
            "제외 기록", type=["csv"], key="r121_exclusions"
        )
    exclusions = read_exclusions(exclusion_upload, images)

    analysis_signature = object_hash({
        "files": file_signature,
        "config": config,
        "exclusions": exclusions,
        "core": core.ENGINE_SHA256,
        "module": MODULE_SHA256,
    })
    if st.button("이미지 분석 실행", type="primary", key="r121_run"):
        records, failures = run_images(images, config, exclusions)
        st.session_state["r121_analysis"] = (
            analysis_signature, records, failures
        )

    saved = st.session_state.get("r121_analysis")
    if not saved or saved[0] != analysis_signature:
        st.info("입력 또는 설정이 변경되었습니다. 이미지 분석을 실행하세요.")
        return
    records, failures = saved[1], saved[2]
    if not failures.empty:
        st.warning("일부 이미지 분석이 실패했습니다. 실패 자료를 삭제한 것으로 취급하지 않습니다.")
        st.dataframe(failures, hide_index=True)
    if not records:
        st.error("분석 완료 이미지가 없습니다.")
        return

    reviewed = st.checkbox(
        "모든 격자와 QC 표시를 검토했고, 촬영·반응·DNA 투입/희석 조건의 "
        "비교 가능성을 확인했습니다.",
        key="r121_review_"+analysis_signature[:16]
    )

    zw = model["well_rule"]["threshold_z"] if model else None
    decision = None
    cv_table = None
    fold_table = None
    manifest = None

    if workflow == "시료 분석":
        st.caption(
            "여러 장을 선택했다면 같은 독립 specimen의 계획된 시야들이어야 합니다. "
            "다른 농도/반복 시료들을 한 번에 합쳐 판정하지 마세요."
        )
        if model:
            st.write(
                f"Model {model['model_id']} | Zw={zw:.5g} | "
                f"Ts={model['sample_rule']['cutoff']:.5g}% | "
                f"specimen당 이미지 수={model['sample_rule']['expected_images_per_specimen']}"
            )
            if reviewed and failures.empty:
                try:
                    decision = predict_specimen(records, list(images), model)
                    if decision["predicted_positive"]:
                        st.success(
                            "GMO SCREENING POSITIVE — "
                            f"predicted ≥{model['study_boundary_pct']:g}% study class"
                        )
                    else:
                        st.info(
                            f"Predicted <{model['study_boundary_pct']:g}% study class "
                            "— GMO 불검출 인증이 아닙니다."
                        )
                    st.write(
                        f"Specimen qZ={decision['qZ_pct']:.4f}% | "
                        f"Ts={decision['sample_cutoff']:.4f}%"
                    )
                    if decision["training_image_match"]:
                        st.warning("개발에 사용한 이미지가 포함됩니다. 독립 검증이 아닙니다.")
                    if decision["outside_training_score_range"]:
                        st.warning("점수가 개발 관찰 범위 밖입니다. 모델의 일반화 근거를 확인하세요.")
                except core.AnalysisError as exc:
                    st.warning("시료 판정 보류: " + str(exc))
            else:
                st.warning("최종 시료 판정은 QC 검토 완료 및 모든 입력 이미지 분석 후 적용됩니다.")
        else:
            st.info("정규화 신호까지만 표시합니다. Zw/Ts가 없으므로 GMO 분류를 하지 않습니다.")

    else:
        st.subheader("표준 라벨과 독립 반복 단위")
        initial = []
        for name in images:
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", name)
            initial.append({
                "image_id": name,
                "mixture_pct": float(match.group(1)) if match else None,
                "specimen_id": "",
                "cv_group": "",
            })
        manifest = st.data_editor(
            pd.DataFrame(initial),
            disabled=["image_id"],
            hide_index=True,
            key="r121_manifest_"+file_signature[:16],
            column_config={
                "mixture_pct": st.column_config.NumberColumn(
                    "실제 혼합비 (%)", min_value=0.0, max_value=100.0
                ),
                "specimen_id": st.column_config.TextColumn("독립 specimen ID"),
                "cv_group": st.column_config.TextColumn("같이 제외할 CV group"),
            },
        )
        st.caption(
            "예: 0_R1의 여러 시야는 같은 specimen_id. "
            "같은 실험 run에 속해 함께 제외해야 하는 자료는 같은 cv_group. "
            "파일명의 숫자로 독립 반복을 자동 추정하지 않습니다."
        )
        a, b, c = st.columns(3)
        quantile = a.number_input(
            "0% 대조군 Z 백분위 (%)", 90.0, 99.99, 99.0, 0.1,
            key="r121_control_quantile"
        ) / 100
        views = b.number_input(
            "Specimen당 계획 이미지 수", 1, 100, 1,
            key="r121_expected_views"
        )
        study = c.number_input(
            "연구 혼합비 경계 (%)", 0.01, 99.99, 3.0, 0.1,
            key="r121_study"
        )
        labels_ok = st.checkbox(
            "실제 혼합비, 독립 시료 단위, CV group 및 계획 이미지 수를 확인했습니다.",
            key="r121_labels_ok_"+file_signature[:16]
        )
        model_signature = object_hash({
            "analysis": analysis_signature,
            "manifest": manifest.to_csv(index=False),
            "quantile": quantile, "views": int(views), "study": study,
        })
        left, right = st.columns(2)

        if left.button(
            "개발 모델 생성", disabled=not (reviewed and labels_ok),
            key="r121_fit_model"
        ):
            st.session_state.pop("r121_fitted", None)
            try:
                meta = validate_manifest(manifest, list(images))
                fitted, units = fit_model(
                    records, meta, config, quantile, int(views), study
                )
                st.session_state["r121_fitted"] = (
                    model_signature, fitted, units
                )
            except Exception as exc:
                st.error(str(exc))

        if right.button(
            "CV group 하나씩 제외하여 내부 검증",
            disabled=not (reviewed and labels_ok),
            key="r121_crossvalidate"
        ):
            st.session_state.pop("r121_cv", None)
            try:
                meta = validate_manifest(manifest, list(images), require_cv=True)
                predictions, folds = internal_cv(
                    records, meta, config, quantile, int(views), study
                )
                st.session_state["r121_cv"] = (
                    model_signature, predictions, folds
                )
            except Exception as exc:
                st.error(str(exc))

        fitted_saved = st.session_state.get("r121_fitted")
        if fitted_saved and fitted_saved[0] == model_signature:
            model, units = fitted_saved[1], fitted_saved[2]
            zw = model["well_rule"]["threshold_z"]
            st.write(
                f"Zw={zw:.5g} | development gap="
                f"{model['sample_rule']['training_gap']:.5g}"
            )
            st.dataframe(units, hide_index=True)
            if model["status"] == "ready":
                st.success(f"개발 모델 잠금 가능: Ts={model['sample_rule']['cutoff']:.5g}%")
                st.download_button(
                    "Robust 모델 JSON 저장", json_bytes(model),
                    file_name="robust_screening_model.json",
                    mime="application/json", key="r121_download_model"
                )
                if st.button("이 세션에서 모델 사용", key="r121_activate_model"):
                    st.session_state["r121_active_model"] = model
                    st.success("왼쪽에서 시료 분석 → 세션/기본 모델을 선택하세요.")
            else:
                st.error("MODEL NOT SEPARABLE: 기준 전후가 겹쳐 모델을 저장하지 않습니다.")
            st.warning("개발 모델의 분리는 독립 검증 정확도가 아닙니다.")

        cv_saved = st.session_state.get("r121_cv")
        if cv_saved and cv_saved[0] == model_signature:
            cv_table, fold_table = cv_saved[1], cv_saved[2]
            st.subheader("내부 교차검증 결과")
            st.dataframe(fold_table, hide_index=True)
            st.dataframe(cv_table, hide_index=True)
            st.json(clean_json(classification_summary(cv_table)))
            st.download_button(
                "내부 검증 CSV 저장", csv_bytes(cv_table),
                file_name="internal_cv_predictions.csv",
                mime="text/csv", key="r121_cv_csv"
            )
            fig = Figure(figsize=(7, 4), layout="constrained")
            ax = fig.subplots()
            for fold, part in cv_table.groupby("heldout_group"):
                good = part.dropna(subset=["decision_ratio"])
                ax.scatter(
                    good.mixture_pct, good.decision_ratio,
                    label=str(fold), s=30
                )
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.axvline(study, linestyle=":", linewidth=1)
            ax.set_xlabel("Assigned mixture (%)")
            ax.set_ylabel("Held-out decision ratio qZ / Ts")
            ax.legend(frameon=False)
            st.pyplot(fig)
            st.warning(
                "실패 fold와 판정 보류도 표에 남깁니다. "
                "이 자료를 보며 방법을 고른 이력이 있으므로, "
                "이 결과는 탐색 후 내부 검증이며 외부 독립 검증은 아닙니다."
            )

    st.subheader("이미지별 측정 결과")
    st.dataframe(summary_table(records, zw), hide_index=True)
    render_record(records, images, zw, decision, reviewed)

    export_signature = object_hash({
        "analysis": analysis_signature,
        "model": model["model_id"] if model else None,
        "manifest": manifest.to_csv(index=False) if manifest is not None else None,
        "cv": cv_table.to_csv(index=False) if cv_table is not None else None,
    })
    if st.button("전체 분석 결과 ZIP 준비", key="r121_results_prepare"):
        bundle = result_zip(
            records, failures, model, manifest, cv_table, fold_table
        )
        st.session_state["r121_export"] = (export_signature, bundle)
    exported = st.session_state.get("r121_export")
    if exported and exported[0] == export_signature:
        st.download_button(
            "분석 결과 ZIP 저장", exported[1],
            file_name="WellScope_ROBUST_RESULTS_PRIVATE.zip",
            mime="application/zip", key="r121_results_download"
        )
    st.caption(EVIDENCE_NOTE)


if __name__ == "__main__":
    try:
        main()
    except core.AnalysisError as exc:
        st.error(str(exc))
