"""Error analysis and visualization for saved QCrank eHANDS 3D classification runs.

Load `.npz` files written by `ehands_qcrank_vertex_classification_V2.py` and
produce error CSVs, residual/slice plots, and marching-cubes meshes with glyphs.
Does not execute quantum circuits.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_MC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "exploration_notebooks",
    "3D-testing",
)
if _MC_DIR not in sys.path:
    sys.path.insert(0, _MC_DIR)

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    import vtkmodules.vtkRenderingFreeType  # noqa: F401
    from custom_marching_cubes import marching_cubes as _marching_cubes

    _VTK_MC_AVAILABLE = True
except Exception as exc:
    vtk = None
    numpy_to_vtk = None
    vtk_to_numpy = None
    _marching_cubes = None
    _VTK_MC_AVAILABLE = False
    print(f"VTK / labeled marching cubes unavailable ({exc}); 3D mesh extract disabled.")


CLASSIFICATION_RUNS_DIR = "classification_runs"

def classify_vertex_classical(input_val, isolevel):
    """VTK-style inside test: 1 iff value >= isolevel (notebook)."""
    inside = np.asarray(input_val) >= isolevel
    if inside.ndim == 0:
        return int(inside)
    return inside.astype(np.uint8, copy=False)


def quantum_classes_to_mc_labels(y, shape):
    """Map quantum class 0 (inside) to VTK label 1."""
    n = int(np.prod(shape))
    pred = np.asarray(y, dtype=int).reshape(-1)[:n]
    labels = (pred == 0).astype(np.uint8)
    return labels.reshape(shape)


BOUNDARY_EPS_ORIG = 0.1  # |value - isolevel| band on the original [0, 1] volume scale
ERROR_CSV_FIELDS = (
    "n_voxels",
    "n_errors",
    "tn",
    "fp",
    "fn",
    "tp",
    "error_rate",
    "precision_inside",
    "recall_inside",
    "f1_inside",
    "mae_ev",
    "mae_ev_incorrect",
    "mean_abs_dist_to_iso_errors",
    "n_boundary",
    "n_errors_boundary",
    "error_rate_boundary",
    "n_tris_classical",
    "n_tris_quantum",
    "n_tris_delta",
    "boundary_eps",
)


def _safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return float("nan")
    return float(num) / den


def compute_volume_error_metrics(
    *,
    y_true,
    y_pred,
    volume_orig,
    isolevel_orig,
    subtraction_vals,
    quantum_ev,
    boundary_eps=BOUNDARY_EPS_ORIG,
):
    """Confusion, analog-error, and boundary-band stats plus a per-incorrect-voxel table."""
    volume_orig = np.asarray(volume_orig, dtype=np.float64)
    shape = tuple(int(s) for s in volume_orig.shape)
    n = int(np.prod(shape))
    y_true = np.asarray(y_true, dtype=int).reshape(-1)[:n]
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)[:n]
    vol = volume_orig.reshape(-1)[:n]
    sub = np.asarray(subtraction_vals, dtype=np.float64).reshape(-1)[:n]
    ev = np.asarray(quantum_ev, dtype=np.float64).reshape(-1)[:n]
    iso = float(isolevel_orig)
    eps = float(boundary_eps)

    incorrect = y_true != y_pred
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    n_errors = int(fp + fn)
    n_voxels = int(n)
    error_rate = _safe_div(n_errors, n_voxels)

    precision_inside = _safe_div(tn, tn + fn)
    recall_inside = _safe_div(tn, tn + fp)
    f1_inside = _safe_div(
        2.0 * precision_inside * recall_inside,
        precision_inside + recall_inside,
    )

    abs_ev_err = np.abs(ev - sub)
    abs_dist_iso = np.abs(vol - iso)
    mae_ev = float(np.mean(abs_ev_err)) if n_voxels else float("nan")
    mae_ev_incorrect = (
        float(np.mean(abs_ev_err[incorrect])) if n_errors else float("nan")
    )
    mean_abs_dist_to_iso_errors = (
        float(np.mean(abs_dist_iso[incorrect])) if n_errors else float("nan")
    )

    boundary = abs_dist_iso < eps
    n_boundary = int(np.sum(boundary))
    n_errors_boundary = int(np.sum(incorrect & boundary))
    error_rate_boundary = _safe_div(n_errors_boundary, n_boundary)

    xs, ys, zs = np.indices(shape)
    err_idx = np.flatnonzero(incorrect)
    kind = np.empty(err_idx.size, dtype=object)
    kind[y_true[err_idx] == 0] = "false_outside"
    kind[y_true[err_idx] == 1] = "false_inside"
    error_table = {
        "x": xs.reshape(-1)[err_idx].astype(int),
        "y": ys.reshape(-1)[err_idx].astype(int),
        "z": zs.reshape(-1)[err_idx].astype(int),
        "value": vol[err_idx],
        "isolevel": np.full(err_idx.size, iso, dtype=np.float64),
        "classical_residual": sub[err_idx],
        "quantum_ev": ev[err_idx],
        "y_true": y_true[err_idx],
        "y_pred": y_pred[err_idx],
        "abs_ev_err": abs_ev_err[err_idx],
        "abs_dist_to_iso": abs_dist_iso[err_idx],
        "error_kind": kind,
    }

    metrics = {
        "n_voxels": n_voxels,
        "n_errors": n_errors,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "error_rate": error_rate,
        "precision_inside": precision_inside,
        "recall_inside": recall_inside,
        "f1_inside": f1_inside,
        "mae_ev": mae_ev,
        "mae_ev_incorrect": mae_ev_incorrect,
        "mean_abs_dist_to_iso_errors": mean_abs_dist_to_iso_errors,
        "n_boundary": n_boundary,
        "n_errors_boundary": n_errors_boundary,
        "error_rate_boundary": error_rate_boundary,
        "n_tris_classical": "",
        "n_tris_quantum": "",
        "n_tris_delta": "",
        "boundary_eps": eps,
    }
    return metrics, error_table


def write_error_voxels_csv(error_table, out_name):
    """Write one row per misclassified voxel."""
    fieldnames = [
        "x",
        "y",
        "z",
        "value",
        "isolevel",
        "classical_residual",
        "quantum_ev",
        "y_true",
        "y_pred",
        "abs_ev_err",
        "abs_dist_to_iso",
        "error_kind",
    ]
    n_rows = int(np.asarray(error_table.get("x", [])).reshape(-1).size) if error_table else 0
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    with open(out_name, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(
                {
                    "x": int(error_table["x"][i]),
                    "y": int(error_table["y"][i]),
                    "z": int(error_table["z"][i]),
                    "value": float(error_table["value"][i]),
                    "isolevel": float(error_table["isolevel"][i]),
                    "classical_residual": float(error_table["classical_residual"][i]),
                    "quantum_ev": float(error_table["quantum_ev"][i]),
                    "y_true": int(error_table["y_true"][i]),
                    "y_pred": int(error_table["y_pred"][i]),
                    "abs_ev_err": float(error_table["abs_ev_err"][i]),
                    "abs_dist_to_iso": float(error_table["abs_dist_to_iso"][i]),
                    "error_kind": str(error_table["error_kind"][i]),
                }
            )
    print(f"Wrote {n_rows} incorrect-voxel rows to: {out_name}")


def error_metrics_to_csv_fields(metrics):
    """Round error-metric dict values for a summary CSV row."""
    if not metrics:
        return {k: "" for k in ERROR_CSV_FIELDS}
    out = {}
    int_keys = {
        "n_voxels",
        "n_errors",
        "tn",
        "fp",
        "fn",
        "tp",
        "n_boundary",
        "n_errors_boundary",
        "n_tris_classical",
        "n_tris_quantum",
        "n_tris_delta",
    }
    for k in ERROR_CSV_FIELDS:
        v = metrics.get(k, "")
        if v is None or v == "":
            out[k] = ""
            continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            out[k] = ""
            continue
        if k in int_keys:
            out[k] = int(round(float(v)))
        elif isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def average_error_metrics(rows):
    """Average rates across iterations; keep last-iteration integer counts."""
    if not rows:
        return {}
    last = dict(rows[-1])
    rate_keys = (
        "n_errors",
        "error_rate",
        "precision_inside",
        "recall_inside",
        "f1_inside",
        "mae_ev",
        "mae_ev_incorrect",
        "mean_abs_dist_to_iso_errors",
        "n_errors_boundary",
        "error_rate_boundary",
    )
    for k in rate_keys:
        vals = []
        for row in rows:
            v = row.get(k)
            if v is None or v == "":
                continue
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        if vals:
            last[k] = float(np.mean(vals))
    return last


def print_volume_error_metrics(dataset_name, metrics):
    """Print the boundary-aware error summary for one classification pass."""
    if not metrics:
        return
    print(
        f"\n{dataset_name} error summary: "
        f"{metrics['n_errors']} / {metrics['n_voxels']} voxels "
        f"(error_rate={metrics['error_rate']:.6f})"
    )
    print(
        f"  CM tn/fp/fn/tp = {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}  "
        f"precision_inside={metrics['precision_inside']:.4f}  "
        f"recall_inside={metrics['recall_inside']:.4f}  "
        f"f1_inside={metrics['f1_inside']:.4f}"
    )
    mae_bad = metrics["mae_ev_incorrect"]
    mae_bad_s = f"{mae_bad:.4f}" if mae_bad == mae_bad else "nan"
    dist_s = (
        f"{metrics['mean_abs_dist_to_iso_errors']:.4f}"
        if metrics["mean_abs_dist_to_iso_errors"] == metrics["mean_abs_dist_to_iso_errors"]
        else "nan"
    )
    print(
        f"  mae_ev={metrics['mae_ev']:.4f}  mae_ev_incorrect={mae_bad_s}  "
        f"mean_|value-iso|_errors={dist_s}"
    )
    print(
        f"  boundary |value-iso|<{metrics['boundary_eps']}: "
        f"{metrics['n_errors_boundary']} / {metrics['n_boundary']} "
        f"(error_rate_boundary={metrics['error_rate_boundary']:.6f})"
        if metrics["n_boundary"]
        else f"  boundary |value-iso|<{metrics['boundary_eps']}: empty band"
    )


def classification_output_paths(
    dataset_name: str,
    volume_shape: tuple[int, int, int],
    circuit_size: int,
    save_name: str,
) -> tuple[str, str, str, str, str, str]:
    """
    Return (summary, slices, residual, mesh, voxel-error CSV, error-vs-iso) paths.
    """
    summary_dir = "classification_summaries"
    side_dir = "side-by-sides"
    residual_dir = "residual_plots"
    mesh_dir = "marching_cubes_meshes"
    error_dir = "error_analysis"
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(side_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    nx, ny, nz = volume_shape
    if save_name is not None:
        stem = save_name
    else:
        stem = f"{dataset_name}_{nx}x{ny}x{nz}_circ{circuit_size}"
    return (
        f"{summary_dir}/{stem}_classification_summary.png",
        f"{side_dir}/{stem}_volume_vs_classification_slices.png",
        f"{residual_dir}/{stem}_classical_minus_isovalue_vs_quantum_ev.png",
        f"{mesh_dir}/{stem}_marching_cubes_classical_vs_quantum.png",
        f"{error_dir}/{stem}_incorrect_voxels.csv",
        f"{error_dir}/{stem}_error_vs_isodistance.png",
    )


def plot_full_image_vs_classification(
    input_image,
    true_image,
    predicted_image,
    out_name,
    *,
    input_value_range=(-1.0, 1.0),
    region_size_hw=None,
):
    """
    Plot input grayscale, true classes, and predicted classes side-by-side to `out_name`.
    
    Credit to CursorAI for the following code.
    """
    font_size_delta = 5
    input_image = np.asarray(input_image, dtype=np.float32)
    true_image = np.asarray(true_image, dtype=np.int32)
    predicted_image = np.asarray(predicted_image, dtype=np.int32)
    h, w = input_image.shape
    fig_w = min(22.0, max(10.0, w / 32.0 + 4.0))
    fig_h = min(14.0, max(5.0, h / 32.0 + 2.0))
    fig, axes = plt.subplots(1, 3, figsize=(1.65 * fig_w, fig_h))

    ax_input, ax_true, ax_pred = axes
    in_vmin, in_vmax = float(input_value_range[0]), float(input_value_range[1])
    im0 = ax_input.imshow(
        input_image, cmap="gray", vmin=in_vmin, vmax=in_vmax, origin="upper", zorder=1
    )

    n_pad = 0
    if region_size_hw is not None:
        rh_i, rw_i = int(region_size_hw[0]), int(region_size_hw[1])
        cy = np.arange(h)[:, np.newaxis]
        cx = np.arange(w)[np.newaxis, :]
        pad_mask = (cy >= rh_i) | (cx >= rw_i)
        n_pad = int(np.sum(pad_mask))
        if n_pad > 0:
            for ax in (ax_input, ax_true, ax_pred):
                ov = np.zeros((h, w, 4), dtype=np.float32)
                ov[pad_mask] = (0.95, 0.15, 0.65, 0.55)
                ax.imshow(ov, origin="upper", interpolation="nearest", zorder=2)
            ax_input.set_title("Input Image Grayscale (magenta = padded band, -1)")
        else:
            ax_input.set_title(
                "Input Image Grayscale"
            )
    else:
        ax_input.set_title("Input Image Grayscale (full region)")
    ax_input.set_xlabel("x")
    ax_input.set_ylabel("y")
    # Keep subplot widths symmetric by attaching the colorbar inside the first panel.
    cax0 = ax_input.inset_axes([1.02, 0.08, 0.025, 0.84])
    cbar0 = fig.colorbar(
        im0,
        cax=cax0,
        ticks=[in_vmin, 0.5 * (in_vmin + in_vmax), in_vmax],
    )
    cbar0.set_ticklabels([f"{in_vmin:g}", f"{0.5 * (in_vmin + in_vmax):g}", f"{in_vmax:g}"])
    for tick in cbar0.ax.get_yticklabels():
        tick.set_fontsize(tick.get_fontsize() + font_size_delta)

    ax_true.imshow(
        true_image, cmap="viridis_r", vmin=0, vmax=1, origin="upper", zorder=1
    )
    if region_size_hw is not None:
        if n_pad > 0:
            ax_true.set_title("Classical Classifications (magenta = padded band)")
        else:
            ax_true.set_title("Classical Classifications")
    else:
        ax_true.set_title("Classical Classifications")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    cmap = plt.get_cmap("viridis_r")
    ax_true.legend(
        handles=[
            Patch(facecolor=cmap(0.0), edgecolor="black", label="Inside"),
            Patch(facecolor=cmap(1.0), edgecolor="black", label="Outside"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    im1 = ax_pred.imshow(
        predicted_image, cmap="viridis_r", vmin=0, vmax=1, origin="upper", zorder=1
    )
    if region_size_hw is not None:
        if n_pad > 0:
            ax_pred.set_title("Quantum Classifications (magenta = padded band)")
        else:
            ax_pred.set_title("Quantum Classifications")
    else:
        ax_pred.set_title("Quantum Classifications")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.legend(
        handles=[
            Patch(facecolor=cmap(0.0), edgecolor="black", label="Inside"),
            Patch(facecolor=cmap(1.0), edgecolor="black", label="Outside"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    xt = axis_ticks(w)
    yt = axis_ticks(h)
    for ax in (ax_input, ax_true, ax_pred):
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        increase_axis_text_size(ax, delta_points=font_size_delta)

    fig.tight_layout()

    fig.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved full image vs classification plot to: {out_name}")
    plt.close(fig)


def _midplane(volume, axis):
    vol = np.asarray(volume)
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis), idx


def plot_volume_slices_vs_classification(
    input_volume,
    true_volume,
    predicted_volume,
    out_name,
    *,
    input_value_range=(-1.0, 1.0),
    dataset_name="volume",
):
    """Plot mid-plane slices of input, classical labels, quantum labels, and error maps."""
    input_volume = np.asarray(input_volume, dtype=np.float32)
    true_volume = np.asarray(true_volume, dtype=np.int32)
    predicted_volume = np.asarray(predicted_volume, dtype=np.int32)
    error_volume = (predicted_volume - true_volume).astype(np.int32)
    in_vmin, in_vmax = float(input_value_range[0]), float(input_value_range[1])
    cmap = plt.get_cmap("viridis_r")
    plane_axes = ((2, "xy"), (1, "xz"), (0, "yz"))

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.5))
    for row, (axis, name) in enumerate(plane_axes):
        inp, idx = _midplane(input_volume, axis)
        tru, _ = _midplane(true_volume, axis)
        pred, _ = _midplane(predicted_volume, axis)
        err, _ = _midplane(error_volume, axis)
        im0 = axes[row, 0].imshow(inp.T, cmap="gray", vmin=in_vmin, vmax=in_vmax, origin="lower")
        axes[row, 1].imshow(tru.T, cmap="viridis_r", vmin=0, vmax=1, origin="lower")
        axes[row, 2].imshow(pred.T, cmap="viridis_r", vmin=0, vmax=1, origin="lower")
        im_err = axes[row, 3].imshow(err.T, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        axes[row, 0].set_title(f"{dataset_name} input  {name} mid ({idx})")
        axes[row, 1].set_title(f"Classical  {name}")
        axes[row, 2].set_title(f"Quantum  {name}")
        axes[row, 3].set_title(f"Error  {name}")
        for ax in axes[row]:
            ax.set_xlabel("i")
            ax.set_ylabel("j")
        cax = axes[row, 0].inset_axes([1.02, 0.08, 0.04, 0.84])
        fig.colorbar(im0, cax=cax)
        if row == 0:
            cax_err = axes[row, 3].inset_axes([1.02, 0.08, 0.04, 0.84])
            fig.colorbar(im_err, cax=cax_err)

    handles = [
        Patch(facecolor=cmap(0.0), edgecolor="black", label="Inside (class 0)"),
        Patch(facecolor=cmap(1.0), edgecolor="black", label="Outside (class 1)"),
    ]
    error_handles = [
        Patch(facecolor=plt.get_cmap("RdBu_r")(0.0), edgecolor="black", label="False inside (−1)"),
        Patch(facecolor=plt.get_cmap("RdBu_r")(1.0), edgecolor="black", label="False outside (+1)"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper right", framealpha=0.95)
    axes[0, 3].legend(handles=error_handles, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved volume slice classification plot to: {out_name}")
    plt.close(fig)


def _mesh_stats(poly):
    n_points = int(poly.GetNumberOfPoints()) if poly is not None else 0
    n_tris = int(poly.GetNumberOfPolys()) if poly is not None else 0
    return n_points, n_tris


def _vtk_mesh_actor(polydata, color=(0.35, 0.72, 0.95), wireframe=False):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
    else:
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetAmbient(0.22)
        actor.GetProperty().SetDiffuse(0.72)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(28)
    return actor


def _vtk_error_glyph_actor(xyz, magnitudes, color, base_radius=0.42):
    """Spheres at misclassified voxels; radius scales with analog-error magnitude."""
    if vtk is None or numpy_to_vtk is None:
        return None
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return None
    mags = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if mags.shape[0] != xyz.shape[0]:
        mags = np.ones(xyz.shape[0], dtype=np.float64)
    p90 = float(np.percentile(mags, 90)) if mags.size else 1.0
    p90 = max(p90, 1e-6)
    scales = base_radius * (0.65 + 0.85 * np.clip(mags / p90, 0.0, 2.0))

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(xyz), deep=True))
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(points)
    scale_arr = numpy_to_vtk(np.ascontiguousarray(scales), deep=True)
    scale_arr.SetName("glyph_scale")
    pdata.GetPointData().SetScalars(scale_arr)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(12)
    sphere.SetPhiResolution(12)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(pdata)
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetScaleFactor(1.0)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(0.92)
    actor.GetProperty().SetAmbient(0.35)
    actor.GetProperty().SetDiffuse(0.7)
    return actor


def _vtk_outline_actor(nx, ny, nz):
    source = vtk.vtkCubeSource()
    source.SetBounds(0, nx - 1, 0, ny - 1, 0, nz - 1)
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(source.GetOutputPort())
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    actor.GetProperty().SetLineWidth(1.5)
    return actor


def _vtk_text_actor(text, x, y):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    prop = actor.GetTextProperty()
    prop.SetFontSize(16)
    prop.SetBold(1)
    prop.SetColor(1.0, 1.0, 1.0)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(x, y)
    return actor


def _vtk_rgb_screenshot(render_window):
    render_window.Render()
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    vtk_img = w2if.GetOutput()
    width, height, _ = vtk_img.GetDimensions()
    rgb = vtk_to_numpy(vtk_img.GetPointData().GetScalars()).reshape(height, width, 3)
    return np.ascontiguousarray(np.flipud(rgb))


def plot_marching_cubes_classical_vs_quantum(
    volume,
    isolevel,
    true_classes,
    pred_classes,
    out_name,
    dataset_name="volume",
    interactive=True,
    error_table=None,
):
    """Extract labeled marching cubes and save a PNG; overlay error glyphs; optional VTK window."""
    empty_stats = {
        "n_tris_classical": 0,
        "n_tris_quantum": 0,
        "n_tris_delta": 0,
    }
    if not _VTK_MC_AVAILABLE:
        print("Skipping marching-cubes figure (VTK / labeled extension not available).")
        return empty_stats

    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float64))
    true_labels = quantum_classes_to_mc_labels(true_classes, volume.shape)
    pred_labels = quantum_classes_to_mc_labels(pred_classes, volume.shape)
    classical_mesh = _marching_cubes(volume, float(isolevel), classified=true_labels)
    quantum_mesh = _marching_cubes(volume, float(isolevel), classified=pred_labels)
    n_pts_c, n_tris_c = _mesh_stats(classical_mesh)
    n_pts_q, n_tris_q = _mesh_stats(quantum_mesh)
    print(
        f"{dataset_name} marching cubes @ isolevel={isolevel:.4f}: "
        f"classical {n_pts_c} pts / {n_tris_c} tris, "
        f"quantum {n_pts_q} pts / {n_tris_q} tris"
    )
    mesh_stats = {
        "n_tris_classical": int(n_tris_c),
        "n_tris_quantum": int(n_tris_q),
        "n_tris_delta": int(n_tris_q - n_tris_c),
    }

    fp_xyz = np.zeros((0, 3), dtype=np.float64)
    fn_xyz = np.zeros((0, 3), dtype=np.float64)
    fp_mag = np.zeros((0,), dtype=np.float64)
    fn_mag = np.zeros((0,), dtype=np.float64)
    if error_table and int(np.asarray(error_table.get("x", [])).reshape(-1).size):
        xyz = np.column_stack(
            (
                np.asarray(error_table["x"], dtype=np.float64),
                np.asarray(error_table["y"], dtype=np.float64),
                np.asarray(error_table["z"], dtype=np.float64),
            )
        )
        mags = np.asarray(error_table["abs_ev_err"], dtype=np.float64)
        kinds = np.asarray(error_table["error_kind"]).astype(str)
        fp_mask = kinds == "false_outside"
        fn_mask = kinds == "false_inside"
        fp_xyz, fp_mag = xyz[fp_mask], mags[fp_mask]
        fn_xyz, fn_mag = xyz[fn_mask], mags[fn_mask]
    n_fp = int(fp_xyz.shape[0])
    n_fn = int(fn_xyz.shape[0])

    nx, ny, nz = volume.shape
    camera = vtk.vtkCamera()
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(nx * 2.4, ny * -2.1, nz * 1.8)
    camera.SetFocalPoint((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)

    def _add_error_glyphs(renderer):
        fp_actor = _vtk_error_glyph_actor(fp_xyz, fp_mag, color=(0.95, 0.28, 0.12))
        fn_actor = _vtk_error_glyph_actor(fn_xyz, fn_mag, color=(0.15, 0.82, 0.92))
        if fp_actor is not None:
            renderer.AddActor(fp_actor)
        if fn_actor is not None:
            renderer.AddActor(fn_actor)

    ren_left = vtk.vtkRenderer()
    ren_left.SetViewport(0.0, 0.0, 0.5, 1.0)
    ren_left.SetBackground(0.12, 0.12, 0.14)
    ren_left.SetActiveCamera(camera)
    ren_left.AddActor(_vtk_mesh_actor(classical_mesh))
    if n_tris_c:
        ren_left.AddActor(_vtk_mesh_actor(classical_mesh, color=(0.08, 0.18, 0.28), wireframe=True))
    ren_left.AddActor(_vtk_outline_actor(nx, ny, nz))
    ren_left.AddActor(
        _vtk_text_actor(
            f"{dataset_name} — classical MC ({n_tris_c} tris)",
            0.04,
            0.93,
        )
    )

    ren_right = vtk.vtkRenderer()
    ren_right.SetViewport(0.5, 0.0, 1.0, 1.0)
    ren_right.SetBackground(0.12, 0.12, 0.14)
    ren_right.SetActiveCamera(camera)
    ren_right.AddActor(_vtk_mesh_actor(quantum_mesh, color=(0.95, 0.62, 0.28)))
    if n_tris_q:
        ren_right.AddActor(_vtk_mesh_actor(quantum_mesh, color=(0.28, 0.12, 0.04), wireframe=True))
    _add_error_glyphs(ren_right)
    ren_right.AddActor(_vtk_outline_actor(nx, ny, nz))
    ren_right.AddActor(_vtk_text_actor(f"quantum MC ({n_tris_q} tris)", 0.04, 0.93))
    ren_right.AddActor(
        _vtk_text_actor(
            f"glyphs: {n_fp} false outside (red), {n_fn} false inside (cyan)",
            0.04,
            0.04,
        )
    )

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren_left)
    render_window.AddRenderer(ren_right)
    render_window.SetSize(1100, 520)
    render_window.SetWindowName(f"{dataset_name} marching cubes (classical | quantum)")
    camera.OrthogonalizeViewUp()
    ren_left.ResetCameraClippingRange()
    ren_right.ResetCameraClippingRange()

    if interactive:
        render_window.SetOffScreenRendering(0)
    else:
        render_window.SetOffScreenRendering(1)

    rgb = _vtk_rgb_screenshot(render_window)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(
        f"{dataset_name}  |  isolevel={isolevel:.3f}  |  "
        f"classical {n_tris_c} tris vs quantum {n_tris_q} tris  |  "
        f"{n_fp + n_fn} error glyphs"
    )
    fig.tight_layout()
    fig.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved marching-cubes comparison to: {out_name}")
    plt.close(fig)

    if interactive:
        print(
            f"Interactive mesh for {dataset_name}: left-drag rotate, scroll zoom, "
            "middle-drag pan. Close the VTK window to continue."
        )
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        render_window.Render()
        interactor.Initialize()
        interactor.Start()

    return mesh_stats


def axis_ticks(n, step=10):
    """
    Compute simple integer tick marks for an axis of length `n`.
    
    Credit to CursorAI for the following code.
    """
    if n <= 0:
        return np.array([], dtype=int)
    return np.asarray(list(range(0, n, step)), dtype=int)


def increase_axis_text_size(ax, delta_points=5):
    """
    Increase title/label/tick/legend font sizes for a matplotlib axis.
    
    Credit to CursorAI for the following code.
    """
    ax.title.set_fontsize(ax.title.get_fontsize() + delta_points)
    ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() + delta_points)
    ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() + delta_points)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(tick.get_fontsize() + delta_points)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() + delta_points)
        legend_title = legend.get_title()
        if legend_title is not None:
            legend_title.set_fontsize(legend_title.get_fontsize() + delta_points)

    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() + delta_points)

def plot_correct_incorrect_input_histogram(all_correct_vals, all_incorrect_vals, bins=20, ax=None):
    """
    Plot histograms of weighted subtraction values for correct vs incorrect predictions.
    
    Credit to CursorAI for the following code.
    """
    if ax is None:
        ax = plt.gca()

    if not (all_correct_vals or all_incorrect_vals):
        ax.set_title("Quantum Classification: correct vs incorrect classifications")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    concat_correct = (
        np.concatenate(all_correct_vals) if all_correct_vals else np.array([])
    )
    concat_incorrect = (
        np.concatenate(all_incorrect_vals) if all_incorrect_vals else np.array([])
    )

    parts = [a for a in (concat_correct, concat_incorrect) if a.size]
    bin_edges = np.histogram_bin_edges(np.concatenate(parts), bins=bins)

    if concat_correct.size:
        ax.hist(concat_correct, bins=bin_edges, alpha=0.6, label="Correct", color="tab:blue")
    if concat_incorrect.size:
        ax.hist(concat_incorrect, bins=bin_edges, alpha=0.6, label="Incorrect", color="tab:orange")

    ax.set_xlabel("Input data value after weighted subtraction")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Quantum Classification: correct vs incorrect classifications")
    ax.legend()


def plot_aggregated_confusion_matrix(total_cm, title="Aggregated Confusion Matrix", ax=None):
    """
    Render an aggregated 2x2 confusion matrix with color-coded diagonal/off-diagonal.
    
    Credit to CursorAI for the following code.
    """
    if ax is None:
        ax = plt.gca()

    total_cm = np.asarray(total_cm, dtype=float)
    cmap_blue = plt.colormaps["Blues"]
    cmap_orange = plt.colormaps["Oranges"]

    diag_max = max(total_cm[0, 0], total_cm[1, 1])
    if diag_max <= 0:
        diag_max = 1.0
    rgba = np.zeros((2, 2, 4), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            val = total_cm[i, j]
            if i == j:
                t = val / diag_max
                rgba[i, j] = cmap_blue(0.22 + 0.73 * t)
            else:
                t = val / diag_max
                rgba[i, j] = cmap_orange(0.28 + 0.67 * t)

    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=10)

    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["Pred Inside", "Pred Outside"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["True Inside", "True Outside"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    for i in range(2):
        for j in range(2):
            val = int(total_cm[i, j])
            r, g, b, _ = rgba[i, j]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color=("black" if luminance > 0.6 else "white"),
            )


def plot_aggregated_predicted_class_counts(agg_counts, ax=None):
    """
    Plot total predicted class counts (class 0 vs 1).
    
    Credit to CursorAI for the following code.
    """
    if ax is None:
        ax = plt.gca()

    labels = ["0", "1"]
    values = [agg_counts["0"], agg_counts["1"]]
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_xlabel("Predicted class (final bit)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Aggregated Predicted Class Counts")


def plot_true_class_input_histogram(all_true_inside_vals, all_true_outside_vals, bins=20, ax=None):
    """
    Plot histogram of true-class weighted subtraction values (classical baseline).
    
    Credit to CursorAI for the following code.
    """
    if ax is None:
        ax = plt.gca()

    if not (all_true_inside_vals or all_true_outside_vals):
        ax.set_title("True input data distribution")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    concat_true_inside = np.concatenate(all_true_inside_vals) if all_true_inside_vals else np.array([])
    concat_true_outside = np.concatenate(all_true_outside_vals) if all_true_outside_vals else np.array([])
    all_true_vals = np.concatenate([a for a in (concat_true_inside, concat_true_outside) if a.size])
    bin_edges = np.histogram_bin_edges(all_true_vals, bins=bins)

    ax.hist(
        all_true_vals,
        bins=bin_edges,
        alpha=0.75,
        color="tab:blue",
    )

    ax.set_xlabel("Input data value after weighted subtraction")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Classical Classification")


def print_per_datapoint_classification_table(data_vals, subtraction_vals, y_true, y_pred):
    """
    Print a per-datapoint ASCII table and return (correct_vals, incorrect_vals) arrays.
    
    Credit to CursorAI for the following code.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    data_vals = np.asarray(data_vals).reshape(-1)
    subtraction_vals = np.asarray(subtraction_vals).reshape(-1)

    print("\nPer-data-point classifications")
    print("+--------+---------------+------------------+--------------+----------------+")
    print("| Index  | Input Value   | Subtraction Vals | True Class   | Pred Class     |")
    print("+--------+---------------+------------------+--------------+----------------+")
    for idx, (val, sub_val, y_t, y_p) in enumerate(
        zip(data_vals, subtraction_vals, y_true, y_pred)
    ):
        print(f"| {idx:<6d} | {val:<13.6f} | {sub_val:<16.6f} | {int(y_t):<12d} | {int(y_p):<14d} |")
    print("+--------+---------------+------------------+--------------+----------------+")

    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_vals = subtraction_vals[correct_mask] if np.any(correct_mask) else None
    incorrect_vals = subtraction_vals[incorrect_mask] if np.any(incorrect_mask) else None

    if correct_vals is not None:
        print(
            "Correct classifications value range: "
            f"[{correct_vals.min():.6f}, {correct_vals.max():.6f}]"
        )
    else:
        print("No correct classifications in this run.")

    if incorrect_vals is not None:
        print(
            "Incorrect classifications value range: "
            f"[{incorrect_vals.min():.6f}, {incorrect_vals.max():.6f}]"
        )
    else:
        print("No incorrect classifications in this run.")

    return correct_vals, incorrect_vals


def print_datapoint_classification_tables_if_any(pre):
    """
    Print any captured per-datapoint tables stored on `pre` (typically first tile only).
    
    Credit to CursorAI for the following code.
    """
    for payload in pre.datapoint_table_payloads:
        print_per_datapoint_classification_table(**payload)


def plot_classification_summary_figure(
    region_width,
    region_height,
    tile_width,
    tile_height,
    acc_list,
    cm_list,
    agg_counts,
    all_correct_vals,
    all_incorrect_vals,
    all_true_inside_vals,
    all_true_outside_vals,
    out_name,
    bins=20,
    volume_shape=None,
    n_circuits=None,
    dataset_name=None,
):
    """
    Create and save the 3-panel summary figure (histograms + aggregated confusion matrix).
    """
    font_size_delta = 5
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    if volume_shape is not None:
        shape_str = "x".join(str(int(s)) for s in volume_shape)
        n_circ = n_circuits if n_circuits is not None else len(acc_list)
        ds = dataset_name or "volume"
        title = (
            f"{ds} {shape_str}: mean accuracy {mean_acc:.3f} "
            f"({n_circ} circuits of {tile_width} addresses)"
        )
    else:
        title = (
            f"Mean accuracy over {region_width}x{region_height} region, with "
            f"{len(acc_list)} ({tile_width}x{tile_height}) tiles: {mean_acc:.3f}"
        )
    print(f"\n{title}\n")

    fig, ax_arr = plt.subplots(1, 3, figsize=(18, 5))
    ax_true_hist, ax_hist, ax_cm = ax_arr

    # Classical Classification
    plot_true_class_input_histogram(
        all_true_inside_vals,
        all_true_outside_vals,
        bins=bins,
        ax=ax_true_hist,
    )

    # Quantum Classification
    plot_correct_incorrect_input_histogram(
        all_correct_vals,
        all_incorrect_vals,
        bins=bins,
        ax=ax_hist,
    )

    # Confusion Matrix
    if cm_list:
        total_cm = np.sum(np.stack(cm_list, axis=0), axis=0)
        print(
            "Aggregated confusion matrix over all runs "
            "[[true0->pred0, true0->pred1], [true1->pred0, true1->pred1]]:"
        )
        print(total_cm)
        n_cm_iter = max(len(cm_list), 1)
        n_err = int(round((float(total_cm[0, 1]) + float(total_cm[1, 0])) / n_cm_iter))
        n_tot = int(round(float(np.sum(total_cm)) / n_cm_iter))
        plot_aggregated_confusion_matrix(
            total_cm,
            ax=ax_cm,
            title=f"Aggregated Confusion Matrix\n{n_err} errors / {n_tot} voxels",
        )
    else:
        ax_cm.set_title("Aggregated Confusion Matrix")
        ax_cm.text(0.5, 0.5, "No CM data", ha="center", va="center")
        ax_cm.axis("off")

    #plot_aggregated_predicted_class_counts(agg_counts, ax=ax_bar)
    for ax in ax_arr:
        increase_axis_text_size(ax, delta_points=font_size_delta)

    fig.tight_layout()
    fig.savefig(out_name, dpi=300)
    print(f"Saved plots to: {out_name}")
    plt.close(fig)
    return mean_acc


def plot_classical_minus_isovalue_vs_quantum_ev(
    all_classical_minus_iso_vals,
    all_quantum_ev_vals,
    out_name,
    all_y_true=None,
    all_y_pred=None,
    class_threshold=0.0,
):
    """
    Plot classical weighted subtraction vs recovered quantum EV with optional best-fit line.
    When labels are provided, correct points are grey and classification errors are highlighted.
    
    Credit to CursorAI for the following code.
    """
    if not all_classical_minus_iso_vals or not all_quantum_ev_vals:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.set_title("Classical weighted subtraction vs Quantum ev")
        ax.text(0.5, 0.5, "No tile accuracy data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_name, dpi=300)
        print(f"Saved residual plot to: {out_name}")
        plt.close(fig)
        return

    classical_vals = np.concatenate(all_classical_minus_iso_vals).astype(float).reshape(-1)
    quantum_ev_vals = np.concatenate(all_quantum_ev_vals).astype(float).reshape(-1)
    n = min(classical_vals.size, quantum_ev_vals.size)
    classical_vals = classical_vals[:n]
    quantum_ev_vals = quantum_ev_vals[:n]

    y_true = None
    y_pred = None
    if all_y_true and all_y_pred:
        y_true = np.concatenate(all_y_true).astype(int).reshape(-1)[:n]
        y_pred = np.concatenate(all_y_pred).astype(int).reshape(-1)[:n]
        if y_true.size != n or y_pred.size != n:
            y_true = None
            y_pred = None

    min_val = float(min(np.min(quantum_ev_vals), np.min(classical_vals)))
    max_val = float(max(np.max(quantum_ev_vals), np.max(classical_vals)))
    if np.isclose(min_val, max_val):
        min_val = min_val - 0.05
        max_val = max_val + 0.05

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    if y_true is not None:
        correct = y_true == y_pred
        fp = (y_true == 0) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
        ax.scatter(
            classical_vals[correct],
            quantum_ev_vals[correct],
            color="0.62",
            alpha=0.28,
            s=10,
            label="Correct",
            zorder=1,
        )
        if np.any(fp):
            ax.scatter(
                classical_vals[fp],
                quantum_ev_vals[fp],
                color="tab:red",
                alpha=0.9,
                s=28,
                label="False outside",
                zorder=3,
            )
        if np.any(fn):
            ax.scatter(
                classical_vals[fn],
                quantum_ev_vals[fn],
                color="tab:cyan",
                alpha=0.9,
                s=28,
                label="False inside",
                zorder=3,
            )
    else:
        ax.scatter(
            classical_vals,
            quantum_ev_vals,
            color="tab:blue",
            alpha=0.85,
            label="Data points",
        )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Ideal (y = x)",
    )
    ax.axhline(
        float(class_threshold),
        color="0.35",
        linestyle=":",
        linewidth=1.0,
        label=f"EV threshold ({float(class_threshold):.2f})",
    )

    if quantum_ev_vals.size >= 2 and float(np.std(classical_vals)) > 1e-12:
        coefficients = np.polyfit(classical_vals, quantum_ev_vals, 1)
        fit_y = np.poly1d(coefficients)(classical_vals)
        ax.plot(classical_vals, fit_y, color="red", label="Line of Best Fit")
        print(f"Slope of line of best fit: {coefficients[0]:.4f}")
    else:
        mean_q = float(np.mean(quantum_ev_vals))
        ax.axhline(
            mean_q,
            color="red",
            linewidth=1.3,
            label=f"Mean quantum EV ({mean_q:.4f})",
        )
        print("Line of best fit skipped (classical values are constant).")

    ax.set_xlabel("Classical weighted subtraction value")
    ax.set_ylabel("Quantum EV")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_name, dpi=300)
    print(
        "Difference stats "
        f"(quantum_ev - weighted_subtraction): "
        f"mean={(quantum_ev_vals - classical_vals).mean():.4f}, "
        f"min={(quantum_ev_vals - classical_vals).min():.4f}, "
        f"max={(quantum_ev_vals - classical_vals).max():.4f}"
    )
    print(f"Saved residual plot to: {out_name}")
    plt.close(fig)


def plot_error_vs_isodistance(
    volume_orig,
    isolevel_orig,
    y_true,
    y_pred,
    subtraction_vals,
    quantum_ev,
    out_name,
    *,
    dataset_name="volume",
):
    """Scatter of |value − isolevel| vs analog error, with mistakes highlighted."""
    vol = np.asarray(volume_orig, dtype=np.float64).reshape(-1)
    n = vol.size
    y_true = np.asarray(y_true, dtype=int).reshape(-1)[:n]
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)[:n]
    sub = np.asarray(subtraction_vals, dtype=np.float64).reshape(-1)[:n]
    ev = np.asarray(quantum_ev, dtype=np.float64).reshape(-1)[:n]
    n = min(n, y_true.size, y_pred.size, sub.size, ev.size)
    vol, y_true, y_pred, sub, ev = vol[:n], y_true[:n], y_pred[:n], sub[:n], ev[:n]
    dist = np.abs(vol - float(isolevel_orig))
    abs_ev = np.abs(ev - sub)
    incorrect = y_true != y_pred
    correct = ~incorrect

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(
        dist[correct],
        abs_ev[correct],
        c="0.65",
        s=8,
        alpha=0.22,
        label="correct",
        zorder=1,
    )
    if np.any(incorrect):
        ax.scatter(
            dist[incorrect],
            abs_ev[incorrect],
            c="tab:red",
            s=22,
            alpha=0.9,
            label="errors",
            zorder=3,
        )
        max_err_dist = float(np.max(dist[incorrect]))
        ax.axvline(
            max_err_dist,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
            label=f"furthest misclassification ({max_err_dist:.3f})",
        )
    ax.set_xlabel("|value − isolevel|")
    ax.set_ylabel("|quantum EV − classical (value − isolevel)|")
    ax.set_title(f"{dataset_name}: quantum–classical mismatch vs distance to isosurface")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    fig.savefig(out_name, dpi=300)
    print(f"Saved error-vs-isodistance plot to: {out_name}")
    plt.close(fig)


def _npz_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(-1)[0]
        item = item.item() if hasattr(item, "item") else item
    else:
        return arr
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.str_):
        return str(item)
    return item


def load_classification_run(path):
    """Load one execution `.npz` into a plain dict of arrays and metadata."""
    with np.load(path, allow_pickle=False) as data:
        volume_orig = np.asarray(data["volume_orig"], dtype=np.float64)
        y_true = np.asarray(data["y_true"], dtype=np.int32)
        y_pred = np.asarray(data["y_pred"], dtype=np.int32)
        run = {
            "path": os.path.abspath(path),
            "volume_orig": volume_orig,
            "region_proc": np.asarray(data["region_proc"], dtype=np.float32),
            "y_true": y_true,
            "y_pred": y_pred,
            "vals": np.asarray(data["vals"], dtype=np.float64).reshape(-1),
            "subtraction_vals": np.asarray(data["subtraction_vals"], dtype=np.float64).reshape(-1),
            "quantum_ev": np.asarray(data["quantum_ev"], dtype=np.float64).reshape(-1),
            "confusion_matrix": np.asarray(data["confusion_matrix"], dtype=np.int32),
            "accuracy": float(_npz_scalar(data["accuracy"])),
            "isolevel_orig": float(_npz_scalar(data["isolevel_orig"])),
            "class_threshold": float(_npz_scalar(data["class_threshold"])),
            "compose_weight": float(_npz_scalar(data["compose_weight"])),
            "circuit_size": int(_npz_scalar(data["circuit_size"])),
            "n_circuits": int(_npz_scalar(data["n_circuits"])),
            "n_valid": int(_npz_scalar(data["n_valid"])),
            "shots_coef": int(_npz_scalar(data["shots_coef"])),
            "mean_data_rec_err": float(_npz_scalar(data["mean_data_rec_err"])),
            "iteration": int(_npz_scalar(data["iteration"])),
            "dataset_name": str(_npz_scalar(data["dataset_name"])),
            "c_mode": str(_npz_scalar(data["c_mode"])),
            "save_name": str(_npz_scalar(data["save_name"])),
            "isovalue_mode": str(_npz_scalar(data["isovalue_mode"])),
            "volume_shape": tuple(int(s) for s in volume_orig.shape),
        }
        for optional in ("preprocess_s", "classification_s", "postprocess_s"):
            if optional in data.files:
                run[optional] = float(_npz_scalar(data[optional]))
    return run


def subtraction_class_histograms(y_true, y_pred, subtraction_vals):
    """Rebuild the histogram lists used by the classification summary figure."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    sub = np.asarray(subtraction_vals, dtype=float).reshape(-1)
    n = min(y_true.size, y_pred.size, sub.size)
    y_true, y_pred, sub = y_true[:n], y_pred[:n], sub[:n]
    correct = y_true == y_pred
    return {
        "all_correct_vals": [sub[correct]] if np.any(correct) else [],
        "all_incorrect_vals": [sub[~correct]] if np.any(~correct) else [],
        "all_true_inside_vals": [sub[y_true == 0]] if np.any(y_true == 0) else [],
        "all_true_outside_vals": [sub[y_true == 1]] if np.any(y_true == 1) else [],
        "agg_counts": {
            "0": int(np.sum(y_pred == 0)),
            "1": int(np.sum(y_pred == 1)),
        },
    }


def discover_run_npz_files(runs_dir, run_paths=None):
    """Return sorted `.npz` paths from explicit files and/or a directory."""
    found: list[str] = []
    if run_paths:
        for item in run_paths:
            if os.path.isdir(item):
                found.extend(
                    os.path.join(item, name)
                    for name in sorted(os.listdir(item))
                    if name.endswith(".npz")
                )
            elif os.path.isfile(item):
                found.append(item)
            else:
                raise FileNotFoundError(f"Run path not found: {item}")
    elif runs_dir:
        if not os.path.isdir(runs_dir):
            raise FileNotFoundError(
                f"Runs directory {runs_dir!r} does not exist. "
                "Run ehands_qcrank_vertex_classification_V2.py first to save classification data."
            )
        found.extend(
            os.path.join(runs_dir, name)
            for name in sorted(os.listdir(runs_dir))
            if name.endswith(".npz")
        )
    if not found:
        raise FileNotFoundError("No classification .npz files to analyze.")
    return found


def analyze_classification_run(run, *, interactive_mesh=False):
    """Write error tables and charts for one saved classification run."""
    metrics, error_table = compute_volume_error_metrics(
        y_true=run["y_true"],
        y_pred=run["y_pred"],
        volume_orig=run["volume_orig"],
        isolevel_orig=run["isolevel_orig"],
        subtraction_vals=run["subtraction_vals"],
        quantum_ev=run["quantum_ev"],
        boundary_eps=BOUNDARY_EPS_ORIG,
    )
    hists = subtraction_class_histograms(
        run["y_true"], run["y_pred"], run["subtraction_vals"]
    )
    (
        summary_out,
        side_by_side_out,
        residual_out,
        mesh_out,
        errors_csv_out,
        error_iso_out,
    ) = classification_output_paths(
        run["dataset_name"],
        run["volume_shape"],
        run["circuit_size"],
        run["save_name"],
    )

    print(f"\nAnalyzing {run['path']}")
    print_volume_error_metrics(run["dataset_name"], metrics)
    write_error_voxels_csv(error_table, errors_csv_out)

    plot_classification_summary_figure(
        region_width=run["volume_shape"][0],
        region_height=run["volume_shape"][1],
        tile_width=run["circuit_size"],
        tile_height=1,
        acc_list=[run["accuracy"]],
        cm_list=[run["confusion_matrix"]],
        agg_counts=hists["agg_counts"],
        all_correct_vals=hists["all_correct_vals"],
        all_incorrect_vals=hists["all_incorrect_vals"],
        all_true_inside_vals=hists["all_true_inside_vals"],
        all_true_outside_vals=hists["all_true_outside_vals"],
        out_name=summary_out,
        bins=20,
        volume_shape=run["volume_shape"],
        n_circuits=run["n_circuits"],
        dataset_name=run["dataset_name"],
    )

    input_value_range = (0.0, 1.0) if run["c_mode"] == "2" else (-1.0, 1.0)
    plot_volume_slices_vs_classification(
        run["region_proc"],
        run["y_true"],
        run["y_pred"],
        out_name=side_by_side_out,
        input_value_range=input_value_range,
        dataset_name=run["dataset_name"],
    )
    mesh_stats = plot_marching_cubes_classical_vs_quantum(
        volume=run["volume_orig"],
        isolevel=run["isolevel_orig"],
        true_classes=run["y_true"],
        pred_classes=run["y_pred"],
        out_name=mesh_out,
        dataset_name=run["dataset_name"],
        interactive=interactive_mesh,
        error_table=error_table,
    ) or {}
    plot_error_vs_isodistance(
        volume_orig=run["volume_orig"],
        isolevel_orig=run["isolevel_orig"],
        y_true=run["y_true"],
        y_pred=run["y_pred"],
        subtraction_vals=run["subtraction_vals"],
        quantum_ev=run["quantum_ev"],
        out_name=error_iso_out,
        dataset_name=run["dataset_name"],
    )
    plot_classical_minus_isovalue_vs_quantum_ev(
        all_classical_minus_iso_vals=[run["subtraction_vals"]],
        all_quantum_ev_vals=[run["quantum_ev"]],
        out_name=residual_out,
        all_y_true=[np.asarray(run["y_true"]).reshape(-1)],
        all_y_pred=[np.asarray(run["y_pred"]).reshape(-1)],
        class_threshold=run["class_threshold"],
    )

    merged = dict(metrics)
    merged.update(mesh_stats)
    return merged


def _run_group_key(run):
    return (run["dataset_name"], int(run["shots_coef"]), int(run["circuit_size"]))


def write_analysis_summary_csv(rows, out_name):
    fieldnames = [
        "dataset",
        "shots_coef",
        "shot_scale_2_pow_k",
        "circuit_size",
        "n_circuits",
        "volume_shape",
        "iteration",
        "run_npz",
        "mean_accuracy",
        "avg_data_recErr",
        *ERROR_CSV_FIELDS,
    ]
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    with open(out_name, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote analysis summary to: {out_name}")


def analyze_saved_runs(npz_paths, *, interactive_mesh=False, save_name=None):
    """Load every run file, emit per-run charts, then aggregated residual/CSV."""
    runs = [load_classification_run(path) for path in npz_paths]
    csv_rows: list[dict] = []
    grouped: dict[tuple, list] = {}

    for run in runs:
        metrics = analyze_classification_run(run, interactive_mesh=interactive_mesh)
        row = {
            "dataset": run["dataset_name"],
            "shots_coef": run["shots_coef"],
            "shot_scale_2_pow_k": 2 ** int(run["shots_coef"]),
            "circuit_size": run["circuit_size"],
            "n_circuits": run["n_circuits"],
            "volume_shape": "x".join(str(s) for s in run["volume_shape"]),
            "iteration": run["iteration"],
            "run_npz": run["path"],
            "mean_accuracy": round(float(run["accuracy"]), 6),
            "avg_data_recErr": round(float(run["mean_data_rec_err"]), 6),
            **error_metrics_to_csv_fields(metrics),
        }
        csv_rows.append(row)
        grouped.setdefault(_run_group_key(run), []).append(run)

    for (dataset_name, shots_coef, circuit_size), group in grouped.items():
        stem = group[-1]["save_name"]
        residual_out = (
            f"residual_plots/{stem}_classical_minus_isovalue_vs_quantum_ev.png"
        )
        plot_classical_minus_isovalue_vs_quantum_ev(
            all_classical_minus_iso_vals=[r["subtraction_vals"] for r in group],
            all_quantum_ev_vals=[r["quantum_ev"] for r in group],
            out_name=residual_out,
            all_y_true=[np.asarray(r["y_true"]).reshape(-1) for r in group],
            all_y_pred=[np.asarray(r["y_pred"]).reshape(-1) for r in group],
            class_threshold=group[-1]["class_threshold"],
        )

    if len(runs) > 1:
        agg_stem = save_name if save_name else "volume3d"
        sc_vals = sorted({int(r["shots_coef"]) for r in runs})
        sc_tag = f"sc{sc_vals[0]}" if len(sc_vals) == 1 else "sc-mixed"
        plot_classical_minus_isovalue_vs_quantum_ev(
            all_classical_minus_iso_vals=[r["subtraction_vals"] for r in runs],
            all_quantum_ev_vals=[r["quantum_ev"] for r in runs],
            out_name=(
                f"residual_plots/{agg_stem}_{sc_tag}_tile-test"
                f"_classical_minus_isovalue_vs_quantum_ev.png"
            ),
            all_y_true=[np.asarray(r["y_true"]).reshape(-1) for r in runs],
            all_y_pred=[np.asarray(r["y_pred"]).reshape(-1) for r in runs],
        )

    csv_stem = save_name if save_name else "volume3d"
    sc_vals = sorted({int(r["shots_coef"]) for r in runs})
    if len(sc_vals) == 1:
        csv_out = f"{csv_stem}_sc{sc_vals[0]}_analysis.csv"
    else:
        csv_out = f"{csv_stem}_analysis.csv"
    write_analysis_summary_csv(csv_rows, csv_out)
    return csv_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved QCrank eHANDS 3D classification runs. "
            "Writes error CSVs, charts, and marching-cubes meshes with error glyphs. "
            "Does not run quantum circuits."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=CLASSIFICATION_RUNS_DIR,
        help="Directory of classification_runs/*.npz files (default: classification_runs).",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="Specific .npz file(s) or directories to analyze instead of --runs-dir.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Stem for the aggregated analysis CSV / residual plot (default: volume3d).",
    )
    parser.add_argument(
        "--interactive-mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each marching-cubes PNG, open a rotatable VTK window. "
            "Default is off so batch analysis can finish unattended."
        ),
    )
    args = parser.parse_args()
    npz_paths = discover_run_npz_files(args.runs_dir, args.run)
    print(f"Found {len(npz_paths)} classification run file(s).")
    analyze_saved_runs(
        npz_paths,
        interactive_mesh=bool(args.interactive_mesh),
        save_name=args.save_name,
    )

