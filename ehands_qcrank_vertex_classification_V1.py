import argparse
import io
from contextlib import redirect_stdout
from typing import NamedTuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import sys
import re
from dotenv import load_dotenv
import os
import time

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions

print(f"Qiskit version: {qiskit.__version__}")

load_dotenv()
data_encoder_circuits_path = os.getenv("DATA_ENCODER_CIRCUITS_PATH")

if data_encoder_circuits_path is None:
    raise EnvironmentError(
        "DATA_ENCODER_CIRCUITS_PATH environment variable is not set. "
        "Please set it to the path of data-encoder-circuits."
    )

print(f"Using DATA_ENCODER_CIRCUITS_PATH from environment variable: {data_encoder_circuits_path}")
circuits_path = data_encoder_circuits_path

if circuits_path not in sys.path:
    sys.path.insert(0, circuits_path)

print(f"Added to sys.path: {circuits_path}")
print(f"Current working directory: {os.getcwd()}")

from datacircuits.ParametricQCrankV2 import ParametricQCrankV2 as QCrankV2

print("imports complete")


# -------------------------------- Data + circuit encoding --------------------------------


class DataInfo:
    __slots__ = (
        "n_data",
        "nq_addr",
        "nq_data",
        "num_q",
        "n_circuits",
        "addr_qL",
        "data_qL",
        "data_inp",
    )

    def __init__(
        self,
        data_range,
        n_circuits=1,
        isovalue=0.5,
        image_path=None,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
        image_array=None,
    ):
        self.n_data = image_width * image_height

        self.nq_addr = (self.n_data - 1).bit_length()
        self.nq_data = 1
        self.n_circuits = n_circuits
        self.data_inp = np.full(
            (2**self.nq_addr, self.nq_data, self.n_circuits), -1.0, dtype=np.float32
        )
        if image_array is not None:
            self.data_inp = self.normalized_array_to_data(
                image_array, image_width, image_height
            )
        else:
            self.data_inp = self.image_to_data(
                image_path,
                image_width,
                image_height,
                image_x_offset,
                image_y_offset,
            )

        self.data_inp[self.n_data :, :, 0] = -1.0
        self.num_q = self.nq_addr + self.nq_data
        self.addr_qL = list(range(self.nq_addr))
        self.data_qL = list(range(self.nq_addr, self.nq_addr + self.nq_data))

    def normalized_array_to_data(self, arr, image_width, image_height):
        a = np.asarray(arr, dtype=np.float32)
        if a.shape != (image_height, image_width):
            raise ValueError(
                f"image_array shape {a.shape} != ({image_height}, {image_width})"
            )
        flat = a.ravel()
        n_take = min(self.n_data, flat.shape[0])
        out = np.full(
            (2**self.nq_addr, self.nq_data, self.n_circuits), -1.0, dtype=np.float32
        )
        out[:n_take, 0, 0] = flat[:n_take]
        return out

    def image_to_data(
        self,
        image_path,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
    ):
        image = Image.open(image_path).convert("L")
        if image_x_offset < 0 or image_y_offset < 0:
            raise ValueError("image_x_offset and image_y_offset must be non-negative integers.")
        if image_x_offset >= image.width or image_y_offset >= image.height:
            raise ValueError(
                f"Offset ({image_x_offset}, {image_y_offset}) is outside image bounds "
                f"({image.width}x{image.height})."
            )

        crop_width = image_width if image_width is not None else (image.width - image_x_offset)
        crop_height = image_height if image_height is not None else (image.height - image_y_offset)
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("image_width and image_height must be positive integers.")
        if image_x_offset + crop_width > image.width or image_y_offset + crop_height > image.height:
            raise ValueError(
                f"Requested crop at ({image_x_offset}, {image_y_offset}) with size "
                f"({crop_width}x{crop_height}) exceeds image size ({image.width}x{image.height})."
            )

        image = image.crop(
            (
                image_x_offset,
                image_y_offset,
                image_x_offset + crop_width,
                image_y_offset + crop_height,
            )
        )
        data = np.asarray(image, dtype=np.float32)
        data = (data / 127.5) - 1.0

        flat = data.ravel()
        n_take = min(self.n_data, flat.shape[0])

        out = np.full(
            (2**self.nq_addr, self.nq_data, self.n_circuits), -1.0, dtype=np.float32
        )
        out[:n_take, 0, 0] = flat[:n_take]
        return out


class EncodedQData:
    __slots__ = ("qc", "qcEL", "nq_addr", "nq_data", "qcrank_obj")

    def __init__(self, di, useCZ=False, measure=True, barrier=True, verbose=False):
        self.nq_addr = di.nq_addr
        self.nq_data = di.nq_data

        self.qcrank_obj = QCrankV2(
            self.nq_addr, self.nq_data, useCZ=useCZ, measure=measure, barrier=barrier
        )

        self.qc = self.qcrank_obj.circuit

        self.qcrank_obj.bind_data(di.data_inp)

        self.qcEL = self.qcrank_obj.instantiate_circuits()

        if verbose:
            print(f"Created {len(self.qcEL)} circuits")


class VertexClassifier:
    def __init__(self, isovalue):
        self.isovalue = isovalue
        self.classification_threshold = 0.0
        self.uses_method3 = False

        self.di = None
        self.eqd = None
        self.qc_main = None

    def ehands_addition(self, qc, q_a, q_b, weight, negation=False, verbose=False):
        if not (0.0 <= float(weight) <= 1.0):
            raise ValueError(f"weight must be in range [0, 1], got {weight}")
        alpha = np.arccos(1 - 2 * weight)

        qc_add = QuantumCircuit(2)

        if negation:
            qc_add.x(1)

        qc_add.rz(pi / 2, 1)
        qc_add.cx(0, 1)
        qc_add.ry(alpha / 2, 0)
        qc_add.cx(1, 0)
        qc_add.ry(-alpha / 2, 0)

        return qc.compose(qc_add, qubits=[q_a, q_b])

    def add_iso_qubit_for_ehands_add(self, qc, data_q, placement_q, weight, negation=True, verbose=False):
        qc_iso = QuantumCircuit(1, 1)
        if self.uses_method3:
            # Method 3 uses a constant +1 contribution on the second operand.
            qc_iso.x(0)
        else:
            qc_iso.ry(np.arccos(self.isovalue), 0)

        qc.compose(qc_iso, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_addition(qc, data_q, placement_q, weight=weight, negation=negation, verbose=verbose)
        qc.barrier()

        return qc

    def init_data(
        self,
        data_range=(-0.99, 0.99),
        image_path=None,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
        image_array=None,
    ):
        self.di = DataInfo(
            data_range,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
            image_array=image_array,
        )

    def encode_c_classify(self, verbose=False):
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)

        total_q = self.di.num_q + 1
        self.qc_main = QuantumCircuit(total_q, total_q)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)

    def compose_iso_qubits(self, weight, verbose=False):
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        self.qc_main = self.add_iso_qubit_for_ehands_add(self.qc_main, q_a, q_b, weight, verbose=verbose)

    def add_meas(self):
        self.qc_main.barrier()
        self.qc_main.measure(list(range(self.di.num_q)), reversed(list(range(self.di.num_q))))

        self.eqd.qc = self.qc_main
        self.eqd.qcEL = [self.qc_main]

    def recover_data(self, n_shots, countsL, all_data_list, all_rec_list, verbose=False):
        # ParametricQCrankV2.reco_from_yields prints to stdout; mute unless verbose.
        if verbose:
            data_rec, data_recErr = self.eqd.qcrank_obj.reco_from_yields(countsL)
        else:
            with redirect_stdout(io.StringIO()):
                data_rec, data_recErr = self.eqd.qcrank_obj.reco_from_yields(countsL)

        shpad = n_shots / 2**self.di.nq_addr
        if verbose:
            print(f"Shots per address: {shpad:.1f}, relative error ~ {1/np.sqrt(shpad):.3f}")

        self.construct_data_lists(data_rec, all_data_list, all_rec_list)

        return all_data_list, all_rec_list, data_rec, data_recErr

    def construct_data_lists(self, data_rec, all_data_list, all_rec_list):
        for i in range(self.di.nq_data):
            data_slice = self.di.data_inp[:, i : i + 1, :]
            rec_slice = data_rec[:, i : i + 1, :]
            all_data_list[i].append(data_slice)
            all_rec_list[i].append(rec_slice)
        return all_data_list, all_rec_list

    def c_classify(self, all_rec_list):
        latest = [subl[-1] for subl in all_rec_list]
        rec = np.concatenate(latest, axis=1)
        classifications = np.where(rec[:, 0, 0] >= self.classification_threshold, 0, 1)
        return classifications

    def compare_against_input(self, pred_classes, weight):
        y_pred = np.asarray(pred_classes, dtype=int).reshape(-1)

        vals = self.di.data_inp[:, 0, 0]
        if self.uses_method3:
            # Method 3 is equivalent to thresholding x' against t' directly.
            subtraction_vals = vals - self.isovalue
            y_true = np.where(subtraction_vals >= 0.0, 0, 1).astype(int)
        else:
            subtraction_vals = weight * vals - (1.0 - weight) * self.isovalue
            y_true = np.where(subtraction_vals >= 0.0, 0, 1).astype(int)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"pred_classes length {y_pred.shape[0]} != n_data {y_true.shape[0]}"
            )

        accuracy = float(np.mean(y_pred == y_true))

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        cm = np.array([[tn, fp], [fn, tp]], dtype=int)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }


# -------------------------------- Simulation --------------------------------


def configure_aer_sim():
    sim = AerSimulator()
    print(sim)
    print(f"\nConfiguration: {sim.configuration()}")
    if hasattr(sim, "available_methods"):
        print(f"Available methods: {sim.available_methods()}")
    else:
        print(sim.configuration())
    if hasattr(sim, "available_devices"):
        print(f"Available devices: {sim.available_devices()}")
    else:
        print(sim.configuration())
    return sim


def configure_qcrank_sampler(sim, n_shots):
    options = SamplerOptions()
    options.default_shots = n_shots
    sampler = Sampler(mode=sim, options=options)
    return sampler, options


def run_sim_job_qcrank(eqd, sim, n_shots=2**12, verbose=False):
    sampler, options = configure_qcrank_sampler(sim, n_shots)

    job = sampler.run(tuple(eqd.qcEL))
    jobRes = job.result()

    countsL = [jobRes[0].data.c.get_counts()]

    if verbose:
        qc = eqd.qc
        cxDepth = qc.depth(filter_function=lambda x: x.operation.name == "cx")
        print(f".... PARAMETRIZED CIRCUIT .............., cx-depth={cxDepth}")
        print("Gates count:", qc.count_ops())
        fig_qc = qc.draw("mpl")
        fig_qc.savefig("qc.png")
        plt.close(fig_qc)

    return countsL


def _image_region_dimensions(
    image_path,
    image_x_offset,
    image_y_offset,
    region_width=None,
    region_height=None,
):
    image = Image.open(image_path)
    max_w = image.width - image_x_offset
    max_h = image.height - image_y_offset
    if max_w <= 0 or max_h <= 0:
        raise ValueError(
            f"Offset ({image_x_offset}, {image_y_offset}) leaves no region in image "
            f"({image.width}x{image.height})."
        )
    rw = region_width if region_width is not None else max_w
    rh = region_height if region_height is not None else max_h
    if rw <= 0 or rh <= 0:
        raise ValueError("region_width and region_height must be positive when set.")
    if rw > max_w or rh > max_h:
        raise ValueError(
            f"Requested region {rw}x{rh} from offset ({image_x_offset}, {image_y_offset}) "
            f"exceeds image bounds (available {max_w}x{max_h})."
        )
    return rw, rh


def auto_isovalue_median(normalized_pixels_1d):
    flat = np.asarray(normalized_pixels_1d, dtype=np.float64).ravel()
    v = float(np.median(flat))
    return float(np.clip(v, -1.0, 1.0))


def load_normalized_grayscale_region(image_path, image_x_offset, image_y_offset, width, height):
    image = Image.open(image_path).convert("L")
    image = image.crop(
        (
            image_x_offset,
            image_y_offset,
            image_x_offset + width,
            image_y_offset + height,
        )
    )
    data = np.asarray(image, dtype=np.float32)
    return (data / 127.5) - 1.0


def _safe_image_stem(image_path: str) -> str:
    """Filesystem-safe basename without extension for output file naming."""
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = stem.strip().strip(".")
    return stem or "image"


def classification_output_paths(
    image_path: str,
    rw: int,
    rh: int,
    tile_width: int,
    tile_height: int,
    image_x_offset: int,
    image_y_offset: int,
    save_name: str,
) -> tuple[str, str, str]:
    """
    Paths under classification_summaries/ and side-by-sides/ with a unique name
    derived from the source image file, crop offset, region size, and tile size.
    """
    summary_dir = "classification_summaries"
    side_dir = "side-by-sides"
    residual_dir = "residual_plots"
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(side_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)
    if save_name is not None:
        return (
            summary_dir + "/" + save_name + "_classification_summary.png",
            side_dir + "/" + save_name + "_image_vs_classification_side_by_side.png",
            residual_dir + "/" + save_name + "_classical_minus_isovalue_vs_quantum_ev.png",
        )
    else:
        stem = _safe_image_stem(image_path)
        tag = (
            f"{stem}_off{image_x_offset}x{image_y_offset}_{rw}x{rh}_"
            f"tile{tile_width}x{tile_height}"
        )
        return (
            summary_dir + "/" + f"classification_summary_{tag}.png",
            side_dir + "/" + f"image_vs_classification_side_by_side_{tag}.png",
            residual_dir + "/" + f"classical_minus_isovalue_vs_quantum_ev_{tag}.png",
        )


class QcrankImageTilingPreprocess(NamedTuple):
    """Region, isovalue/method settings, empty canvases and metric accumulators before the tile loop."""

    rw: int
    rh: int
    n_tx: int
    n_ty: int
    n_tiles: int
    region_proc: np.ndarray
    use_method3: bool
    method3_weight: float
    image_isovalue_proc: float
    class_threshold: float
    padded_canvas_gray: np.ndarray
    padded_canvas_true: np.ndarray
    padded_canvas_pred: np.ndarray
    all_data_list: list
    all_rec_list: list
    agg_counts: dict
    cm_list: list
    acc_list: list
    all_correct_vals: list
    all_incorrect_vals: list
    all_true_inside_vals: list
    all_true_outside_vals: list
    all_classical_minus_iso_vals: list
    all_quantum_ev_vals: list
    datapoint_table_payloads: list


def prepare_qcrank_ehands_vertex_classification_image(
    isovalue,
    weight,
    image_path,
    tile_width,
    tile_height,
    image_x_offset=0,
    image_y_offset=0,
    region_width=None,
    region_height=None,
    isovalue_mode="auto_median",
    classification_mode="auto",
    inside_bias=0,
):
    """
    Load and normalize the region, choose isovalue and method-1 vs method-3 processing,
    allocate full canvases and per-tile accumulator lists. Does not run the quantum tile loop.
    """
    print(
        f"inputs (weight: {weight}, tile: {tile_width}x{tile_height}, "
        f"region origin ({image_x_offset}, {image_y_offset}), isovalue_mode={isovalue_mode})"
    )
    if isovalue_mode == "fixed":
        print(f"  fixed isovalue (all tiles): {isovalue}")

    rw, rh = _image_region_dimensions(
        image_path, image_x_offset, image_y_offset, region_width, region_height
    )
    if rw < 1 or rh < 1:
        raise ValueError(f"Region size must be positive; got {rw}x{rh}.")
    n_tx = (rw + tile_width - 1) // tile_width
    n_ty = (rh + tile_height - 1) // tile_height

    n_tiles = n_tx * n_ty
    print(
        f"Processing {n_tiles} tiles ({n_tx}x{n_ty}) over full region {rw}x{rh} "
        f"(edge tiles padded with -1.0 to {tile_width}x{tile_height} where needed)."
    )

    region_gray = load_normalized_grayscale_region(
        image_path, image_x_offset, image_y_offset, rw, rh
    )
    if isovalue_mode == "auto_median":
        iso_before_bias = auto_isovalue_median(region_gray.ravel())
        print(f"Auto-selected isovalue (median of full region, all tiles): {iso_before_bias:.4f}")
    elif isovalue_mode == "fixed":
        iso_before_bias = float(isovalue)
    else:
        raise ValueError(
            f"Unknown isovalue_mode {isovalue_mode!r}; use 'auto_median', 'auto_median_tile', or 'fixed'."
        )

    image_isovalue = float(np.clip(iso_before_bias - inside_bias, -1.0, 1.0))
    if inside_bias != 0.0:
        print(
            f"Inside bias: effective isovalue = median/fixed ({iso_before_bias:.4f}) "
            f"- inside_bias ({inside_bias:.4f}) = {image_isovalue:.4f} (more 'inside' / class 0)"
        )

    if classification_mode not in ("auto", "1", "3"):
        raise ValueError("classification_mode must be one of: 'auto', '1', '3'.")

    if classification_mode == "auto":
        use_method3 = image_isovalue < 0.5
    elif classification_mode == "3":
        use_method3 = True
    else:
        use_method3 = False

    print(f"Classification mode: {classification_mode}")
    if use_method3:
        if image_isovalue >= 1.0:
            raise ValueError("Method 3 requires isovalue t < 1.0 for w = 1/(1-t).")
        candidate_w = 1.0 / (1.0 - image_isovalue)
        # if 0.0 <= candidate_w <= 1.0:
        t_prime = (image_isovalue + 1.0) / 2.0
        method3_weight = 1.0 / 2 * (1.0 - t_prime)
        # method3_weight = candidate_w
        region_proc = (region_gray + 1.0) / 2.0
        image_isovalue_proc = (image_isovalue + 1.0) / 2.0
        class_threshold = 0.5
        print(
            "Method 3 enabled: shifted x,t from [-1,1] to [0,1], "
            f"classification threshold set to 0.5, and weight set to w=1/(1-t)={method3_weight:.4f}."
        )
        # else:
        #     if classification_mode == "3":
        #         raise ValueError(
        #             "Method 3 was forced on, but computed w=1/(1-t)="
        #             f"{candidate_w:.4f} is outside [0,1]. "
        #             "Choose --classification-mode 1/auto or adjust isovalue/inside-bias."
        #         )
    else:
        method3_weight = weight
        region_proc = region_gray
        image_isovalue_proc = image_isovalue
        class_threshold = 0.0

    canvas_h = n_ty * tile_height
    canvas_w = n_tx * tile_width
    padded_canvas_gray = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    padded_canvas_true = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    padded_canvas_pred = np.zeros((canvas_h, canvas_w), dtype=np.int32)

    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    agg_counts = {"0": 0, "1": 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
    all_true_inside_vals = []
    all_true_outside_vals = []
    all_classical_minus_iso_vals = []
    all_quantum_ev_vals = []
    datapoint_table_payloads = []

    return QcrankImageTilingPreprocess(
        rw=rw,
        rh=rh,
        n_tx=n_tx,
        n_ty=n_ty,
        n_tiles=n_tiles,
        region_proc=region_proc,
        use_method3=use_method3,
        method3_weight=method3_weight,
        image_isovalue_proc=image_isovalue_proc,
        class_threshold=class_threshold,
        padded_canvas_gray=padded_canvas_gray,
        padded_canvas_true=padded_canvas_true,
        padded_canvas_pred=padded_canvas_pred,
        all_data_list=all_data_list,
        all_rec_list=all_rec_list,
        agg_counts=agg_counts,
        cm_list=cm_list,
        acc_list=acc_list,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        all_true_inside_vals=all_true_inside_vals,
        all_true_outside_vals=all_true_outside_vals,
        all_classical_minus_iso_vals=all_classical_minus_iso_vals,
        all_quantum_ev_vals=all_quantum_ev_vals,
        datapoint_table_payloads=datapoint_table_payloads,
    )


class ClassificationTileStitchRecord(NamedTuple):
    """One tile's grayscale input and true/pred label grids for canvas stitching."""

    ty: int
    tx: int
    padded_tile: np.ndarray
    true_tile: np.ndarray
    pred_tile: np.ndarray


def stitch_classification_tiles_into_canvases(pre, tile_width, tile_height, records):
    """
    Post-process: write each tile patch into the full-region canvases on ``pre``.
    ``prepare_qcrank_ehands_vertex_classification_image`` must have allocated the arrays.
    """
    for rec in records:
        ty0 = rec.ty * tile_height
        tx0 = rec.tx * tile_width
        ty1 = ty0 + tile_height
        tx1 = tx0 + tile_width
        pre.padded_canvas_gray[ty0:ty1, tx0:tx1] = rec.padded_tile
        pre.padded_canvas_true[ty0:ty1, tx0:tx1] = rec.true_tile
        pre.padded_canvas_pred[ty0:ty1, tx0:tx1] = rec.pred_tile


def build_classification_plot_context(
    pre,
    image_path,
    tile_width,
    tile_height,
    image_x_offset,
    image_y_offset,
    save_name,
):
    """Bundle ``pre`` state and path metadata for plotting (call after stitching)."""
    return {
        "image_path": image_path,
        "rw": pre.rw,
        "rh": pre.rh,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "image_x_offset": image_x_offset,
        "image_y_offset": image_y_offset,
        "save_name": save_name,
        "acc_list": pre.acc_list,
        "cm_list": pre.cm_list,
        "agg_counts": pre.agg_counts,
        "all_correct_vals": pre.all_correct_vals,
        "all_incorrect_vals": pre.all_incorrect_vals,
        "all_true_inside_vals": pre.all_true_inside_vals,
        "all_true_outside_vals": pre.all_true_outside_vals,
        "padded_canvas_gray": pre.padded_canvas_gray,
        "padded_canvas_true": pre.padded_canvas_true,
        "padded_canvas_pred": pre.padded_canvas_pred,
        "use_method3": pre.use_method3,
        "image_isovalue_proc": pre.image_isovalue_proc,
        "n_tiles": pre.n_tiles,
        "all_classical_minus_iso_vals": pre.all_classical_minus_iso_vals,
        "all_quantum_ev_vals": pre.all_quantum_ev_vals,
    }


# -------------------------------- Test --------------------------------
def qcrank_ehands_vertex_classification_image_driver(
    isovalue,
    weight,
    sim,
    image_path,
    tile_width,
    tile_height,
    image_x_offset=0,
    image_y_offset=0,
    region_width=None,
    region_height=None,
    isovalue_mode="auto_median",
    classification_mode="auto",
    inside_bias=0,
    save_name=None,
):

    # Thanks to CursorAI for the reorganization of the code to add timers and separate sections.
    print("RUNNING TEST: CLASSICAL CLASSIFICATION ON IMAGE (TILED)")

    ############## PREPROCESS SECTION ##############
    preprocess_start_time = time.time()
    pre = prepare_qcrank_ehands_vertex_classification_image(
        isovalue=isovalue,
        weight=weight,
        image_path=image_path,
        tile_width=tile_width,
        tile_height=tile_height,
        image_x_offset=image_x_offset,
        image_y_offset=image_y_offset,
        region_width=region_width,
        region_height=region_height,
        isovalue_mode=isovalue_mode,
        classification_mode=classification_mode,
        inside_bias=inside_bias,
    )
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - preprocess_start_time

    ############## CLASSIFICATION SECTION ##############
    classification_start_time = time.time()
    all_rec_list, all_data_list, pre, stitch_records = qcrank_ehands_vertex_classification_image(
        pre,
        weight,
        sim,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    classification_end_time = time.time()
    classification_time = classification_end_time - classification_start_time
    
    ############## POSTPROCESS SECTION ##############

    postprocess_start_time = time.time()
    stitch_classification_tiles_into_canvases(pre, tile_width, tile_height, stitch_records)
    plot_ctx = build_classification_plot_context(
        pre,
        image_path,
        tile_width,
        tile_height,
        image_x_offset,
        image_y_offset,
        save_name,
    )
    postprocess_end_time = time.time()
    postprocess_time = postprocess_end_time - postprocess_start_time

    ############## PLOT SECTION ##############
    print_datapoint_classification_tables_if_any(pre)

    summary_out, side_by_side_out, residual_out = classification_output_paths(
        plot_ctx["image_path"],
        plot_ctx["rw"],
        plot_ctx["rh"],
        plot_ctx["tile_width"],
        plot_ctx["tile_height"],
        plot_ctx["image_x_offset"],
        plot_ctx["image_y_offset"],
        plot_ctx["save_name"],
    )

    print(f"Saving classification summary to: {summary_out}")
    print(f"Saving side-by-side figure to: {side_by_side_out}")
    print(f"Saving residual plot to: {residual_out}")
    plot_classification_summary_figure(
        region_width=plot_ctx["rw"],
        region_height=plot_ctx["rh"],
        tile_width=plot_ctx["tile_width"],
        tile_height=plot_ctx["tile_height"],
        acc_list=plot_ctx["acc_list"],
        cm_list=plot_ctx["cm_list"],
        agg_counts=plot_ctx["agg_counts"],
        all_correct_vals=plot_ctx["all_correct_vals"],
        all_incorrect_vals=plot_ctx["all_incorrect_vals"],
        all_true_inside_vals=plot_ctx["all_true_inside_vals"],
        all_true_outside_vals=plot_ctx["all_true_outside_vals"],
        out_name=summary_out,
        bins=20,
    )
    plot_full_image_vs_classification(
        plot_ctx["padded_canvas_gray"],
        plot_ctx["padded_canvas_true"],
        plot_ctx["padded_canvas_pred"],
        out_name=side_by_side_out,
        input_value_range=(0.0, 1.0) if plot_ctx["use_method3"] else (-1.0, 1.0),
        region_size_hw=(plot_ctx["rh"], plot_ctx["rw"]),
    )
    plot_classical_minus_isovalue_vs_quantum_ev(
        all_classical_minus_iso_vals=plot_ctx["all_classical_minus_iso_vals"],
        all_quantum_ev_vals=plot_ctx["all_quantum_ev_vals"],
        out_name=residual_out,
    )

    ############## SUMMARY ##############

    total_time = preprocess_time + classification_time + postprocess_time

    print(f"\nPreprocess time: {preprocess_time:.2f} seconds")
    print(f"Classification time: {classification_time:.2f} seconds")
    print(f"Postprocess time: {postprocess_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")

    return all_rec_list, all_data_list


def qcrank_ehands_vertex_classification_image(pre, weight, sim, tile_width, tile_height):
    stitch_records: list[ClassificationTileStitchRecord] = []
    tile_index = 0
    for ty in range(pre.n_ty):
        for tx in range(pre.n_tx):
            #verbose = tile_index == 0
            verbose = False

            x0 = tx * tile_width
            y0 = ty * tile_height
            x1 = min(x0 + tile_width, pre.rw)
            y1 = min(y0 + tile_height, pre.rh)
            w_sub = x1 - x0
            h_sub = y1 - y0
            pad_value = 0.0 if pre.use_method3 else -1.0
            padded_tile = np.full((tile_height, tile_width), pad_value, dtype=np.float32)
            padded_tile[:h_sub, :w_sub] = pre.region_proc[y0:y1, x0:x1]

            vc = VertexClassifier(0.0)
            vc.init_data(
                image_path=None,
                image_array=padded_tile,
                image_width=tile_width,
                image_height=tile_height,
                image_x_offset=0,
                image_y_offset=0,
            )

            vc.isovalue = pre.image_isovalue_proc
            vc.classification_threshold = pre.class_threshold
            vc.uses_method3 = pre.use_method3

            vc.encode_c_classify(verbose)
            vc.compose_iso_qubits(pre.method3_weight, verbose)
            vc.add_meas()

            n_shots = vc.di.n_data * (2**12)
            countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

            vc.recover_data(n_shots, countsL, pre.all_data_list, pre.all_rec_list, verbose)

            classifications = vc.c_classify(pre.all_rec_list)
            comp = vc.compare_against_input(classifications, weight)

            pre.cm_list.append(comp["confusion_matrix"])
            pre.acc_list.append(comp["accuracy"])

            data_vals = vc.di.data_inp[:, 0, 0]
            quantum_ev_vals = np.asarray(pre.all_rec_list[0][-1])[:, 0, 0]
            if pre.use_method3:
                subtraction_vals = data_vals - vc.isovalue
            else:
                subtraction_vals = (
                    pre.method3_weight * data_vals - (1.0 - pre.method3_weight) * vc.isovalue
                )

            pre.all_quantum_ev_vals.append(np.asarray(quantum_ev_vals).reshape(-1))
            pre.all_classical_minus_iso_vals.append(np.asarray(subtraction_vals).reshape(-1))

            y_t = np.asarray(comp["y_true"]).reshape(-1)
            y_p = np.asarray(comp["y_pred"]).reshape(-1)
            sub = np.asarray(subtraction_vals).reshape(-1)
            correct_mask = y_t == y_p
            incorrect_mask = ~correct_mask
            true_inside_mask = y_t == 0
            true_outside_mask = y_t == 1

            if np.any(true_inside_mask):
                pre.all_true_inside_vals.append(sub[true_inside_mask])
            if np.any(true_outside_mask):
                pre.all_true_outside_vals.append(sub[true_outside_mask])

            if np.any(correct_mask):
                pre.all_correct_vals.append(sub[correct_mask])
            if np.any(incorrect_mask):
                pre.all_incorrect_vals.append(sub[incorrect_mask])

            # One ASCII table (first tile only) when the circuit is small enough to read.
            if tile_index == 0 and vc.di.n_data <= 64:
                pre.datapoint_table_payloads.append(
                    {
                        "data_vals": np.asarray(data_vals, dtype=np.float32).copy(),
                        "subtraction_vals": np.asarray(subtraction_vals, dtype=np.float32).copy(),
                        "y_true": np.asarray(comp["y_true"]).copy(),
                        "y_pred": np.asarray(comp["y_pred"]).copy(),
                    }
                )

            pre.agg_counts["0"] += int(np.sum(classifications == 0))
            pre.agg_counts["1"] += int(np.sum(classifications == 1))

            n_pix = tile_width * tile_height
            true_tile = (
                np.asarray(comp["y_true"], dtype=int).reshape(-1)[:n_pix].reshape(tile_height, tile_width)
            )
            pred_tile = (
                np.asarray(comp["y_pred"], dtype=int).reshape(-1)[:n_pix].reshape(tile_height, tile_width)
            )
            stitch_records.append(
                ClassificationTileStitchRecord(
                    ty=ty,
                    tx=tx,
                    padded_tile=padded_tile,
                    true_tile=true_tile,
                    pred_tile=pred_tile,
                )
            )

            tile_index += 1

    return pre.all_rec_list, pre.all_data_list, pre, stitch_records


# -------------------------------- Plots --------------------------------
# Credit to CursorAI for the following plot functions

def plot_full_image_vs_classification(
    input_image,
    true_image,
    predicted_image,
    out_name,
    *,
    input_value_range=(-1.0, 1.0),
    region_size_hw=None,
):
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

def axis_ticks(n, step=10):
    """Pixel-axis tick positions; labels only at multiples of `step` (no extra edge tick)."""
    if n <= 0:
        return np.array([], dtype=int)
    return np.asarray(list(range(0, n, step)), dtype=int)


def increase_axis_text_size(ax, delta_points=5):
    """Increase axis/legend/tick/text sizes by a fixed point delta."""
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
    """Diagonal = correct (Blues, scaled within diagonal); off-diagonal = error (Oranges, scaled within errors)."""
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
    if ax is None:
        ax = plt.gca()

    labels = ["0", "1"]
    values = [agg_counts["0"], agg_counts["1"]]
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_xlabel("Predicted class (final bit)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Aggregated Predicted Class Counts")


def plot_true_class_input_histogram(all_true_inside_vals, all_true_outside_vals, bins=20, ax=None):
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
    """Deferred per-datapoint ASCII tables (after classification; first tile only when n_data <= 64)."""
    for payload in pre.datapoint_table_payloads:
        print_per_datapoint_classification_table(**payload)


def plot_classification_summary_figure(region_width, region_height, tile_width, tile_height, acc_list, cm_list, agg_counts, all_correct_vals, all_incorrect_vals, all_true_inside_vals, all_true_outside_vals, out_name, bins=20):
    font_size_delta = 5
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    title = f"Mean accuracy over {region_width}x{region_height} region, with {len(acc_list)} ({tile_width}x{tile_height}) tiles: {mean_acc:.3f}"
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
        plot_aggregated_confusion_matrix(total_cm, ax=ax_cm)
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
):
    """
    Plot style mirrors notebook residual plotting:
      x-axis: classical weighted subtraction
              w*x - (1-w)*isovalue (or x-isovalue in Method 3)
      y-axis: quantum recovered expectation value (ev)
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

    min_val = float(min(np.min(quantum_ev_vals), np.min(classical_vals)))
    max_val = float(max(np.max(quantum_ev_vals), np.max(classical_vals)))
    if np.isclose(min_val, max_val):
        min_val = min_val - 0.05
        max_val = max_val + 0.05

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
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
    #ax.set_title("Classical weighted subtraction vs Quantum EV")
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


# -------------------------------- Main --------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QCrank eHANDS classical classification on a tiled image (flat layout)."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="test_images/Plant_tissue_sections_64x64.png",
        help="Image file (tiled over region from offset; see --region-*).",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=4,
        help="Tile width in pixels (edge tiles padded with -1.0 if needed).",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=4,
        help="Tile height in pixels (edge tiles padded with -1.0 if needed).",
    )
    parser.add_argument(
        "--image-x-offset",
        type=int,
        default=0,
        help="Left edge of crop region.",
    )
    parser.add_argument(
        "--image-y-offset",
        type=int,
        default=0,
        help="Top edge of the tiled region (upper-left corner of region).",
    )
    parser.add_argument(
        "--region-width",
        type=int,
        default=None,
        help="Width of region to cover with tiles (default: to right edge of image).",
    )
    parser.add_argument(
        "--region-height",
        type=int,
        default=None,
        help="Height of region to cover with tiles (default: to bottom edge of image).",
    )
    parser.add_argument(
        "--isovalue-mode",
        type=str,
        choices=("auto_median", "fixed"),
        default="auto_median",
        help=(
            "auto_median: median of all pixels in the region (one isovalue for every tile). "
            "fixed: use --isovalue for every tile."
        ),
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=-0.5,
        help="Isovalue when --isovalue-mode fixed.",
    )
    parser.add_argument(
        "--inside-bias",
        type=float,
        default=0.0,
        help=(
            "Subtract this from the chosen isovalue (after auto median or fixed). "
            "Positive values favor class 0 ('inside'). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--classification-mode",
        type=str,
        choices=("auto", "1", "3"),
        default="auto",
        help=(
            "Classification method selection mode. "
            "auto: enable Method 3 when isovalue < 0.5 (with safety fallback if w is invalid). "
            "1: always use Method 1 baseline. "
            "3: force Method 3 regardless of isovalue (errors if w is outside [0,1])."
        ),
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Path to save the results.",
    )
    args = parser.parse_args()

    weight = 0.5

    sim = configure_aer_sim()

    qcrank_ehands_vertex_classification_image_driver(
        args.isovalue,
        weight,
        sim,
        image_path=args.image_path,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        image_x_offset=args.image_x_offset,
        image_y_offset=args.image_y_offset,
        region_width=args.region_width,
        region_height=args.region_height,
        isovalue_mode=args.isovalue_mode,
        classification_mode=args.classification_mode,
        inside_bias=args.inside_bias,
        save_name=args.save_name,
    )
