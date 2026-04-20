"""
QCrank eHANDS classical (C) vertex classification on a tiled image — flat buffer layout.
"""

import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import sys
import re
from dotenv import load_dotenv
import os

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
        data_rec, data_recErr = self.eqd.qcrank_obj.reco_from_yields(countsL)

        shpad = n_shots / 2**self.di.nq_addr
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
) -> tuple[str, str]:
    """
    Paths under classification_summaries/ and side-by-sides/ with a unique name
    derived from the source image file, crop offset, region size, and tile size.
    """
    summary_dir = "classification_summaries"
    side_dir = "side-by-sides"
    if save_name is not None:
        return summary_dir + "/" + save_name + "_classification_summary.png", side_dir + "/" + save_name + "_image_vs_classification_side_by_side.png"
    else:
        stem = _safe_image_stem(image_path)
        tag = (
            f"{stem}_off{image_x_offset}x{image_y_offset}_{rw}x{rh}_"
            f"tile{tile_width}x{tile_height}"
        )
        return summary_dir + "/" + f"classification_summary_{tag}.png", side_dir + "/" + f"image_vs_classification_side_by_side_{tag}.png"


# -------------------------------- Test --------------------------------


def qcrank_ehands_vertex_classification_image(
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
    """
    Classify image data in non-overlapping tiles of size tile_width x tile_height.

    isovalue_mode:
      - 'auto_median': one isovalue for the whole image,
        the median of all normalized pixels in the region (constant per tile).
      - 'fixed': use the provided `isovalue` for every tile.

    inside_bias: subtracted from that isovalue (clipped to [-1, 1]). With weight 0.5,
      class 0 ('inside') requires val >= isovalue, so a positive bias lowers the
      threshold and yields more 'inside' classifications.

    The region processed starts at (image_x_offset, image_y_offset); its size is
    region_width x region_height if given, otherwise the remaining image extent.
    Edge tiles are padded to tile_width x tile_height with normalized black (-1.0) so
    padded pixels bias toward "outside" under the weighted isovalue rule; the full
    region is covered without cropping.
    """

    print("RUNNING TEST: CLASSICAL CLASSIFICATION ON IMAGE (TILED)")
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
        if 0.0 <= candidate_w <= 1.0:
            method3_weight = candidate_w
            region_proc = (region_gray + 1.0) / 2.0
            image_isovalue_proc = (image_isovalue + 1.0) / 2.0
            class_threshold = 0.5
            print(
                "Method 3 enabled: shifted x,t from [-1,1] to [0,1], "
                f"classification threshold set to 0.5, and weight set to w=1/(1-t)={method3_weight:.4f}."
            )
        else:
            if classification_mode == "3":
                raise ValueError(
                    "Method 3 was forced on, but computed w=1/(1-t)="
                    f"{candidate_w:.4f} is outside [0,1]. "
                    "Choose --classification-mode 1/auto or adjust isovalue/inside-bias."
                )
            use_method3 = False
            method3_weight = weight
            region_proc = region_gray
            image_isovalue_proc = image_isovalue
            class_threshold = 0.0
            print(
                "Method 3 skipped: computed w=1/(1-t)="
                f"{candidate_w:.4f}, which is outside [0,1]. Falling back to baseline method."
            )
    else:
        method3_weight = weight
        region_proc = region_gray
        image_isovalue_proc = image_isovalue
        class_threshold = 0.0

    canvas_h = n_ty * tile_height
    canvas_w = n_tx * tile_width
    padded_canvas_gray = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    padded_canvas_pred = np.zeros((canvas_h, canvas_w), dtype=np.int32)

    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    agg_counts = {"0": 0, "1": 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []

    tile_index = 0
    for ty in range(n_ty):
        for tx in range(n_tx):
            verbose = tile_index == 0

            x0 = tx * tile_width
            y0 = ty * tile_height
            x1 = min(x0 + tile_width, rw)
            y1 = min(y0 + tile_height, rh)
            w_sub = x1 - x0
            h_sub = y1 - y0
            pad_value = 0.0 if use_method3 else -1.0
            padded_tile = np.full((tile_height, tile_width), pad_value, dtype=np.float32)
            padded_tile[:h_sub, :w_sub] = region_proc[y0:y1, x0:x1]

            vc = VertexClassifier(0.0)
            vc.init_data(
                image_path=None,
                image_array=padded_tile,
                image_width=tile_width,
                image_height=tile_height,
                image_x_offset=0,
                image_y_offset=0,
            )

            vc.isovalue = image_isovalue_proc
            vc.classification_threshold = class_threshold
            vc.uses_method3 = use_method3

            vc.encode_c_classify(verbose)
            vc.compose_iso_qubits(method3_weight, verbose)
            vc.add_meas()

            n_shots = vc.di.n_data * (2**12)
            countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

            all_data_list, all_rec_list, data_rec, data_recErr = vc.recover_data(
                n_shots, countsL, all_data_list, all_rec_list, verbose
            )

            classifications = vc.c_classify(all_rec_list)
            comp = vc.compare_against_input(classifications, weight)

            cm_list.append(comp["confusion_matrix"])
            acc_list.append(comp["accuracy"])

            data_vals = vc.di.data_inp[:, 0, 0]
            if use_method3:
                subtraction_vals = data_vals - vc.isovalue
            else:
                subtraction_vals = method3_weight * data_vals - (1.0 - method3_weight) * vc.isovalue

            print_tile_table = n_tiles == 1 and vc.di.n_data <= 64
            if print_tile_table:
                correct_vals, incorrect_vals = print_per_datapoint_classification_table(
                    data_vals=data_vals,
                    subtraction_vals=subtraction_vals,
                    y_true=comp["y_true"],
                    y_pred=comp["y_pred"],
                )
                if correct_vals is not None:
                    all_correct_vals.append(correct_vals)
                if incorrect_vals is not None:
                    all_incorrect_vals.append(incorrect_vals)
            else:
                y_t = np.asarray(comp["y_true"]).reshape(-1)
                y_p = np.asarray(comp["y_pred"]).reshape(-1)
                sub = np.asarray(subtraction_vals).reshape(-1)
                correct_mask = y_t == y_p
                incorrect_mask = ~correct_mask
                if np.any(correct_mask):
                    all_correct_vals.append(sub[correct_mask])
                if np.any(incorrect_mask):
                    all_incorrect_vals.append(sub[incorrect_mask])

            agg_counts["0"] += int(np.sum(classifications == 0))
            agg_counts["1"] += int(np.sum(classifications == 1))

            n_pix = tile_width * tile_height
            pred_tile = (
                np.asarray(comp["y_pred"], dtype=int).reshape(-1)[:n_pix].reshape(tile_height, tile_width)
            )
            ty0 = ty * tile_height
            tx0 = tx * tile_width
            ty1 = ty0 + tile_height
            tx1 = tx0 + tile_width
            padded_canvas_gray[ty0:ty1, tx0:tx1] = padded_tile
            padded_canvas_pred[ty0:ty1, tx0:tx1] = pred_tile

            tile_index += 1

    summary_out, side_by_side_out = classification_output_paths(
        image_path,
        rw,
        rh,
        tile_width,
        tile_height,
        image_x_offset,
        image_y_offset,
        save_name,
    )
    print(f"Saving classification summary to: {summary_out}")
    print(f"Saving side-by-side figure to: {side_by_side_out}")

    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name=summary_out,
        bins=20,
    )
    plot_full_image_vs_classification(
        padded_canvas_gray,
        padded_canvas_pred,
        out_name=side_by_side_out,
        input_value_range=(0.0, 1.0) if use_method3 else (-1.0, 1.0),
        region_size_hw=(rh, rw),
        tile_size_hw=(tile_height, tile_width),
        isovalue=image_isovalue_proc,
        n_tiles=n_tiles,
    )

    print("Returning data and recovered data lists (last tile only)")
    return all_rec_list, all_data_list


# -------------------------------- Plots --------------------------------


def plot_full_image_vs_classification(
    input_image,
    predicted_image,
    out_name,
    *,
    input_value_range=(-1.0, 1.0),
    isovalue=None,
    region_size_hw=None,
    tile_size_hw=None,
    n_tiles=None,
):
    input_image = np.asarray(input_image, dtype=np.float32)
    predicted_image = np.asarray(predicted_image, dtype=np.int32)
    h, w = input_image.shape
    fig_w = min(22.0, max(10.0, w / 32.0 + 4.0))
    fig_h = min(14.0, max(5.0, h / 32.0 + 2.0))
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * fig_w, fig_h))

    ax_input, ax_pred = axes
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
            for ax in (ax_input, ax_pred):
                ov = np.zeros((h, w, 4), dtype=np.float32)
                ov[pad_mask] = (0.95, 0.15, 0.65, 0.55)
                ax.imshow(ov, origin="upper", interpolation="nearest", zorder=2)
            ax_input.set_title("Input (magenta = padded band, -1)")
        else:
            ax_input.set_title(
                "Input (no padded band — region is a multiple of tile size)"
            )
    else:
        ax_input.set_title("Input (full region, normalized grayscale)")
    ax_input.set_xlabel("x (column)")
    ax_input.set_ylabel("y (row)")
    cbar0 = fig.colorbar(
        im0, ax=ax_input, ticks=[in_vmin, 0.5 * (in_vmin + in_vmax), in_vmax], fraction=0.046, pad=0.04
    )
    cbar0.set_ticklabels([f"{in_vmin:g}", f"{0.5 * (in_vmin + in_vmax):g}", f"{in_vmax:g}"])

    im1 = ax_pred.imshow(
        predicted_image, cmap="viridis_r", vmin=0, vmax=1, origin="upper", zorder=1
    )
    if region_size_hw is not None:
        if n_pad > 0:
            ax_pred.set_title("Predicted (magenta = same padded band)")
        else:
            ax_pred.set_title(
                "Predicted (no padded band — canvas matches region)"
            )
    else:
        ax_pred.set_title("Predicted classification (stitched tiles)")
    ax_pred.set_xlabel("x (column)")
    ax_pred.set_ylabel("y (row)")
    cmap = plt.get_cmap("viridis_r")
    ax_pred.legend(
        handles=[
            Patch(facecolor=cmap(0.0), edgecolor="black", label="0 = inside"),
            Patch(facecolor=cmap(1.0), edgecolor="black", label="1 = outside"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    def _axis_ticks(n, max_ticks=17):
        if n <= max_ticks:
            return np.arange(n)
        step = max(1, int(np.ceil(n / max_ticks)))
        return np.arange(0, n, step)

    xt = _axis_ticks(w)
    yt = _axis_ticks(h)
    for ax in (ax_input, ax_pred):
        ax.set_xticks(xt)
        ax.set_yticks(yt)

    has_canvas_caption = (
        region_size_hw is not None and tile_size_hw is not None
    )
    if has_canvas_caption:
        rh_s, rw_s = int(region_size_hw[0]), int(region_size_hw[1])
        th_s, tw_s = int(tile_size_hw[0]), int(tile_size_hw[1])
        cap = (
            f"Canvas {w}×{h} px, region {rw_s}×{rh_s} px, tile {th_s}×{tw_s} px. "
            f"Padded pixels (beyond region): {n_pad}. "
            f"Number of tiles: {n_tiles}. "
            f"Isovalue: {isovalue:.3f}"
        )
        if n_pad == 0:
            cap += (
                " No extra band — width and height are multiples of the tile size, "
                "so the tile grid fills the region exactly."
            )

    if has_canvas_caption:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout()

    if has_canvas_caption:
        fig.canvas.draw()
        p_in = ax_input.get_position()
        p_pr = ax_pred.get_position()
        x_mid = (p_in.x0 + p_pr.x1) / 2.0
        fig.suptitle(cap, fontsize=9, ha="center", x=x_mid, y=0.96)

    fig.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved full image vs classification plot to: {out_name}")


def plot_correct_incorrect_input_histogram(all_correct_vals, all_incorrect_vals, bins=20, ax=None):
    if ax is None:
        ax = plt.gca()

    if not (all_correct_vals or all_incorrect_vals):
        ax.set_title("Input distribution: correct vs incorrect classifications")
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

    ax.set_xlabel("Input value after weighted subtraction")
    ax.set_ylabel("Count over all runs")
    ax.set_title("Input value distribution: correct vs incorrect classifications")
    ax.legend()


def plot_aggregated_confusion_matrix(total_cm, title="Aggregated Confusion Matrix", ax=None, cmap="Blues"):
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(total_cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)

    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    norm = im.norm
    cm = im.get_cmap()
    for i in range(2):
        for j in range(2):
            val = int(total_cm[i, j])
            rgba = cm(norm(val))
            r, g, b = rgba[0], rgba[1], rgba[2]
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
    ax.set_ylabel("Total count over all runs")
    ax.set_title("Aggregated Predicted Class Counts")


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


def plot_classification_summary_figure(acc_list, cm_list, agg_counts, all_correct_vals, all_incorrect_vals, out_name, bins=20):
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    print(f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}")

    fig, ax_arr = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}",
        fontsize=16,
    )
    ax_hist, ax_cm, ax_bar = ax_arr

    plot_correct_incorrect_input_histogram(
        all_correct_vals,
        all_incorrect_vals,
        bins=bins,
        ax=ax_hist,
    )

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

    plot_aggregated_predicted_class_counts(agg_counts, ax=ax_bar)

    fig.tight_layout()
    fig.savefig(out_name, dpi=300)
    print(f"Saved plots to: {out_name}")
    plt.close(fig)
    return mean_acc


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

    qcrank_ehands_vertex_classification_image(
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
