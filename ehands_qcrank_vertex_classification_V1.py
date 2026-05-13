import argparse
import csv
import io
from contextlib import redirect_stdout
from typing import NamedTuple
from PIL import Image
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import sys
import re
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeTorino
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit.transpiler import generate_preset_pass_manager

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
        """
        Build normalized input tensor for QCrank address/data qubits.

        Accepts either an already-normalized grayscale array (`image_array`) or an image file
        (`image_path`) and produces `data_inp` with shape
        ``(2**nq_addr, nq_data, n_circuits)`` padded with -1 for unused addresses.
        """
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
        """Convert a normalized grayscale array into the QCrank `data_inp` format."""
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
        """
        Load a grayscale image crop and return it encoded into `data_inp` format.

        The crop is converted to float32 and normalized to [-1, 1]. Values are flattened
        into the address basis and padded with -1.0 to the next power-of-two length.

        Credit to CursorAI for the following code.
        """
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
        """Create a QCrank circuit and instantiate bound circuits for the given `DataInfo`."""
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
        """Classifier wrapper that builds circuits and maps recovered EVs to binary classes."""
        self.isovalue = isovalue
        self.classification_threshold = 0.0
        self.di = None
        self.eqd = None
        self.qc_main = None

    def ehands_addition(self, qc, q_a, q_b, weight, negation=False, verbose=False):
        """Compose the eHANDS weighted-addition gadget onto `qc` using qubits `q_a`, `q_b`."""
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

    def add_iso_qubit_for_ehands_add(self, qc, data_q, placement_q, weight, c_mode="1", negation=True, verbose=False):
        """Prepare an isovalue qubit (mode-dependent) and run the eHANDS addition step."""
        if c_mode == "1":
            qc_iso = QuantumCircuit(1, 1)
            qc_iso.ry(np.arccos(self.isovalue), 0)
            qc.compose(qc_iso, placement_q, inplace=True)
        else:
            negation = False

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
        """Initialize `DataInfo` from an image file or a pre-normalized tile array."""
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
        """Encode the data with QCrank and allocate the main circuit with an iso qubit slot."""
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)

        total_q = self.di.num_q + 1
        self.qc_main = QuantumCircuit(total_q, total_q)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)

    def compose_iso_qubits(self, weight, c_mode="1", verbose=False):
        """Insert the isovalue qubit preparation and addition gadget into `qc_main`."""
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        self.qc_main = self.add_iso_qubit_for_ehands_add(self.qc_main, q_a, q_b, weight, c_mode=c_mode, verbose=verbose)

    def add_meas(self):
        """Add measurement for the data/address register and finalize `EncodedQData` fields."""
        self.qc_main.barrier()
        self.qc_main.measure(list(range(self.di.num_q)), reversed(list(range(self.di.num_q))))

        self.eqd.qc = self.qc_main
        self.eqd.qcEL = [self.qc_main]

    def recover_data(self, n_shots, countsL, all_data_list, all_rec_list, verbose=False):
        """Recover expectation values from sampler counts and append to accumulator lists."""
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
        """Append the latest input and recovered slices into `all_*_list` accumulators."""
        for i in range(self.di.nq_data):
            data_slice = self.di.data_inp[:, i : i + 1, :]
            rec_slice = data_rec[:, i : i + 1, :]
            all_data_list[i].append(data_slice)
            all_rec_list[i].append(rec_slice)
        return all_data_list, all_rec_list

    def c_classify(self, all_rec_list):
        """Convert recovered EVs to classes using `classification_threshold` (0/1 labels)."""
        latest = [subl[-1] for subl in all_rec_list]
        rec = np.concatenate(latest, axis=1)
        classifications = np.where(rec[:, 0, 0] >= self.classification_threshold, 0, 1)
        return classifications

    def compare_against_input(self, pred_classes, weight):
        """Compute labels from classical weighted subtraction and compare to predictions."""
        y_pred = np.asarray(pred_classes, dtype=int).reshape(-1)

        vals = self.di.data_inp[:, 0, 0]
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
    """
    Create and print an `AerSimulator` configuration for local runs.
    
    Credit to CursorAI for the following code.
    """
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


def build_sim_backend(backend: str):
    """
    Return a simulation backend instance by name ('aer', 'fake_torino', 'fake_marrakesh').
    
    Credit to CursorAI for the following code.
    """
    key = backend.lower().replace("-", "_")
    if key == "aer":
        return configure_aer_sim()
    if key == "fake_torino":
        sim = FakeTorino()
        print(sim)
        print(f"\nConfiguration: {sim.configuration()}")
        return sim
    if key == "fake_marrakesh":
        sim = FakeMarrakesh()
        print(sim)
        print(f"\nConfiguration: {sim.configuration()}")
        return sim
    raise ValueError(
        f"Unknown backend {backend!r}; expected 'aer', 'fake_torino', or 'fake_marrakesh'."
    )


def configure_qcrank_sampler(sim, n_shots):
    """Configure a runtime `Sampler` with the given backend and shot count."""
    options = SamplerOptions()
    options.default_shots = n_shots
    sampler = Sampler(mode=sim, options=options)
    return sampler, options


def run_sim_job_qcrank(eqd, sim, n_shots=2**12, verbose=False):
    """Transpile and execute the circuit list with the runtime `Sampler` (simulation mode)."""
    sampler, options = configure_qcrank_sampler(sim, n_shots)

    qc_run = tuple(transpile(q, sim) for q in eqd.qcEL)
    job = sampler.run(qc_run)
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


def run_sim_job_qcrank_batch(circuits, sim, n_shots, verbose=False):
    """Transpile and execute a flat list of circuits in a single Sampler job (simulation)."""
    sampler, options = configure_qcrank_sampler(sim, n_shots)
    qc_run = tuple(transpile(q, sim) for q in circuits)
    job = sampler.run(qc_run)
    jobRes = job.result()
    return [jobRes[i].data.c.get_counts() for i in range(len(circuits))]


def make_json_serializable(obj):
    """
    Recursively convert common Python/numpy containers into JSON-serializable objects.
    
    Credit to CursorAI for the following code.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_job_submission_info(submission_info_dict: dict, *, output_dir: str = "JobOutputs"):
    """
    Append a hardware job submission record to `job_submission_info.json`.

    Mirrors Chris's `ehands_qcrank_qsobelV3.py`.
    
    Credit to CursorAI for the following code.
    """
    os.makedirs(output_dir, exist_ok=True)
    submission_file = os.path.join(output_dir, "job_submission_info.json")

    key = str(submission_info_dict.get("job_id", "")) or datetime.now().isoformat()
    payload = {key: make_json_serializable(submission_info_dict)}

    if os.path.exists(submission_file):
        try:
            with open(submission_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        existing = {}

    if isinstance(existing, dict):
        existing.update(payload)
    else:
        existing = payload

    with open(submission_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"Job submission info saved to {submission_file}")


def build_hardware_backend(*, account_name: str, backend_name: str):
    """
    Create a `QiskitRuntimeService` and resolve an IBM backend for hardware runs.

    Mirrors Chris's `ehands_qcrank_qsobelV3.py`.
    
    Credit to CursorAI for the following code.
    """
    if not account_name:
        raise ValueError("account_name must be provided for hardware runs.")
    if not backend_name:
        raise ValueError("backend_name must be provided for hardware runs.")
    if "ibm" not in backend_name.lower():
        raise ValueError("Hardware backend must look like 'ibm_<name>' (for example: ibm_marrakesh).")

    service = QiskitRuntimeService(name=account_name)
    backend = service.backend(backend_name)
    return service, backend


def run_hardware_job_qcrank_batch(
    circuits,
    *,
    backend,
    n_shots: int,
    rc: int = 0,
    optimization_level: int = 3,
    seed_transpiler: int | None = None,
    submit_only: bool = False,
    verbose: bool = False,
    submission_context: dict | None = None,
):
    """
    Transpile and submit a flat list of circuits in a single Sampler job on IBM hardware.

    Returns a list of count dicts (one per circuit) or a job ID string if submit_only.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be positive, got {n_shots}.")

    options = SamplerOptions()
    options.default_shots = int(n_shots)

    if rc and int(rc) > 0:
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True
        options.twirling.num_randomizations = int(rc)

    pm = generate_preset_pass_manager(
        optimization_level=int(optimization_level),
        backend=backend,
        seed_transpiler=seed_transpiler,
    )
    qc_run = [pm.run(q) for q in circuits]

    sampler = Sampler(mode=backend, options=options)
    job = sampler.run(qc_run)

    try:
        gate_count = qc_run[0].count_ops() if qc_run else {}
        depth = qc_run[0].depth() if qc_run else None
        two_qb_depth = (
            qc_run[0].depth(filter_function=lambda x: x.operation.num_qubits > 1)
            if qc_run
            else None
        )
        width = qc_run[0].num_qubits if qc_run else None
    except Exception:
        gate_count, depth, two_qb_depth, width = {}, None, None, None

    submit_info = {
        "job_id": job.job_id(),
        "local_submission_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend_name": (
            getattr(backend, "name", None)
            if not callable(getattr(backend, "name", None))
            else backend.name()
        ),
        "total_shots": int(n_shots),
        "rc": int(rc),
        "num_circuits": len(qc_run),
        "hw_opt_level": int(optimization_level),
        "hw_seed_transpiler": seed_transpiler,
        "transpiled_gate_count": gate_count,
        "transpiled_depth": depth,
        "transpiled_2qb_depth": two_qb_depth,
        "transpiled_circuit_width": width,
    }
    if submission_context:
        submit_info.update(make_json_serializable(submission_context))
    save_job_submission_info(submit_info)

    if submit_only:
        return job.job_id()

    jobRes = job.result()
    return [jobRes[i].data.c.get_counts() for i in range(len(circuits))]


def _decompress_count_keys(compressed_counts, total_qubits):
    """Reverse the compression applied by ``retrieve_save_results.ipynb``.

    The notebook's ``compress_count_keys`` reverses the Qiskit bitstring
    before converting to an integer.  This function inverts that process to
    recover the original bitstring-keyed counts dict.
    """
    decompressed: dict[str, int] = {}
    for key, value in compressed_counts.items():
        bin_key = bin(int(key))[2:].zfill(total_qubits)
        bin_key = bin_key[::-1]
        decompressed[bin_key] = value
    return decompressed


_DOUBLE_SLICE_SPAN_RE = re.compile(
    r"DoubleSliceSpan\(\s*<start='([^']+)',\s*stop='([^']+)',"
)


def _parse_double_slice_span_total_seconds(result_data: dict) -> float | None:
    """
    Sum (stop - start) over all ``DoubleSliceSpan`` entries embedded in
    ``result_data['results']['raw']`` (as saved by the retrieval notebook).

    Returns None if the raw string is missing or no spans match.

    Credit to CursorAI for the following code.
    """
    raw = None
    results_block = result_data.get("results")
    if isinstance(results_block, dict):
        raw = results_block.get("raw")
    if not raw or not isinstance(raw, str):
        return None
    spans = _DOUBLE_SLICE_SPAN_RE.findall(raw)
    if not spans:
        return None
    total = 0.0
    for start_s, stop_s in spans:
        try:
            t0 = datetime.strptime(start_s.strip(), "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(stop_s.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        total += (t1 - t0).total_seconds()
    return total


def load_hw_results_from_dir(results_dir):
    """
    Load hardware job results from a directory of ``job_results_*.json`` files.

    Cross-references ``job_submission_info.json`` (looked up in the parent
    directory first, then in *results_dir* itself) to obtain per-job tile
    metadata.

    Returns ``{tile_width: {"counts": [...], "n_shots": int, "shots_coef_k": int,
    "c_mode": str, "job_id": str, "n_tiles": int,
    "execution_s": float | None}}`` keyed by square tile edge length.
    ``execution_s`` is the sum of ``DoubleSliceSpan`` durations from the job JSON
    (IBM-reported execution window), or None if not parseable.

    Credit to CursorAI for the following code.
    """
    results_dir = os.path.normpath(results_dir)

    submission_info = {}
    for candidate in [
        os.path.join(os.path.dirname(results_dir), "job_submission_info.json"),
        os.path.join(results_dir, "job_submission_info.json"),
    ]:
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                submission_info = json.load(f)
            print(f"Loaded submission info from {candidate}")
            break
    if not submission_info:
        raise FileNotFoundError(
            "Could not find job_submission_info.json in the parent or results directory. "
            "This file is required to map job results to tile sizes."
        )

    results_by_tile_size: dict[int, dict] = {}
    for filename in sorted(os.listdir(results_dir)):
        if not filename.startswith("job_results_") or not filename.endswith(".json"):
            continue

        job_id = filename.replace("job_results_", "").replace(".json", "")
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        num_bits = result_data.get("total_qubits")
        if num_bits is None:
            print(f"WARNING: no total_qubits in {filename}; skipping")
            continue
        counts = [
            _decompress_count_keys(d, num_bits) for d in result_data["counts"]
        ]

        info = submission_info.get(job_id, {})
        tile_w = info.get("tile_width")
        if tile_w is None:
            print(f"WARNING: no submission info for job {job_id}; skipping {filename}")
            continue

        execution_s = _parse_double_slice_span_total_seconds(result_data)
        if execution_s is None:
            print(
                f"  WARNING: could not parse DoubleSliceSpan times from {filename}; "
                f"CSV avg_classification_s will use local wall-clock time."
            )

        results_by_tile_size[int(tile_w)] = {
            "counts": counts,
            "n_shots": info.get("total_shots"),
            "shots_coef_k": info.get("shots_coef_k"),
            "c_mode": info.get("classification_mode", "1"),
            "job_id": job_id,
            "n_tiles": info.get("n_tiles"),
            "execution_s": execution_s,
        }
        exec_note = f", IBM span total {execution_s:.1f}s" if execution_s is not None else ""
        n_circuits = len(counts)
        per_circuit = info.get("total_shots")
        if per_circuit is not None and n_circuits:
            total_region_shots = int(per_circuit) * n_circuits
            shot_note = (
                f"{n_circuits} circuits × {per_circuit} shots/circuit "
                f"= {total_region_shots:,} total shots (full region)"
            )
        else:
            shot_note = f"{n_circuits} circuits, {per_circuit} shots/circuit"
        print(f"  Loaded {filename}: tile {tile_w}x{tile_w}, {shot_note}{exec_note}")

    if not results_by_tile_size:
        raise FileNotFoundError(
            f"No matching job_results_*.json files found in {results_dir}"
        )

    print(
        f"Loaded hardware results for tile sizes: "
        f"{sorted(results_by_tile_size.keys())}"
    )
    return results_by_tile_size


def image_region_dimensions(
    image_path,
    image_x_offset,
    image_y_offset,
    region_width=None,
    region_height=None,
):
    """
    Validate and compute the width/height of the requested crop region.
    
    Credit to CursorAI for the following code.
    """
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
    """Pick an isovalue as the median of normalized pixel values (clipped to [-1, 1])."""
    flat = np.asarray(normalized_pixels_1d, dtype=np.float64).ravel()
    v = float(np.median(flat))
    return float(np.clip(v, -1.0, 1.0))


def load_normalized_grayscale_region(image_path, image_x_offset, image_y_offset, width, height):
    """Load a grayscale crop and normalize to [-1, 1] float32."""
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


def safe_image_stem(image_path: str) -> str:
    """
    Create a filesystem-safe stem from an image filename (for output naming).
    
    Credit to CursorAI for the following code.
    """
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
    Return (summary, side-by-side, residual) output file paths and ensure directories exist.
    
    Credit to CursorAI for the following code.
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


class QcrankImageTilingPreprocess:
    __slots__ = (
        "rw",
        "rh",
        "n_tx",
        "n_ty",
        "n_tiles",
        "region_proc",
        "compose_weight",
        "image_isovalue_proc",
        "class_threshold",
        "padded_canvas_gray",
        "padded_canvas_true",
        "padded_canvas_pred",
        "all_data_list",
        "all_rec_list",
        "agg_counts",
        "cm_list",
        "acc_list",
        "all_correct_vals",
        "all_incorrect_vals",
        "all_true_inside_vals",
        "all_true_outside_vals",
        "all_classical_minus_iso_vals",
        "all_quantum_ev_vals",
        "datapoint_table_payloads",
    )

    def __init__(
        self,
        *,
        rw,
        rh,
        n_tx,
        n_ty,
        n_tiles,
        region_proc,
        compose_weight,
        image_isovalue_proc,
        class_threshold,
        padded_canvas_gray,
        padded_canvas_true,
        padded_canvas_pred,
        all_data_list,
        all_rec_list,
        agg_counts,
        cm_list,
        acc_list,
        all_correct_vals,
        all_incorrect_vals,
        all_true_inside_vals,
        all_true_outside_vals,
        all_classical_minus_iso_vals,
        all_quantum_ev_vals,
        datapoint_table_payloads,
    ):
        """Container for region/tile preprocessing outputs and per-run accumulators."""
        self.rw = rw
        self.rh = rh
        self.n_tx = n_tx
        self.n_ty = n_ty
        self.n_tiles = n_tiles
        self.region_proc = region_proc
        self.compose_weight = compose_weight
        self.image_isovalue_proc = image_isovalue_proc
        self.class_threshold = class_threshold
        self.padded_canvas_gray = padded_canvas_gray
        self.padded_canvas_true = padded_canvas_true
        self.padded_canvas_pred = padded_canvas_pred
        self.all_data_list = all_data_list
        self.all_rec_list = all_rec_list
        self.agg_counts = agg_counts
        self.cm_list = cm_list
        self.acc_list = acc_list
        self.all_correct_vals = all_correct_vals
        self.all_incorrect_vals = all_incorrect_vals
        self.all_true_inside_vals = all_true_inside_vals
        self.all_true_outside_vals = all_true_outside_vals
        self.all_classical_minus_iso_vals = all_classical_minus_iso_vals
        self.all_quantum_ev_vals = all_quantum_ev_vals
        self.datapoint_table_payloads = datapoint_table_payloads


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
    inside_bias=0,
):
    """
    Load and normalize the region, choose isovalue, allocate full canvases and per-tile
    accumulator lists. Does not run the quantum tile loop.
    
    Credit to CursorAI for the following code.
    """
    print(
        f"inputs (weight: {weight}, tile: {tile_width}x{tile_height}, "
        f"region origin ({image_x_offset}, {image_y_offset}), isovalue_mode={isovalue_mode})"
    )
    if isovalue_mode == "fixed":
        print(f"  fixed isovalue (all tiles): {isovalue}")

    rw, rh = image_region_dimensions(
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

    compose_weight = weight
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
        compose_weight=compose_weight,
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
    ty: int
    tx: int
    padded_tile: np.ndarray
    true_tile: np.ndarray
    pred_tile: np.ndarray


def stitch_classification_tiles_into_canvases(pre, tile_width, tile_height, records):
    """
    Post-process: write each tile patch into the full-region canvases on ``pre``.
    ``prepare_qcrank_ehands_vertex_classification_image`` must have allocated the arrays.
    
    Credit to CursorAI for the following code.
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
    """Assemble a plotting context dict from preprocessing outputs and accumulators."""
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
        "image_isovalue_proc": pre.image_isovalue_proc,
        "n_tiles": pre.n_tiles,
        "all_classical_minus_iso_vals": pre.all_classical_minus_iso_vals,
        "all_quantum_ev_vals": pre.all_quantum_ev_vals,
    }


# -------------------------------- Test --------------------------------


def test_shot_count_loop_vertex_classification_image_driver(
    *,
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
    inside_bias=0,
    save_name=None,
    tile_sizes=(2, 4, 8, 16, 64),
    iterations=5,
    shots_coef=(8, 10, 12),
    c_mode="1",
    run_mode: str = "sim",
    hw_backend=None,
    hw_rc: int = 0,
    hw_opt_level: int = 3,
    hw_seed_transpiler: int | None = None,
    hw_submit_only: bool = False,
    hw_results: dict | None = None,
):
    """Run a shots-coefficient sweep over a tiled image classification experiment.

    Calls the tile-size/iteration driver for each shots exponent and writes a summary CSV.
    """
    if isinstance(shots_coef, (int, float)):
        shots_coef_iter = (int(shots_coef),)
    else:
        shots_coef_iter = tuple(int(x) for x in shots_coef)

    base_save = save_name if save_name is not None else safe_image_stem(image_path)
    tile_test_csv_rows: list[dict[str, object]] = []
    last_tile_size_mean_accuracy = 0.0

    for sc in shots_coef_iter:
        print(f"\n{'#' * 60}\nShots coefficient k = {sc}  (shot scale 2**{sc} = {2**sc})\n{'#' * 60}")
        last_tile_size_mean_accuracy, rows = test_tile_size_iteration_loop_vertex_classification_image_driver(
            sc=sc,
            isovalue=isovalue,
            weight=weight,
            sim=sim,
            image_path=image_path,
            tile_width=tile_width,
            tile_height=tile_height,
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
            region_width=region_width,
            region_height=region_height,
            isovalue_mode=isovalue_mode,
            inside_bias=inside_bias,
            save_name=base_save,
            tile_sizes=tile_sizes,
            iterations=iterations,
            c_mode=c_mode,
            run_mode=run_mode,
            hw_backend=hw_backend,
            hw_rc=hw_rc,
            hw_opt_level=hw_opt_level,
            hw_seed_transpiler=hw_seed_transpiler,
            hw_submit_only=hw_submit_only,
            hw_results=hw_results,
        )
        tile_test_csv_rows.extend(rows)

    results_path = f"{base_save}_results.csv"
    fieldnames = [
        "image_path",
        "shots_coef",
        "shot_scale_2_pow_k",
        "tile_size",
        "iterations",
        "mean_accuracy",
        "avg_preprocess_s",
        "avg_classification_s",
        "avg_postprocess_s",
        "avg_total_s",
        "total_time_5iter_s",
    ]
    with open(results_path, "w", encoding="utf-8", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tile_test_csv_rows)
    print(f"\nWrote per-(shots_coef, tile size) summary to: {results_path}")

    return last_tile_size_mean_accuracy


def test_tile_size_iteration_loop_vertex_classification_image_driver(
    *,
    sc: int,
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
    inside_bias=0,
    save_name=None,
    tile_sizes=(2, 4, 8, 16, 64),
    iterations=5,
    c_mode="1",
    run_mode: str = "sim",
    hw_backend=None,
    hw_rc: int = 0,
    hw_opt_level: int = 3,
    hw_seed_transpiler: int | None = None,
    hw_submit_only: bool = False,
    hw_results: dict | None = None,
):
    """Run a tile-size sweep with multiple iterations and emit plots/CSV per configuration."""
    base_save = save_name if save_name is not None else safe_image_stem(image_path)
    last_tile_size_mean_accuracy = 0.0
    tile_test_csv_rows: list[dict[str, object]] = []
    submitted_job_ids: list[str] = []

    # Aggregated residual data across the entire tile test (all tile sizes, all iterations).
    test_classical_minus_iso_vals: list[np.ndarray] = []
    test_quantum_ev_vals: list[np.ndarray] = []

    # Iterate over each tile size
    for tile_sz in tile_sizes:
        tw = th = int(tile_sz)
        run_save_name = f"{base_save}_sc{sc}_{tw}x{th}"
        print(f"\n{'=' * 60}\n[k={sc}] Tile size {tw}x{th}  (save stem: {run_save_name})\n{'=' * 60}")

        total_preprocess_time = 0.0
        total_classification_time = 0.0
        total_postprocess_time = 0.0
        total_mean_accuracy = 0.0

        # Per-tile-size accumulators across all iterations of this tile size.
        size_classical_minus_iso_vals: list[np.ndarray] = []
        size_quantum_ev_vals: list[np.ndarray] = []

        for i in range(iterations):
            print(f"\nIteration {i + 1}:")
            ############## PREPROCESS SECTION ##############
            preprocess_start_time = time.time()
            pre = prepare_qcrank_ehands_vertex_classification_image(
                isovalue=isovalue,
                weight=weight,
                image_path=image_path,
                tile_width=tw,
                tile_height=th,
                image_x_offset=image_x_offset,
                image_y_offset=image_y_offset,
                region_width=region_width,
                region_height=region_height,
                isovalue_mode=isovalue_mode,
                inside_bias=inside_bias,
            )
            if c_mode == "2":
                # make t'
                pre.image_isovalue_proc = (pre.image_isovalue_proc + 1) / 2

                # make x'
                pre.region_proc = (pre.region_proc + 1) / 2

                # make w'
                pre.compose_weight = 1 / (2 * (1 - pre.image_isovalue_proc))

                # set new classification threshold
                pre.class_threshold = 0.5

            preprocess_end_time = time.time()
            preprocess_time = preprocess_end_time - preprocess_start_time

            ############## CLASSIFICATION SECTION ##############
            classification_start_time = time.time()

            hw_tile_data = (hw_results or {}).get(tw)
            classification_result = qcrank_ehands_vertex_classification_image(
                pre,
                weight,
                sim,
                tile_width=tw,
                tile_height=th,
                c_mode=c_mode,
                shots_coef=hw_tile_data["shots_coef_k"] if hw_tile_data else sc,
                run_mode=run_mode,
                hw_backend=hw_backend,
                hw_rc=hw_rc,
                hw_opt_level=hw_opt_level,
                hw_seed_transpiler=hw_seed_transpiler,
                hw_submit_only=hw_submit_only,
                preloaded_counts=hw_tile_data["counts"] if hw_tile_data else None,
                preloaded_n_shots=hw_tile_data["n_shots"] if hw_tile_data else None,
            )

            if isinstance(classification_result, str):
                submitted_job_ids.append(classification_result)
                continue

            all_rec_list, all_data_list, pre, stitch_records = classification_result

            classification_end_time = time.time()
            classification_time = classification_end_time - classification_start_time

            ############## POSTPROCESS SECTION ##############
            postprocess_start_time = time.time()
            stitch_classification_tiles_into_canvases(pre, tw, th, stitch_records)
            plot_ctx = build_classification_plot_context(
                pre,
                image_path,
                tw,
                th,
                image_x_offset,
                image_y_offset,
                run_save_name,
            )
            postprocess_end_time = time.time()
            postprocess_time = postprocess_end_time - postprocess_start_time

            ############## PLOT SECTION ##############
            print_datapoint_classification_tables_if_any(pre)

            summary_out, side_by_side_out, _ = classification_output_paths(
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
            mean_accuracy = plot_classification_summary_figure(
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

            if c_mode == "1":
                input_value_range = (-1.0, 1.0)
            else:
                input_value_range = (0.0, 1.0)
            plot_full_image_vs_classification(
                plot_ctx["padded_canvas_gray"],
                plot_ctx["padded_canvas_true"],
                plot_ctx["padded_canvas_pred"],
                out_name=side_by_side_out,
                input_value_range=input_value_range,
                region_size_hw=(plot_ctx["rh"], plot_ctx["rw"]),
            )

            size_classical_minus_iso_vals.extend(plot_ctx["all_classical_minus_iso_vals"])
            size_quantum_ev_vals.extend(plot_ctx["all_quantum_ev_vals"])

            ############## SUMMARY ##############

            total_time = preprocess_time + classification_time + postprocess_time
            total_preprocess_time += preprocess_time
            total_classification_time += classification_time
            total_postprocess_time += postprocess_time
            total_mean_accuracy += mean_accuracy

            print(f"\nPreprocess time for iteration {i + 1}: {preprocess_time:.2f} seconds")
            print(f"Classification time for iteration {i + 1}: {classification_time:.2f} seconds")
            print(f"Postprocess time for iteration {i + 1}: {postprocess_time:.2f} seconds")
            print(f"Total time for iteration {i + 1}: {total_time:.2f} seconds")

        if hw_submit_only:
            continue

        n = iterations
        average_mean_accuracy = total_mean_accuracy / n
        average_preprocess_time = total_preprocess_time / n
        average_classification_time = total_classification_time / n
        hw_tile_for_csv = (hw_results or {}).get(tw)
        exec_from_json = (
            hw_tile_for_csv.get("execution_s")
            if hw_tile_for_csv is not None
            else None
        )
        if exec_from_json is not None:
            average_classification_time = float(exec_from_json)
        average_postprocess_time = total_postprocess_time / n
        average_total_time = average_preprocess_time + average_classification_time + average_postprocess_time
        if exec_from_json is not None:
            overall_total_time = (
                total_preprocess_time + float(exec_from_json) + total_postprocess_time
            )
        else:
            overall_total_time = (
                total_preprocess_time + total_classification_time + total_postprocess_time
            )
        last_tile_size_mean_accuracy = average_mean_accuracy

        print(f"\nAverage times and mean accuracy for k={sc}, tile {tw}x{th} over {n} iterations:\n")
        print(f"Mean accuracy: {average_mean_accuracy:.3f}")
        print(f"Average preprocess time: {average_preprocess_time:.2f} seconds")
        if exec_from_json is not None:
            print(
                f"Average classification time: {average_classification_time:.2f} seconds "
                f"(IBM DoubleSliceSpan total from job JSON)"
            )
        else:
            print(f"Average classification time: {average_classification_time:.2f} seconds")
        print(f"Average postprocess time: {average_postprocess_time:.2f} seconds")
        print(f"Average total time: {average_total_time:.2f} seconds")
        print(f"Total time for k={sc}, tile {tw}x{th}: {overall_total_time:.2f} seconds")

        tile_test_csv_rows.append(
            {
                "image_path": image_path,
                "shots_coef": sc,
                "shot_scale_2_pow_k": 2**sc,
                "tile_size": f"{tw}x{th}",
                "iterations": n,
                "mean_accuracy": round(average_mean_accuracy, 6),
                "avg_preprocess_s": round(average_preprocess_time, 2),
                "avg_classification_s": round(average_classification_time, 2),
                "avg_postprocess_s": round(average_postprocess_time, 2),
                "avg_total_s": round(average_total_time, 2),
                "total_time_5iter_s": round(overall_total_time, 2),
            }
        )

        # Per-tile-size residual plot: aggregates across all iterations for this tile size.
        size_residual_dir = "residual_plots"
        os.makedirs(size_residual_dir, exist_ok=True)
        size_residual_out = (
            f"{size_residual_dir}/{base_save}_sc{sc}_{tw}x{th}"
            f"_classical_minus_isovalue_vs_quantum_ev.png"
        )
        try:
            n_classical = sum(np.asarray(a).size for a in size_classical_minus_iso_vals)
            n_quantum = sum(np.asarray(a).size for a in size_quantum_ev_vals)
            print(
                f"\n[k={sc}] Tile {tw}x{th} aggregated residual plot inputs: "
                f"classical points={n_classical}, quantum points={n_quantum}, "
                f"chunks={len(size_classical_minus_iso_vals)} (over {n} iterations)"
            )
            plot_classical_minus_isovalue_vs_quantum_ev(
                all_classical_minus_iso_vals=size_classical_minus_iso_vals,
                all_quantum_ev_vals=size_quantum_ev_vals,
                out_name=size_residual_out,
            )
        except Exception as exc:
            import traceback
            print(f"!!! Residual plot generation failed for {size_residual_out}: {exc}")
            traceback.print_exc()

        # Roll this tile size's data into the test-level accumulators.
        test_classical_minus_iso_vals.extend(size_classical_minus_iso_vals)
        test_quantum_ev_vals.extend(size_quantum_ev_vals)

    if submitted_job_ids:
        print(f"\nAll hardware jobs submitted ({len(submitted_job_ids)} total):")
        for jid in submitted_job_ids:
            print(f"  {jid}")
        raise SystemExit(0)

    # Write tile-test summary CSV (matches plant_*_results.csv reference format).
    results_path = f"{base_save}_sc{sc}_results.csv"
    fieldnames = [
        "image_path",
        "tile_size",
        "iterations",
        "mean_accuracy",
        "avg_preprocess_s",
        "avg_classification_s",
        "avg_postprocess_s",
        "avg_total_s",
        "total_time_5iter_s",
    ]
    with open(results_path, "w", encoding="utf-8", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tile_test_csv_rows)
    print(f"\nWrote per-tile-size summary to: {results_path}")

    # Aggregated tile-test residual plot covering every tile size and iteration in this run.
    test_residual_dir = "residual_plots"
    os.makedirs(test_residual_dir, exist_ok=True)
    test_residual_out = (
        f"{test_residual_dir}/{base_save}_sc{sc}_tile-test"
        f"_classical_minus_isovalue_vs_quantum_ev.png"
    )
    try:
        n_classical = sum(np.asarray(a).size for a in test_classical_minus_iso_vals)
        n_quantum = sum(np.asarray(a).size for a in test_quantum_ev_vals)
        print(
            f"\nTile-test aggregated residual plot inputs: "
            f"classical points={n_classical}, quantum points={n_quantum}, "
            f"chunks={len(test_classical_minus_iso_vals)} "
            f"(over {len(tile_sizes)} tile sizes x {iterations} iterations)"
        )
        plot_classical_minus_isovalue_vs_quantum_ev(
            all_classical_minus_iso_vals=test_classical_minus_iso_vals,
            all_quantum_ev_vals=test_quantum_ev_vals,
            out_name=test_residual_out,
        )
    except Exception as exc:
        import traceback
        print(f"!!! Tile-test residual plot generation failed for {test_residual_out}: {exc}")
        traceback.print_exc()

    return last_tile_size_mean_accuracy, tile_test_csv_rows


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
    inside_bias=0,
    save_name=None,
    *,
    tile_sizes=(2, 4, 8, 16, 64),
    iterations=5,
    shots_coef=(8, 10, 12),
    c_mode="1",
    run_mode: str = "sim",
    hw_backend=None,
    hw_rc: int = 0,
    hw_opt_level: int = 3,
    hw_seed_transpiler: int | None = None,
    hw_submit_only: bool = False,
    hw_results: dict | None = None,
):
    """
    Top-level driver for the tiled image classification experiment (dispatches to tests).
    
    Thanks to CursorAI for the reorganization of the code to add timers and separate sections.
    """
    print("RUNNING TEST: CLASSICAL CLASSIFICATION ON IMAGE (TILED)")
    if isinstance(shots_coef, (int, float)):
        shots_coef_iter = (int(shots_coef),)
    else:
        shots_coef_iter = tuple(int(x) for x in shots_coef)
    print(
        f"Shots exponents k (n_shots = n_data * 2**k per tile): {shots_coef_iter}; "
        f"tile sizes: {tile_sizes}; {iterations} iterations per (k, tile size)."
    )

    return test_shot_count_loop_vertex_classification_image_driver(
        isovalue=isovalue,
        weight=weight,
        sim=sim,
        image_path=image_path,
        tile_width=tile_width,
        tile_height=tile_height,
        image_x_offset=image_x_offset,
        image_y_offset=image_y_offset,
        region_width=region_width,
        region_height=region_height,
        isovalue_mode=isovalue_mode,
        inside_bias=inside_bias,
        save_name=save_name,
        tile_sizes=tile_sizes,
        iterations=iterations,
        shots_coef=shots_coef,
        c_mode=c_mode,
        run_mode=run_mode,
        hw_backend=hw_backend,
        hw_rc=hw_rc,
        hw_opt_level=hw_opt_level,
        hw_results=hw_results,
        hw_seed_transpiler=hw_seed_transpiler,
        hw_submit_only=hw_submit_only,
    )

def qcrank_ehands_vertex_classification_image(
    pre,
    weight,
    sim,
    tile_width,
    tile_height,
    c_mode="1",
    shots_coef=12,
    *,
    run_mode: str = "sim",
    hw_backend=None,
    hw_rc: int = 0,
    hw_opt_level: int = 3,
    hw_seed_transpiler: int | None = None,
    hw_submit_only: bool = False,
    preloaded_counts: list[dict] | None = None,
    preloaded_n_shots: int | None = None,
):
    """
    Build all per-tile circuits, submit them as a single batch job, then recover EVs, classify, and accumulate.

    When *preloaded_counts* is provided (a list of count-dicts, one per tile
    circuit), Phase 2 (job submission) is skipped and the given counts are used
    directly for reconstruction and classification (Phase 3).
    
    Credit to CursorAI for helping with the batch job submission.
    """
    stitch_records: list[ClassificationTileStitchRecord] = []

    # Phase 1: Build all tile circuits (needed for QCrank reconstruction objects)
    tile_build_data: list[tuple[VertexClassifier, np.ndarray, int, int]] = []
    all_circuits: list[QuantumCircuit] = []

    for ty in range(pre.n_ty):
        for tx in range(pre.n_tx):
            x0 = tx * tile_width
            y0 = ty * tile_height
            x1 = min(x0 + tile_width, pre.rw)
            y1 = min(y0 + tile_height, pre.rh)
            w_sub = x1 - x0
            h_sub = y1 - y0
            padded_tile = np.full((tile_height, tile_width), -1.0, dtype=np.float32)
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

            vc.encode_c_classify(verbose=False)
            vc.compose_iso_qubits(pre.compose_weight, c_mode=c_mode, verbose=False)
            vc.add_meas()

            all_circuits.extend(vc.eqd.qcEL)
            tile_build_data.append((vc, padded_tile, ty, tx))

    print(f"Total circuits in batch: {len(all_circuits)}")

    # Phase 2: Obtain counts — either from preloaded hardware results or by running jobs
    n_shots = tile_build_data[0][0].di.n_data * (2**shots_coef)

    if preloaded_counts is not None:
        all_counts = preloaded_counts
        if preloaded_n_shots is not None:
            n_shots = preloaded_n_shots
        print(
            f"Using preloaded hardware counts ({len(all_counts)} circuits, "
            f"n_shots={n_shots})"
        )
    elif run_mode == "hardware":
        if hw_backend is None:
            raise ValueError("hw_backend must be provided when run_mode='hardware'.")
        result = run_hardware_job_qcrank_batch(
            all_circuits,
            backend=hw_backend,
            n_shots=n_shots,
            rc=hw_rc,
            optimization_level=hw_opt_level,
            seed_transpiler=hw_seed_transpiler,
            submit_only=hw_submit_only,
            submission_context={
                "n_tiles": len(tile_build_data),
                "tile_width": int(tile_width),
                "tile_height": int(tile_height),
                "shots_coef_k": int(shots_coef),
                "classification_mode": str(c_mode),
            },
        )
        if hw_submit_only:
            print(f"Submitted hardware job id: {result}")
            return result
        all_counts = result
    else:
        all_counts = run_sim_job_qcrank_batch(all_circuits, sim, n_shots)

    # Phase 3: Process results per-tile
    for tile_index, (vc, padded_tile, ty, tx) in enumerate(tile_build_data):
        countsL = [all_counts[tile_index]]

        vc.recover_data(n_shots, countsL, pre.all_data_list, pre.all_rec_list, verbose=False)

        classifications = vc.c_classify(pre.all_rec_list)
        comp = vc.compare_against_input(classifications, weight)

        pre.cm_list.append(comp["confusion_matrix"])
        pre.acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        quantum_ev_vals = np.asarray(pre.all_rec_list[0][-1])[:, 0, 0]
        subtraction_vals = (
            pre.compose_weight * data_vals - (1.0 - pre.compose_weight) * vc.isovalue
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


def plot_classification_summary_figure(region_width, region_height, tile_width, tile_height, acc_list, cm_list, agg_counts, all_correct_vals, all_incorrect_vals, all_true_inside_vals, all_true_outside_vals, out_name, bins=20):
    """
    Create and save the 3-panel summary figure (histograms + aggregated confusion matrix).
    
    Credit to CursorAI for the following code.
    """
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
    Plot classical weighted subtraction vs recovered quantum EV with optional best-fit line.
    
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
        "--test",
        type=str,
        choices=("full", "shots", "tile"),
        default="full",
        help=(
            "Which test driver to run. "
            "full: original full driver; "
            "shots: only the shots-coefficient loop driver; "
            "tile: only the tile-size loop driver (uses --shots-coef-k)."
        ),
    )
    parser.add_argument(
        "--shots-coef-k",
        type=int,
        default=12,
        help="Shots exponent k used when --test tile (n_shots = n_data * 2**k per tile).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run for each tile size.",
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
        default=None,
        help=(
            "Square tile edge length (pixels). If set together with --tile-height, only that "
            "size is run instead of the default multi-size tile test. Must equal --tile-height."
        ),
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Must match --tile-width when either is set (see --tile-width).",
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
        "--save-name",
        type=str,
        default=None,
        help="Path to save the results.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("aer", "fake_torino", "fake_marrakesh"),
        default="aer",
        help=(
            "Simulation backend for qiskit_ibm_runtime Sampler: "
            "aer, fake_torino, or fake_marrakesh."
        ),
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=("sim", "hardware"),
        default="sim",
        help=(
            "Execution target. "
            "sim: run via Aer/Fake backends (uses --backend). "
            "hardware: run on an IBM QPU (uses --ibm-backend and --account-name)."
        ),
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default=None,
        help=(
            "IBM backend name for --run-mode hardware (example: ibm_marrakesh). "
            "Must contain 'ibm'."
        ),
    )
    parser.add_argument(
        "--account-name",
        type=str,
        default="",
        help=(
            "QiskitRuntimeService account name (as configured in your local IBM Runtime setup). "
            "Required for --run-mode hardware."
        ),
    )
    parser.add_argument(
        "--rc",
        type=int,
        default=0,
        help=(
            "Pauli twirling/randomized compilation count for --run-mode hardware. "
            "0 disables; >0 enables twirling in Sampler options."
        ),
    )
    parser.add_argument(
        "--hw-opt-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
        help="Transpiler optimization level for --run-mode hardware.",
    )
    parser.add_argument(
        "--hw-seed-transpiler",
        type=int,
        default=None,
        help="Optional transpiler seed for --run-mode hardware.",
    )
    parser.add_argument(
        "--hw-submit-only",
        action="store_true",
        help=(
            "For --run-mode hardware: submit the job and exit after printing the job id "
            "(does not wait for results / does not generate plots)."
        ),
    )
    parser.add_argument(
        "--c-mode",
        type=str,
        choices=("1", "2"),
        default="auto",
        help=(
            "Classification method selection mode. "
            "1: Use base approach."
            "2: Use iso-weight encoding."
        ),
    )
    parser.add_argument(
        "--hw-results-dir",
        type=str,
        default=None,
        help=(
            "Directory containing job_results_*.json files from a previous hardware run. "
            "When set, skips circuit execution and uses the saved counts for post-processing. "
            "Tile sizes and shots-coef are auto-detected from the results metadata. "
            "Requires job_submission_info.json in the parent directory (JobOutputs/)."
        ),
    )
    args = parser.parse_args()

    #default_tile_sizes = (2, 4, 8, 16, 64)
    default_tile_sizes = (2, 4, 8, 16)
    iterations = args.iterations

    # --- Hardware results post-processing path ---
    hw_results = None
    if args.hw_results_dir is not None:
        hw_results = load_hw_results_from_dir(args.hw_results_dir)

        hw_tile_sizes = tuple(sorted(hw_results.keys()))
        first_entry = hw_results[hw_tile_sizes[0]]
        hw_sc = first_entry["shots_coef_k"]
        hw_c_mode = first_entry["c_mode"]

        if args.c_mode == "auto":
            args.c_mode = hw_c_mode
            print(f"Auto-detected c_mode from hardware results: {args.c_mode}")
        if args.c_mode != hw_c_mode:
            print(
                f"WARNING: --c-mode {args.c_mode} differs from hardware results "
                f"c_mode {hw_c_mode}; using --c-mode value."
            )

        print(
            f"Hardware post-processing: tile sizes {hw_tile_sizes}, "
            f"shots_coef_k={hw_sc}, c_mode={args.c_mode}, iterations=1"
        )

    tw, th = args.tile_width, args.tile_height
    if hw_results is not None:
        tile_sizes = tuple(sorted(hw_results.keys()))
        iterations = 1
    elif tw is not None or th is not None:
        if tw is None or th is None:
            parser.error("Use both --tile-width and --tile-height together, or omit both for the default tile test.")
        if tw < 1 or th < 1:
            parser.error("Tile width and height must be positive.")
        tile_sizes = (tw,)
    else:
        tile_sizes = default_tile_sizes

    weight = 0.5

    if args.c_mode == "2":
        if args.isovalue > 0.5:
            parser.error("Isovalue must be less than 0.5 for iso-weight encoding.")

    sim = None
    hw_backend = None
    if hw_results is not None:
        pass
    elif args.run_mode == "sim":
        sim = build_sim_backend(args.backend)
    else:
        if not args.ibm_backend:
            parser.error("--ibm-backend is required when --run-mode hardware.")
        if not args.account_name:
            parser.error("--account-name is required when --run-mode hardware.")
        _, hw_backend = build_hardware_backend(
            account_name=args.account_name, backend_name=args.ibm_backend
        )

    common_kwargs = dict(
        isovalue=args.isovalue,
        weight=weight,
        sim=sim,
        image_path=args.image_path,
        tile_width=tw if tw is not None else 4,
        tile_height=th if th is not None else 4,
        image_x_offset=args.image_x_offset,
        image_y_offset=args.image_y_offset,
        region_width=args.region_width,
        region_height=args.region_height,
        isovalue_mode=args.isovalue_mode,
        inside_bias=args.inside_bias,
        save_name=args.save_name,
        tile_sizes=tile_sizes,
        iterations=iterations,
        c_mode=args.c_mode,
        run_mode=args.run_mode,
        hw_backend=hw_backend,
        hw_rc=int(args.rc),
        hw_opt_level=int(args.hw_opt_level),
        hw_seed_transpiler=args.hw_seed_transpiler,
        hw_submit_only=bool(args.hw_submit_only),
        hw_results=hw_results,
    )

    if hw_results is not None:
        sc = hw_results[tile_sizes[0]]["shots_coef_k"]
        test_tile_size_iteration_loop_vertex_classification_image_driver(
            **common_kwargs,
            sc=int(sc),
        )
    elif args.test == "full":
        qcrank_ehands_vertex_classification_image_driver(**common_kwargs)
    elif args.test == "shots":
        test_shot_count_loop_vertex_classification_image_driver(
            **common_kwargs,
            shots_coef=(8, 10, 12),
        )
    else:
        test_tile_size_iteration_loop_vertex_classification_image_driver(
            **common_kwargs,
            sc=int(args.shots_coef_k),
        )
