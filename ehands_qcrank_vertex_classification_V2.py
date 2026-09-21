import argparse
import csv
import io
from contextlib import redirect_stdout
from typing import NamedTuple
from PIL import Image
import numpy as np
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

try:
    from qiskit_ibm_runtime.fake_provider import FakeMiami
except ImportError:
    FakeMiami = None
try:
    from qiskit_ibm_runtime.fake_provider import FakeBoston
except ImportError:
    FakeBoston = None
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


def _is_power_of_two(n: int) -> bool:
    return int(n) >= 1 and (int(n) & (int(n) - 1)) == 0


def flatten_and_partition_1d(values, circuit_size):
    """
    Flatten `values` to 1D and pack into QCrank circuits of `circuit_size` addresses.

    Returns ``(data_inp, n_valid, n_circuits)`` where `data_inp` has shape
    ``(circuit_size, 1, n_circuits)`` and unused tail addresses are -1.0.
    Circuit 0 holds samples ``[0:circuit_size]``, circuit 1 holds the next block, etc.
    """
    if not _is_power_of_two(circuit_size):
        raise ValueError(f"circuit_size must be a power of 2, got {circuit_size}")
    n_addr = int(circuit_size)
    flat = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), -1.0, 1.0)
    n_valid = int(flat.size)
    if n_valid < 1:
        raise ValueError("flat input must contain at least one sample.")
    n_circuits = (n_valid + n_addr - 1) // n_addr
    padded = np.full(n_circuits * n_addr, -1.0, dtype=np.float32)
    padded[:n_valid] = flat
    data_inp = np.empty((n_addr, 1, n_circuits), dtype=np.float32)
    data_inp[:, 0, :] = padded.reshape(n_circuits, n_addr).T
    return data_inp, n_valid, n_circuits


def packed_circuits_to_1d(arr, n_valid=None):
    """Undo `flatten_and_partition_1d`: ``(n_addr, n_circuits)`` → C-order 1D."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[:, 0, :]
    flat = np.asarray(a, dtype=np.float64).reshape(-1, order="F")
    if n_valid is not None:
        return flat[: int(n_valid)]
    return flat


class DataInfo:
    __slots__ = (
        "n_data",
        "n_valid",
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
        flat_data=None,
        circuit_size=None,
    ):
        """
        Build normalized input tensor for QCrank address/data qubits.

        Flattened volumes (`flat_data`) are partitioned into `circuit_size`-address
        circuits. Image inputs are flattened the same way when `circuit_size` is set;
        otherwise one circuit is used with power-of-two address padding.
        `data_inp` has shape ``(2**nq_addr, nq_data, n_circuits)``.
        """
        self.nq_data = 1
        if flat_data is not None:
            if circuit_size is None:
                n_valid = int(np.asarray(flat_data).size)
                circuit_size = 1 << (n_valid - 1).bit_length()
            self.data_inp, self.n_valid, self.n_circuits = flatten_and_partition_1d(
                flat_data, circuit_size
            )
            self.n_data = int(circuit_size)
            self.nq_addr = int(circuit_size).bit_length() - 1
        elif image_array is not None or image_path is not None:
            if image_array is not None:
                a = np.asarray(image_array, dtype=np.float32)
                if image_width is not None and image_height is not None and a.shape != (image_height, image_width):
                    raise ValueError(
                        f"image_array shape {a.shape} != ({image_height}, {image_width})"
                    )
                flat = a.reshape(-1)
            else:
                flat = self._load_image_flat(
                    image_path, image_width, image_height, image_x_offset, image_y_offset
                )
            if circuit_size is None:
                circuit_size = 1 << (max(int(flat.size), 1) - 1).bit_length()
            self.data_inp, self.n_valid, self.n_circuits = flatten_and_partition_1d(
                flat, circuit_size
            )
            self.n_data = int(circuit_size)
            self.nq_addr = int(circuit_size).bit_length() - 1
        else:
            raise ValueError("Provide flat_data, image_array, or image_path.")

        self.num_q = self.nq_addr + self.nq_data
        self.addr_qL = list(range(self.nq_addr))
        self.data_qL = list(range(self.nq_addr, self.nq_addr + self.nq_data))

    def _load_image_flat(
        self,
        image_path,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
    ):
        """Load a grayscale crop, normalize to [-1, 1], and return a 1D array."""
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
        return ((data / 127.5) - 1.0).reshape(-1)


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
        self.qcEL_main = None

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
        flat_data=None,
        circuit_size=None,
    ):
        """Initialize `DataInfo` from a flattened volume, image file, or pre-normalized array."""
        self.di = DataInfo(
            data_range,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
            image_array=image_array,
            flat_data=flat_data,
            circuit_size=circuit_size,
        )

    def encode_c_classify(self, verbose=False):
        """Encode all partitioned circuits with QCrank and allocate an iso-qubit slot on each."""
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)

        total_q = self.di.num_q + 1
        self.qcEL_main = []
        for qc_bound in self.eqd.qcEL:
            qc_main = QuantumCircuit(total_q, total_q)
            qc_main.compose(qc_bound, list(range(self.di.num_q)), inplace=True)
            self.qcEL_main.append(qc_main)
        self.qc_main = self.qcEL_main[0]

    def compose_iso_qubits(self, weight, c_mode="1", verbose=False):
        """Insert the isovalue qubit preparation and addition gadget into every circuit."""
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        composed = []
        for qc in self.qcEL_main:
            composed.append(
                self.add_iso_qubit_for_ehands_add(
                    qc, q_a, q_b, weight, c_mode=c_mode, verbose=verbose
                )
            )
        self.qcEL_main = composed
        self.qc_main = self.qcEL_main[0]

    def add_meas(self):
        """Add measurement for the data/address register on every circuit."""
        measured = []
        for qc in self.qcEL_main:
            qc.barrier()
            qc.measure(list(range(self.di.num_q)), reversed(list(range(self.di.num_q))))
            measured.append(qc)
        self.qcEL_main = measured
        self.qc_main = measured[0]
        self.eqd.qc = self.qc_main
        self.eqd.qcEL = self.qcEL_main

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
        classifications = np.where(rec[:, 0, :] >= self.classification_threshold, 0, 1)
        return packed_circuits_to_1d(classifications, n_valid=None)

    def compare_against_input(self, pred_classes, weight, n_valid=None):
        """Compute labels from classical weighted subtraction and compare to predictions."""
        y_pred = np.asarray(pred_classes, dtype=int).reshape(-1)
        vals = packed_circuits_to_1d(self.di.data_inp[:, 0, :], n_valid=None)
        subtraction_vals = weight * vals - (1.0 - weight) * self.isovalue
        y_true = np.where(subtraction_vals >= 0.0, 0, 1).astype(int)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"pred_classes length {y_pred.shape[0]} != packed length {y_true.shape[0]}"
            )

        n_keep = int(n_valid) if n_valid is not None else int(y_true.shape[0])
        y_pred = y_pred[:n_keep]
        y_true = y_true[:n_keep]
        vals = vals[:n_keep]
        subtraction_vals = subtraction_vals[:n_keep]

        accuracy = float(np.mean(y_pred == y_true)) if n_keep else float("nan")

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
            "vals": vals,
            "subtraction_vals": subtraction_vals,
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
    Return a simulation backend instance by name (
        'aer', 'fake_torino', 'fake_marrakesh', 'fake_miami', 'fake_boston'
    ).
    
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
    if key == "fake_miami":
        if FakeMiami is None:
            raise ValueError(
                "FakeMiami is not available in this qiskit-ibm-runtime version."
            )
        sim = FakeMiami()
        print(sim)
        print(f"\nConfiguration: {sim.configuration()}")
        return sim
    if key == "fake_boston":
        if FakeBoston is None:
            raise ValueError(
                "FakeBoston is not available in this qiskit-ibm-runtime version."
            )
        sim = FakeBoston()
        print(sim)
        print(f"\nConfiguration: {sim.configuration()}")
        return sim
    raise ValueError(
        f"Unknown backend {backend!r}; expected 'aer', 'fake_torino', 'fake_marrakesh', "
        "'fake_miami', or 'fake_boston'."
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


def _find_job_submission_info(results_dir: str) -> str | None:
    """Walk up from *results_dir* until ``job_submission_info.json`` is found."""
    current = os.path.abspath(results_dir)
    while True:
        candidate = os.path.join(current, "job_submission_info.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def load_hw_results_from_dir(results_dir):
    """
    Load hardware job results from a directory of ``job_results_*.json`` files.

    Cross-references ``job_submission_info.json`` (walked up from *results_dir*
    through ancestor directories, e.g. ``JobOutputs/`` for nested layouts like
    ``JobOutputs/<backend>/<shape>/``) to obtain per-job tile metadata.

    Returns ``{tile_width: {"counts": [...], "n_shots": int, "shots_coef_k": int,
    "c_mode": str, "job_id": str, "n_tiles": int,
    "execution_s": float | None}}`` keyed by square tile edge length.
    ``execution_s`` is the sum of ``DoubleSliceSpan`` durations from the job JSON
    (IBM-reported execution window), or None if not parseable.

    Credit to CursorAI for the following code.
    """
    results_dir = os.path.normpath(results_dir)

    submission_info_path = _find_job_submission_info(results_dir)
    if submission_info_path is None:
        raise FileNotFoundError(
            f"Could not find job_submission_info.json at or above {results_dir}. "
            "This file is required to map job results to tile sizes."
        )
    with open(submission_info_path, "r", encoding="utf-8") as f:
        submission_info = json.load(f)
    print(f"Loaded submission info from {submission_info_path}")

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
        tile_w = info.get("circuit_size", info.get("tile_width"))
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
                f"{n_circuits} circuits x {per_circuit} shots/circuit "
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


def filter_tile_sizes_for_region(tile_sizes, rw: int, rh: int) -> tuple[int, ...]:
    """
    Keep only square tile edges ``t`` with ``1 <= t <= rw`` and ``t <= rh``.

    Larger values would pad an entire ``t x t`` circuit tile for a smaller region
    (see ``prepare_qcrank_ehands_vertex_classification_image``); those sizes are dropped.
    """
    kept: list[int] = []
    dropped: list[int] = []
    for t in tile_sizes:
        ti = int(t)
        if ti < 1:
            dropped.append(ti)
            continue
        if ti > rw or ti > rh:
            dropped.append(ti)
            continue
        kept.append(ti)
    if dropped:
        uniq = sorted({int(x) for x in dropped})
        print(
            f"Tile-size filter for {rw}x{rh} region: removed {uniq} "
            f"(tile edge exceeds region width or height). Using {tuple(kept)}."
        )
    if not kept:
        raise ValueError(
            f"No tile sizes remain after filtering {tuple(tile_sizes)!r} for region {rw}x{rh}."
        )
    return tuple(kept)


def auto_isovalue_median(normalized_pixels_1d):
    """Pick an isovalue as the median of normalized pixel values (clipped to [-1, 1])."""
    flat = np.asarray(normalized_pixels_1d, dtype=np.float64).ravel()
    v = float(np.median(flat))
    return float(np.clip(v, -1.0, 1.0))


def make_sphere_volume(size):
    """Smooth radial field on a size^3 grid. Intensity is 1 at the center (notebook)."""
    c = (size - 1) / 2.0
    x, y, z = np.ogrid[:size, :size, :size]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    radius = size * 0.32
    return np.clip(1.0 - r / (radius * 1.25), 0.0, 1.0).astype(np.float64)


def make_torus_volume(size):
    """Smooth torus field on a size^3 grid (notebook)."""
    c = (size - 1) / 2.0
    x, y, z = np.ogrid[:size, :size, :size]
    x, y, z = x - c, y - c, z - c
    major = size * 0.28
    minor = size * 0.11
    q = np.sqrt(x**2 + y**2)
    torus_r = np.sqrt((q - major) ** 2 + z**2)
    return np.clip(1.0 - torus_r / (minor * 1.4), 0.0, 1.0).astype(np.float64)


DEFAULT_3D_DATASETS = ("sphere_32", "torus_32", "sphere_64", "torus_64")


def build_3d_volume_datasets(names=None):
    """Return the notebook sphere/torus volumes at 32^3 and 64^3, keyed by name."""
    catalog = {
        "sphere_32": lambda: make_sphere_volume(32),
        "torus_32": lambda: make_torus_volume(32),
        "sphere_64": lambda: make_sphere_volume(64),
        "torus_64": lambda: make_torus_volume(64),
    }
    requested = tuple(DEFAULT_3D_DATASETS if names is None else names)
    out = {}
    for name in requested:
        key = str(name).strip()
        if key not in catalog:
            raise ValueError(
                f"Unknown 3D dataset {key!r}; expected one of {list(catalog)}."
            )
        out[key] = catalog[key]()
    return out


def encode_volume_to_qcrank_range(volume):
    """Map a notebook [0, 1] volume into QCrank's [-1, 1] encoding range."""
    vol = np.asarray(volume, dtype=np.float64)
    return np.clip(2.0 * vol - 1.0, -1.0, 1.0).astype(np.float32)


def encoded_isovalue_to_original(iso_encoded):
    """Map an encoded [-1, 1] isovalue back to the original [0, 1] volume scale."""
    return float(np.clip((float(iso_encoded) + 1.0) / 2.0, 0.0, 1.0))


def original_isovalue_to_encoded(iso_orig):
    """Map a notebook [0, 1] isolevel into QCrank's [-1, 1] encoding range."""
    return float(np.clip(2.0 * float(iso_orig) - 1.0, -1.0, 1.0))



def filter_circuit_sizes(circuit_sizes) -> tuple[int, ...]:
    """Keep positive power-of-two circuit address counts."""
    kept: list[int] = []
    dropped: list[int] = []
    for t in circuit_sizes:
        ti = int(t)
        if not _is_power_of_two(ti) or ti < 2:
            dropped.append(ti)
            continue
        kept.append(ti)
    if dropped:
        print(
            f"Circuit-size filter: removed {sorted(set(dropped))} "
            f"(must be a power of 2, >= 2). Using {tuple(kept)}."
        )
    if not kept:
        raise ValueError(
            f"No circuit sizes remain after filtering {tuple(circuit_sizes)!r}."
        )
    return tuple(kept)


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



class QcrankImageTilingPreprocess:
    __slots__ = (
        "rw",
        "rh",
        "n_tx",
        "n_ty",
        "n_tiles",
        "n_circuits",
        "n_valid",
        "circuit_size",
        "dataset_name",
        "volume_orig",
        "volume_shape",
        "isolevel_orig",
        "region_proc",
        "compose_weight",
        "image_isovalue_proc",
        "class_threshold",
        "padded_canvas_gray",
        "padded_canvas_true",
        "padded_canvas_pred",
        "all_data_list",
        "all_rec_list",
        "cm_list",
        "acc_list",
        "vals",
        "subtraction_vals",
        "quantum_ev",
    )

    def __init__(
        self,
        *,
        rw,
        rh,
        n_tx,
        n_ty,
        n_tiles,
        n_circuits,
        n_valid,
        circuit_size,
        dataset_name,
        volume_orig,
        volume_shape,
        isolevel_orig,
        region_proc,
        compose_weight,
        image_isovalue_proc,
        class_threshold,
        padded_canvas_gray,
        padded_canvas_true,
        padded_canvas_pred,
        all_data_list,
        all_rec_list,
        cm_list,
        acc_list,
        vals,
        subtraction_vals,
        quantum_ev,
    ):
        """Container for volume preprocessing outputs and stitched classification arrays."""
        self.rw = rw
        self.rh = rh
        self.n_tx = n_tx
        self.n_ty = n_ty
        self.n_tiles = n_tiles
        self.n_circuits = n_circuits
        self.n_valid = n_valid
        self.circuit_size = circuit_size
        self.dataset_name = dataset_name
        self.volume_orig = volume_orig
        self.volume_shape = volume_shape
        self.isolevel_orig = isolevel_orig
        self.region_proc = region_proc
        self.compose_weight = compose_weight
        self.image_isovalue_proc = image_isovalue_proc
        self.class_threshold = class_threshold
        self.padded_canvas_gray = padded_canvas_gray
        self.padded_canvas_true = padded_canvas_true
        self.padded_canvas_pred = padded_canvas_pred
        self.all_data_list = all_data_list
        self.all_rec_list = all_rec_list
        self.cm_list = cm_list
        self.acc_list = acc_list
        self.vals = vals
        self.subtraction_vals = subtraction_vals
        self.quantum_ev = quantum_ev


def prepare_qcrank_ehands_vertex_classification_image(
    isovalue,
    weight,
    volume,
    circuit_size,
    dataset_name="volume",
    isovalue_mode="fixed",
    inside_bias=0,
):
    """
    Encode a 3D volume to [-1, 1], choose isovalue, and allocate classification canvases.

    The volume is not tiled spatially. Downstream classification flattens it to 1D and
    partitions the samples into circuits of `circuit_size` addresses.
    """
    volume_orig = np.asarray(volume, dtype=np.float64)
    if volume_orig.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume_orig.shape}")
    nx, ny, nz = (int(s) for s in volume_orig.shape)
    n_valid = nx * ny * nz
    n_circuits = (n_valid + int(circuit_size) - 1) // int(circuit_size)
    print(
        f"inputs (dataset={dataset_name}, shape={volume_orig.shape}, "
        f"weight={weight}, circuit_size={circuit_size}, n_circuits={n_circuits}, "
        f"isovalue_mode={isovalue_mode})"
    )

    region_proc = encode_volume_to_qcrank_range(volume_orig)
    if isovalue_mode == "auto_median":
        iso_before_bias = auto_isovalue_median(region_proc.ravel())
        print(f"Auto-selected isovalue (median of encoded volume): {iso_before_bias:.4f}")
    elif isovalue_mode == "fixed":
        iso_before_bias = float(isovalue)
        print(f"  fixed encoded isovalue: {iso_before_bias}")
    else:
        raise ValueError(
            f"Unknown isovalue_mode {isovalue_mode!r}; use 'auto_median' or 'fixed'."
        )

    image_isovalue = float(np.clip(iso_before_bias - inside_bias, -1.0, 1.0))
    isolevel_orig = encoded_isovalue_to_original(image_isovalue)
    if inside_bias != 0.0:
        print(
            f"Inside bias: effective encoded isovalue = median/fixed ({iso_before_bias:.4f}) "
            f"- inside_bias ({inside_bias:.4f}) = {image_isovalue:.4f} "
            f"(original-scale isolevel {isolevel_orig:.4f})"
        )
    else:
        print(f"Original-scale isolevel (for marching cubes): {isolevel_orig:.4f}")

    padded_canvas_gray = region_proc.copy()
    padded_canvas_true = np.zeros(volume_orig.shape, dtype=np.int32)
    padded_canvas_pred = np.zeros(volume_orig.shape, dtype=np.int32)

    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    return QcrankImageTilingPreprocess(
        rw=nx,
        rh=ny,
        n_tx=n_circuits,
        n_ty=1,
        n_tiles=n_circuits,
        n_circuits=n_circuits,
        n_valid=n_valid,
        circuit_size=int(circuit_size),
        dataset_name=str(dataset_name),
        volume_orig=volume_orig,
        volume_shape=tuple(int(s) for s in volume_orig.shape),
        isolevel_orig=isolevel_orig,
        region_proc=region_proc,
        compose_weight=weight,
        image_isovalue_proc=image_isovalue,
        class_threshold=0.0,
        padded_canvas_gray=padded_canvas_gray,
        padded_canvas_true=padded_canvas_true,
        padded_canvas_pred=padded_canvas_pred,
        all_data_list=all_data_list,
        all_rec_list=all_rec_list,
        cm_list=[],
        acc_list=[],
        vals=None,
        subtraction_vals=None,
        quantum_ev=None,
    )


class ClassificationTileStitchRecord(NamedTuple):
    ty: int
    tx: int
    padded_tile: np.ndarray
    true_tile: np.ndarray
    pred_tile: np.ndarray


def stitch_classification_tiles_into_canvases(pre, tile_width, tile_height, records):
    """No-op for flattened 3D volumes; canvases are written during classification."""
    del tile_width, tile_height, records



CLASSIFICATION_RUNS_DIR = "classification_runs"


def classification_run_npz_path(save_stem: str, iteration: int) -> str:
    """Return `classification_runs/<stem>_iterXX.npz`, creating the directory if needed."""
    os.makedirs(CLASSIFICATION_RUNS_DIR, exist_ok=True)
    return os.path.join(
        CLASSIFICATION_RUNS_DIR, f"{save_stem}_iter{int(iteration):02d}.npz"
    )


def save_classification_run(
    *,
    path,
    volume_orig,
    region_proc,
    y_true,
    y_pred,
    vals,
    subtraction_vals,
    quantum_ev,
    confusion_matrix,
    accuracy,
    isolevel_orig,
    class_threshold,
    compose_weight,
    circuit_size,
    n_circuits,
    n_valid,
    shots_coef,
    mean_data_rec_err,
    dataset_name,
    c_mode,
    save_name,
    isovalue_mode,
    iteration,
    preprocess_s=0.0,
    classification_s=0.0,
    postprocess_s=0.0,
):
    """Persist stitched 3D labels and recovered EVs for offline analysis."""
    np.savez_compressed(
        path,
        volume_orig=np.asarray(volume_orig, dtype=np.float64),
        region_proc=np.asarray(region_proc, dtype=np.float32),
        y_true=np.asarray(y_true, dtype=np.int32),
        y_pred=np.asarray(y_pred, dtype=np.int32),
        vals=np.asarray(vals, dtype=np.float64).reshape(-1),
        subtraction_vals=np.asarray(subtraction_vals, dtype=np.float64).reshape(-1),
        quantum_ev=np.asarray(quantum_ev, dtype=np.float64).reshape(-1),
        confusion_matrix=np.asarray(confusion_matrix, dtype=np.int32),
        accuracy=np.float64(accuracy),
        isolevel_orig=np.float64(isolevel_orig),
        class_threshold=np.float64(class_threshold),
        compose_weight=np.float64(compose_weight),
        circuit_size=np.int32(circuit_size),
        n_circuits=np.int32(n_circuits),
        n_valid=np.int32(n_valid),
        shots_coef=np.int32(shots_coef),
        mean_data_rec_err=np.float64(mean_data_rec_err),
        iteration=np.int32(iteration),
        preprocess_s=np.float64(preprocess_s),
        classification_s=np.float64(classification_s),
        postprocess_s=np.float64(postprocess_s),
        dataset_name=np.asarray(str(dataset_name)),
        c_mode=np.asarray(str(c_mode)),
        save_name=np.asarray(str(save_name)),
        isovalue_mode=np.asarray(str(isovalue_mode)),
        volume_shape=np.asarray(tuple(int(s) for s in np.asarray(volume_orig).shape), dtype=np.int32),
    )
    print(f"Saved classification run to: {path}")


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
    volumes: dict | None = None,
):
    """Run a shots-coefficient sweep over flattened 3D volume classification.

    Calls the circuit-size/iteration driver for each shots exponent and writes a summary CSV.
    """
    if isinstance(shots_coef, (int, float)):
        shots_coef_iter = (int(shots_coef),)
    else:
        shots_coef_iter = tuple(int(x) for x in shots_coef)

    if hw_results is None:
        tile_sizes = filter_circuit_sizes(tile_sizes)

    volumes = volumes if volumes is not None else build_3d_volume_datasets()
    base_save = save_name if save_name is not None else "volume3d"
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
            volumes=volumes,
        )
        tile_test_csv_rows.extend(rows)

    results_path = f"{base_save}_results.csv"
    fieldnames = [
        "dataset",
        "shots_coef",
        "shot_scale_2_pow_k",
        "circuit_size",
        "n_circuits",
        "volume_shape",
        "iterations",
        "iterations_completed",
        "mean_accuracy",
        "mean_accuracy_sem",
        "avg_data_recErr",
        "avg_preprocess_s",
        "avg_classification_s",
        "avg_postprocess_s",
        "avg_total_s",
        "total_time_5iter_s",
        "run_npz",
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
    volumes: dict | None = None,
):
    """Run a circuit-size sweep over flattened 3D volumes and save stitched run data."""
    volumes = volumes if volumes is not None else build_3d_volume_datasets()
    base_save = save_name if save_name is not None else "volume3d"
    last_tile_size_mean_accuracy = 0.0
    tile_test_csv_rows: list[dict[str, object]] = []
    submitted_job_ids: list[str] = []

    jobs: list[tuple[str, np.ndarray, int]] = []
    for dataset_name, volume in volumes.items():
        for tile_sz in tile_sizes:
            jobs.append((str(dataset_name), np.asarray(volume), int(tile_sz)))

    for dataset_name, volume, tw in jobs:
        th = tw
        run_save_name = f"{base_save}_{dataset_name}_sc{sc}_circ{tw}"
        print(
            f"\n{'=' * 60}\n[k={sc}] {dataset_name} {tuple(volume.shape)} "
            f"circuit_size={tw}  (save stem: {run_save_name})\n{'=' * 60}"
        )

        total_preprocess_time = 0.0
        total_classification_time = 0.0
        total_postprocess_time = 0.0
        total_mean_data_rec_err = 0.0
        mean_accuracy_per_iteration: list[float] = []
        run_npz_paths: list[str] = []

        tile_aborted = False
        for i in range(iterations):
            print(f"\nIteration {i + 1}:")
            try:
                ############## PREPROCESS SECTION ##############
                preprocess_start_time = time.time()
                pre = prepare_qcrank_ehands_vertex_classification_image(
                    isovalue=isovalue,
                    weight=weight,
                    volume=volume,
                    circuit_size=tw,
                    dataset_name=dataset_name,
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

                all_rec_list, all_data_list, pre, stitch_records, mean_data_rec_err = (
                    classification_result
                )

                classification_end_time = time.time()
                classification_time = classification_end_time - classification_start_time

                ############## POSTPROCESS SECTION ##############
                postprocess_start_time = time.time()
                stitch_classification_tiles_into_canvases(pre, tw, th, stitch_records)
                run_npz = classification_run_npz_path(run_save_name, i + 1)
                postprocess_end_time = time.time()
                postprocess_time = postprocess_end_time - postprocess_start_time
                save_classification_run(
                    path=run_npz,
                    volume_orig=pre.volume_orig,
                    region_proc=pre.region_proc,
                    y_true=pre.padded_canvas_true,
                    y_pred=pre.padded_canvas_pred,
                    vals=pre.vals,
                    subtraction_vals=pre.subtraction_vals,
                    quantum_ev=pre.quantum_ev,
                    confusion_matrix=pre.cm_list[-1],
                    accuracy=pre.acc_list[-1],
                    isolevel_orig=pre.isolevel_orig,
                    class_threshold=pre.class_threshold,
                    compose_weight=pre.compose_weight,
                    circuit_size=pre.circuit_size,
                    n_circuits=pre.n_circuits,
                    n_valid=pre.n_valid,
                    shots_coef=hw_tile_data["shots_coef_k"] if hw_tile_data else sc,
                    mean_data_rec_err=mean_data_rec_err,
                    dataset_name=pre.dataset_name,
                    c_mode=c_mode,
                    save_name=run_save_name,
                    isovalue_mode=isovalue_mode,
                    iteration=i + 1,
                    preprocess_s=preprocess_time,
                    classification_s=classification_time,
                    postprocess_s=postprocess_time,
                )
                run_npz_paths.append(run_npz)

                ############## SUMMARY ##############

                total_time = preprocess_time + classification_time + postprocess_time
                total_preprocess_time += preprocess_time
                total_classification_time += classification_time
                total_postprocess_time += postprocess_time
                mean_accuracy = float(pre.acc_list[-1]) if pre.acc_list else 0.0
                mean_accuracy_per_iteration.append(float(mean_accuracy))
                total_mean_data_rec_err += mean_data_rec_err

                print(f"\nPreprocess time for iteration {i + 1}: {preprocess_time:.2f} seconds")
                print(f"Classification time for iteration {i + 1}: {classification_time:.2f} seconds")
                print(f"Postprocess time for iteration {i + 1}: {postprocess_time:.2f} seconds")
                print(f"Total time for iteration {i + 1}: {total_time:.2f} seconds")
            except Exception as exc:
                import traceback

                tile_aborted = True
                print(
                    f"\n!!! Error during {dataset_name} circuit_size={tw} (k={sc}), iteration {i + 1}: {exc}\n"
                    f"    Skipping remaining iterations for this circuit size; "
                    f"run will continue with other datasets/circuit sizes."
                )
                traceback.print_exc()
                break

        if hw_submit_only:
            continue

        n = iterations
        ni = len(mean_accuracy_per_iteration)
        if tile_aborted or ni != n:
            if ni == 0:
                print(
                    f"\n{dataset_name} circuit_size={tw}: no iterations completed successfully; "
                    f"writing a CSV row with zeroed metrics (planned {n} iterations)."
                )
            else:
                print(
                    f"\n{dataset_name} circuit_size={tw} cancelled or incomplete: {ni}/{n} iterations completed; "
                    f"writing CSV row from completed iterations only."
                )

        average_mean_accuracy = float(np.mean(mean_accuracy_per_iteration)) if ni else 0.0
        if ni > 1:
            mean_accuracy_sem = float(
                np.std(mean_accuracy_per_iteration, ddof=1) / np.sqrt(ni)
            )
        else:
            mean_accuracy_sem = 0.0
        average_mean_data_rec_err = (total_mean_data_rec_err / ni) if ni else 0.0
        average_preprocess_time = (total_preprocess_time / ni) if ni else 0.0
        average_classification_time = (total_classification_time / ni) if ni else 0.0
        hw_tile_for_csv = (hw_results or {}).get(tw)
        exec_from_json = (
            hw_tile_for_csv.get("execution_s")
            if hw_tile_for_csv is not None
            else None
        )
        if exec_from_json is not None and ni > 0:
            average_classification_time = float(exec_from_json)
        average_postprocess_time = (total_postprocess_time / ni) if ni else 0.0
        average_total_time = average_preprocess_time + average_classification_time + average_postprocess_time
        if exec_from_json is not None and ni > 0:
            overall_total_time = (
                total_preprocess_time + float(exec_from_json) + total_postprocess_time
            )
        else:
            overall_total_time = (
                total_preprocess_time + total_classification_time + total_postprocess_time
            )
        if ni > 0:
            last_tile_size_mean_accuracy = average_mean_accuracy

        iter_label = f"{ni} completed (of {n} planned)" if ni != n else f"{n}"
        n_valid_vol = int(np.prod(volume.shape))
        n_circuits_csv = (n_valid_vol + tw - 1) // tw
        print(
            f"\nAverage times and mean accuracy for k={sc}, {dataset_name} "
            f"circuit_size={tw} over {iter_label} iterations:\n"
        )
        print(f"Mean accuracy: {average_mean_accuracy:.3f} (SEM over iterations: {mean_accuracy_sem:.4g})")
        print(f"Average mean data_recErr (over tiles, then iterations): {average_mean_data_rec_err:.6g}")
        print(f"Average preprocess time: {average_preprocess_time:.2f} seconds")
        if exec_from_json is not None and ni > 0:
            print(
                f"Average classification time: {average_classification_time:.2f} seconds "
                f"(IBM DoubleSliceSpan total from job JSON)"
            )
        else:
            print(f"Average classification time: {average_classification_time:.2f} seconds")
        print(f"Average postprocess time: {average_postprocess_time:.2f} seconds")
        print(f"Average total time: {average_total_time:.2f} seconds")
        print(f"Total time for k={sc}, {dataset_name} circuit_size={tw}: {overall_total_time:.2f} seconds")

        tile_test_csv_rows.append(
            {
                "dataset": dataset_name,
                "shots_coef": sc,
                "shot_scale_2_pow_k": 2**sc,
                "circuit_size": tw,
                "n_circuits": n_circuits_csv,
                "volume_shape": "x".join(str(s) for s in volume.shape),
                "iterations": n,
                "iterations_completed": ni,
                "mean_accuracy": round(average_mean_accuracy, 6),
                "mean_accuracy_sem": round(mean_accuracy_sem, 6),
                "avg_data_recErr": round(average_mean_data_rec_err, 6),
                "avg_preprocess_s": round(average_preprocess_time, 2),
                "avg_classification_s": round(average_classification_time, 2),
                "avg_postprocess_s": round(average_postprocess_time, 2),
                "avg_total_s": round(average_total_time, 2),
                "total_time_5iter_s": round(overall_total_time, 2),
                "run_npz": ";".join(run_npz_paths),
            }
        )

    if submitted_job_ids:
        print(f"\nAll hardware jobs submitted ({len(submitted_job_ids)} total):")
        for jid in submitted_job_ids:
            print(f"  {jid}")
        raise SystemExit(0)

    # Write tile-test summary CSV (matches plant_*_results.csv reference format).
    results_path = f"{base_save}_sc{sc}_results.csv"
    fieldnames = [
        "dataset",
        "circuit_size",
        "n_circuits",
        "volume_shape",
        "iterations",
        "iterations_completed",
        "mean_accuracy",
        "mean_accuracy_sem",
        "avg_data_recErr",
        "avg_preprocess_s",
        "avg_classification_s",
        "avg_postprocess_s",
        "avg_total_s",
        "total_time_5iter_s",
        "run_npz",
    ]
    with open(results_path, "w", encoding="utf-8", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tile_test_csv_rows)
    print(f"\nWrote per-tile-size summary to: {results_path}")
    print(
        "Classification data saved under classification_runs/. "
        "Run ehands_qcrank_vertex_classification_V2_analysis.py to generate charts "
        "and error analysis without re-executing circuits."
    )

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
    volumes: dict | None = None,
):
    """
    Top-level driver for 3D volume classification (flatten to 1D, partition into circuits).
    """
    print("RUNNING TEST: VERTEX CLASSIFICATION ON 3D VOLUMES (FLATTENED CIRCUITS)")
    if isinstance(shots_coef, (int, float)):
        shots_coef_iter = (int(shots_coef),)
    else:
        shots_coef_iter = tuple(int(x) for x in shots_coef)

    print(
        f"Shots exponents k (n_shots = circuit_size * 2**k per circuit): {shots_coef_iter}; "
        f"circuit sizes: {tile_sizes}; {iterations} iterations per (k, dataset, circuit size)."
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
        volumes=volumes,
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
    Flatten the 3D volume to 1D, partition into QCrank circuits, run one batch job,
    then recover EVs, classify, and reshape labels back onto the volume.

    `tile_width` is treated as the per-circuit address count (`circuit_size`).
    """
    del tile_height
    circuit_size = int(pre.circuit_size if pre.circuit_size else tile_width)

    vc = VertexClassifier(0.0)
    vc.init_data(
        flat_data=np.asarray(pre.region_proc, dtype=np.float32).reshape(-1),
        circuit_size=circuit_size,
    )
    vc.isovalue = pre.image_isovalue_proc
    vc.classification_threshold = pre.class_threshold
    vc.encode_c_classify(verbose=False)
    vc.compose_iso_qubits(pre.compose_weight, c_mode=c_mode, verbose=False)
    vc.add_meas()

    all_circuits = list(vc.eqd.qcEL)
    pre.n_circuits = int(vc.di.n_circuits)
    pre.n_tiles = pre.n_circuits
    pre.n_valid = int(vc.di.n_valid)
    print(
        f"Flattened {pre.dataset_name} {pre.volume_shape} -> {pre.n_valid} samples, "
        f"{pre.n_circuits} circuits x {circuit_size} addresses "
        f"(nq_addr={vc.di.nq_addr})"
    )
    print(f"Total circuits in batch: {len(all_circuits)}")

    n_shots = int(circuit_size) * (2**int(shots_coef))

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
                "n_tiles": pre.n_circuits,
                "n_circuits": pre.n_circuits,
                "circuit_size": int(circuit_size),
                "tile_width": int(circuit_size),
                "tile_height": 1,
                "dataset_name": pre.dataset_name,
                "volume_shape": list(pre.volume_shape),
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

    _, _, _, data_recErr = vc.recover_data(
        n_shots, all_counts, pre.all_data_list, pre.all_rec_list, verbose=False
    )
    mean_data_rec_err = float(np.mean(np.asarray(data_recErr)))

    classifications = vc.c_classify(pre.all_rec_list)
    comp = vc.compare_against_input(
        classifications, pre.compose_weight, n_valid=pre.n_valid
    )

    pre.cm_list.append(comp["confusion_matrix"])
    pre.acc_list.append(comp["accuracy"])

    data_vals = np.asarray(comp["vals"]).reshape(-1)
    subtraction_vals = np.asarray(comp["subtraction_vals"]).reshape(-1)
    quantum_ev_vals = packed_circuits_to_1d(
        np.asarray(pre.all_rec_list[0][-1])[:, 0, :], n_valid=pre.n_valid
    )
    y_t = np.asarray(comp["y_true"]).reshape(-1)
    y_p = np.asarray(comp["y_pred"]).reshape(-1)

    pre.vals = data_vals
    pre.subtraction_vals = subtraction_vals
    pre.quantum_ev = np.asarray(quantum_ev_vals).reshape(-1)

    shape = pre.volume_shape
    pre.padded_canvas_gray = np.asarray(pre.region_proc, dtype=np.float32)
    pre.padded_canvas_true = y_t.reshape(shape)
    pre.padded_canvas_pred = y_p.reshape(shape)

    stitch_records: list[ClassificationTileStitchRecord] = []
    return pre.all_rec_list, pre.all_data_list, pre, stitch_records, mean_data_rec_err


# -------------------------------- Main --------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "QCrank eHANDS vertex classification on 3D sphere/torus volumes. "
            "Volumes are flattened to 1D and partitioned into circuits (no spatial tiling). "
            "Saves stitched classification data; charts are generated separately by "
            "ehands_qcrank_vertex_classification_V2_analysis.py."
        )
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=("full", "shots", "tile"),
        default="full",
        help=(
            "Which test driver to run. "
            "full: shots-coefficient loop; "
            "shots: only the shots-coefficient loop driver; "
            "tile: circuit-size loop for a single --shots-coef-k."
        ),
    )
    parser.add_argument(
        "--shots-coef-k",
        type=int,
        default=12,
        help="Shots exponent k used when --test tile (n_shots = circuit_size * 2**k per circuit).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run for each (dataset, circuit size).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_3D_DATASETS),
        help=(
            "3D volumes to classify. Default: sphere_32 torus_32 sphere_64 torus_64 "
            "(32^3 and 64^3 analytic fields from the notebook)."
        ),
    )
    parser.add_argument(
        "--circuit-size",
        type=int,
        default=None,
        help=(
            "Addresses per circuit (power of 2). If omitted, the default sweep is "
            "256 then 4096 addresses. Alias: --tile-width."
        ),
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help="Alias for --circuit-size (addresses per circuit, not a 2D tile edge).",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Ignored; retained so older command lines still parse.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-x-offset",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-y-offset",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--region-width",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--region-height",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--isovalue-mode",
        type=str,
        choices=("auto_median", "fixed"),
        default="fixed",
        help=(
            "fixed: use --isovalue in encoded [-1, 1] (default 0.0 = notebook isolevel 0.5). "
            "auto_median: median of the encoded volume."
        ),
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=0.0,
        help="Encoded-space isovalue when --isovalue-mode fixed. 0.0 maps to 0.5 on the [0, 1] volume.",
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
        choices=("aer", "fake_torino", "fake_marrakesh", "fake_miami", "fake_boston"),
        default="aer",
        help=(
            "Simulation backend for qiskit_ibm_runtime Sampler: "
            "aer, fake_torino, fake_marrakesh, fake_miami, or fake_boston "
            "(fake_miami/fake_boston require qiskit-ibm-runtime>=0.47.0)."
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
            "(does not wait for results / does not save classification run data)."
        ),
    )
    parser.add_argument(
        "--c-mode",
        type=str,
        choices=("1", "2", "auto"),
        default="1",
        help=(
            "Classification method selection mode. "
            "1: Use base approach. "
            "2: Use iso-weight encoding. "
            "auto: take c_mode from hardware results when --hw-results-dir is set."
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
            "Requires job_submission_info.json in JobOutputs/ (walked up from this path)."
        ),
    )
    args = parser.parse_args()

    default_circuit_sizes = (256, 4096)
    iterations = args.iterations
    volumes = build_3d_volume_datasets(args.datasets)

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
            f"Hardware post-processing: circuit sizes {hw_tile_sizes}, "
            f"shots_coef_k={hw_sc}, c_mode={args.c_mode}, iterations=1"
        )

    if args.c_mode == "auto":
        args.c_mode = "1"

    circuit_size_arg = args.circuit_size if args.circuit_size is not None else args.tile_width
    if hw_results is not None:
        tile_sizes = tuple(sorted(hw_results.keys()))
        iterations = 1
    elif circuit_size_arg is not None:
        if int(circuit_size_arg) < 2:
            parser.error("circuit size must be a power of 2, >= 2.")
        tile_sizes = (int(circuit_size_arg),)
    else:
        tile_sizes = default_circuit_sizes

    if hw_results is None:
        tile_sizes = filter_circuit_sizes(tile_sizes)

    tw = int(tile_sizes[0])
    th = 1
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
        volumes=volumes,
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
