"""
TODO: Add header comments
"""


import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import sys
from dotenv import load_dotenv
import os
from contextlib import redirect_stdout

import qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import array_to_latex
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeMarrakesh


print(f"Qiskit version: {qiskit.__version__}")

# Adapted from Chris Pestano's QSobel
load_dotenv()
data_encoder_circuits_path = os.getenv('DATA_ENCODER_CIRCUITS_PATH')

if data_encoder_circuits_path is None:
    raise EnvironmentError("DATA_ENCODER_CIRCUITS_PATH environment variable is not set. Please set it to the path of data-encoder-circuits.")

print(f"Using DATA_ENCODER_CIRCUITS_PATH from environment variable: {data_encoder_circuits_path}")
circuits_path = data_encoder_circuits_path

# Add to sys.path if not already present
if circuits_path not in sys.path:
    sys.path.insert(0, circuits_path)

print(f"Added to sys.path: {circuits_path}")
print(f"Current working directory: {os.getcwd()}")

from datacircuits.ParametricQCrankV2 import ParametricQCrankV2 as QCrankV2, analyze_qcrank_residuals

print("imports complete")


#--------------------------------Info Classes--------------------------------#
class DataInfo:
    __slots__ = ('n_data', 'nq_addr', 'nq_data', 'num_q', 'n_circuits', 'addr_qL', 'data_qL', 'data_inp')
    def __init__(
        self,
        n_cubes,
        data_range,
        n_circuits=1,
        isovalue=0.5,
        use_image=False,
        image_path=None,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
    ):
        if use_image:
            self.n_data = image_width * image_height
        else:
            self.n_data = n_cubes * 8 # data per array

        self.nq_addr = (self.n_data - 1).bit_length()
        self.nq_data = 1
        self.n_circuits = n_circuits
        self.data_inp = np.zeros((2**self.nq_addr, self.nq_data, self.n_circuits))
        if use_image:
            self.data_inp = self.image_to_data(
                image_path,
                image_width,
                image_height,
                image_x_offset,
                image_y_offset,
            )
        else:
            self.data_inp[:self.n_data, :, 0] = np.random.uniform(
            data_range[0], data_range[1], size=(self.n_data, 1))

        self.data_inp[self.n_data :, :, 0] = 0.0
        self.num_q = self.nq_addr + self.nq_data # Number of qubits in QCrank array
        self.addr_qL = list(range(self.nq_addr))
        self.data_qL = list(range(self.nq_addr, self.nq_addr + self.nq_data))
    
    def image_to_data(
        self,
        image_path,
        image_width=None,
        image_height=None,
        image_x_offset=0,
        image_y_offset=0,
    ):
        """
        Convert an image to grayscale data normalized to [-1, 1].
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

        # PIL uses (0, 0) as top-left origin; crop box is (left, upper, right, lower).
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

        # Match the same shape expected by the rest of the pipeline:
        # (2**nq_addr, nq_data, n_circuits), with zero-padding already in place.
        out = np.zeros((2**self.nq_addr, self.nq_data, self.n_circuits), dtype=np.float32)
        out[:n_take, 0, 0] = flat[:n_take]
        return out

class EncodedQData:
    __slots__ = ('qc', 'qcEL', 'nq_addr', 'nq_data', 'qcrank_obj')
    def __init__(self, di, useCZ=False, measure=True, barrier=True, verbose=False):
        self.nq_addr = di.nq_addr
        self.nq_data = di.nq_data

        self.qcrank_obj = QCrankV2(self.nq_addr, self.nq_data, useCZ=useCZ, measure=measure, barrier=barrier)

        self.qc = self.qcrank_obj.circuit
        
        self.qcrank_obj.bind_data(di.data_inp)

        self.qcEL = self.qcrank_obj.instantiate_circuits()

        if verbose:
            print(f"Created {len(self.qcEL)} circuits")


#--------------------------------VertexClassifier Class--------------------------------#
class VertexClassifier:
    def __init__(self, n_cubes, isovalue):
        self.n_cubes = n_cubes
        self.isovalue = isovalue

        # to be initialized by other setup methods
        self.di = None
        self.eqd = None
        self.qc_main = None

    def ehands_addition(self, qc, q_a, q_b, weight, negation=False, verbose=False):
        """ 
        Perform an EHands addition of two qubits with a given weight and optional negation.

        :param qc: QuantumCircuit to which the addition will be applied
        :param q_a: Index of the first qubit
        :param q_b: Index of the second qubit
        :param weight: Weight for the addition (between 0 and 1)
        :param negation: If True, negate the second qubit before addition

        :return: QuantumCircuit with the eHANDS addition applied   
        """
        alpha = np.arccos(1 - 2 * weight)

        qc_add = QuantumCircuit(2)

        if negation:
            qc_add.x(1)

        qc_add.rz(pi/2, 1)
        qc_add.cx(0, 1)
        qc_add.ry(alpha/2, 0)
        qc_add.cx(1, 0) 
        qc_add.ry(-alpha/2, 0)

        if verbose:
            fig = qc_add.draw("mpl")
            fig.show()
        
        return qc.compose(qc_add, qubits=[q_a, q_b])

    def ehands_multiplication(self, qc, q_a, q_b, verbose=False):
        """
        Output is on q_a
        """
        qc_mult = QuantumCircuit(2)

        qc_mult.rz(np.pi/2, 0)
        qc_mult.cx(1, 0)

        if verbose:
            print("Multiplication Circuit")
            fig = qc_mult.draw("mpl")
            fig.show()

        return qc.compose(qc_mult, qubits=[q_a, q_b])

    
    def add_iso_qubit_for_ehands_add(self, qc, data_q, placement_q, weight, negation=True, verbose=False):
        qc_iso = QuantumCircuit(1, 1)
        qc_iso.ry(np.arccos(self.isovalue), 0)

        qc.compose(qc_iso, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_addition(qc, data_q, placement_q, weight=weight, negation=negation, verbose=verbose)
        qc.barrier()

        return qc

    def add_anc_qubit_for_ehands_add(self, qc, data_q, placement_q, negation=True, verbose=False):
        qc_anc = QuantumCircuit(1, 1)
        iso_encoded_weight = 1.0 / (1.0 + self.isovalue)

        qc.compose(qc_anc, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_addition(qc, data_q, placement_q, weight=iso_encoded_weight, negation=negation, verbose=verbose)
        qc.barrier()

        return qc
    
    def add_iso_qubit_for_ehands_mult(self, qc, data_q, placement_q, verbose=False):
        qc_iso = QuantumCircuit(1, 1)
        k = -self.isovalue/4
        print(f"k: {k}")
        qc_iso.ry(np.arccos(k), 0)
        qc.compose(qc_iso, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_multiplication(qc, data_q, placement_q, verbose=verbose)
        qc.barrier()

        return qc
    
    def init_data(self, data_range=(-0.99, 0.99), use_image=False, image_path=None, image_width=None, image_height=None, image_x_offset=0, image_y_offset=0):
        self.di = DataInfo(self.n_cubes, data_range, use_image=use_image, image_path=image_path, image_width=image_width, image_height=image_height, image_x_offset=image_x_offset, image_y_offset=image_y_offset)

    def encode_q_classify(self, operations, verbose=False):
        # encode sample data into qcrank
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose) 
        total_q = self.di.num_q + operations * self.n_cubes + 1
        #print(f"total q: {total_q}")
        # For classification, we want all address qubits + the classify qubit measured.
        # Classical bits layout (indices):
        #   0          : classification bit
        #   1..nq_addr : address bits
        self.qc_main = QuantumCircuit(total_q, self.di.nq_addr + 1)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)

    def encode_c_classify(self, verbose=False):
        # encode sample data into qcrank
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)
   
        total_q = self.di.num_q + 1
        #print(f"total q: {total_q}")
        self.qc_main = QuantumCircuit(total_q, total_q)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)
    
    def compose_iso_qubits(self, weight, verbose=False):
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        self.qc_main = self.add_iso_qubit_for_ehands_add(self.qc_main, q_a, q_b, weight, verbose=verbose)

    def compose_ancilla_qubits_add(self, verbose=False):
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        self.qc_main = self.add_anc_qubit_for_ehands_add(self.qc_main, q_a, q_b, verbose=verbose)

    def compose_ancilla_qubits_mult(self, weight, verbose=False):
        q_a = self.di.data_qL[0]
        q_b = self.di.num_q
        self.qc_main = self.add_iso_qubit_for_ehands_mult(self.qc_main, q_a, q_b, verbose=verbose)
            
    def add_meas(self, q_classify=False, c_classify=False):
        self.qc_main.barrier()
        if c_classify:
            self.qc_main.measure(list(range(self.di.num_q)), reversed(list(range(self.di.num_q))))
        elif q_classify:
            for i, q in enumerate(self.di.addr_qL):
                self.qc_main.measure(q, i + 1)
            self.qc_main.measure(self.di.classify_q, 0)
        else:
            raise ValueError("No classification type specified")

        self.eqd.qc = self.qc_main
        self.eqd.qcEL = [self.qc_main]

    def recover_data(self, n_shots, countsL, all_data_list, all_rec_list, verbose=False):
        data_rec, data_recErr = self.eqd.qcrank_obj.reco_from_yields(countsL)
        
        shpad = n_shots / 2**self.di.nq_addr
        print(f'Shots per address: {shpad:.1f}, relative error ~ {1/np.sqrt(shpad):.3f}')

        self.construct_data_lists(data_rec, all_data_list, all_rec_list)

        return all_data_list, all_rec_list, data_rec, data_recErr
    
    def construct_data_lists(self, data_rec, all_data_list, all_rec_list):
        for i in range(self.di.nq_data):
            data_slice = self.di.data_inp[:, i:i+1, :]
            rec_slice = data_rec[:, i:i+1, :]
            all_data_list[i].append(data_slice)
            all_rec_list[i].append(rec_slice)
        return all_data_list, all_rec_list

    def analyze_all_qcrank_residuals(self, data_rec, verbose=False):
        # Iterate over all input arrays
        for i in range(self.di.nq_data):
            # Save data and recovered to lists
            data_slice = self.di.data_inp[:, i:i+1, :]
            rec_slice = data_rec[:, i:i+1, :]

            # QCrank analysis
            if verbose:
                with open(os.devnull, "w") as f:
                    with redirect_stdout(f):
                        analyze_qcrank_residuals(data_slice, rec_slice)
            else:
                analyze_qcrank_residuals(data_slice, rec_slice)
            
            if verbose:
                print(f'\nCube original data:\n', data_slice)
                print(f'Reconstructed data:\n', rec_slice)
                print(f'Difference:\n', (data_slice - rec_slice))
                if i > 2: 
                    break
    
    def c_classify(self, all_rec_list):
        # Latest slice per data qubit (each list grows by one append per recover_data call).
        latest = [subl[-1] for subl in all_rec_list]
        rec = np.concatenate(latest, axis=1)
        # Match compare_against_input, which uses the first data channel only.
        classifications = np.where(rec[:, 0, 0] >= 0, 0, 1)
        return classifications


    def q_classify(self, countsL):
        """
        Classify each data point based on the final bit of the measured
        bitstrings, using the address bits to map outcomes back to data indices.

        Bitstrings are assumed to come from `nq_addr` address qubits followed by
        one classification qubit.

        :param countsL: List of count dictionaries as returned by
                        `run_sim_job_qcrank`. Keys are bitstrings, values are
                        shot counts.
        :return: A NumPy array of shape (n_data,) with entries 0 or 1 giving the
                 classification for each data point.
        """
        n_addr = self.di.nq_addr
        n_data = self.di.n_data

        # Totals per data index for final-bit 0 vs 1
        zero_totals = np.zeros(n_data, dtype=int)
        one_totals = np.zeros(n_data, dtype=int)

        for counts in countsL:
            for bitstring, n_shots in counts.items():
                if not bitstring:
                    continue

                # Expect address bits + 1 classify bit; ignore anything else
                if len(bitstring) < n_addr + 1:
                    continue

                addr_bits = bitstring[:-1]
                last_bit = bitstring[-1]

                # Map address bits to data index (0 .. n_data-1)
                try:
                    data_idx = int(addr_bits, 2)
                except ValueError:
                    continue

                if data_idx < 0 or data_idx >= n_data:
                    continue

                if last_bit == '0':
                    zero_totals[data_idx] += n_shots
                elif last_bit == '1':
                    one_totals[data_idx] += n_shots

        # For each data point, classify based on which final bit is more frequent
        classifications = np.where(one_totals > zero_totals, 1, 0)
        return classifications

    def compare_against_input(self, pred_classes, weight):
        """
        Compare predicted classes (per address) against classes derived directly
        from the input data values, applying a weighted subtraction of the
        isolevel before classification.

        Weighted subtraction rule:
          subtraction_val = weight * value - (1 - weight) * isolevel
          class = 0 if subtraction_val >= 0 else 1

        :param pred_classes: array-like of shape (n_data,)
        :return: dict with y_true, y_pred, accuracy, confusion_matrix
        """
        y_pred = np.asarray(pred_classes, dtype=int).reshape(-1)

        # For now, compare against the first data channel/cube.
        # self.di.data_inp shape: (n_data, nq_data, 1)
        vals = self.di.data_inp[:, 0, 0]
        subtraction_vals = weight * vals - (1.0 - weight) * self.isovalue
        y_true = np.where(subtraction_vals >= 0, 0, 1).astype(int)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"pred_classes length {y_pred.shape[0]} != n_data {y_true.shape[0]}")

        accuracy = float(np.mean(y_pred == y_true))

        # confusion matrix in order: [[true0->pred0, true0->pred1],
        #                             [true1->pred0, true1->pred1]]
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


#--------------------------------Simulation Utility Functions--------------------------------#
def configure_aer_sim(type=None):
    match type:
        case "FakeTorino":
            sim = FakeTorino()
        case "FakeMarrakesh":
            sim = FakeMarrakesh()
        case _:
            sim = AerSimulator()

    print(sim)
    print(f"\nConfiguration: {sim.configuration()}")

    if hasattr(sim, 'available_methods'):
        print(f"Available methods: {sim.available_methods()}")
    else:
        print(sim.configuration())

    if hasattr(sim, 'available_devices'):
        print(f"Available devices: {sim.available_devices()}")
    else:
        print(sim.configuration())   
    
    return sim

def configure_qcrank_sampler(sim, n_shots):
    options = SamplerOptions()
    options.default_shots=n_shots
    sampler = Sampler(mode=sim, options=options)
    return sampler, options


def run_sim_job_qcrank(eqd, sim, n_shots = 2**12, verbose=False):
    """ 
    Run a quantum circuit on the simulator and return the counts.
    
    :param encoding_args: dictionary including information about the QC of encoded data
    :param n_shots: number of shots to run
    :param verbose: print out extra details and draw circuit

    :return: Counts of the measurement outcomes
    """
    sampler, options = configure_qcrank_sampler(sim, n_shots)

    job = sampler.run(tuple(eqd.qcEL))
    jobRes = job.result()

    #countsL = [jobRes[0].data.meas.get_counts()]
    countsL = [jobRes[0].data.c.get_counts()]

    if verbose:
        qc = eqd.qc
        cxDepth=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
        print(f'.... PARAMETRIZED CIRCUIT .............., cx-depth={cxDepth}')
        print('Gates count:', qc.count_ops())
        
        #print(qc.draw('text'))
        # draw circuit as a png
        qc.draw('mpl').savefig('qc.png')
        
    return countsL


#--------------------------------Tests--------------------------------#
def test_qcrank_ehands_c_classify_flat(n_cubes, isovalue, weight, sim):
    print("RUNNING TEST: CLASSICAL CLASSIFICATION WITH FLAT DATA STRUCTURE")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True
    
    # One list per data-qubit channel (matches DataInfo.nq_data and construct_data_lists).
    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    # Accumulate statistics over all iterations
    agg_counts = {'0': 0, '1': 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
  
    for _ in range(100):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data()
        vc.encode_c_classify(verbose)

        # add iso value qubit 
        vc.compose_iso_qubits(weight, verbose)

        # Add measurement
        vc.add_meas(q_classify=False, c_classify=True)

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        all_data_list, all_rec_list, data_rec, data_recErr = vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)

        classifications = vc.c_classify(all_rec_list)
        comp = vc.compare_against_input(classifications, weight)

        cm_list.append(comp["confusion_matrix"])
        acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

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

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # Analyze the residuals
        #vc.analyze_all_qcrank_residuals(data_rec, verbose=verbose)
        
        # verbose for first iteration only
        verbose = False
    
    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name="flat_c_classification_summary_10x_shots.png",
        bins=20,
    )
    
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list


def test_qcrank_ehands_q_classify_flat(n_cubes, isovalue, weight, sim):
    print("RUNNING TEST: CLASSIFY WITH FLAT DATA STRUCTURE")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True

    all_data_list = []

    # Accumulate statistics over all iterations
    agg_counts = {'0': 0, '1': 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
  
    for _ in range(100):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data(data_range=(-1, 1))
        vc.encode_q_classify(1, verbose)

        # add iso value qubit 
        vc.compose_iso_qubits(weight, verbose=False)

        vc.qc_main.cx(vc.di.data_qL[0], vc.di.classify_q)

        # Add measurement
        vc.add_meas(q_classify=True, c_classify=False)

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        classifications = vc.q_classify(countsL)
        comp = vc.compare_against_input(classifications, weight)

        # Store stats
        cm_list.append(comp["confusion_matrix"])
        acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

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

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # verbose for first iteration only
        verbose = False
    
    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name="q_classification_summary.png",
        bins=20,
    )

    print("Returning data and recovered data lists")
    return agg_counts, all_data_list

def test_qcrank_ehands_c_classify_flat_ancilla(n_cubes, isovalue, weight, sim):
    print("RUNNING TEST: CLASSICAL CLASSIFICATION WITH FLAT DATA STRUCTURE AND ANCILLA WEIGHT")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True
    
    # One list per data-qubit channel (matches DataInfo.nq_data and construct_data_lists).
    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    # Accumulate statistics over all iterations
    agg_counts = {'0': 0, '1': 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
  
    for _ in range(100):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data()
        vc.encode_c_classify(verbose)

        # add iso value qubit 
        vc.compose_ancilla_qubits_add(verbose=verbose)

        # Add measurement
        vc.add_meas(q_classify=False, c_classify=True)

        # Run Simulation
        n_shots = 10 * vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        all_data_list, all_rec_list, data_rec, data_recErr = vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)

        classifications = vc.c_classify(all_rec_list)
        comp = vc.compare_against_input(classifications, weight)

        cm_list.append(comp["confusion_matrix"])
        acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

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

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # Analyze the residuals
        #vc.analyze_all_qcrank_residuals(data_rec, verbose=verbose)
        
        # verbose for first iteration only
        verbose = False
    
    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name="flat_c_classification_summary_ancilla_weight_10x_shots.png",
        bins=20,
    )
    
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list

def test_qcrank_ehands_c_classify_flat_mult(n_cubes, isovalue, weight, sim):
    print("RUNNING TEST: CLASSICAL CLASSIFICATION WITH FLAT DATA STRUCTURE AND ISO WEIGHT")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True
    
    # One list per data-qubit channel (matches DataInfo.nq_data and construct_data_lists).
    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    # Accumulate statistics over all iterations
    agg_counts = {'0': 0, '1': 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
  
    for _ in range(100):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data()
        vc.encode_c_classify(verbose)

        # add iso value qubit 
        vc.compose_ancilla_qubits_mult(weight, verbose)

        # Add measurement
        vc.add_meas(q_classify=False, c_classify=True)

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        all_data_list, all_rec_list, data_rec, data_recErr = vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)

        classifications = vc.c_classify(all_rec_list)
        comp = vc.compare_against_input(classifications, weight)

        cm_list.append(comp["confusion_matrix"])
        acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

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

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # Analyze the residuals
        #vc.analyze_all_qcrank_residuals(data_rec, verbose=verbose)
        
        # verbose for first iteration only
        verbose = False
    
    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name="flat_mult_c_classification_summary.png",
        bins=20,
    )
    
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list
    
def test_qcrank_ehands_c_classify_flat_image(n_cubes, isovalue, weight, sim, image_path, image_width, image_height):
    print("RUNNING TEST: CLASSICAL CLASSIFICATION WITH FLAT DATA STRUCTURE")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True
    
    # One list per data-qubit channel (matches DataInfo.nq_data and construct_data_lists).
    nq_data = 1
    all_data_list = [[] for _ in range(nq_data)]
    all_rec_list = [[] for _ in range(nq_data)]

    # Accumulate statistics over all iterations
    agg_counts = {'0': 0, '1': 0}
    cm_list = []
    acc_list = []
    all_correct_vals = []
    all_incorrect_vals = []
  
    for _ in range(1):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data(use_image=True, image_path=image_path, image_width=image_width, image_height=image_height, image_x_offset=36, image_y_offset=0)
        vc.encode_c_classify(verbose)

        # add iso value qubit 
        vc.compose_iso_qubits(weight, verbose)

        # Add measurement
        vc.add_meas(q_classify=False, c_classify=True)

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        all_data_list, all_rec_list, data_rec, data_recErr = vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)

        classifications = vc.c_classify(all_rec_list)
        comp = vc.compare_against_input(classifications, weight)

        cm_list.append(comp["confusion_matrix"])
        acc_list.append(comp["accuracy"])

        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

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

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # Analyze the residuals
        #vc.analyze_all_qcrank_residuals(data_rec, verbose=verbose)
        
        # verbose for first iteration only
        verbose = False
    
    plot_classification_summary_figure(
        acc_list=acc_list,
        cm_list=cm_list,
        agg_counts=agg_counts,
        all_correct_vals=all_correct_vals,
        all_incorrect_vals=all_incorrect_vals,
        out_name="flat_c_classification_summary_10x_shots.png",
        bins=20,
    )
    input_tile = vc.di.data_inp[:vc.di.n_data, 0, 0].reshape(image_height, image_width)
    predicted_tile = np.asarray(comp["y_pred"], dtype=int).reshape(image_height, image_width)
    plot_input_tile_and_classification(
        input_tile=input_tile,
        predicted_tile=predicted_tile,
        out_name="flat_c_image_tile_vs_classification.png",
    )
    
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list


#--------------------------------Plots--------------------------------#
def plot_input_tile_and_classification(input_tile, predicted_tile, out_name):
    """
    Plot the processed image tile and predicted classes side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax_input, ax_pred = axes
    im0 = ax_input.imshow(input_tile, cmap="gray", vmin=-1.0, vmax=1.0, origin="upper")
    ax_input.set_title("Input tile (normalized grayscale)")
    ax_input.set_xlabel("x")
    ax_input.set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=ax_input, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
    cbar0.set_ticklabels(["-1", "0", "1"])
    ax_input.set_xticks(np.arange(input_tile.shape[1]))
    ax_input.set_yticks(np.arange(input_tile.shape[0]))

    im1 = ax_pred.imshow(predicted_tile, cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax_pred.set_title("Predicted classification")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.set_xticks(np.arange(predicted_tile.shape[1]))
    ax_pred.set_yticks(np.arange(predicted_tile.shape[0]))
    cmap = plt.get_cmap("viridis")
    ax_pred.legend(
        handles=[
            Patch(facecolor=cmap(0.0), edgecolor="black", label="0 = inside"),
            Patch(facecolor=cmap(1.0), edgecolor="black", label="1 = outside"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved image tile comparison plot to: {out_name}")


def plot_correct_incorrect_input_histogram(all_correct_vals, all_incorrect_vals, bins=20, ax=None):
    """
    Plot a histogram of input values for correct vs incorrect classifications.

    Notes:
    - This function does not call `plt.show()` so multiple plots can be shown
      together by the caller.
    """
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

    # One shared bin grid so correct vs incorrect use the same bin width and x alignment.
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
    """
    Plot a 2x2 confusion matrix heatmap.

    Notes:
    - This function does not call `plt.show()` so multiple plots can be shown
      together by the caller.
    """
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

    # Annotate cells with counts, using luminance so text stays readable for any colormap.
    norm = im.norm
    cm = im.get_cmap()
    for i in range(2):
        for j in range(2):
            val = int(total_cm[i, j])
            rgba = cm(norm(val))  # (r,g,b,a) in 0..1
            r, g, b = rgba[0], rgba[1], rgba[2]
            # Relative luminance (sRGB approximation); higher = lighter background.
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
    Plot aggregated predicted class counts as a bar chart.

    Notes:
    - This function does not call `plt.show()` so multiple plots can be shown
      together by the caller.
    """
    if ax is None:
        ax = plt.gca()

    labels = ["0", "1"]
    values = [agg_counts["0"], agg_counts["1"]]
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_xlabel("Predicted class (final bit)")
    ax.set_ylabel("Total count over all runs")
    ax.set_title("Aggregated Predicted Class Counts")


def print_per_datapoint_classification_table(data_vals, subtraction_vals, y_true, y_pred):
    """
    Print a per-data-point table and return (correct_vals, incorrect_vals) where
    each is a NumPy array of subtraction values (or None if empty).
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
    plt.show()
    return mean_acc


#--------------------------------Main--------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QCrank eHANDS vertex reconstruction / classification tests."
    )
    parser.add_argument(
        "test_num",
        nargs="?",
        type=int,
        default=0,
        choices=(range(1, 5)),
        metavar="N",
        help=(
            "Which test to run: 1=c_classify_flat, 2=c_classify_flat_mult, 3=c_classify_flat_ancilla, 4=c_classify_flat_image"
        ),
    )
    args = parser.parse_args()
    test_num = args.test_num

    n_cubes = 4
    isovalue = -0.5
    weight = 0.5

    sims = ["AerSimulator", "FakeTorino", "FakeMarrakesh"]
    sim = configure_aer_sim(sims[0])

    if n_cubes < 1:
        print("n_cubes must be 1 or more")
        sys.exit(1)

    match test_num:
        case 1:
            all_rec_list, all_data_list = test_qcrank_ehands_c_classify_flat(n_cubes, isovalue, weight, sim)
        case 2:
            all_rec_list, all_data_list = test_qcrank_ehands_c_classify_flat_mult(n_cubes, isovalue, weight, sim)
        case 3:
            all_rec_list, all_data_list = test_qcrank_ehands_c_classify_flat_ancilla(n_cubes, isovalue, weight, sim)
        case 4:
            all_rec_list, all_data_list = test_qcrank_ehands_c_classify_flat_image(n_cubes, isovalue, weight, sim, image_path="test_images/Plant_tissue_sections_-_39815344093.jpg", image_width=4, image_height=4)
        case _:
            print("Invalid test number or no test specified")
            sys.exit(1)