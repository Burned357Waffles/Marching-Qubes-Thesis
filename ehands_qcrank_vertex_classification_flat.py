"""
TODO: Add header comments
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import sys
from dotenv import load_dotenv
import os
from contextlib import redirect_stdout

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_distribution, array_to_latex
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


class DataInfo:
    __slots__ = ('n_data', 'nq_addr', 'nq_data', 'num_q', 'addr_qL', 'data_qL', 'data_inp')
    def __init__(self, n_cubes, data_range, isovalue=0.5):
        self.n_data = n_cubes * 8 # data per array
        self.nq_addr = (self.n_data - 1).bit_length()
        self.nq_data = 1
        # Address space is 2^nq_addr rows; indices [0, n_data) get random data, [n_data, ...) stay 0.
        self.data_inp = np.zeros((2**self.nq_addr, 1, 1))
        self.data_inp[:self.n_data, :, 0] = np.random.uniform(
            data_range[0], data_range[1], size=(self.n_data, 1)
        )
        self.data_inp[self.n_data :, :, 0] = 0.0
        self.num_q = self.nq_addr + self.nq_data # Number of qubits in QCrank array
        self.addr_qL = list(range(self.nq_addr))
        self.data_qL = list(range(self.nq_addr, self.nq_addr + self.nq_data))

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

    
    def add_iso_qubit_for_ehands_add(self, qc, data_q, placement_q, weight, negation=True, verbose=False):
        qc_iso = QuantumCircuit(1, 1)
        qc_iso.ry(np.arccos(self.isovalue), 0)

        qc.compose(qc_iso, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_addition(qc, data_q, placement_q, weight=weight, negation=negation, verbose=verbose)
        qc.barrier()

        return qc
    
    def init_data(self, data_range=(-0.99, 0.99)):
        self.di = DataInfo(self.n_cubes, data_range)

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
        self.qc_main = self.add_iso_qubit_for_ehands_add(self.qc_main, q_a, q_b, self.isovalue, weight, verbose=verbose)
            

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

def display_statevector(qc):
    """
    Display the statevector, unitary matrix, and Bloch sphere representation of a quantum circuit.

    :param qc: QuantumCircuit to analyze

    :return: Statevector of the quantum circuit
    """
    # Get Statevector
    state = Statevector.from_instruction(qc)

    # Display Statevector
    print("\nStatevector:")
    print(state.draw("latex"))

    matrix_form = np.array(state).reshape(-1, 1)
    print("\nStatevector as a matrix:")
    print(array_to_latex(matrix_form))

    # Display the Unitary Matrix
    print("Unitary matrix:")
    print(Operator(qc).draw("latex"))

    # Display Bloch Sphere
    fig = plot_bloch_multivector(state)
    plt.show()

    return state


def configure_qcrank_sampler(n_shots):
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
    sampler, options = configure_qcrank_sampler(n_shots)

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


#---------------------------tests---------------------------#

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

        # For every run, print a table of each data point and its classification
        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

        print("\nPer-data-point classifications")
        print("+--------+---------------+------------------+--------------+----------------+")
        print("| Index  | Input Value   | Subtraction Vals | True Class   | Pred Class     |")
        print("+--------+---------------+------------------+--------------+----------------+")
        for idx, (val, sub_val, y_t, y_p) in enumerate(
            zip(data_vals, subtraction_vals, comp["y_true"], comp["y_pred"])
        ):
            print(f"| {idx:<6d} | {val:<13.6f} | {sub_val:<16.6f} | {y_t:<12d} | {y_p:<14d} |")
        print("+--------+---------------+------------------+--------------+----------------+")

        # Also show value ranges where the classifier is correct vs incorrect
        correct_mask = comp["y_true"] == comp["y_pred"]
        incorrect_mask = ~correct_mask

        if np.any(correct_mask):
            correct_vals = subtraction_vals[correct_mask]
            all_correct_vals.append(correct_vals)
            print(f"Correct classifications value range: "
                  f"[{correct_vals.min():.6f}, {correct_vals.max():.6f}]")
        else:
            print("No correct classifications in this run.")

        if np.any(incorrect_mask):
            incorrect_vals = subtraction_vals[incorrect_mask]
            all_incorrect_vals.append(incorrect_vals)
            print(f"Incorrect classifications value range: "
                  f"[{incorrect_vals.min():.6f}, {incorrect_vals.max():.6f}]")
        else:
            print("No incorrect classifications in this run.")

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # Analyze the residuals
        #vc.analyze_all_qcrank_residuals(data_rec, verbose=verbose)
        
        # verbose for first iteration only
        verbose = False
    
    # After all iterations, summarize and plot using matplotlib
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    print(f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}")

    # Plot summaries into a single subplots figure, then save as PNG.
    fig, ax_arr = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}", fontsize=16)
    ax_hist, ax_cm, ax_bar = ax_arr

    plot_correct_incorrect_input_histogram(
        all_correct_vals,
        all_incorrect_vals,
        bins=20,
        ax=ax_hist
    )

    if cm_list:
        total_cm = np.sum(np.stack(cm_list, axis=0), axis=0)
        print("Aggregated confusion matrix over all runs "
              "[[true0->pred0, true0->pred1], [true1->pred0, true1->pred1]]:")
        print(total_cm)
        plot_aggregated_confusion_matrix(total_cm, ax=ax_cm)
    else:
        ax_cm.set_title("Aggregated Confusion Matrix")
        ax_cm.text(0.5, 0.5, "No CM data", ha="center", va="center")
        ax_cm.axis("off")

    plot_aggregated_predicted_class_counts(agg_counts, ax=ax_bar)

    fig.tight_layout()
    out_name = "flat_c_classification_summary.png"
    fig.savefig(out_name, dpi=300)
    print(f"Saved plots to: {out_name}")
    plt.show()
    
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

        # For every run, print a table of each data point and its classification
        data_vals = vc.di.data_inp[:, 0, 0]
        # Subtraction value used for the "true" label:
        #   subtraction_val = weight * input_val - (1 - weight) * isolevel
        subtraction_vals = weight * data_vals - (1.0 - weight) * vc.isovalue

        print("\nPer-data-point classifications")
        print("+--------+---------------+------------------+--------------+----------------+")
        print("| Index  | Input Value   | Subtraction Vals | True Class   | Pred Class     |")
        print("+--------+---------------+------------------+--------------+----------------+")
        for idx, (val, sub_val, y_t, y_p) in enumerate(
            zip(data_vals, subtraction_vals, comp["y_true"], comp["y_pred"])
        ):
            print(f"| {idx:<6d} | {val:<13.6f} | {sub_val:<16.6f} | {y_t:<12d} | {y_p:<14d} |")
        print("+--------+---------------+------------------+--------------+----------------+")

        # Also show value ranges where the classifier is correct vs incorrect
        correct_mask = comp["y_true"] == comp["y_pred"]
        incorrect_mask = ~correct_mask

        if np.any(correct_mask):
            correct_vals = subtraction_vals[correct_mask]
            all_correct_vals.append(correct_vals)
            print(f"Correct classifications value range: "
                  f"[{correct_vals.min():.6f}, {correct_vals.max():.6f}]")
        else:
            print("No correct classifications in this run.")

        if np.any(incorrect_mask):
            incorrect_vals = subtraction_vals[incorrect_mask]
            all_incorrect_vals.append(incorrect_vals)
            print(f"Incorrect classifications value range: "
                  f"[{incorrect_vals.min():.6f}, {incorrect_vals.max():.6f}]")
        else:
            print("No incorrect classifications in this run.")

        # Aggregate class counts over all iterations
        agg_counts["0"] += int(np.sum(classifications == 0))
        agg_counts["1"] += int(np.sum(classifications == 1))

        # verbose for first iteration only
        verbose = False
    
    # After all iterations, summarize and plot using matplotlib
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    print(f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}")

    # Plot summaries into a single subplots figure, then save as PNG.
    fig, ax_arr = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Mean accuracy over {len(acc_list)} runs (0 if >=0 else 1): {mean_acc:.3f}", fontsize=16)
    ax_hist, ax_cm, ax_bar = ax_arr

    plot_correct_incorrect_input_histogram(
        all_correct_vals,
        all_incorrect_vals,
        bins=20,
        ax=ax_hist
    )

    if cm_list:
        total_cm = np.sum(np.stack(cm_list, axis=0), axis=0)
        print("Aggregated confusion matrix over all runs "
              "[[true0->pred0, true0->pred1], [true1->pred0, true1->pred1]]:")
        print(total_cm)
        plot_aggregated_confusion_matrix(total_cm, ax=ax_cm)
    else:
        ax_cm.set_title("Aggregated Confusion Matrix")
        ax_cm.text(0.5, 0.5, "No CM data", ha="center", va="center")
        ax_cm.axis("off")

    plot_aggregated_predicted_class_counts(agg_counts, ax=ax_bar)

    fig.tight_layout()
    out_name = "q_classification_summary.png"
    fig.savefig(out_name, dpi=300)
    print(f"Saved plots to: {out_name}")
    plt.show()

    print("Returning data and recovered data lists")
    return agg_counts, all_data_list
    
#---------------------------plots---------------------------#


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


def plot_aggregated_confusion_matrix(total_cm, title="Aggregated Confusion Matrix", ax=None):
    """
    Plot a 2x2 confusion matrix heatmap.

    Notes:
    - This function does not call `plt.show()` so multiple plots can be shown
      together by the caller.
    """
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)

    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate cells with counts
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(total_cm[i, j]), ha="center", va="center", color="black")


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


def plot_residuals(actual, theory, title, x_label, y_label, legend):
    min_val = min(min(actual), min(theory))
    max_val = max(max(actual), max(theory))

    plt.figure()
    plt.scatter(theory, actual, label=legend)   
    # calculate slope of line of best fit
    coefficients = np.polyfit(theory, actual, 1)
    print(f"Slope of line of best fit: {coefficients[0]:.4f}")
    # plot line of best fit
    plt.plot(theory, np.poly1d(coefficients)(theory), color='red', label='Line of Best Fit')    

    plt.xlabel(x_label)    
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.grid(True)

    plt.show()

def plot_residuals_subplots(all_actual, all_theory, labels=None, title_prefix="Residual Comparison", x_label="Original", y_label="Reconstructed", filename="residuals.png"):
    """
    Plots residuals for multiple cubes in a single figure with subplots.

    Parameters:
        all_actual: list of np.arrays, each containing reconstructed data for a cube
        all_theory: list of np.arrays, each containing original data for a cube
        labels: list of strings for legend titles per cube (optional)
        title_prefix: prefix for subplot titles
        x_label, y_label: axis labels
    """
    n_cubes = len(all_actual)
    n_cols = min(3, n_cubes)  # max 3 per row
    n_rows = math.ceil(n_cubes / n_cols)

    # Compute global min and max for consistent square axes
    combined_actual = np.concatenate([a.flatten() for a in all_actual])
    combined_theory = np.concatenate([t.flatten() for t in all_theory])
    min_val = min(combined_actual.min(), combined_theory.min())
    max_val = max(combined_actual.max(), combined_theory.max())
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D
    
    for i in range(n_cubes):
        ax = axes[i]
        actual = all_actual[i].flatten()
        theory = all_theory[i].flatten()
        lbl = labels[i] if labels else f"Cube {i+1}"
        
        ax.scatter(theory, actual, label=lbl)
        # line of best fit
        coeffs = np.polyfit(theory, actual, 1)
        slope = coeffs[0]
        ax.plot(theory, np.poly1d(coeffs)(theory), linestyle='--', color='red')

        # add slope as text in top-left corner of subplot
        ax.text(0.05, 0.95, f"Slope = {slope:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
        
        ax.set_title(f"{title_prefix} {i+1}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
    
    fig.savefig(filename)
    
    # Hide any unused axes
    for j in range(n_cubes, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_residual_analysis(n_cubes, all_data_list, all_rec_list, table, filename="residuals.png"):
    for i in range(n_cubes):
        if table:
            print(f"-----------------------------Analysis of Original Cube {i + 1} vs Reconstructed----------------------------------")
            print("+----------------+----------------+----------------+")
            print("| Original Data  | Reconstructed  | Difference     |")
            print("+----------------+----------------+----------------+")
            
        avg_dif = 0
        for o, r in zip(np.concatenate(all_data_list[i]).flatten(),
                        np.concatenate(all_rec_list[i]).flatten()):
            dif = o - r
            avg_dif += dif
            if table:
                print(f"| {o:<14.6f} | {r:<14.6f} | {dif:<14.6f} |")
        if table:
            print("+----------------+----------------+----------------+")

        n_points = len(np.concatenate(all_data_list[i]).flatten())
        avg_dif /= n_points

        print(f"Average Difference: {avg_dif}")
        """
        title = f'Original Data Cube {i + 1} vs Reconstructed Data'
        x_label = 'Original Data'
        y_label = 'Reconstructed Data'
        legend = 'Data Points'
        plot_residuals(np.concatenate(all_rec_list[i]).flatten(),
                        np.concatenate(all_data_list[i]).flatten(), 
                        title, x_label, y_label, legend)
        """

    all_actual = [np.concatenate(all_rec_list[i]).flatten() for i in range(n_cubes)]
    all_theory  = [np.concatenate(all_data_list[i]).flatten() for i in range(n_cubes)]
    labels = [f"Cube {i+1}" for i in range(n_cubes)]

    plot_residuals_subplots(all_actual, all_theory, labels=labels, filename=filename)
        

#---------------------------main---------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QCrank eHANDS vertex reconstruction / classification tests."
    )
    parser.add_argument(
        "test_num",
        nargs="?",
        type=int,
        default=0,
        choices=(range(1, 3)),
        metavar="N",
        help=(
            "Which test to run: 1=c_classify_flat"
        ),
    )
    args = parser.parse_args()
    test_num = args.test_num

    n_cubes = 4
    isovalue = 0.5
    weight = 0.5
    k = 0.5

    sims = ["AerSimulator", "FakeTorino", "FakeMarrakesh"]
    sim = configure_aer_sim(sims[0])

    if n_cubes < 1:
        print("n_cubes must be 1 or more")
        sys.exit(1)

    match test_num:
        case 1:
            all_rec_list, all_data_list = test_qcrank_ehands_c_classify_flat(n_cubes, isovalue, weight, sim)

        case _:
            print("Invalid test number or no test specified")
            sys.exit(1)