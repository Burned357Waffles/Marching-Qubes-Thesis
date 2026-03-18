"""
TODO: Add header comments
"""


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
    __slots__ = ('n_data', 'nq_addr', 'nq_data', 'num_q', 'addr_qL', 'data_qL', 'data_inp', 'classify_q')
    def __init__(self, n_cubes, data_range, make_iso=False, isovalue=0.5):
        self.n_data = 2**3 # data per array
        self.nq_addr = math.ceil(math.log2(self.n_data))
        self.nq_data = n_cubes
            
        self.data_inp = np.random.uniform(data_range[0], data_range[1], size=(self.n_data, self.nq_data, 1))

        if make_iso:
            iso_array = np.full((self.n_data, 1, 1), isovalue)
            self.data_inp = np.concatenate((self.data_inp, iso_array), axis=1)

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
            print("Addition Circuit:")
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
    
    def add_iso_qubit_for_ehands_add(self, qc, data_q, placement_q, weight, negation=True, verbose=False, reset=False):
        qc_iso = QuantumCircuit(1, 1)
        qc_iso.ry(np.arccos(self.isovalue), 0)

        qc.compose(qc_iso, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_addition(qc, data_q, placement_q, weight=weight, negation=negation, verbose=verbose)
        qc.barrier()

        #qc.reset(placement_q)
        qc_iso.ry(np.arccos(self.isovalue), 0)

        return qc

    def add_k_qubit_for_ehands_mult(self, qc, data_q, placement_q, k, verbose=False):
        qc_k = QuantumCircuit(1, 1)
        qc_k.ry(np.arccos(k), 0)

        qc.compose(qc_k, placement_q, inplace=True)

        qc.barrier()
        qc = self.ehands_multiplication(qc, data_q, placement_q, verbose=False)
        qc.barrier()

        return qc
    
    def init_data(self, make_iso, data_range=(-0.99, 0.99)):
        self.di = DataInfo(self.n_cubes, data_range, make_iso=make_iso)

    def encode(self, operations, verbose=False, reset=False):
        # encode sample data into qcrank
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)
        if reset:
            total_q = self.di.num_q + 1
        else:    
            total_q = self.di.num_q + operations * self.n_cubes
        #print(f"total q: {total_q}")
        self.qc_main = QuantumCircuit(total_q, total_q)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)
    
    def encode_v2(self, operations, verbose=False, reset=False):
        # encode sample data into qcrank
        self.eqd = EncodedQData(self.di, measure=False, verbose=verbose)
        if reset:
            total_q = self.di.num_q + 2
            self.di.classify_q = total_q - 1 # classifier qubit will be the final qubit in the circuit
        else:    
            total_q = self.di.num_q + operations * self.n_cubes + 1
        #print(f"total q: {total_q}")
        self.qc_main = QuantumCircuit(total_q, 1)
        self.qc_main.compose(self.eqd.qcEL[0], list(range(self.di.num_q)), inplace=True)
    
    def compose_iso_qubits(self, weight, verbose=False, reset=False):
        for i in range(self.n_cubes):
            q_a = self.di.data_qL[i]
            if reset: # if reset, we only use 1 iso qubit and reset it after the addition
                q_b = self.di.num_q
            else:
                q_b = self.di.num_q + i 
            self.qc_main = self.add_iso_qubit_for_ehands_add(self.qc_main, q_a, q_b, self.isovalue, weight, verbose=verbose, reset=reset)
            
    def compose_k_qubits(self, k, verbose=False):
        for i in range(self.n_cubes):
            q_a = self.di.data_qL[i]
            q_b = self.di.num_q + self.n_cubes + i
            self.qc_main = self.add_k_qubit_for_ehands_mult(self.qc_main, q_a, q_b, k, verbose=verbose)

    def add_meas(self, classify=False):
        self.qc_main.barrier()
        if classify:
            self.qc_main.measure(self.di.classify_q, 0)
        else:
            self.qc_main.measure(list(range(self.di.num_q)), reversed(list(range(self.di.num_q))))

        self.eqd.qc = self.qc_main
        self.eqd.qcEL = [self.qc_main]

    def recover_data(self, n_shots, countsL, all_data_list, all_rec_list, verbose=False):
        data_rec, data_recErr = self.eqd.qcrank_obj.reco_from_yields(countsL)
        
        shpad = n_shots / 2**self.di.nq_addr
        print(f'Shots per address: {shpad:.1f}, relative error ~ {1/np.sqrt(shpad):.3f}')

        self.analyze_all_qcrank_residuals(data_rec, all_data_list, all_rec_list, verbose=verbose)

    def analyze_all_qcrank_residuals(self, data_rec, all_data_list, all_rec_list, verbose=False):
        # Iterate over all input arrays
        for i in range(self.di.nq_data):
            # Save data and recovered to lists
            data_slice = self.di.data_inp[:, i:i+1, :]
            rec_slice = data_rec[:, i:i+1, :]

            all_data_list[i].append(data_slice)
            all_rec_list[i].append(rec_slice)

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
        
        print(qc.draw('text'))
        
    return countsL


#---------------------------tests---------------------------#


def test_qcrank_ehands_add_reset(n_cubes, isovalue, weight, reset, sim):
    print("RUNNING TEST: ADD WITH RESET ENABLED")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True
    
    all_data_list = [[] for _ in range(n_cubes)]
    all_rec_list = [[] for _ in range(n_cubes)]
  
    for _ in range(20):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data(False)
        vc.encode(1, verbose, reset)

        # add iso value qubit 
        vc.compose_iso_qubits(weight, verbose, reset)

        #if verbose:
         #   display_statevector(vc.qc_main)

        # Add measurement
        vc.add_meas()

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)
        
        # verbose for first iteration only
        verbose = False
        
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list

def test_qcrank_ehands_add_mult(n_cubes, isovalue, weight, k, sim):
    print("RUNNING TEST: MULTIPLY AFTER ADD")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight}), k: {k}")
    verbose = True

    all_data_list = [[] for _ in range(n_cubes)]
    all_rec_list = [[] for _ in range(n_cubes)]
    
    for _ in range(20):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        vc.init_data(False)
        vc.encode(2, verbose)

        # add iso value qubits
        vc.compose_iso_qubits(weight, verbose)
        # add k qubits
        vc.compose_k_qubits(k)

        #if verbose:
            #display_statevector(vc.qc_main)

        # Add measurement
        vc.add_meas()

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        # Recover the data from QC
        vc.recover_data(n_shots, countsL, all_data_list, all_rec_list, verbose)
        
        # verbose for first iteration only
        verbose = False
        
    print("Returning data and recovered data lists")
    return all_rec_list, all_data_list

def test_qcrank_ehands_classify(n_cubes, isovalue, weight, reset, sim):
    print("RUNNING TEST: CLASSIFY WITH RESET ENABLED")
    print(f"inputs (n_cubes: {n_cubes}, isovalue: {isovalue}, weight: {weight})")
    verbose = True

    all_data_list = []
    counts = {'0': 0, '1': 0}

    fig = None
  
    for _ in range(20):
        # initialize data and isovalue arrays
        vc = VertexClassifier(n_cubes, isovalue)
        #vc.init_data(False, data_range=(-1, 0))
        vc.init_data(False, data_range=(0, 1))
        vc.encode_v2(1, verbose, reset)

        # add iso value qubit 
        vc.compose_iso_qubits(weight, verbose, False)

        vc.qc_main.cx(vc.di.data_qL[0], vc.di.classify_q)

        #if verbose:
            #display_statevector(vc.qc_main)

        # Add measurement
        vc.add_meas(classify=True)

        # Run Simulation
        n_shots = vc.di.n_data * (2**12)
        countsL = run_sim_job_qcrank(vc.eqd, sim, n_shots, verbose)

        for k in counts:
            counts[k] += countsL[0][k]
        
        # verbose for first iteration only
        verbose = False
    
    fig = plot_distribution(counts)
    plt.show()
    print("Returning data and recovered data lists")
    return counts, all_data_list
    

#---------------------------plots---------------------------#


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
n_cubes = 3
isovalue = 0.5
weight = 0.5
k = 0.5

test_num = 2
sims = ["AerSimulator", "FakeTorino", "FakeMarrakesh"]
sim = configure_aer_sim(sims[0])

if n_cubes >= 1:
    match test_num:
        case 1:
            reset = False
            all_rec_list, all_data_list = test_qcrank_ehands_add_reset(n_cubes, isovalue, weight, reset, sim)
            display_residual_analysis(n_cubes, all_data_list, all_rec_list, False, filename="residuals_sub.png")
        case 2:
            reset = True
            all_rec_list, all_data_list = test_qcrank_ehands_add_reset(n_cubes, isovalue, weight, reset, sim)
            display_residual_analysis(n_cubes, all_data_list, all_rec_list, False, filename="residuals_sub_reset.png")
        case 3:
            all_rec_list, all_data_list = test_qcrank_ehands_add_mult(n_cubes, isovalue, weight, k, sim)
            display_residual_analysis(n_cubes, all_data_list, all_rec_list, False, filename="residuals_sub_mult.png")
        case 4:
            counts, all_data_list = test_qcrank_ehands_classify(n_cubes, isovalue, weight, True, sim)

else: 
    print("n_cubes must be 1 or more")
