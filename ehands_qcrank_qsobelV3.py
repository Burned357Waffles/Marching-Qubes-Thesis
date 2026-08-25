import sys
import os
# ewb 11/21/2025, thanks GPT
# get value of an environment variable
default_data_encoder_circuits_path = r'C:\Users\pesta\Documents\GithubRepos\data-encoder-circuits'
data_encoder_circuits_path = os.getenv('DATA_ENCODER_CIRCUITS_PATH')
# test if data_encoder_circuits_path is set or empty    
if not data_encoder_circuits_path: 
    print("DATA_ENCODER_CIRCUITS_PATH environment variable not set. Using default paths.")
    sys.path.append(default_data_encoder_circuits_path)
else:
    print(f"Using DATA_ENCODER_CIRCUITS_PATH from environment variable: {data_encoder_circuits_path}")
    sys.path.append(data_encoder_circuits_path) 
sys.path.append('../data-encoder-circuits')
print("sys.path:", sys.path) # debug
from datacircuits import qcrank, frqi
from datacircuits import ParametricQCrankV2
from datacircuits.ParametricQCrankV2 import  ParametricQCrankV2 as qcrank_reco_from_yields

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image
import matplotlib.patches as mpatches
import math
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from matplotlib import pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library.standard_gates import RYGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit.transpiler import generate_preset_pass_manager
from qiskit import transpile
import time
import json
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~UTILITY FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_image_intensity_addresses(img_2d: np.ndarray, file_path = ""):
    """
    QIMP specific function
    Plot a line graph with linear pixel addresses on the x-axis and intensity on the y-axis,
    and display a small textbox with min/max intensity values (min forced to 0).
    Used for generating clean paper friendly image results
    """
    if img_2d.ndim != 2:
        raise ValueError("img_2d must be a 2D numpy array")

    # Compute stats from the full image (not the downsampled series)
    img_min = np.min(img_2d)
    img_max = np.max(img_2d)

    # Flatten in row-major (C-order) so addresses are 0..H*W-1
    intensities = img_2d.ravel(order='C')
    addresses = np.arange(intensities.size, dtype=np.int64)

    # --- Force minimum to 0 for plotting ---
    intensities_to_plot = np.clip(intensities, 0, None)  # clamp anything below 0 up to 0
    y_max = float(max(0.0, img_max))
    if y_max == 0.0:
        y_max = 1.0  # avoid a flat axis when the image is all <= 0
    y_max *= 1.05  # small headroom

    # NEW: add a little room below zero so the 0 gridline isn't on the frame
    y_min = -0.05 * y_max  # ~5% of the top limit below zero

    fig, ax = plt.subplots()
    ax.plot(addresses, intensities_to_plot)
    ax.set_xlabel("Pixel address (row-major index)")
    ax.set_ylabel("Intensity")
    ax.set_title("Image Intensity vs. Linear Pixel Address")
    ax.set_ylim(y_min, y_max)

    # Add a small textbox with min/max (min is forced to 0 for display)
    txt = f"min: {img_min:.3f} \nmax: {img_max:.3f}"
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round', alpha=0.6, pad=0.3)
    )

    fig.tight_layout()
    if file_path!="":
        plt.savefig(file_path)
    plt.show()


def calculate_mape(y_true, y_pred):
    """
    EWB Alternative to calculating MSE; Not currently used in this implementation
    
    :param y_true: measured value
    :param y_pred: predicted value
    """
    # Add a constant offset to ensure all values are > 1.0
    C = np.min(y_true)
    if C < 0.0:
        C *= -1.0
    C += 1.0
    y_true_shifted = y_true + C
    y_pred_shifted = y_pred + C
    # Calculate MAPE
    mape = np.mean(np.abs((y_true_shifted - y_pred_shifted) / y_true_shifted)) * 100
    return mape


# for use with classical sobel if it also needs to imitate behavior of image borders
def wrap_pad_image(img, pad_top, pad_bottom, pad_left, pad_right):
    """
    QIMP specific function
    img: 2D (H, W) grayscale or 3D (H, W, C) color image as a NumPy array
    pad_*: how many pixels of padding to add on each side
    """
    pad_width = (
        (pad_top, pad_bottom),   # pad along height (rows)
        (pad_left, pad_right),   # pad along width (cols)
    )
    
    # if it's color (H,W,C), we don't pad channels
    if img.ndim == 3:
        pad_width = pad_width + ((0, 0),)

    return np.pad(img, pad_width, mode="wrap")

def im_show_no_border_ticks(image, scaled_down = False):
    """
    QIMP specific function
    Used to clean up image outputs for formal use; removes image boundary labels
    """
    fig, ax = plt.subplots()
    normal_max = math.sqrt(16+4) # The max magnitude of the individual gradients is 4 if the input image values are between 0 and 1
    scaled_down_max = normal_max / 8# scaled down because of weighted summations; used for quantum circuits 
    if not scaled_down:
        ax.imshow(image, cmap='gray', vmin=0, vmax=normal_max)
        print("Max value: ", normal_max)
    else:
        ax.imshow(image, cmap='gray', vmin=0, vmax=scaled_down_max)  
        print("Max value: ", scaled_down_max)
    # Get the current ticks that Matplotlib chose
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    # Remove the first and last tick from each axis
    new_xticks = xticks[2:-2]
    new_yticks = yticks[2:-2]
    ax.set_xticks(new_xticks)
    ax.set_yticks(new_yticks)
    plt.show()


def sobel_classical(image, mode=5):
    """ 
    QIMP specific function
    Creates a classically calculated Sobel output for comparison with the quantum circuit outputs
    Modes 2-5 currently unsued due to circuit width being too large for simulation and hardware runs
    mode 0: returns Gx
    mode 1: returns Gy
    mode 2: returns Gx^2 
    mode 3: returns Gy^2
    mode 4: returns Gx^2 + Gy^2
    mode 5: returns sqrt(Gx^2 + Gy^2)
    """
    if type(image) == str:
        image = Image.open(image)
        image = image.convert('L')
        image = np.array(image)
    # Scale down to range 0, 1 to match circuit inputs
    image = image/255
    sobel_image = np.ones_like(image)
    for i in range(0, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gx = (image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1]) - \
                 (image[i - 1, j - 1] + 2 * image[i, j - 1] + image[i + 1, j - 1])
            gy = (image[i + 1, j - 1] + 2 * image[i + 1, j] + image[i + 1, j + 1]) - \
                 (image[i - 1, j - 1] + 2 * image[i - 1, j] + image[i - 1, j + 1])
            if mode == 0:
                sobel_image[i, j] = gx
            elif mode == 1:
                sobel_image[i, j] = gy
            elif mode == 2:
                sobel_image[i, j] = gx * gx
            elif mode == 3:
                sobel_image[i, j] = gy * gy
            elif mode == 4:
                sobel_image[i, j] = gx * gx + gy * gy
            elif mode == 5:
                sobel_image[i, j] = np.sqrt(gx * gx + gy * gy)  
    # make the borders white
    sobel_image[0, :] = 1
    sobel_image[-1, :] = 1
    sobel_image[:, 0] = 1
    sobel_image[:, -1] = 1
    return sobel_image


def group_gates(gate_dict):
    """
    Debugging function to group gates by type and count them.
    """
    one_qubit_gates = 0
    two_qubit_gates = 0
    for key, value in gate_dict.items():
        if key == 'ry':
            one_qubit_gates += value
        elif key == 'cx':
            two_qubit_gates += value
        elif key == 'h':
            one_qubit_gates += value
        elif key == 'barrier':
            # ignore barriers for now
            pass
        elif key == 'measure':
            one_qubit_gates += value
        elif key == 'rz':
            one_qubit_gates += value
        elif key == 'cz':
            two_qubit_gates += value
        elif key == 'x':
            one_qubit_gates += value
        elif key == 'reset' or key == 'if_else':
            pass
        else:
            raise ValueError(f"Unknown gate type: {key}")
    return {'1-qb gates': one_qubit_gates, '2-qb gates': two_qubit_gates}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File Save and Loading ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
With this collection of function, IBM QPU job results are saved and loaded with the JobOutputs subdirectory
"""


# fun with GPT generated code for serializing job results so that it can be dumped to a json file
def make_serializable(obj):
    """Recursively convert objects to JSON-serializable structures."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return make_serializable(vars(obj))
    elif hasattr(obj, "_asdict"):  # namedtuple
        return make_serializable(obj._asdict())
    elif hasattr(obj, "tolist"):  # NumPy arrays
        return obj.tolist()
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    else:
        return obj

def clean_runtime_result(result):
    """Clean Qiskit Runtime result into human-friendly JSON-safe format."""
    cleaned = {"results": []}

    # Case 1: Dictionary result (common for runtime jobs retrieved by job_id)
    if isinstance(result, dict):
        for exp in result.get("results", []):
            cleaned_exp = {}
            if "quasi_dists" in exp:  # Sampler results
                cleaned_exp["quasi_dists"] = [
                    {str(k): float(v) for k, v in dist.items()}
                    for dist in exp["quasi_dists"]
                ]
            if "values" in exp:  # Estimator results
                cleaned_exp["values"] = [float(v) for v in exp["values"]]
            if "metadata" in exp:
                cleaned_exp["metadata"] = make_serializable(exp["metadata"])
            cleaned["results"].append(cleaned_exp)
        if "metadata" in result:
            cleaned["metadata"] = make_serializable(result["metadata"])

    # Case 2: Object with .results attribute (PrimitiveResult-like)
    elif hasattr(result, "results"):
        for exp in result.results:
            exp_data = {}
            if hasattr(exp, "data") and hasattr(exp.data, "get"):
                if "quasi_dists" in exp.data:  # Sampler
                    exp_data["quasi_dists"] = [
                        {str(k): float(v) for k, v in dist.items()}
                        for dist in exp.data["quasi_dists"]
                    ]
                if "values" in exp.data:  # Estimator
                    exp_data["values"] = [float(v) for v in exp.data["values"]]
            if hasattr(exp, "metadata"):
                exp_data["metadata"] = make_serializable(exp.metadata)
            cleaned["results"].append(exp_data)
        if hasattr(result, "metadata"):
            cleaned["metadata"] = make_serializable(result.metadata)

    # Case 3: Legacy qiskit.result.Result
    elif hasattr(result, "to_dict"):
        cleaned = make_serializable(result.to_dict())

    else:
        cleaned["raw"] = str(result)
    return cleaned


def compress_count_keys(counts):
    """
    Changes the key values of the counts dictionary from bit strings to an int string for storage effiency 
    """
    compressed_counts = {}
    for key, value in counts.items():
        # reverse the key string
        key = key[::-1]
        # Remove leading zeros
        compressed_key = key.lstrip('0')
        # If the key becomes empty after stripping, it means it was all zeros
        if compressed_key == '':
            compressed_key = '0'
        # convert bit string into integer
        compressed_key = str(int(compressed_key, 2))
        compressed_counts[compressed_key] = value
    return compressed_counts


def decompress_count_keys(compressed_counts, total_qubits):
    """
    Reverses the compressing process of compress_count_keys when loading counts data
    """
    decompressed_counts = {}
    for key, value in compressed_counts.items():
        # convert integer key back to binary string
        bin_key = bin(int(key))[2:]  # remove '0b' prefix
        # Pad with leading zeros to match total_qubits
        bin_key = bin_key.zfill(total_qubits)
        # reverse the key string back to original order
        bin_key = bin_key[::-1]
        decompressed_counts[bin_key] = value
    return decompressed_counts


def save_job_submission(submission_info_dict):
    # Save job submission time to a file
    output_dir = "JobOutputs"   # no leading slash = relative folder
    # Check if directory exists; if not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    submission_file = os.path.join(output_dir, "job_submission_info.json")
    # update job_submission_info.json if it exists, else create new
    if os.path.exists(submission_file):
        with open(submission_file, "r") as f:
            existing_info = json.load(f)
        existing_info.update(submission_info_dict)
        submission_info_dict = existing_info
    # if file doesn't exist, create new
    else:
        with open(submission_file, "w") as f:
            json.dump(submission_info_dict, f, indent=4)
    print(f"Job submission time saved to {submission_file}")

# load the job if it exists
def load_job_result(job_id):
    # Load the data if it was already saved
    json_jobid_fname = f"job_results_{job_id}.json"
    # check if json_jobid_fname was created and return counts if so
    output_dir = "JobOutputs"  
    # Check if directory exists; if not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, json_jobid_fname)):
        print(f"Job results file {json_jobid_fname} already exists. Loading counts from file.")
        with open(os.path.join(output_dir, json_jobid_fname), "r") as f:
            final_export = json.load(f)
        print("Length of counts: ", len(final_export["counts"]))
        compressed_count_list = final_export["counts"]
        counts = []
        for count in compressed_count_list:
            # decompress count keys
            total_qubits = final_export["total_qubits"]
            counts.append(decompress_count_keys(count, total_qubits))
        return counts
    else: 
        return -1

def load_job_backend(job_id):
    # Load the data if it was already saved
    json_jobid_fname = f"job_results_{job_id}.json"
    # check if json_jobid_fname was created and return counts if so
    output_dir = "JobOutputs"  
    # Check if directory exists; if not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, json_jobid_fname)):
        print(f"Job results file {json_jobid_fname} already exists. Loading counts from file.")
        with open(os.path.join(output_dir, json_jobid_fname), "r") as f:
            final_export = json.load(f)
        backend = final_export["job_info"]["backend"]
        return backend
    else: 
        return -1

# Retrieve the job
def save_job_results(job_id, service):
    json_jobid_fname = f"job_results_{job_id}.json"
    # check if json_jobid_fname was created and return counts if so
    output_dir = "JobOutputs" 
    # Check if directory exists; if not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    job = service.job(job_id)
    # Fetch the results
    result = job.result()
    # Get and clean results
    raw_result = job.result()
    cleaned_result = clean_runtime_result(raw_result)
    # Job metadata
    creation_date = job.creation_date
    status_attr = job.status if not callable(job.status) else job.status()
    if hasattr(status_attr, "name"):  # Enum-like
        status_str = status_attr.name
    else:  # Already a string
        status_str = str(status_attr)
    counts = [result[i].data.c.get_counts()  for i in range(len(result))]
    compressed_count = []
    for count in counts:
        # compress count keys
        compressed_count.append(compress_count_keys(count))
    job_info = {
        "job_id": job.job_id(),
        "program_id": getattr(job, "program_id", None),
        "backend": job.backend().name if job.backend() else None,
        "creation_date": creation_date.isoformat() if isinstance(creation_date, datetime) else str(creation_date),
        "status": status_str
    }
    print("Counts dict of length:", len(counts))
    # Combine job info + results
    total_qubits = len(list(counts[0].keys())[0])
    final_export = {
        "counts":  compressed_count,
        # length of one key string in counts
        "total_qubits": total_qubits,
        "job_info": job_info,
        "results": cleaned_result #NOTE: This can take up a lot of memory depending on the job TODO: Figure out whats actually needed from the raw results and trim it down
    }
    # (Optional) Save dictionary to a JSON file
    with open(os.path.join(output_dir, json_jobid_fname), "w") as f:
        json.dump(final_export, f, indent=4)
    print(f"Universal cleaned job results saved to {json_jobid_fname}")
    return counts

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Output Chart FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def verification_charts(output_list, comparison_list, calculation_mode, data, hw, trim=0, file_path="", precalibration=False, title_overide=["",""], comp_chart = True, res_chart = True):
    """
    QIMP specific function
    Outputs a pair of analysis charts:
        a histogram for residual standard deviation to make sure the stdev is as expected given the shot count per address
        an expected vs measured plot with a best fit line to make sure the scaling is correct
    """
    # Title generation
    if title_overide != ["",""]:
        res_title = title_overide[0]
        real_vs_exp_title = title_overide[1]
    else:
        real_vs_exp_title = "Sobel Values Comparison: Classic vs Quantum\n" + calculation_mode + " Circuit for " + str(len(data)) + "x" + str(len(data)) + " Image"
        #  chart_real_vs_expected(adjusted_stitched_output, comparison, title = real_vs_expected_title)
        res_title = "Sobel Value Residuals: Classical vs Quantum\n" + calculation_mode + " Circuit for " + str(len(data)) + "x" + str(len(data)) + " Image"
        if hw == "ideal":
            real_vs_exp_title += " on Ideal Simulator"
            res_title += " on Ideal Simulator"
        else:
            real_vs_exp_title += " on " + hw
            res_title += " on " + hw
            if precalibration:
                real_vs_exp_title += " (Precalibration)"
                res_title += " (Precalibration)"
            else:
                real_vs_exp_title += " (Postcalibration)"
                res_title += " (Postcalibration)"

    # If a trim value is passed in, the calculations will ignore that amount of boundary values in the circuit output and comparison list arrays
    if trim>0:
        print("Ignoring", trim, "pixels from each edge for calculations")
        comparison_list = comparison_list[trim:-trim, trim:-trim]
        output_list = output_list[trim:-trim, trim:-trim]
    print("Total datapoints for verification: ", output_list.size)
    
    if comp_chart and res_chart:
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    elif res_chart and not comp_chart:
        # Create a single figure with one subplot
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    elif comp_chart and not res_chart:
        # Create a single figure with one subplot
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    else:
        raise ValueError("At least one of comp_chart or res_chart must be True.")
    residuals = output_list - comparison_list
    residual_min = np.min(residuals)
    residual_max = np.max(residuals)
    residuals_std = np.std(residuals)
    residuals_mean = np.mean(residuals)
    min_output = np.min(output_list)

    if res_chart:
    # Plot residuals on first subplot
        ax1.tick_params(axis='both', which='both', labelsize=15)
        ax1.axvline(residuals_mean, color='red', linestyle='dashed', linewidth=3, label='Mean')
        ax1.hist(residuals.ravel(), bins=24, range=(residual_min, residual_max))
        ax1.set_title(res_title, fontsize=18)
        ax1.set_xlabel("Residual Value", fontsize=18)
        ax1.set_ylabel("Count", fontsize=18)
        y_max = ax1.get_ylim()[1]
        chart_text = f'Mean: {residuals_mean:.5f}\n Std: {residuals_std:.3f}'
        ax1.text(residual_max*.65, y_max*.8, chart_text, fontsize=18)
        print(f"Residuals mean: {residuals_mean} \nstd: {residuals_std}")

    if comp_chart:
    # Plot comparison of real vs expected
        comparison_list = comparison_list.flatten()
        output_list = output_list.flatten()
        ax2.tick_params(axis='both', which='both', labelsize=15)
        # Plot comparison on second subplot
        m, b = np.polyfit(comparison_list, output_list, 1)
        theta = np.arctan(m) * 180 / np.pi
        ax2.plot(comparison_list, output_list, 'o', markersize=4, alpha=0.5)
        ax2.plot(comparison_list, m * comparison_list + b, color='red', linewidth=3, alpha=0.5)
        ax2.set_title(real_vs_exp_title, fontsize=18)
        ax2.set_xlabel("Comparison Value (Calculated Classically)", fontsize=18)
        ax2.set_ylabel("Measured EV", fontsize=18)
        # x and y axis ranges should be the same
        overall_min = min(np.min(comparison_list), np.min(output_list))
        overall_max = max(np.max(comparison_list), np.max(output_list))
        ax2.set_xlim(overall_min-abs(overall_min)*0.05, overall_max+abs(overall_max)*0.05)
        ax2.set_ylim(overall_min-abs(overall_min)*0.05, overall_max+abs(overall_max)*0.05)
        chart_text = f'Correlation: {np.corrcoef(comparison_list, output_list)[0, 1]:.2f}\n' \
                    f'Best fit line: y = {m:.2f}x + {b:.2f}\n theta = {theta:.2f} degrees'
        ax2.text((min_output - 0.2*min_output), 0.65*np.max(output_list), chart_text, fontsize=18)
        print(f"Correlation coefficient: {np.corrcoef(comparison_list, output_list)[0, 1]}")
        print(f"Best fit line: y = {m}x + {b}\ntheta = {theta} degrees")
    plt.tight_layout()

    if file_path != "":
        # save as pdf
        plt.savefig(file_path) 
    plt.show()

def analyze_np_array_image(img_2d: np.ndarray, image_file, title_overide=["",""], trim = 0):
    """
    QIMP specific function
    Display an image represented as a 2D numpy array in a more paper friendly format
    :param img_2d: 2D np array of the image
    :param image_file: original image file sobel_classical() will use to create a comparison
    :param title_overide: Changes the title, if provided
    :param trim: trims boundary rows and columns of the image, if provided
    """

    # Display and save image
    image_file_path = image_file.replace("TestImages/", "").replace(".jpg", "sobel_output.pdf").replace(".png", "_sobel_output.pdf")
    image_file_path = os.path.join("OutputCharts", image_file_path)
    # plot_image_intensity_addresses(img_2d, file_path=os.path.join("OutputCharts", image_file_path))
    fig, ax = plt.subplots()
    im = ax.imshow(img_2d[trim:-trim, trim:-trim], cmap='gray')
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
    cb.set_label('Intensity')  # optional label
    plt.tight_layout()
    fig.tight_layout()
    plt.savefig(image_file_path)
    # Generate, save, and display charts
    comparison = sobel_classical(image_file, mode=5)/4
    chart_file_path = image_file.replace("TestImages/", "").replace(".jpg", "_output_charts.pdf").replace(".png", "_np_array_output_charts.pdf")
    chart_file_path = os.path.join("OutputCharts", chart_file_path)
    verification_charts(img_2d, comparison, "Sobel Magnitude", img_2d, trim=trim, hw="", file_path=chart_file_path, title_overide=title_overide, res_chart = False)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~QSobel Ehands FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_nq_addr(data):
    """
    Given a numpy array, returns the amount of address qubits needed to represent it if encoded with basis encoding

    """
    # Assume square images for now
    width = math.sqrt(data.shape[0])
    height = math.sqrt(data.shape[0])
    print("height, width: ", height, width)
    print("n_addr qubits:", int(math.log2(height) + math.log2(width)))
    return int(math.log2(height) + math.log2(width))


def tile_image(data, n_tiles, mode=0, tile_len=None): # replace n_tiles with tile_len for non-square images
    """
    QIMP specific function
    Given the data 2D numpy array, create 6 shifted images based on the chosen mode, then break them into the specified amount of square tiles 

    img: 2D numpy array representing the image.
    n_tiles: total number of tiles along both dimensions.
    Tile an image into n_tiles x n_tiles pieces.
    """
    # hard coded for now for testing; remove later 
    channels = 6
    compact = True
    height, width = data.shape
    if height == width:
        image_len = data.shape[0]
        print("image_len=", image_len)
        n_tiles_1D = int(math.sqrt(n_tiles))
        tile_len = image_len // n_tiles_1D
    else:
        # tile_len must be provided for non-square images
        n_tiles_1D = width // tile_len
    # TODO: Add Value errors like these for other implicit requirements for my implementation when the rest of the codebase is done
    # if tile_len == image_len:
    #     print("Tile length is equal to the image length, no need to tile. Processing the image as is.")
    #     return [data]
    # if tile_len < 4:
    #     raise ValueError("Tile length must be larger than the 3x3 sobel window. Currently: ", tile_len)
    # if tile_len > image_len/2:
    #     raise ValueError("Tile length must be less than half the original image size. Currently: ", tile_len)
    if not math.log2(tile_len).is_integer():
        raise ValueError("Tile length must be a power of 2. Received tile length: ", tile_len)
    starts = [int(i*tile_len) for i in range(n_tiles_1D)]
    print("Starts list: ", starts)
    n_img = n_tiles

    # create the 6 relevant shifted versions of the data array, then stack them into 3D numpy array of dimensions (width, height, channels)
    # set up tiles array 
    tiles = np.ones((tile_len**2, channels, n_img)) 
    if channels == 6:
        # shifted_image_array is a numpy array of shape (width*height, channels)
        shifted_image_array = build_data_array(data, mode=mode, compact=compact)
        # Reshape to from (width*height, 6) to (width, height, 6)
        shifted_image_array = shifted_image_array.reshape((width, height, channels))
    else:
        raise ValueError("Currently only 6 channel data_array is supported.")
    # Break up the channel array into tiles
    for i in range(n_tiles):
        for j in range(channels):
            y = starts[i// n_tiles_1D]
            x = starts[i % n_tiles_1D]
            chunk_vals = shifted_image_array[y : y + tile_len, x : x + tile_len, j].copy()
            chunk_vals = chunk_vals.flatten()
            tiles[:,j,i] = chunk_vals

    # E-Hands affects the scaling. Range adjusted to [-1,1]
    # Max value is no longer used in ParametricQCrankV2
    # Scale the image to [0,1] range since arccos conversion replaces the QCrank handling of the data with max_val
    # For ParametricQCrankV2 arccos is applied internally; range just needs to be between [-1, 1]
    # Assume only two possible max vals are 1.0 or 255
    # data_array will contain 6 shifted versions of the full image
    if np.max(tiles) > 1.0:
        tiles = tiles / 255.0
        tiles = (tiles-0.5)/0.5
    return tiles


def reconstruct_from_tiles(tiles, n_tiles, data, mode=0, padding=True):
    """
    QIMP specific function
    Takes a list of circuit results and stitches them into a larger image
    tiles:   list of tiles
    n_tiles: the original number of *interior* (non‐overlapping) tiles passed in
    mode:    
      0 = Overlap with replacement (interiors only)  
      1 = Overlap with averaging   (interiors only)  
      2 = No overlap (paste back‐to‐back)
    """
    print("Using overlap mode:", mode)
    # If only one tile was passed, force no‐overlap:
    if len(tiles) == 1:
        print("Only one tile passed in, forcing no‐overlap mode")
        mode = 2
    tile_len  = tiles[0].shape[0] # assumes all tiles are the same square dimensions
    stride    = tile_len  # just tile_len apart
    if data is None: 
        print("Data is None, assuming image is square")
        n_out_1D = int(math.sqrt(len(tiles)))
        # print("Tile reconstruction without overlap\n\n")
        n_row = n_out_1D
        n_col = n_out_1D
    else:
        print("Data argument provided, using its shape for reconstruction")
        print("data shape:", data.shape)
        r, c = data.shape
        print("r, c, tile_len:", r, c, tile_len)
        n_row = r // tile_len
        n_col = c // tile_len
    print("Tiles array of dimensions:", len(tiles), "x", tile_len, "x", tile_len)
    print("Reconstructing image of size:", n_row*tile_len, "x", n_col*tile_len)
    recon = np.zeros((n_row * tile_len, n_col * tile_len), dtype=tiles[0].dtype)
    for i in range(n_row):
        for j in range(n_col):
            y0, x0 = i*stride, j*stride
            tile   = tiles[i * n_col + j]
            recon[y0 : y0+tile_len,x0 : x0+tile_len] = tile[0 : tile_len,0 : tile_len]
    # Set first and last rows and columns to 1 to account for sobel window
    if padding:
        recon[0,:] = 1
        recon[-1,:] = 1
        recon[:,0] = 1
        recon[:,-1] = 1
    return recon

def get_named_backend(named_backend, account_name):
    """
    returns the relevant backend info if using a simulated fake backend or a real QPU
    """
    # account_credentials_name = "pestano_free_account"
    # account_credentials_name = "LBNL QCAN Instance"  # ewb account
    # account_credentials_name = "pestano-ORNL-account"
    if account_name == '':
        raise ValueError("Please provide an account name")
    account_credentials_name = account_name
    # echo check, ewb
    print("get_named_backend:: account_credentials_name=", account_credentials_name)
    service = QiskitRuntimeService(name=account_credentials_name)
    if "fake" in named_backend:
        # NOTE: Defaults to Density Matrix; cannot simulate images larger than 8x8
        named_backend = named_backend.replace('fake_','ibm_')
        hw_backend = service.backend(named_backend)
        backend = AerSimulator.from_backend(hw_backend)
        if ('GPU' in AerSimulator().available_devices()):
            backend.set_options(device='GPU')
    elif "ibm" in named_backend:
        backend = service.backend(named_backend)
    else:
        raise ValueError("Named backend must contain 'fake_' or 'ibm_' prefix")
    return backend


def get_backend(printing = False, seeded=True, force_CPU = False, named_backend=None):
    """
    returns the relevant information for an ideal backend
    """
    if force_CPU:
        if seeded:
            backend= AerSimulator(method='statevector', seed_simulator=777)
        else:
            backend= AerSimulator(method='statevector')
        if printing:
            print('Using CPU')
    elif ('GPU' in AerSimulator().available_devices()):
        if seeded:
            backend= AerSimulator(method='statevector', device='GPU', seed_simulator=777)
        else:
            backend= AerSimulator(method='statevector', device='GPU')
        if printing:
            print('Using GPU')
    else:
        if seeded:
            backend= AerSimulator(method='statevector', seed_simulator=777)
        else:
            backend= AerSimulator(method='statevector')
        if printing:
            print('Using CPU')
    return backend


def shifter(image, position):
    """
    QIMP specific function
    Shifts the original image so address will be able to get information from its chosen neighbor
    Assumes a 3x3 window (eg: position a is (Y-1,X-1), position b is (Y-1,X), etc.)
    Input: the original image and position of neighbor of interest
    Output: shifted numpy array corresponding to the neighbor position

    |A B C|
    |D E F|
    |G H I|
    
    To obtain the neighbor position, it should shift in the OPPOSITE direction of the neighbor position

    eg right shift so left neighbor value is stored in addr:

    |O O O|       |O O O|
    |X O O|  ==>  |O X O|
    |O O O|       |O O O|
    """
    shift_map = {
        'A': (-1, -1),  # (Y-1,X-1)
        'B': (-1, 0),   # (Y-1,X)
        'C': (-1, 1),   # (Y-1,X+1)

        'D': (0, -1),   # (Y,X-1)
        'F': (0, 1),    # (Y,X+1)

        'G': (1, -1),   # (Y+1,X-1)
        'H': (1, 0),    # (Y+1,X)
        'I': (1, 1)     # (Y+1,X+1)
    }
    if position not in shift_map:
        raise ValueError("Invalid position")
    # Get the shift values for the specified position
    shift_y, shift_x = shift_map[position]
    # Shift the image in the opposite direction of the target position
    shifted_image = np.roll(image, (-shift_y, -shift_x), axis=(0, 1))
    return shifted_image


def mapper(image, mapping):
    """
    QIMP specific function (but mapping logic relevant for any multi-channel QCrank work)

    Converts input array into several arrays images based on mapping list 
    Input: the original image and python list representing a mapping of neighbor qubits to data qubits from LSQB to MSQB 
    Output: a 2D numpy array
    """
    # Initialize an empty 2D array of shape (flattened image, len(mapping))
    # flattened image represents the data, 
    # len mapping = the number of data qubit channels that will be created in the QCrank circuit
    mapped_image = np.zeros((image.shape[0] * image.shape[1], len(mapping)))
    # Iterate through each position in the mapping, and fill the corresponding column in the mapped_image with shifter
    for idx, position in enumerate(mapping):
        shifted_image = shifter(image, position)
        # input for a circuit that will have data channels of count idx
        mapped_image[:, idx] = shifted_image.flatten() # dimension 0 = data, dimension 1 = qubit data channel number
    return mapped_image


def build_data_array(image, mode = 0, compact=True):
    """
    QIMP specific function
    Creates 2D array of shifted images (l*w, n_img) that will be input for QCrank circuit
    NOTE: These still optionally need to be broken up into tiles
    Modes:  0 = Gx only
            1 = Gy only
            2 = Gx^2
            3 = Gy^2
            4 = Gx^2 + Gy^2(Not recommended for simulation use and is untested on QPUs)
    """
    if compact:
        # Compact mapping original proposed by Jan in April 2025
        # gx_array = mapper(image, ['C', 'I', 'A', 'G', 'F', 'D'])
        # gy_array = mapper(image, ['A', 'C', 'G', 'I', 'B', 'H'])

        # # New mapping proposed by Jan in July 2025
        gx_array = mapper(image, ['A', 'G', 'C', 'I', 'D', 'F'])
        gy_array = mapper(image, ['A', 'C', 'G', 'I', 'B', 'H'])
    if mode == 0: # Gx
        #produces a 2D numpy array of shape (l*l, 6)
        data_array = gx_array
    elif mode == 1: # Gy
        data_array = gy_array
    elif mode == 2: # Gx^2
        data_array = gx_array
        data_array = np.append(data_array, gx_array, axis=1)
    elif mode == 3: # Gy^2
        data_array = gy_array
        data_array = np.append(data_array, gy_array, axis=1)
    elif mode == 4: # Gx^2 + Gy^2
        data_array = gy_array
        data_array = np.append(data_array, gy_array, axis=1)
        data_array = np.append(data_array, gx_array, axis=1)
        data_array = np.append(data_array, gx_array, axis=1)
    return data_array      

def ehands_product_with_memory(ckt, multiplicand, multiplier, barrier=False, reverse=False):
    """ 
    Assumes phase rotations are already applied to the inputs outside of this function.
    ckt: quantum circuit to be modified
    multiplicand: address of multiplicand qubit (value will be retained)
    multiplier: address of multiplier qubit (product will be stored and measured from here)
    """
    if reverse:
        ckt.cx(multiplicand, multiplier)
        ckt.rz(-np.pi/2, multiplier)
        if barrier:
            ckt.barrier()
    else:    
        ckt.rz(np.pi/2, multiplier)
        ckt.cx(multiplicand, multiplier)
        if barrier:
            ckt.barrier()
    return ckt


def ehands_parity_flip(ckt, target, ancilla, cr, reset=False):
    if reset:
        #print("Resetting ancilla qubit")
        ckt.reset(ancilla)
        ckt.h(ancilla)
        ckt.measure(ancilla, cr)
        with ckt.if_test((cr, 1)):
            ckt.z(target)
    # New Implementation (Doesn't require an H gate)
    else:
        ckt.h(ancilla)
        ckt.cz(ancilla, target)
        # Dr. Balewski uses this method in his updated paper, but because I did not develop it I'll use my implementation
        #ckt.cx(target, ancilla)
    return ckt


def ehands_addition(ckt, addend1, addend2, weight, leading_negation = False, negation = False, ancilla=-1, reverse=False, reset_parity=False, cr=0, barrier=False, target_0=True):
    """
    Weighted sum
    Assumes phase rotations are already applied to the inputs outside of this function.
    ckt: quantum circuit to be modified
    addend1: number of first addend qubit (sum will be stored and measured from here)
    addend2: number of second addend qubit 
    weight: relative weight
    negation: if addend2 is negated to create a subtraction operation
    ancilla: if an ancilla is provided, the addition is being used in a sequence of operations, and a parity flip, 
            which uses the ancilla, is needed to eliminate unwanted terms in the quantum state   
            note ancilla qubits can be data qubits not currently storing meaning info, but should be reset first
            it should also not be used in the final addition operation in a sequence
    """
    if reverse:     
        """   
        if an ancilla is provided, the addition is being used in a sequence of operations, 
        and a parity flip, which uses the ancilla, is needed to elminate unwanted terms in the quantum state    
        """
        if ancilla>=0:
            # reset the ancilla qubit in case it was previously utilized for other operations
            # parity flip applies to the qubit that stores the final value, addend1
            if target_0:
                ckt = ehands_parity_flip(ckt,addend1, ancilla, cr, reset_parity) #, reverse=True)
            else: 
                ckt = ehands_parity_flip(ckt,addend2, ancilla, cr, reset_parity)
        if barrier:
            ckt.barrier()
        alpha = np.arccos(1-2*weight)
        ckt.ry(alpha/2, addend1)
        ckt.cx(addend2, addend1)
        ckt.ry(-alpha/2, addend1)
        ckt.cx(addend1, addend2)
        ckt.rz(-np.pi/2, addend2)
        if negation:
            ckt.x(addend2)
        if leading_negation:
            ckt.x(addend1)
    else:
        alpha = np.arccos(1-2*weight)
        if leading_negation:
            ckt.x(addend1)
        if negation:
            #print(addend2)
            ckt.x(addend2)
        ckt.rz(np.pi/2, addend2)
        ckt.cx(addend1, addend2)
        ckt.ry(alpha/2, addend1)
        ckt.cx(addend2, addend1) 
        ckt.ry(-alpha/2, addend1)
        """   
        if an ancilla is provided, the addition is being used in a sequence of operations, 
        and a parity flip, which uses the ancilla, is needed to elminate unwanted terms in the quantum state    
        """
        if ancilla>=0:
            # reset the ancilla qubit in case it was previously utilized for other operations
            # parity flip applies to the qubit that stores the final value, addend1
            if target_0:
                ckt = ehands_parity_flip(ckt,addend1, ancilla, cr, reset_parity) #, reverse=True)
            else: 
                ckt = ehands_parity_flip(ckt,addend2, ancilla, cr, reset_parity)
        if barrier:
            ckt.barrier()
    return ckt
    

def get_target_counts(addr_bits_str, counts, target_qubit, with_reset=False):    
    """
    addr_bits_str: string of bits that are not the target qubit
    counts: dictionary of counts from the measurement
    target_qubit: string representing index of the target qubit alongside all other data qubits eg: '010'
    POTENTIALLY: offset: number of qubits used in the measurement, which will be between the data and addr qubits (add if nq_addr starts to get used) 
    ancilla: supporting qubits not tied to data or addr qubits, assume located in between them. MIGHT NOT BE NEEDED, LEAVE AS DEFAULT FOR CALCS FOR NOW
    # """
    print(counts)
    full_bit_str_0 = addr_bits_str + '0' * (len(target_qubit))
    full_bit_str_1 = addr_bits_str + target_qubit
    if with_reset: # an additional bit is included for count that represents the midcircuit measurements outcome. Assume it is the MSB
        full_bit_str_0_leading_0 = '0' + full_bit_str_0
        full_bit_str_1_leading_0 = '0' + full_bit_str_1
        c_0= counts.get(full_bit_str_0_leading_0, 0)
        c_1= counts.get(full_bit_str_1_leading_0, 0)
        full_bit_str_0_leading_1 = '1' + full_bit_str_0
        full_bit_str_1_leading_1 = '1' + full_bit_str_1
        c_0 += counts.get(full_bit_str_0_leading_1, 0)
        c_1 += counts.get(full_bit_str_1_leading_1, 0)
    else:
        c_0 = counts.get(full_bit_str_0, 0)
        c_1 = counts.get(full_bit_str_1, 0)
    if c_0 is None:
        c_0 = 0
    if c_1 is None:
        c_1 = 0
    return c_0, c_1

def get_EV(counts, nq_addr, target_qubit = '1'):
    """ 
    WITH MULTIPLE DATA QUBITS, c_0 AND c_1 NEED TO BE THE SUMS OF ALL BASIS STATE COUNTS WITH THE TARGET QUBIT SET TO 0 OR 1, RESPECTIVELY 
    The above can be achieved trivially by applying measurement gates to only the data qubit that contains the final value (target qubit) and the address qubits
    Replaces counts with the expected values of the basis state
    Y coordinates make up the lower half of the addr qubits
    Count is a list of dicts, where each dict is for a different circuit
    """
    length = int(2**(nq_addr/2))
    #upper_bits = 0
    output_image = np.zeros((length, length), dtype=float)
    for i in range(length**2): # assumes square image
        addr_bits_str = bin(i)[2:].zfill(nq_addr)
        c_0, c_1 = get_target_counts(addr_bits_str, counts, target_qubit)
        if c_0 == 0 and c_1 == 0:
            # If both probabilities are zero, set expectation value to zero
            expectation_value = 0
        else:
            """
            EV = (p_0 - p_1)/(p_0 + p_1)
            Replacing p_1 with p and p_0 with 1-p:
            EV = (1-p - p)/(1-p + p) 
            EV = (1-2p)/(1) 
            EV= 1-2p 
            """
            expectation_value = 1 - (2*c_1/(c_0 + c_1))
        x_coords = i % length
        y_coords = i // length
        output_image[y_coords][x_coords] = expectation_value #* 2 / np.pi   
        #upper_bits += 1
    return output_image

def get_target_qubit(channels, target_qubit_index, offset): 
    """
    Builds a string representing the target qubit based on the number of channels and the target qubit index.
    MSQB First; offset appended at the end
    """
    target_qubit = ''
    for i in range(channels):
        if i == target_qubit_index:
            target_qubit += '1'
        else:
            target_qubit += '0'
    for i in range(offset):
        target_qubit += '0'
    return target_qubit


def apply_compact_sobel(ckt, qubit_list, offset=2, ancilla_addr=1, last_ancilla=True, with_resets = False, barrier=False, circuit_width=0):
    """  
    QIMP specific function (but is an example of how to use predefined EHands circuit blocks to create a subcircuit that handles computations)

    This implementation computes the sobel kernel without the need for scaling qubits.
    NOTE: This circuit leads to 1/8 scaling of EV relative to its intended value.
    NOTE: If the EHands encoded values are from range [-1, 1] instead of [0, 1], the scaling factor will be 1/4 instead of 1/8
    NOTE: Reset variant not tested due to poor simulation performance.
    """
    if with_resets:
        ancilla_inc = 0
    else:
        ancilla_inc = 1
    # New implementation, Jan's proposal July 2025
    # final value stored in Least Significant Bit Data Qubit (0)
    # TODO: Manually overrid ancilla address location; fix later
    w0 = 1/2
    # S1: (D0+D1)/2
    ckt = ehands_addition(ckt, qubit_list[0]+offset, qubit_list[1]+offset, w0, reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # S2: (D2+D3)/2
    ckt = ehands_addition(ckt, qubit_list[2]+offset, qubit_list[3]+offset, w0, ancilla=ancilla_addr, reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    ancilla_addr += ancilla_inc
    # S3: (D4-D5)/2 
    ckt = ehands_addition(ckt, qubit_list[4]+offset, qubit_list[5]+offset, w0, ancilla=ancilla_addr, leading_negation=True,  reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    #ancilla_addr += ancilla_inc
    # S4: (D0+D2-D5-D4)/4
    ckt = ehands_addition(ckt, qubit_list[0]+offset, qubit_list[2]+offset, w0, leading_negation=True,  reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # S5: (D3+2*D1+D2-A-2*D0-D4)/8
    # TODO: Manually overrid the ancilla choice for this design; fix later
    ckt = ehands_addition(ckt, qubit_list[0]+offset, qubit_list[4]+offset, w0, reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    return ckt
   
    # Previous Flipped Orientation
    # S1: (D5+D4)/2
    # ckt = ehands_addition(ckt, qubit_list[5]+offset, qubit_list[4]+offset, w0, reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # # S2: (D3+D2)/2
    # ckt = ehands_addition(ckt, qubit_list[3]+offset, qubit_list[2]+offset, w0, ancilla=ancilla_addr, reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # ancilla_addr += ancilla_inc
    # # S3: (D0-D1)/2 
    # ckt = ehands_addition(ckt, qubit_list[1]+offset, qubit_list[0]+offset, w0, ancilla=ancilla_addr, leading_negation=True,  reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # #ancilla_addr += ancilla_inc
    # # S4:  (G+I-A-C) = (D3+D2-D5-D4)/4
    # ckt = ehands_addition(ckt, qubit_list[5]+offset, qubit_list[3]+offset, w0, leading_negation=True,  reset_parity=with_resets, barrier=barrier, cr=circuit_width)
    # # S5: ((D3+2*D1+D2-A-2*D0-D4)/8
    # # TODO: Manually overrid the ancilla choice for this design; fix later
    # ckt = ehands_addition(ckt, qubit_list[5]+offset, qubit_list[1]+offset, w0, reset_parity=with_resets, barrier=barrier, cr=circuit_width)

def apply_scaling(ckt, ckt_width, target_qubits, offset=2, barrier=False):
    """   
    Apply scaling to the qubits in target_qubits.
    """
    scaled_ckt = QuantumCircuit(ckt_width, ckt_width)
    scaled_ckt.ry(np.arccos(.5), 0)
    for i in range(len(target_qubits)):
        scaled_ckt = ehands_product_with_memory(scaled_ckt, 0, target_qubits[i] + offset)
    scaled_ckt.compose(ckt, inplace=True)
    if barrier:
        scaled_ckt.barrier()
    return scaled_ckt

def ehands_encoding_to_qcrank(data, transpile_ckt=False, ehands=False, show_ckt=False, direction="UP", channels=1, offset=0, shots=1000000, mode=0, compact=True, tile_length=-1, overlap=True):
    """
    Takes a numpy array and creates a QCrank circuit that encodes its data
    data is a list of numpy arrays of shape (height, width, channels)
    """
    # NOTE: Hardcoded to only work for grayscale images for now
    # NOTE: for irregular channel numbers, the channels are simply repeated values rather than shifted arrays
    # Qcrank expects data in the shape:  (n_pixels, num_channels, num_tiles)
    max_val = 1.0
    # Assume square image for now
    nq_addr = get_nq_addr(data)
    nq_data = channels # number of shifted images
    shots = shots      # number of shots to sample per circuit
    # Derived sizes ---------------------------------------------------------------
    n_addr = 2**nq_addr         # number of different addresses
    # ------------------------------------------------------------------------------ 
    print(nq_addr, "address qubits", nq_data, "data qubits")
    print(f"Data value: ", data.shape)

    # This function is from an outside directory provided by the AQuA-DATA team at LBNL (see data-encoder-circuits directory)
    param_qcrank = ParametricQCrankV2.ParametricQCrankV2( nq_addr, nq_data, useCZ=False,measure=False,barrier=True )
    # Important NOTE: QCrankV2 has address qubits on the LSQBs, V1 had an option to reverse bits during initialization, but that is not present in V2

    param_qcrank.bind_data(data)
    # generate the instantiated circuits
    # rotation gates values are mapped to their respective gates 
    data_circs = param_qcrank.instantiate_circuits()
    addr_qubits = list(range(nq_addr))
    # return the relevant data for the decoding function as a dictionary (arg_dict for the rest of the codebase)
    return {
        'data_circs': data_circs,
        'param_qcrank': param_qcrank,
        'nq_addr': nq_addr,
        'max_val': max_val,
        'channels': channels,
        'shots': shots,
        'addr_qubits': addr_qubits,
        'offset': offset
    }


# Implementation to handle all circuits in a job; does not work if circuits are transpiled beforehand
def add_measurement(arg_dict, ehands=False, target_qubit='', n_ancilla=0):
    """    
    Adds measurement gates to the circuits in arg_dict for the target qubit and address qubits and returns the modified circuit
    arg_dict: dictionary of arguments to be passed to the ehands_to_qcrank function
    """
    if ehands:
        # If target not provided, assume target qubit is always the LSB for now
        if target_qubit == '':
            for i in range(arg_dict['offset']):
                target_qubit += '0'
            target_qubit += '1'
    # for each circuit in the job
    for i in range(len(arg_dict['data_circs'])):
        # get the index for the target qubit
        target_qubit_index = [0]
        # for each qubit in the circuit 
        for j in range(len(target_qubit)):
            if target_qubit[j] == '1':
                print("Adding a measurement for target qubit index: ", j+n_ancilla)
                target_qubit_index[0] = j+n_ancilla
        m_gates = target_qubit_index+arg_dict['addr_qubits']
        # Measure only the target qubit and address qubits
        arg_dict['data_circs'][i].measure(m_gates, m_gates)
        # reverse the bits to match orientation in V1
        arg_dict['data_circs'][i] = arg_dict['data_circs'][i].reverse_bits()
    # run the simulation for all images
    return arg_dict


def execute_sim_job(arg_dict, account_name, show_ckt=False, named_backend=None, force_CPU=False, specifications=False):
    """
    Executes a simulation of circuits found in arg_dicts
    Assumes check for real backend name handled outside this function
    """
    if named_backend:
        backend = get_named_backend(named_backend, account_name)
    else:
        backend = get_backend(printing=True, seeded=True, force_CPU=force_CPU)
    print("Backend: \n", backend)
    # Targetted backends
    if named_backend:
        print(f'.... PARAMETRIZED Transpiled CIRCUIT ..............')
        pm = generate_preset_pass_manager(backend=backend)
        if "fake" in named_backend:
            circuits = [pm.run(circuit) for circuit in arg_dict['data_circs']]
        else:
            raise ValueError(f'Fake backend must be prefixed with "fake".')
    # Ideal simulator 
    else:  
        circuits = arg_dict['data_circs']
    # Add Sampler
    if specifications:
        if not named_backend:
            gate_dict = arg_dict['data_circs'][0].count_ops()
            print(' Gate count:', gate_dict)
            #TODO adjust group gates to work with transpiled circuits, as well 
            grouped_gate_dict = group_gates(gate_dict)
            print('Grouped gate count:', grouped_gate_dict)
            print(' Circuit depth:', arg_dict['data_circs'][0].depth())
            print(' Circuit width:', arg_dict['data_circs'][0].num_qubits)
        else:
            print('Grouped gate count dict:', circuits[0].count_ops())
            print('Transpiled Circuit depth:', circuits[0].depth())
            print('Transpiled Circuit width:', circuits[0].num_qubits)
    options = SamplerOptions()
    options.default_shots=arg_dict['shots']
    sampler = Sampler(mode=backend, options=options)
    #if not named_backend or "ibm" not in named_backend:
    start_time = time.time()
    job = sampler.run(circuits)
    elapsed_time = time.time() - start_time
    # For debugging only need to show the first circuit
    if show_ckt:
        # If GPU is available, assume in Perlmutter and use simple draw method since mpl is not supported
        if ('GPU' in backend.available_devices()):
            display(arg_dict['data_circs'][0].draw())
        else:
            
            display(arg_dict['data_circs'][0].draw("mpl", scale=0.5, fold=100))
    return job, elapsed_time


def submit_real_job(arg_dict, account_name, named_backend, test_run, rc=0, shots_per_iter=-1, specifications=False, t_seed = None):
    """
    Transpiles circuits found in arg_dict then submits them as jobs to the IBM backend from the named_backend argument
    Assumes check for real backend name handled outside this function
    """
    backend = get_named_backend(named_backend, account_name)
    print("Backend: \n", backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler = t_seed)
    if "ibm" in named_backend:
        circuits = [pm.run(circuit) for circuit in arg_dict['data_circs']]
    else:
        raise ValueError(f'Real backend must be prefixed with "ibm".')
    options = SamplerOptions()
    options.default_shots=arg_dict['shots']
    
    # Pauli Twirling/randomized compilation
    # These options only work on real HW; used to reduce HW errors. See https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques
    if rc > 0:
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True
        options.twirling.num_randomizations= rc
        print("RC enabled with", rc, "randomizations and",math.ceil(shots_per_iter/rc), "shots per randomization.")
    print("Debug: specifications: ", specifications)
    if t_seed:
        #layout=circuits[0]._layout.final_index_layout(filter_ancillas=True)
        print('seed=%d'%(t_seed))
    if specifications:
        gate_count = circuits[0].count_ops()
        ckt_depth = circuits[0].depth()
        t2qb_depth = circuits[0].depth(filter_function=lambda x:x.operation.num_qubits > 1)

        print('Transpiled Circuit Specifications:')
        print('Grouped gate count dict:', gate_count)
        print('Transpiled Circuit depth:', ckt_depth)
        print('Transpiled Circuit 2qb depth:', t2qb_depth)
        try:
            physQBlayout = circuits[0]._layout.final_index_layout(filter_ancillas=True)
            nqTotal = len(physQBlayout)
        except Exception as e:
            physQBlayout = [i for i in range(nqTotal)]
        print('Transpiled Circuit width:', nqTotal)
        print("layout:", physQBlayout)

    sampler = Sampler(mode=backend, options=options)

    if test_run:
        print("Test run selected; not submitting to real backend.")
        return -1
    job = sampler.run(circuits)  # ewb 8/7/2025, sending over all circuits
    job_id = job.job_id()
    print(f">>> Job ID: {job_id}") # it will take a while for the job to run , hours to days
    submission_info_dict = {
        'job_id': job_id,
        'local submission time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'backend_name': named_backend,
        'total_shots': arg_dict['shots'],
        'rc': rc,
        'num_circuits': len(circuits),
        'nq_addr': arg_dict['nq_addr'],
        'transpiled gate count': gate_count,
        'transpiled depth': ckt_depth,
        'transpiled 2qb depth': t2qb_depth,
        'transpiled circuit width': nqTotal
    }
    save_job_submission(submission_info_dict)
    return job_id

def qcrank_to_ehands_decoding(counts, nq_addr, ehands=False, target_qubit=''):
    """
    Calculates the EV values for each circuit in the job submission
    """
    tile_list = []
    print(f"Decoding {len(counts)} tiles of size {2**(nq_addr/2)} x {2**(nq_addr/2)}")
    # for each circuit
    for i in range(len(counts)):
        # Calculate the EV for each address
        restored_image = get_EV(counts = counts[i], nq_addr = nq_addr, target_qubit=target_qubit)
        tile_list.append(restored_image)
    return tile_list



def retrieve_real_results(job_id, nq_addr, target_qubit, overlap, n_tiles, account_name, data=None, padding=True):
    """
    When a real IBM QPU job is finished running, this retrieves the results
    Input: a list of jobs ids (max 2 for now) needed to reconstruct a given input. 
    Assume gx is the leading job id if two are provided
    """

    if account_name == '':
        raise ValueError("Please provide an account name")
    account_credentials_name = account_name
    
    counts = load_job_result(job_id)
    if counts == -1:
        # echo check, ewb
        print("account_credentials_name=", account_credentials_name)
        service = QiskitRuntimeService(name=account_credentials_name)
        counts= save_job_results(job_id, service)
    target_qubit = target_qubit
    tile_output = qcrank_to_ehands_decoding(counts, nq_addr, ehands=True, target_qubit=target_qubit)
    # With improvements from V2, overlap is no longer nededed but kept in for now just in case it has applications in the future
    if overlap:
        overlap_mode=0 # TODO: Implement mode 1: overlap with value averaging
    else:
        overlap_mode=2
    stitched_output = reconstruct_from_tiles(tile_output, n_tiles=n_tiles, data=data, mode=overlap_mode, padding=padding)
    #plt.imshow(tile_output, cmap='gray')
    return stitched_output


def sobel_driver(image = 8, mode = 0, compact = False, reset = False, shots_per_pixel = 10000, circuit_display_mode = 1, tile_calc_value = 1, overlap = True, barrier = True, circuit_specifications = 1, output_charts = False, force_CPU = False, named_backend=False, account_name='', rc=0, test_run=False, tile_calc_mode=0, t_seed = None, save_charts = False):
    """
    Primary driver for the quantum Sobel implementation
    
    :param image: either an int representing the length of a square randomized image to be generated or the file path to the image 
    :param mode: the type of circuit to be generated
        0: gx 
        1: gy
        2: gx^2
        3: gy^2
        4: gx^2 + gy^2
    :param compact: whether the circuit will use the compact implementation of the circuit (extended implementation has been removed for V3, so until other implementations are developed it is the only available)
    :param reset: whether the EHands circuit blocks will use the more compact resettable implementation or the reversible implementation that requires unique ancilla qubits 
    :param shots_per_pixel: the number of shots per pixel
    :param circuit_display_mode: if the circuit layout is to be displayed, and if so, how much of the circuit is displayed
        0: No circuit display
        1: Display the entire circuit
        2: Display just the sobel convolution circuit
    :param tile_calc_value: When calculating the amount of tiles, value used to determine how the image is broken up into tiles (used alongside tile_calc_mode)
    :param overlap: whether tile overlapping is used in the circuits (not used in V3, but kept in case it is still useful)
    :param barrier: whether the circuits have barriers included for better readability (not used during actual job execution, as the barriers may affect runtime)
    :param circuit_specifications: whether to include details regarding circuit specifications, such as circuit shape (depth and width) and gate count
    :param output_charts: whether output verification charts are used
    :param force_CPU: force the simulator to exclusively use CPU instead of defaulting to GPU if it is available
    :param named_backend: if specified, the name of the backend to be used (either prefaced with fake_ or ibm_)
    :param account_name: IBM account used for real HW runs
    :param rc: Pauli Twirling feature used for real HW runs
    :param test_run: prevents real HW runs from being submitted to IBM; used to test transpilation and circuit outputs 
    :param tile_calc_mode: When calculating the amount of tiles, determines whether the value passed by tile_calc_value is the number of tiles or the length of each tile
        0: number of tiles
        1: length of the tiles 
    :param t_seed: transpilation seed
    :param save_charts: where to save the output charts
    """
    #-------------------------------Current unimplemented features:-----------------------------------
    if not compact and reset:
        raise ValueError("Reset qubits are not implemented for the extended circuit. Please set reset to False or use the compact circuit.") 
    if mode == 4:
        raise ValueError("Gx^2 + Gy^2 is not recommended for simulation use and is untested on QPUs. Please set mode to 0, 1, 2, or 3.") 
    if barrier and circuit_display_mode == 0:
        raise ValueError("Barriers are only useful for circuit display. Please disable barriers or set circuit_display_mode to 1 or 2.")
    if not compact and reset:
        raise ValueError("Reset qubits are not implemented for the extended circuit. Please set reset to False or use the compact circuit.")
    if not compact and (mode == 2 or mode == 3 or mode == 4):
        raise ValueError("The extended circuit is not implemented for Gx^2 or Gx^2 + Gy^2. Please set mode to 0, 1, or 2.")

    #-------------------------------Calculations for circuit width (does not account qubits needed for addressing; that is handled by QCrank)-----------------------------------
    # Circuit width scaling factors
    # Non squared gradients only need 1 gradient subcircuit, while squared gradients need 2 gradient subcircuits
    if mode == 0 or mode == 1:
        circ_width_scaling = 1
        square_val = 1
    elif mode == 2 or mode == 3:
        circ_width_scaling = 2
        square_val = 2
    # Leave out Gx^2 + Gy^2 for now, as it is untested on QPUs and not recommended for simulation use

    # How much to reduce the classical output or increase the quantum output to make the results comparable
    if compact:
        # NOTE: Scaling down the classical output is preferred, as scaling up the quantum output requires more shots to retain the same theoretical accuracy
        # 1/8x scaling from sequential weighted sum, but 2x scaling for extending input range from [0, 1] to [-1, 1]
        output_scaling = 8
        shots_per_pixel = shots_per_pixel
        # Base offset for the ancilla qubits used for sequential summations
        if reset:
            # All sequential summations use the same ancilla qubits reset after each operation
            parity_ancilla = 1
            reset_offset = 1
        else:
            parity_ancilla = 2 # For the current implementation at most 2 consecutive weighted summations are used
            reset_offset = 0
        # Compact circuit does not use scaling qubits for the data qubits representing the neighboring qubits
        scaling_qbs = 0

    # Define data being acted on
    # if image is None, generate a random square image
    if type(image) == int:
        # Check if image value is a power of 2 greater than 4
        if image < 4 or (image & (image - 1)) != 0:
            raise ValueError("Image size must be a power of 2 greater than 4")
        data = np.random.randint(0, 256, size=(image, image), dtype=np.uint8)
        data = data.reshape(image, image)
    # if image provided, convert it to a numpy array of grayscale values
    else:
        image = Image.open(image)
        gray = image.convert('L')
        gray = np.array(gray)
        data = gray

    # Calculate total number of data qubits/channels
    channels = 6 * circ_width_scaling 

    # In updated July 2025 implementation, the target qubit contains the final EV of the chosen address is now the Least Significant Data qb
    target_qubit_index = 0
    
    image_len = data.shape[0]
    if tile_calc_mode == 0: # tile_calc_value represents the number of tiles
        n_tiles = tile_calc_value
        tile_length = int(image_len/math.sqrt(n_tiles))
    else: # tile_calc_values represents the tile_length 
        tile_length = tile_calc_value
        n_tiles = (image_len//tile_length)**2

    # total shots
    shots = shots_per_pixel*tile_length**2 

    if tile_length > 64 and compact and (mode == 2 or mode == 3) and ('GPU' in AerSimulator().available_devices()):
        raise ValueError("Tile length is too large for the compact Gx^2 or Gy^2 circuit to fit into a GPU. Please increase the number of tiles to allow for smaller tile sizes.")

    # if image is not square and tile length is not specified, raise error 
    if data.shape[0] != data.shape[1] and tile_calc_mode == 0:
        raise ValueError("Non-square images require tile length to be specified. Please set tile_calc_mode to 1 and provide a tile_calc_value.")

    print("\n\n\n\n\n -------mode-------:\n", mode)
    if mode == 0:
        calculation_mode = 'Gx'
    if mode == 1:
        calculation_mode = 'Gy'
    if mode == 2:
        calculation_mode = 'Gx^2'
    if mode == 3:
        calculation_mode = 'Gy^2'

    # Create 6 shifted versions of the data array, then break them up into tiles 
    tile_list = tile_image(data, n_tiles=n_tiles, mode=mode, tile_len=tile_length)
    output_tile_list = []    
    runtime_list = []
    nq_addr  = get_nq_addr(tile_list)

    # if circuit has addr as LSQBs, then the offset is the size of the address qubits
    offset = nq_addr
    print("Offset for target qubit: ", offset)
    if compact and reset:
        offset = 1

    # create target qubit string used when retrieving specific counts for qiskit's result count dictionary
    target_qubit = get_target_qubit(channels, target_qubit_index, parity_ancilla)

    # encode the numpy array containing the tiled data into a multi channel QCrank array
    # each QCrank is responsible for the same tiled region of the main image for all shifted image, each channel encoding data for one of the images
    encoding_args = ehands_encoding_to_qcrank(tile_list, ehands=True, show_ckt=False, channels=channels, offset=offset, shots=shots, mode=mode,compact=compact, tile_length=tile_length, overlap=False)

    #print("Number of tiles: ", (tile_list.shape[2]))
    #print("number of data_circs:", len(encoding_args['data_circs']))
    
    print('_____________________________________\n')
    print('+++++ Quantum Sobel Driver +++++++')
    print('_____________________________________\n')
    print('Beginning Calculations for ', calculation_mode)
    print('Shots per pixel:', shots_per_pixel)
    if rc > 0: # how many shots are allocated to each random iteration if Pauli Twirling is enabled 
        print('Total number of shots per circuit:', math.ceil(shots/rc))
    else:
        print('Total number of shots per circuit:', shots)

    # generate and append the EHands circuit that handles the chosen Sobel calculations to the end of each tiled QCrank circuit
    for i in range(tile_list.shape[2]):
        if (tile_list.shape[2]<=4) or (i % (tile_list.shape[2]//4))==0:
            print('\n_____________________________\n')
            print(f'Processing Tile {i}')
        # TODO: See if offset and channels are still needed for calculations within encoding and decoding functions, if not remove them from parameters and just keep them as an external variable

        # Total circuit with is the number of QCrank data qubits/channels, QCrank address qubits, and number of parity ancilla required to support the EHands Sobel circuits         
        circ_width = encoding_args['channels'] + encoding_args['nq_addr'] + parity_ancilla

        if (not named_backend or "ibm_" not in named_backend) and (circ_width > 34):
            raise ValueError("Circuit width is too large for Simulation. Please increase the number of tiles or use smaller tile sizes.")
        if i == 0:
            print('Circuit width:', circ_width)

        # initialize primary circuit, which is the QCrank width + additional ancilla qubits
        main_circ = QuantumCircuit(circ_width, circ_width+reset_offset)

        # shifted_storage_qubits = list(range(encoding_args['offset'], encoding_args['channels'] + encoding_args['nq_addr'] + encoding_args['offset']))
        sobel_width =  encoding_args['channels'] + encoding_args['nq_addr']
        shifted_storage_qubits = list(range(0, sobel_width))
        # turn this into a for loop here for tileing?

        # insert the QCrank circuit into the full circuit
        main_circ.compose(encoding_args['data_circs'][i], qubits = shifted_storage_qubits, inplace=True)
        encoding_args['data_circs'][i] = main_circ

        # reverse the bits to match orientation in V1
        main_circ.circuit=main_circ.reverse_bits()
        ehands_sobel_circ = QuantumCircuit(circ_width, circ_width+reset_offset)

        if compact:
            # apply the first stencil operation
            x_grad1_coords = [0, 1, 2, 3, 4, 5]
            # x_grad1_coords = [5, 4, 3, 2, 1, 0]     
            ehands_sobel_circ = apply_compact_sobel(ehands_sobel_circ, x_grad1_coords, offset=offset, ancilla_addr=sobel_width, last_ancilla=False, with_resets=reset, barrier=barrier, circuit_width=circ_width)
            # for mode 2 and 3 need Gx^2 and/ Gy^2, we need to apply the stencil operation twice
            if mode == 2 or mode == 3:
                # apply the second x gradient
                x_grad2_coords = [6, 7, 8, 9, 10, 11]
                # x_grad2_coords = [11, 10, 9, 8, 7, 6]
                if reset:
                    ancilla_addr = 0
                else:
                    ancilla_addr = 2
                # apply the second stencil operation
                ehands_sobel_circ = apply_compact_sobel(ehands_sobel_circ, x_grad2_coords, offset=offset, ancilla_addr=ancilla_addr, last_ancilla=False, with_resets=reset, barrier=barrier)
                # multiply both stencil operations together, with 0 storing the product
                data_channel_width = encoding_args['channels']//2
                ehands_sobel_circ = ehands_product_with_memory(ehands_sobel_circ, target_qubit_index+data_channel_width+offset, target_qubit_index+offset)

        

        # Display just the sobel sub circuit
        if circuit_display_mode==2 and i == 0:
            if ('GPU' in AerSimulator().available_devices()):
                display(ehands_sobel_circ.draw())
            else:
                display(ehands_sobel_circ.draw("mpl", scale=1, fold=100))
        main_circ.barrier()
        # append sobel circuit to data qubits of the main QCrank circuit
        all_qubits = list(range(0, circ_width))
        main_circ.compose(ehands_sobel_circ, qubits = all_qubits, inplace=True)     
        
        # Flipped for consistency with previous implementations 
        encoding_args['data_circs'][i] = main_circ
        
    # End for loop here; qcrank is built to parse multiple circuits, so it shouldn't be called multiple times.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # Display the full circuit
    if circuit_display_mode==1:
        show_ckt = True
    else:
        show_ckt = False
    # Display circuit specifications
    if circuit_specifications == 1:
        specifications = True
    else:
        specifications = False
    print(f"Total circuits in job: {len(encoding_args['data_circs'])}")
    # add measurement gates
    encoding_args = add_measurement(encoding_args, ehands=True, target_qubit=target_qubit, n_ancilla=offset)
    # execute a simulation job
    if not named_backend or "ibm" not in named_backend:
        job, runtime = execute_sim_job(encoding_args, account_name, show_ckt=show_ckt, named_backend=named_backend, force_CPU=force_CPU, specifications=specifications)
        jobRes=job.result()
        #print("Job Result-------\n\n", jobRes)
        #TODO See if this needs to be modified; likely not needed
        counts=[jobRes[i].data.c.get_counts()  for i in range(len(encoding_args['data_circs']))]
    # execute a hardware job
    else: 
        # For real backends, we need to run the job and get the results
        if rc > 0:
            jobID = submit_real_job(encoding_args, account_name, named_backend=named_backend, test_run=test_run, rc=rc, shots_per_iter=shots, specifications=specifications, t_seed=t_seed)
        else:
            jobID = submit_real_job(encoding_args, account_name, named_backend=named_backend, test_run=test_run, specifications=specifications, t_seed=t_seed)
        if not test_run:
            print(f'Circuit Queued for execution on IBMQ {named_backend} Job ID: {jobID}')
        # return early with the job ID
        return jobID
    if (tile_list.shape[2]<=4) or (i % (tile_list.shape[2]//4))==0:
        print(f'Quantum runtime: {runtime:.3f} seconds')
    runtime_list.append(runtime)
    # Decoding function built to handle multiple circuits, assuming counts list has multiple dicts
    nq_addr  = get_nq_addr(tile_list)
    # decode the EHands output values, which represent the Sobel values
    output_tile_list = qcrank_to_ehands_decoding(counts, nq_addr, ehands=True, target_qubit=target_qubit)
    if overlap:
        overlap_mode=0
    else:
        overlap_mode=2
    # reassemble the EV values into a single image
    stitched_output = reconstruct_from_tiles(output_tile_list, n_tiles=n_tiles, data=data, mode=overlap_mode, padding=False)
    print(runtime_list)
    total_runtime = sum(runtime_list)
    if n_tiles > 1:
        print("Slowest tile runtime: ", max(runtime_list))
    print("total runtime: ", total_runtime)

    # imitate the value wrapping from the quantum circuit for fair comparison with classical sobel values
    padded_data = wrap_pad_image(data, 1, 1, 1, 1)
    comparison = sobel_classical(padded_data, mode=mode)
    comparison = comparison[1:-1, 1:-1]
    # scale down comparison values    
    comparison = (comparison/(output_scaling*0.5)**square_val)
    output_list = stitched_output

    if output_charts:
        # Display the charts
        file_path = ""
        if save_charts:
            dir = os.path.join("OutputCharts")
            os.makedirs(dir, exist_ok=True)
            file_path = os.path.join(dir, "ideal_" + calculation_mode + "_" + str(len(data)) + "x" + str(len(data)) + "_output_charts.pdf")
            print("Saving to: ", file_path)
        verification_charts(output_list, comparison, calculation_mode, data, hw="ideal", trim=0, file_path=file_path)
    return output_list, total_runtime

def get_real_results(image, job_id, n_tiles, overlap, mode=-1, account_name='', padding=0, trim = 0, save_charts=False, display_charts=False, keep_padding=False):
    """
    QIMP specific functon
    Retrieves the chosen IBM HW job upon its completion and generates analysis charts on its performance
    :param image: the file path to the input image used for the job submission; used to generate the classical Sobel values for comparison with the quantum results
    :param job_id: the job ID of the chosen IBM HW job submission
    :param n_tiles: the number of tiles the input image was broken into for the job submission; used to properly reconstruct the quantum results from their tiled format
    :param overlap: whether tile overlapping was used in the job submission; used to properly reconstruct the quantum results from their tiled format
    :param mode: the type of circuit that was generated for the job submission
        0: gx 
        1: gy
    :param account_name: IBM account used for real HW runs; used to retrieve the job results
    :param padding: the amount of padding that was added to the input image for the job submission; 
    :param trim: the amount of boundary rows/cols to be trimmed from the quantum output and classical comparison values for the analysis charts; used to account for cases where the quantum results are less accurate around the edges of the image, which is often the case when padding is used
    :param save_charts: whether to save the analysis charts generated for the quantum results;
    :param display_charts: whether to display the analysis charts generated for the quantum results; if False, charts will be saved but not displayed
    :param keep_padding: whether to keep the padding in the quantum output and classical comparison values for the analysis charts; if False, padding will be removed
    """
    if mode == 0:
            calculation_mode = 'Gx'
    if mode == 1:
            calculation_mode = 'Gy'
    if mode == 2:
            calculation_mode = 'Gx^2'
    if mode == 3:
            calculation_mode = 'Gy^2'

    # Hardcoded values for now; will need to fix later 
    image = Image.open(image)
    gray = image.convert('L')
    gray = np.array(gray)
    data = gray
    print("Shape of input image:", data.shape)
    tile_list=tile_image(data, n_tiles=n_tiles)
    #print(tile_list)
    channels = 6
    offset = 2
    target_qubit_index = 0
    nq_addr = get_nq_addr(tile_list)
    target_qubit = get_target_qubit(channels, target_qubit_index, offset)

    output_scaling = 8
    square_val = 1
    if mode == 0:
        gx_output = retrieve_real_results(job_id, nq_addr, target_qubit=target_qubit,n_tiles=n_tiles, overlap=overlap, account_name=account_name, padding=padding, data=data)
        gx_output = np.array(gx_output)
        output_list = gx_output #gx_output*((output_scaling)**square_val)
        # print("Gx Output:\n", output_list)
        # print("Gx Shape:", output_list.shape)
        # return output_list
    else:
        gy_output = retrieve_real_results(job_id, nq_addr, target_qubit=target_qubit,n_tiles=n_tiles, overlap=overlap, account_name=account_name, padding=padding, data=data)
        gy_output = np.array(gy_output)
        output_list = gy_output #gy_output*((output_scaling)**square_val)

    if padding>0: 
        comparison_list = sobel_classical(data, mode=mode)
    else: 
        padded_data = wrap_pad_image(data, 1, 1, 1, 1)
        comparison_list = sobel_classical(padded_data, mode=mode)
        comparison_list = comparison_list[1:-1, 1:-1]
    comparison_list = (comparison_list  /(output_scaling*0.5)**square_val)
    if np.corrcoef(comparison_list, output_list)[0, 1] < 0.9:
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
        print("Warning: Low correlation between quantum and classical results:", np.corrcoef(comparison_list, output_list)[0, 1])
        print("If this is lower than expected, could be the incorrect job or would require another job to rerun the circuit")
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
    file_path = ""
    if save_charts:
        # # TODO: See if this output directory works on different OS
        # dir = "./OutputCharts/"
        # file_name = job_id + "_output_charts.pdf"
        # print("Saving figure to: ", file_name)
        dir = os.path.join("OutputCharts")
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, job_id + "precalibration_output_charts.pdf")
        print("Saving to: ", file_path)
    hw_backend = load_job_backend(job_id)
    # ---------------------------------------------------------------------------------------------------------
    # Temporary scaling adjustment based on linear fit. 
    # TODO Replace with as more permanent solution that makes use of a appended set of 10 randomized circuits at the end of the job batch, as Jan suggested
    cl = comparison_list
    ol = output_list
    if padding>0:
        cl = cl[padding:-padding, padding:-padding]
        ol = ol[padding:-padding, padding:-padding]
        if keep_padding:
            print("Keeping padding in final output, but using trimmed data for scaling adjustment")
            comparison_list = comparison_list[padding:-padding, padding:-padding]
            output_list = output_list[padding:-padding, padding:-padding]
       # print("Cl and Ol shapes after trimming for scaling adjustment:", cl.shape, ol.shape)
    cl_flat = cl.flatten()
    ol_flat = ol.flatten()
    m, b = np.polyfit(cl_flat, ol_flat, 1)
    print("Adjusting scaling by factor m:", m)
    #comparison_list = comparison_list * m
    output_list = output_list / m
    # ---------------------------------------------------------------------------------------------------------
    file_path = ""
    if save_charts:
        # # TODO: See if this output directory works on different OS
        dir = os.path.join("OutputCharts")
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, job_id + "_output_charts.pdf")
        print("Saving to: ", file_path)
    if display_charts:
        verification_charts(ol, cl, calculation_mode, data, trim=trim, hw=hw_backend, file_path=file_path, precalibration=True)
        verification_charts(output_list, comparison_list, calculation_mode, data, hw=hw_backend, trim=trim,  file_path=file_path)    
    return output_list