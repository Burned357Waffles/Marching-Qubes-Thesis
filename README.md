# Marching-Qubes-Thesis (QCrank / eHANDS vertex classification)

*Thanks to CursorAI for helping with this file.*

This repository contains experiments for **tiled, per-pixel binary classification** of image regions using a **QCrank-based quantum data encoding** plus an **eHANDS-style weighted subtraction gadget**. The main script in this repo runs the end-to-end pipeline:

- load a grayscale image region and normalize it
- split the region into tiles (with padding for edge tiles)
- build/execute a quantum circuit per tile (simulation or IBM hardware)
- recover expectation values, convert them to binary classes, and score against a classical baseline
- write plots and CSV summaries

## Key files

- `ehands_qcrank_vertex_classification_V1.py`: main driver + plotting utilities (CLI runnable)
- `test_images/`: sample images for runs
- output folders (created automatically):
  - `classification_summaries/`
  - `side-by-sides/`
  - `residual_plots/`
  - `JobOutputs/` (hardware job submission metadata)

## Requirements

### Python

- Python 3.10+ recommended

### Python packages

The main script imports:

- `numpy`, `matplotlib`, `Pillow` (PIL)
- `python-dotenv`
- `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`

Install into a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pillow python-dotenv qiskit qiskit-aer qiskit-ibm-runtime
```

## External dependency: `DATA_ENCODER_CIRCUITS_PATH`

`ehands_qcrank_vertex_classification_V1.py` expects an environment variable named `DATA_ENCODER_CIRCUITS_PATH` that points to a folder containing a Python package named `datacircuits/` with:

- `datacircuits/ParametricQCrankV2.py` (providing `ParametricQCrankV2`)

The script will add `DATA_ENCODER_CIRCUITS_PATH` to `sys.path` at runtime.

### Set it in PowerShell (recommended)

```powershell
$env:DATA_ENCODER_CIRCUITS_PATH="C:\path\to\data-encoder-circuits"
python .\ehands_qcrank_vertex_classification_V1.py --help
```

### Or set it via `.env`

This repo’s `.gitignore` ignores `.env`. You can create/edit `.env` in the repo root:

```env
DATA_ENCODER_CIRCUITS_PATH=C:\path\to\data-encoder-circuits
```

## Running the experiment (simulation)

The script supports multiple “test” modes through `--test`:

- `full`: run the default driver (sweeps shots exponents and tile sizes)
- `shots`: sweep shots exponents only
- `tile`: run the tile-size loop for a single shots exponent `--shots-coef-k`

Example (simulation with Aer for 5 iterations):

```bash
python .\ehands_qcrank_vertex_classification_V1.py --image-path test_images\test_concentric_circles_16x16.png --save-name circle_16x16_method_1_iso_0_aer --isovalue-mode fixed --isovalue 0 --c-mode 1 --test tile --backend aer --iterations 5
```

### Running a single tile size

If you pass `--tile-width` and `--tile-height`, the tile sweep becomes a single-tile-size run.

```bash
python .\ehands_qcrank_vertex_classification_V1.py --image-path test_images\test_concentric_circles_16x16.png --save-name circle_16x16_method_1_iso_0_aer --isovalue-mode fixed --isovalue 0 --c-mode 1 --test tile --backend aer --iterations 5 --tile-width 4 --tile-height 4
```

## Running on IBM hardware

To run on a QPU, set:

- `--run-mode hardware`
- `--ibm-backend` (example: `ibm_marrakesh`)
- `--account-name` (your locally configured `QiskitRuntimeService` account name)

Optional:

- `--rc` to enable twirling/randomized compilation with `rc > 0`
- `--hw-opt-level` and `--hw-seed-transpiler`
- `--hw-submit-only` to submit and exit after printing a job id (no result wait/plots)

Example:

```bash
python .\ehands_qcrank_vertex_classification_V1.py --image-path test_images\test_concentric_circles_16x16.png --save-name circle_16x16_method_1_iso_0_aer --isovalue-mode fixed --isovalue 0 --c-mode 1 --test tile --iterations 1 --shots-coef-k 12 --run-mode hardware --ibm-backend ibm_marrakesh --account-name "<your_account_name>" --hw-submit-only
```

Hardware submissions also write a JSON record to `JobOutputs/job_submission_info.json`.

## Outputs

Depending on the run mode and test selection, the script writes:

- **CSV summaries** in the repo root (e.g. `*_results.csv`)
- **Classification summary figure** to `classification_summaries/`
- **Side-by-side image/classification figure** to `side-by-sides/`
- **Residual scatter plot** (classical weighted subtraction vs quantum EV) to `residual_plots/`

## Notes / troubleshooting

- If you see `DATA_ENCODER_CIRCUITS_PATH environment variable is not set`, set it (see above) and rerun.
- If your editor/linter can’t resolve `datacircuits.*`, that typically means it doesn’t know about `DATA_ENCODER_CIRCUITS_PATH`. The script itself inserts that path at runtime.

