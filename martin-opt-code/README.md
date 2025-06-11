# Perfect linear concept manipulation

This project explores methods for finding linear transformations `P` and optional bias terms `b` that modify a feature representation `x` such that the transformed features `Px + b` become statistically independent of a sensitive or confounding variable `z`, while remaining close to the original features `x`.

## Core Problems

The central tasks involve solving variations of the following optimisation problems:

1.  **Least-Squares Concept Erasure:**
    Find a transformation matrix $P$ that minimises the distortion while ensuring zero covariance between transformed features and sensitive labels:
    $$
    \min_{P} \mathbb{E}_{x}\left[ \| Px - x \|_{M}^{2} \right] \quad \text{subject to} \quad \text{Cov}(Px, z) = 0
    $$

2.  **Least-Squares Concept Switching:**
    Find a transformation matrix $P$ and a bias vector $b$ that minimise the distortion while ensuring zero covariance:
    $$
    \min_{P, b} \mathbb{E}_{x}\left[ \| Px + b - x \|_{M}^{2} \right] \quad \text{subject to} \quad \text{Cov}(Px + b, z) = 0
    $$

where:
- $P$ is the transformation matrix.
- $b$ is the bias vector.
- $x$ represents the input features.
- $z$ represents the sensitive labels or concepts we want to remove correlation with.
- $M$ is a metric (often the identity matrix), defining the norm $\| \cdot \|_M$.
- $\mathbb{E}_{x}[\cdot] denotes the expectation over the distribution of $x$.
- $\text{Cov}(\cdot, z)$ is the cross-covariance between the transformed features and the sensitive labels.

The goal is to find $P$ (and potentially $b$) that minimises the distortion $\| Px (+ b) - x \|_{M}^{2}$ while ensuring the transformed features have zero covariance with $z$.

This project implements variations of the **Condat-Vu primal-dual algorithm** to solve these constrained optimisation problems efficiently.

## Project Structure

```
.
├── LICENSE          # Project licence file
├── README.md        # Project description
├── requirements.txt # Project dependencies
├── data/            # Data files (e.g., x_data.pt, z_labels.pt)
│   ├── data_generation.py # Script to generate synthetic data
│   ├── x_data.pt
│   └── z_labels.pt
├── results/         # Output files (e.g., optimal transformations, visualisations)
│   ├── A_optimal.pt
│   ├── b_optimal.pt
│   ├── new_transformation_visualisation_with_bias.png
│   ├── P_optimal_with_bias.pt
│   ├── P_optimal.pt
│   ├── transformation_visualisation_with_bias.png
│   └── transformation_visualisation.png
├── scripts/         # Utility and main execution scripts
│   ├── compute_covariances.py # Computes and saves covariance matrices
│   ├── concept_erasure_with_bias.py # Solves the optimisation problem with bias
│   ├── concept_erasure.py  # Solves the original optimisation problem (no bias)
│   └── concept_switching.py # Solves a newer/alternative optimisation problem with bias
├── src/             # Source code for support functions
│   └── support_functions.py
└── tests/           # Unit tests
    └── test_support_functions.py
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd perfect-linear-concept-manipulation
    ```
2.  Create and activate a virtual environment (Python 3.9+ recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Generate Data:**
    First, you need the input data (`x_data.pt`) and labels (`z_labels.pt`). If they don't exist, run the data generation script:
    ```bash
    python data/data_generation.py
    ```
    This will create the necessary files in the `data/` directory.

2.  **Compute Covariances (Optional but Recommended):**
    Pre-computing covariance matrices can speed up the optimisation scripts.
    ```bash
    python scripts/compute_covariances.py
    ```
    This script typically loads data from `data/` and saves computed covariances (e.g., `Sigma_xx`, `Sigma_xz`, `Sigma_zz`) potentially back to `data/` or a dedicated cache directory (check script specifics).

3.  **Solve the Optimisation Problem(s):**
    Run the desired script to find the optimal transformation:

    *   **Least-Squares Concept Erasure:**
        ```bash
        python scripts/concept_erasure.py
        ```
        Saves `P_optimal.pt` and `transformation_visualisation.png` to `results/`.

    *   **Least-Squares Concept Erasure with Bias:** (Assuming this corresponds to the old "Concept Switching")
        ```bash
        python scripts/concept_erasure_with_bias.py
        ```
        Saves `P_optimal_with_bias.pt`, `b_optimal.pt`, and `transformation_visualisation_with_bias.png` to `results/`.

    *   **Concept Switching:** (Assuming this corresponds to the old "New/Alternative Problem")
        ```bash
        python scripts/concept_switching.py
        ```
        Check the script for specific outputs (e.g., might save `A_optimal.pt`, `b_optimal.pt`, `new_transformation_visualisation_with_bias.png`).

    These scripts typically:
    - Load data and/or pre-computed covariances.
    - Run the Condat-Vu iterations (or other relevant algorithm).
    - Save the resulting optimal transformation parameters (`P`, `b`, etc.) to the `results/` directory.
    - Generate visualisations comparing data before and after transformation, saved to `results/`.

## Running Tests

To run the unit tests, navigate to the project's root directory in your terminal and run:
```bash
python -m unittest discover tests
```
This command will automatically discover and run all tests within the `tests/` directory.

## Contributing

Information on how to contribute to the project (optional).