

# OR–Odorant Benchmark (M2OR)

Generate **benchmark datasets** for predicting interactions between **olfactory receptors (ORs)** and **odorants/compounds** using the **M2OR** database.
The tool creates “**cold-receptor**” and “**cold-ligand**” splits while controlling per-class imbalance and the similarity thresholds used to cluster receptors and compounds.

---

## Table of contents

* [Features](#features)
* [Repository structure](#repository-structure)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Parameters](#parameters)
* [Outputs](#outputs)
* [Similarity matrices](#similarity-matrices)
* [Customize clustering & splits](#customize-clustering--splits)
* [Data files](#data-files)
* [Notes](#notes)

---

## Features

* Cold-receptor and cold-ligand split strategies.
* Configurable class imbalance per receptor/ligand.
* Threshold-based clustering of receptors and compounds.
* Optional inclusion of mutant entries.
* Plug-in similarity matrices (receptors and compounds).

---

## Repository structure

```
.
├── Main_BMG.py                # Example script / entry point
├── Benchmark_generator_OR.py  # Core benchmark generation logic
├── dataframe_utils.py         # Dataframe helpers, Clustering & splitting utilities (edit linkage here)
├── Similitud_compounds.py     # Compound similarity computation
├── Needleman-Wunsch.py        # Alignment → compound_sim_matrix.csv
├── receptor_sim_matrix.npy    # Receptor similarity matrix (default input)
└── M2OR/
    ├── main_receptors.csv
    ├── main_compounds.csv
    └── pairs.csv              # Base data (';' delimiter, 'id' as index)
```

---

## Installation

```bash
python >= 3.9
pip install numpy pandas matplotlib
```

Place the M2OR CSVs under `M2OR/` and ensure `receptor_sim_matrix.npy` is accessible at the project root (or pass a custom path via `Sim_matrix_r`).

---

## Quick start

```python
from Main_BMG import benchmark_dataset_or

# Defaults
bm_default = benchmark_dataset_or()

# Custom thresholds and higher allowed imbalance
bm_custom = benchmark_dataset_or(Imb_lvl=10, th_c=0.5, th_r=0.7, LOG=True)

# Access splits (contain pair_id assignments)
receptor_split = bm_custom.cold_receptor_split
ligand_split   = bm_custom.cold_ligand_split

# Export parameters and results
info = bm_custom.get_parameters_generation()
```

---

## Parameters

| Name           | Type  | Default                     | Description                                                                   |
| -------------- | ----- | --------------------------- | ----------------------------------------------------------------------------- |
| `Imb_lvl`      | int   | `3`                         | Maximum allowed **class imbalance** per receptor/ligand when building splits. |
| `th_c`         | float | `0.6`                       | **Similarity threshold** for clustering **compounds** (odorants).             |
| `th_r`         | float | `0.6`                       | **Similarity threshold** for clustering **receptors**.                        |
| `LOG`          | bool  | `False`                     | Verbose logging to console.                                                   |
| `Mutations`    | bool  | `False`                     | Include (`True`) or exclude (`False`) **mutant** entries.                     |
| `Sim_matrix_r` | str   | `'receptor_sim_matrix.npy'` | Path to the `.npy` file with the **receptor similarity matrix**.              |

---

## Outputs

* **`cold_receptor_split`** — structure with `pair_id` assignments for the **cold-receptor** split.
* **`cold_ligand_split`** — structure with `pair_id` assignments for the **cold-ligand** split.
* **`List_L`, `List_R`** — helper lists of ligand and receptor clusters/IDs.

> The cold splits ensure that receptors (or ligands) in **test** do **not** appear in **train**.

---

## Similarity matrices

* **Receptors**: loaded from `receptor_sim_matrix.npy` (override with `Sim_matrix_r=...`).
* **Compounds (odorants)**:

  * `Needleman-Wunsch.py` computes alignments and produces `compound_sim_matrix.csv`.
  * `Similitud_compounds.py` contains the logic to build the compound similarity matrix (adjust scoring here if needed).

---

## Customize clustering & splits

Edit these functions in `dataframe_utils.py` to change clustering behavior (e.g., linkage: `single`, `complete`, `average`, `ward`) and sample assignment to splits:

* `clustering_and_split_compounds(...)`
* `clustering_and_split_receptors(...)`

These functions control how clusters are formed using `th_c` and `th_r`, and how samples are assigned to the cold splits.

---

## Data files

* `M2OR/main_receptors.csv`, `M2OR/main_compounds.csv`, `M2OR/pairs.csv`

  * Delimiter: `;`
  * Index column: `id`

---

## Notes

* Ensure `pair_id` values in `pairs.csv` are consistent with receptor and compound IDs in the base tables.
* When `Mutations=True`, decide how mutant entries are grouped and split downstream.

---

> *Tip:* If you expose the class as a package, consider adding docstrings and type hints (already shown above) and publishing API docs with Sphinx or MkDocs.
