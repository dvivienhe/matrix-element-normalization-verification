# Normalization verification of optical and transfer matrix elements in semiconductor heterostructures

Programs and input data for the article

> De He, Zhiyi Huang, Lianhe Li, Haibo Zhang, *Normalization verification of optical and transfer matrix elements in
> semiconductor heterostructures*, submitted to Computer Physics Communications.

Running `run_all.py` regenerates every computed table and the eight figures of the article and compares them
with the expected values in `expected_outputs.json`.

## Installation

Python 3.10 with the packages of `requirements.txt` (NumPy, SciPy, Matplotlib, fontTools) and the Arial font:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
```

## Reproducing the results

```
python run_all.py
```

The programs in `code/` run in order and write to `results/` (2.5–4.5 min in the recorded environment). Tables and figures are compared
byte by byte, JSON results after their run-identity fields are removed; the exit code is 0 only if every program
succeeds and every output matches. Byte identity is claimed for the recorded environment (Windows 11, CPython
3.10.0, `requirements.txt`, Arial); on other platforms the last digits of a few values may differ by round-off.

| Article item | Program | Input data |
|---|---|---|
| Figs. 1-2, Table 2 | `controlled_validation.py` | - |
| Fig. 3 | `reference/circular_branches.py` | - |
| Fig. 4, Table 3, Tables A.2-A.3 | `reference/run_references.py` | `data/reference/` |
| Table 4 (upper half) | `evidence_rows.py` | - |
| Fig. 5, Table 4 (lower half), Table S8 | `case144_analysis.py` | `data/case144/` |
| Fig. 6, Table 5 | `solcore_interface/analyse_states.py` | `data/solcore/` |
| Fig. S1, Table S5 | `mixed_error_methods.py` | - |
| Fig. S2, Tables 6 and S9 | `make_figures.py` (Fig. S2) | `data/public_solver_scan.json` |
| Table S4 | `potential_convention.py` | - |
| Section S4 (wall offset) | `wall_offset.py` | - |
| Table A.4, numbers quoted in the text | `text_values.py` | `data/solcore/identity.json` |
| Figures 1-6, S1, S2 | `make_figures.py` | outputs above |

## Licence and citation

Program code: MIT. Data, tables and figures: CC BY 4.0. See `LICENSE`. Please cite the article and this
repository (`CITATION.cff`); version 1.0.0 is archived on Zenodo, https://doi.org/10.5281/zenodo.24280593.
The program code was developed with the assistance of AI tools; see the declaration in the article.
