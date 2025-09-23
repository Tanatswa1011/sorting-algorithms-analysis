# Sorting Benchmarks Project

This project is part of my dissertation.  
It benchmarks four sorting algorithms: **Bubble Sort, Merge Sort, Quick Sort, and TimSort**.  
The aim is to test them in a resource-constrained environment using Python.

## Files
- `benchmarks.py` → main script to run benchmarks (produces CSVs and optional plots).
- `notebooks/Sorting_Benchmark_Analysis.ipynb` → Jupyter notebook for cleaning and analyzing results.
- `results/` → contains CSV outputs, summaries, and plots after running benchmarks.

## Requirements
Install the needed libraries first:

```bash
pip install numpy pandas matplotlib psutil xlsxwriter
