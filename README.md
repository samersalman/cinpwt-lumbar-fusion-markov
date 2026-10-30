# ciNPWT after posterior lumbar fusion: Markov cost-utility analysis

This repository contains the numerical analysis code for:

> **Cost-Effectiveness of Negative-Pressure Wound Therapy After Posterior Lumbar Fusion: A Markov Cost-Utility and Risk-Threshold Analysis**

The model compares contemporary surgical-site infection (SSI) prevention with a standard dressing against the same prevention strategy plus closed-incision negative-pressure wound therapy (ciNPWT). It uses a cohort Markov cost-utility model from a US commercial-payer perspective, with monthly cycles and a 24-month base-case horizon.

## Included analyses

- Base-case cost, QALY, infection, reoperation, and net-benefit outcomes
- Baseline SSI-risk, device-price, and treatment-effect threshold analyses
- One-way deterministic sensitivity analysis
- Probabilistic sensitivity analysis with 10,000 seeded simulations
- Prespecified scenario analyses and budget impact
- Structural, closed-form, and face-validity checks

This repository intentionally excludes all figure-generation, table-generation, manuscript-generation, and document-formatting code.

## Repository structure

```text
analysis/
  inputs.py              Parameter store and evidence provenance
  model.py               Markov model engine
  run_basecase.py        Base-case analyses
  run_threshold.py       Risk, price, and relative-risk thresholds
  run_dsa.py             Deterministic sensitivity analysis
  run_psa.py             Probabilistic sensitivity analysis
  run_scenarios.py       Scenarios and budget impact
  validation.py          Validation checks
  data/
    cpi_medical.csv      BLS medical-care CPI series
    life_table_2023.csv  2023 US life-table mortality
results/                 Reference numerical outputs
run_analysis.py          One-command analysis runner
```

## Reproduce the analyses

Python 3.12 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

The probabilistic sensitivity analysis is the slowest stage and may take several minutes. It uses NumPy's random generator with seed `12345`. Exact PSA draws depend on the pinned NumPy and SciPy versions.

Each analysis script may also be run independently from the `analysis/` directory. All numerical outputs are written to `results/`; rerunning the pipeline replaces files with the same names.

## Model notes

- Baseline 90-day all-SSI risk is evaluated from 0.5% to 15.0%.
- The reported risk anchors are 2.0%, 6.6%, and 11.4%.
- The base-case ciNPWT price is $250 and relative risk is 0.61.
- Costs are expressed in 2025 US dollars and QALYs and costs are discounted at 3% annually.
- The primary willingness-to-pay threshold is $100,000/QALY.
- Model inputs, distributions, source citations, source locations, and residual assumptions are documented directly in `analysis/inputs.py`.

The two principal unresolved inputs are explicitly modeled as assumptions: the institutional net acquisition price of ciNPWT and the duration of superficial SSI. Users should substitute local acquisition prices and infection risks when applying the model to a specific institution.

## Public data files

`cpi_medical.csv` contains annual values from the US Bureau of Labor Statistics CPI-U medical-care series `CUUR0000SAM`. `life_table_2023.csv` contains sex-specific mortality probabilities from *United States Life Tables, 2023* (National Vital Statistics Reports 74(6), DOI: 10.15620/cdc/174591). Retrieval details are preserved in the file headers.

## Citation

Please cite the associated article and the archived software release. Release-specific citation metadata and the Zenodo DOI can be added to `CITATION.cff` after the first GitHub release is archived by Zenodo.

## License

This software is released under the [MIT License](LICENSE). The included US federal data are public-domain source data and retain their source notices.
