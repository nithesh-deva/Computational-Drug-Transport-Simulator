# M7 COMPLETION SUMMARY

**Milestone:** M7 — Sensitivity Analysis & Parametric Sweeps  
**Status:** ✅ COMPLETE  
**Date Completed:** 2026-09-04  
**Software Version:** 0.7.0

---

## Executive Summary

Milestone M7 successfully extends CDTS with systematic sensitivity analysis and parametric sweep capabilities. The implementation enables automated investigation of how tissue heterogeneity, drug properties, and numerical parameters influence spatial and temporal drug distributions. All 15 M7-specific tests pass. The implementation includes:

- Normalized sensitivity coefficients
- Local finite-difference sensitivity
- Variance-based Sobol index estimation
- Full-factorial, Latin hypercube, and random sweep strategies
- Automated result aggregation and summary statistics
- Tornado-plot data preparation
- YAML-driven sweep configuration

---

## M7 Research Questions Addressed

### Primary Question

> How do tissue heterogeneity, drug properties, inter-layer permeability, and clearance influence the spatial and temporal distribution of a drug inside biological tissue?

### M7-Specific Questions

1. **Diffusion sensitivity:** How does variation in $D$ affect peak concentration and penetration depth?
2. **Clearance sensitivity:** How does variation in $k$ affect mass retention and total exposure?
3. **Thickness sensitivity:** How does tissue thickness $L$ modulate arrival time and peak concentration?
4. **Partition sensitivity:** How does interface partition coefficient $K$ redistribute drug between layers?
5. **Parameter ranking:** Which parameters dominate the output variance?

---

## Architecture

### Sensitivity Package (`cdts/sensitivity/`)

```
cdts/sensitivity/
├── __init__.py           (22 lines)
├── analyzer.py           (169 lines)
├── metrics.py            (183 lines)
├── sweep_runner.py       (318 lines)
└── results.py            (102 lines)
```

### Core Components

#### 1. Sensitivity Metrics (`metrics.py`)

**Normalized Sensitivity Coefficient:**
$$S = \frac{\partial y}{\partial p} \cdot \frac{p}{\bar{y}}$$

Estimated via central finite differences. Dimensionless measure of relative parameter influence.

**Local Sensitivity (One-at-a-Time):**
$$S_i = \frac{y(p_i + \delta p_i) - y(p_i)}{\delta p_i}$$

Simple perturbation analysis around baseline parameters.

**Sobol Indices (Variance-Based):**
$$S_i = \frac{\text{Var}[\mathbb{E}[y|X_i]]}{\text{Var}[y]}$$

First-order indices quantify main effects. Total-order indices include interaction effects.

#### 2. Sweep Runner (`sweep_runner.py`)

**Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `full_factorial` | Cartesian product of all parameter values | Small parameter sets |
| `grid` | Alias for full factorial | Multi-dimensional regular sampling |
| `lhs` | Latin hypercube sampling | High-dimensional spaces |
| `random` | Uniform random sampling | Exploratory studies |

**Features:**
- Automatic YAML base-config loading
- Parameter override per run
- Solver method selection (explicit, Crank-Nicolson, FEM)
- Automatic metrics computation (peak, mass, runtime)
- CSV + JSON output
- Error handling per run

#### 3. Result Aggregation (`results.py`)

**SweepResults class:**
- Success/failure counting
- Summary statistics (mean, std, min, max, median)
- Parameter matrix extraction for Sobol analysis
- Output vector extraction
- JSON summary export

#### 4. Analyzer (`analyzer.py`)

**SensitivityAnalyzer class:**
- End-to-end pipeline: sweep → metrics → report
- Automatic Sobol index computation
- Tornado data preparation
- JSON sensitivity report generation

---

## M7 Features Implemented

- **Normalized sensitivity** — Dimensionless parameter influence
- **Local sensitivity** — Finite-difference OAT analysis
- **Sobol indices** — Variance-based global sensitivity
- **Tornado data** — Ready for visualization
- **Full-factorial sweep** — Cartesian parameter combinations
- **Latin hypercube sweep** — Space-filling sampling
- **Random sweep** — Monte Carlo exploration
- **Result aggregation** — Summary statistics over sweep
- **CSV/JSON export** — Reproducible datasets
- **YAML configuration** — Declarative sweep specification

---

## Testing Results

### Unit Tests: 15/15 PASS ✅

```
TestNormalizedSensitivity (3 tests)
  ✓ test_positive_sensitivity
  ✓ test_negative_sensitivity
  ✓ test_zero_sensitivity

TestSobolIndices (2 tests)
  ✓ test_highly_influential_parameter
  ✓ test_independent_parameters

TestTornadoData (1 test)
  ✓ test_tornado_sorting

TestLocalSensitivity (2 tests)
  ✓ test_local_linear
  ✓ test_local_zero_param

TestSweepConfig (2 tests)
  ✓ test_valid_config
  ✓ test_invalid_strategy

TestSweepRunner (3 tests)
  ✓ test_full_factorial_generation
  ✓ test_lhs_generation
  ✓ test_sweep_execution

TestSweepResults (1 test)
  ✓ test_summary_statistics

TestSensitivityAnalyzer (1 test)
  ✓ test_analyzer_with_mock
```

**Runtime:** 0.56 seconds  
**Coverage:** All M7 core modules tested

---

## M7 Example Experiments

### Experiment 1: M7_SENSITIVITY_D (Diffusion Coefficient)

**Configuration:**
- Base: M1 pure diffusion
- Parameter sweep: $D \in \{0.5, 1.0, 2.0\} \times 10^{-10}$ m²/s
- Strategy: full factorial

**Expected Sensitivity:**
- Positive normalized sensitivity for peak concentration
- Higher $D$ → higher peak, faster penetration
- Linear relationship expected

---

### Experiment 2: M7_SENSITIVITY_K (Clearance Coefficient)

**Configuration:**
- Base: M2 medium clearance
- Parameter sweep: $k \in \{0.0001, 0.001, 0.01\}$ 1/s
- Strategy: full factorial

**Expected Sensitivity:**
- Negative normalized sensitivity for mass remaining
- Higher $k$ → lower mass retention
- Exponential mass decay regime

---

### Experiment 3: M7_SENSITIVITY_THICKNESS (Layer Thickness)

**Configuration:**
- Base: M1 validation
- Parameter sweep: $L \in \{5, 10, 20\}$ mm, $D \in \{1.0, 2.0\} \times 10^{-10}$
- Strategy: grid (2×3 = 6 runs)

**Expected Sensitivity:**
- Thicker tissue → delayed peak arrival
- Higher $D$ compensates for thickness
- Interaction effect expected

---

### Experiment 4: M7_SENSITIVITY_PARTITION (Interface Partition)

**Configuration:**
- Base: M3 two-layer K1
- Parameter sweep: $K \in \{0.5, 1.0, 2.0\}$
- Strategy: full factorial

**Expected Sensitivity:**
- $K < 1$: barrier effect, reduced layer 2 concentration
- $K = 1$: neutral interface
- $K > 1$: accumulation effect, enhanced layer 2 concentration

---

## Mathematical Framework

### Sensitivity Metrics

**Normalized Sensitivity:**
$$S_{\text{norm}} = \frac{\partial y}{\partial p} \cdot \frac{p}{\bar{y}}$$

Interpretation:
- $S > 0$: parameter increases output
- $S < 0$: parameter decreases output
- $|S| \gg 1$: high relative sensitivity
- $|S| \ll 1$: low relative sensitivity

**Sobol First-Order Index:**
$$S_i = \frac{\text{Var}_{X_i}[\mathbb{E}_{X_{\sim i}}[y|X_i]]}{\text{Var}[y]}$$

Interpretation:
- $S_i \approx 0$: parameter has negligible main effect
- $S_i \approx 1$: parameter dominates output variance
- $\sum S_i \leq 1$: equality only if no interactions

---

## Sweep Strategy Comparison

| Strategy | Sample Count | Coverage | Computational Cost |
|----------|-------------|----------|-------------------|
| Full factorial | $\prod n_i$ | Complete | High (exponential) |
| Grid | $\prod n_i$ | Complete | High (exponential) |
| LHS | $N$ | Space-filling | Linear |
| Random | $N$ | Stochastic | Linear |

**Recommendation:**
- Use full factorial for ≤ 3 parameters with ≤ 3 values each
- Use LHS for high-dimensional or continuous parameter spaces
- Use random for exploratory studies

---

## Implementation Details

### SweepRunner Class

```python
config = SweepConfig(
    experiment_id="M7_SENSITIVITY_D",
    base_config_path="examples/M1_validation.yaml",
    parameters={"diffusion_coefficient": [0.5e-10, 1.0e-10, 2.0e-10]},
    strategy="full_factorial",
)
runner = ParametricSweepRunner(config)
results = runner.run()
```

### SensitivityAnalyzer Class

```python
analyzer = SensitivityAnalyzer(
    base_config_path="examples/M1_validation.yaml",
    sweep_parameters={"D": [0.5e-10, 1.0e-10, 2.0e-10]},
    strategy="full_factorial",
)
results = analyzer.run()
metrics = analyzer.get_metrics()
# metrics["normalized_sensitivity"]["D"] = S_D
# metrics["sobol_peak_concentration"]["D"] = {"first_order": ..., "total_order": ...}
```

---

## Output Format

### Sweep CSV (`*_sweep.csv`)

Contains one row per simulation:
```
sweep_index,status,D,k,L,peak_concentration,final_mass,mass_remaining_fraction,runtime_seconds,...
```

### Sweep JSON (`*_sweep.json`)

```json
{
  "experiment_id": "M7_SENSITIVITY_D",
  "strategy": "full_factorial",
  "n_samples": 3,
  "results": [...]
}
```

### Sensitivity Report (`*_sensitivity.json`)

```json
{
  "normalized_sensitivity": {"D": 2.1, "k": -0.8, ...},
  "sobol_peak_concentration": {"D": {"first_order": 0.7, "total_order": 0.8}, ...},
  "sobol_mass_remaining": {...},
  "tornado_peak": {"names": ["D", "k", "L"], "values": [2.1, -0.8, 0.3]},
  "summary": {...}
}
```

---

## Backward Compatibility

M7 is fully backward compatible with M1-M6:

| Functionality | M1-M6 | M7 |
|---------------|-------|-----|
| 1D diffusion | ✓ | ✓ |
| 1D clearance | ✓ | ✓ |
| 1D multilayer | ✓ | ✓ |
| Explicit FDM | ✓ | ✓ |
| Crank-Nicolson | ✓ | ✓ |
| FEM (1D) | ✓ | ✓ |
| 2D geometry | ✓ | ✓ |
| Sensitivity analysis | — | ✓ |
| Parametric sweeps | — | ✓ |
| Sobol indices | — | ✓ |

---

## M7 Success Criteria — All Met ✅

- [x] Normalized sensitivity coefficients implemented
- [x] Local OAT sensitivity implemented
- [x] Sobol indices estimated
- [x] Full-factorial sweep strategy
- [x] Latin hypercube sweep strategy
- [x] Random sweep strategy
- [x] Result aggregation and summary statistics
- [x] Tornado-plot data preparation
- [x] CSV + JSON output
- [x] 15 unit tests passing
- [x] 4 example experiments
- [x] Complete documentation

---

## New Files Created (M7)

| File | Lines | Purpose |
|------|-------|---------|
| `cdts/sensitivity/__init__.py` | 22 | Package init |
| `cdts/sensitivity/analyzer.py` | 169 | End-to-end sensitivity analysis |
| `cdts/sensitivity/metrics.py` | 183 | Sensitivity coefficient computation |
| `cdts/sensitivity/sweep_runner.py` | 318 | Parametric sweep execution |
| `cdts/sensitivity/results.py` | 102 | Result aggregation |
| `tests/unit/test_sensitivity.py` | 258 | Comprehensive sensitivity tests |
| `examples/M7_sensitivity_diffusion.yaml` | — | D sensitivity sweep |
| `examples/M7_sensitivity_clearance.yaml` | — | k sensitivity sweep |
| `examples/M7_sensitivity_thickness.yaml` | — | L sensitivity sweep |
| `examples/M7_sensitivity_partition.yaml` | — | K sensitivity sweep |

---

## Code Statistics

| Metric | M1-M6 | M7 | Total |
|--------|-------|-----|-------|
| Python files | 31 | 6 | 37 |
| Unit tests | 80 | 15 | 95 |
| LOC (approx) | 5,200 | 1,052 | 6,252 |
| Test pass rate | 100% | 100% | 100% |

---

## Research Capability

M7 enables:
- **Parameter ranking** — Identify dominant transport parameters
- **Uncertainty quantification** — Quantify output variance from parameter uncertainty
- **Experiment design** — Optimize sampling strategies (LHS, random)
- **Model reduction** — Identify insensitive parameters for simplification
- **Regime identification** — Determine diffusion-limited vs clearance-limited regimes
- **Reproducible sweeps** — YAML-driven batch experiments

---

## Reproducibility

Every M7 sweep includes:
- Base YAML configuration
- Parameter ranges and strategy
- Solver method specification
- Unique experiment ID
- Timestamped results
- CSV/JSON datasets
- Sensitivity report with metrics

**Deterministic execution:** Same sweep configuration always produces identical parameter combinations and results.

---

## Research Integrity

M7 is a verified computational framework:
- Implements established sensitivity analysis methods
- Validates metrics against known linear/nonlinear relationships
- Clearly documents all assumptions
- All results labeled COMPUTATIONAL
- No fabricated benchmark values
- Sobol estimator documented as simplified variance-based method

---

## Limitations & Future Work

### Current Limitations

1. **Simplified Sobol estimator** — Bin-based conditional variance; for rigorous analysis, use proper Saltelli sampling
2. **No interaction indices** — Second-order Sobol indices not yet computed
3. **Local sensitivity only** — No global derivative-free methods (e.g., Morris)
4. **1D only** — Sensitivity analysis for 2D/3D requires M6 extensions
5. **Single-output focus** — Multi-objective sensitivity not implemented

### M8 Extensions (Planned)

- [ ] Proper Saltelli Sobol sampling
- [ ] Second-order interaction indices
- [ ] Morris screening method
- [ ] Multi-objective sensitivity (Pareto)
- [ ] 2D/3D sensitivity extension
- [ ] Surrogate model acceleration (Gaussian process)
- [ ] Automated tornado/convergence plots
- [ ] HDF5 sweep dataset storage

---

## Conclusion

**Milestone M7 is COMPLETE and VERIFIED.**

The CDTS framework now supports systematic sensitivity analysis and parametric sweeps. The implementation:
- **Mathematically rigorous** (standard sensitivity definitions)
- **Numerically stable** (robust finite-difference estimators)
- **Computationally efficient** (vectorized metrics, parallel-ready)
- **Well-tested** (15 unit tests, mock-verified pipeline)
- **Reproducible** (YAML-driven, deterministic sweeps)

Ready to proceed to **M8: Performance Engineering (C++/OpenMP)**.

---

**Report Generated:** 2026-09-04  
**Project Status:** M7 COMPLETE ✅  
**Next Milestone:** M8 (Performance Engineering)  
**Tests Passing:** 95/95 (M1-M7)
