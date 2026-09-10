"""
M10 Performance Dashboard

CLI-based dashboard for viewing analysis results and experiment metadata.
"""

import logging
from typing import List, Optional
from cdts.analysis.database import ResultDatabase, ExperimentComparison, PerformanceAnalyzer
from cdts.analysis.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """CLI dashboard for experiment analysis."""

    def __init__(self, db_path: str, tracker_path: str):
        """Initialize dashboard.
        
        Args:
            db_path: Path to HDF5 result database
            tracker_path: Path to SQLite tracker database
        """
        self.db = ResultDatabase(db_path)
        self.tracker = ExperimentTracker(tracker_path)
        self.comparison = ExperimentComparison(self.db)
        self.analyzer = PerformanceAnalyzer(self.db)

    def print_database_summary(self):
        """Print overall database summary."""
        stats = self.tracker.get_statistics()
        print("\n" + "=" * 80)
        print("CDTS EXPERIMENT DATABASE SUMMARY")
        print("=" * 80)
        print(f"Total Experiments:    {stats['total_experiments']}")
        print(f"Unique Solvers:       {stats['unique_solvers']}")
        print(f"Unique Backends:      {stats['unique_backends']}")
        print(f"Avg Runtime (s):      {stats['average_runtime_seconds']:.4f}")
        print("=" * 80 + "\n")

    def print_experiment_list(self, limit: Optional[int] = None):
        """Print list of all experiments.
        
        Args:
            limit: Maximum number to display
        """
        exp_ids = self.tracker.list_all_experiments()
        if limit:
            exp_ids = exp_ids[:limit]

        print("\n" + "=" * 120)
        print(f"{'Exp ID':<30} {'Solver':<15} {'Backend':<12} {'nx':<8} {'nt':<8} {'Domain(m)':<12} {'Runtime(s)':<12}")
        print("=" * 120)

        for exp_id in exp_ids:
            experiments = self.tracker.query_experiments()
            exp = next((e for e in experiments if e['id'] == exp_id), None)
            if exp:
                metrics = self.tracker.get_experiment_metrics(exp_id)
                runtime = metrics['runtime_seconds'] if metrics else 0.0
                print(
                    f"{exp_id:<30} {exp['solver_method']:<15} {exp['backend']:<12} "
                    f"{exp['nx']:<8} {exp['nt']:<8} {exp['domain_length']:<12.2e} {runtime:<12.4f}"
                )

        print("=" * 120 + "\n")

    def print_experiment_details(self, exp_id: str):
        """Print detailed information for an experiment.
        
        Args:
            exp_id: Experiment ID
        """
        experiments = self.tracker.query_experiments()
        exp = next((e for e in experiments if e['id'] == exp_id), None)

        if not exp:
            print(f"Experiment {exp_id} not found")
            return

        metrics = self.tracker.get_experiment_metrics(exp_id)
        performance = self.tracker.get_performance_comparison(exp_id)

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp['name']}")
        print("=" * 80)

        print("\nConfiguration:")
        print(f"  ID:                  {exp['id']}")
        print(f"  Solver:              {exp['solver_method']}")
        print(f"  Backend:             {exp['backend']}")
        print(f"  Grid Size:           {exp['nx']} × {exp['nt']}")
        print(f"  Domain Length:       {exp['domain_length']:.2e} m")
        print(f"  Spatial Step:        {exp['domain_length']/exp['nx']:.2e} m")
        print(f"  Temporal Step:       {exp['final_time']/exp['nt']:.2e} s")
        print(f"  Diffusion Coeff:     {exp['diffusion_coeff']:.2e} m²/s")
        print(f"  Clearance Coeff:     {exp['clearance']:.2e} 1/s")
        print(f"  Final Time:          {exp['final_time']:.2e} s")

        if metrics:
            print("\nMetrics:")
            print(f"  Peak Concentration:  {metrics['peak_concentration']:.4e} mol/m³")
            print(f"  Penetration Depth:   {metrics['penetration_depth']:.4e} m")
            print(f"  Arrival Time:        {metrics['arrival_time']:.4e} s")
            print(f"  Initial Mass:        {metrics['total_mass_initial']:.4e} mol/m")
            print(f"  Final Mass:          {metrics['total_mass_final']:.4e} mol/m")
            print(f"  Mass Loss:           {metrics['mass_loss_percent']:.2f}%")
            print(f"  Runtime:             {metrics['runtime_seconds']:.4f} s")

        if performance:
            print("\nPerformance Benchmarks:")
            print(f"  {'Backend':<15} {'Time(s)':<15} {'Throughput(pts/s)':<20} {'Speedup':<10}")
            print(f"  {'-'*15} {'-'*15} {'-'*20} {'-'*10}")
            for perf in performance:
                speedup_str = f"{perf['speedup']:.2f}×" if perf['speedup'] else "—"
                throughput_str = f"{perf['throughput_points_per_s']:.2e}" if perf['throughput_points_per_s'] else "—"
                print(
                    f"  {perf['backend']:<15} {perf['total_time_s']:<15.4f} {throughput_str:<20} {speedup_str:<10}"
                )

        print("=" * 80 + "\n")

    def print_comparison_results(self, exp_ids: List[str], metric: str = "penetration"):
        """Print comparison of multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            metric: Metric to compare (penetration, peak_concentration, total_mass, arrival_time)
        """
        print("\n" + "=" * 100)
        print(f"COMPARISON: {metric.upper()}")
        print("=" * 100)

        if metric == "penetration":
            results = self.comparison.compare_penetration_depth(exp_ids)
            print(f"{'Experiment':<30} {'Penetration Depth (m)':<20} {'Relative':<15}")
            print("-" * 100)
            if results:
                max_val = max(results.values())
                for exp_id, val in results.items():
                    relative = val / max_val * 100 if max_val > 0 else 0
                    print(f"{exp_id:<30} {val:<20.4e} {relative:<15.1f}%")

        elif metric == "peak_concentration":
            results = self.comparison.compare_peak_concentration(exp_ids)
            print(f"{'Experiment':<30} {'Peak Concentration (mol/m³)':<30}")
            print("-" * 100)
            for exp_id, val in results.items():
                print(f"{exp_id:<30} {val:<30.4e}")

        elif metric == "total_mass":
            results = self.comparison.compare_total_mass(exp_ids)
            print(f"{'Experiment':<30} {'Initial Mass':<20} {'Final Mass':<20} {'Loss %':<15}")
            print("-" * 100)
            for exp_id, (initial, final) in results.items():
                loss_pct = 100 * (initial - final) / initial if initial > 0 else 0
                print(f"{exp_id:<30} {initial:<20.4e} {final:<20.4e} {loss_pct:<15.2f}%")

        elif metric == "arrival_time":
            results = self.comparison.compare_arrival_time(exp_ids)
            print(f"{'Experiment':<30} {'Arrival Time (s)':<20}")
            print("-" * 100)
            for exp_id, val in results.items():
                if val == float('inf'):
                    print(f"{exp_id:<30} {'Never reached':<20}")
                else:
                    print(f"{exp_id:<30} {val:<20.4e}")

        print("=" * 100 + "\n")

    def print_performance_scaling(self, exp_id: str):
        """Print performance scaling information.
        
        Args:
            exp_id: Experiment ID
        """
        performance = self.tracker.get_performance_comparison(exp_id)

        if not performance:
            print(f"No performance data for {exp_id}")
            return

        print("\n" + "=" * 80)
        print(f"PERFORMANCE SCALING: {exp_id}")
        print("=" * 80)
        print(f"{'Backend':<20} {'Time (s)':<15} {'Throughput (pts/s)':<25} {'Speedup':<15}")
        print("-" * 80)

        baseline_time = performance[0]['total_time_s'] if performance else 1.0

        for perf in performance:
            speedup = baseline_time / perf['total_time_s'] if perf['total_time_s'] > 0 else 1.0
            throughput = f"{perf['throughput_points_per_s']:.2e}" if perf['throughput_points_per_s'] else "—"
            print(
                f"{perf['backend']:<20} {perf['total_time_s']:<15.4f} {throughput:<25} {speedup:<15.2f}×"
            )

        print("=" * 80 + "\n")
