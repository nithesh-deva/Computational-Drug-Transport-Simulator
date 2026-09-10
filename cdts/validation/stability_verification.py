"""
M6 Stability Verification Framework

Implements rigorous stability analysis for numerical solvers:
- Stability condition checking before execution
- Fourier number (r) analysis
- Safe configuration validation
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StabilityVerifier:
    """Verify stability conditions for numerical methods."""
    
    @staticmethod
    def check_explicit_fdm(
        diffusion_coeff: float,
        dx: float,
        dt: float,
        strict: bool = True
    ) -> Tuple[bool, Dict]:
        """Verify explicit FDM stability condition.
        
        Explicit FDM for diffusion is stable if:
            r = D·Δt/Δx² ≤ 0.5
        
        Args:
            diffusion_coeff: Diffusion coefficient D [m²/s]
            dx: Spatial step [m]
            dt: Time step [s]
            strict: If True, require r < 0.5 (margin); if False, allow r ≤ 0.5
            
        Returns:
            Tuple:
                - is_stable: True if configuration is stable
                - info: Dictionary with stability details
        """
        if diffusion_coeff < 0:
            raise ValueError(f"Diffusion must be non-negative, got {diffusion_coeff}")
        if dx <= 0 or dt <= 0:
            raise ValueError(f"dx and dt must be positive, got dx={dx}, dt={dt}")
        
        r = diffusion_coeff * dt / (dx ** 2)
        stability_limit = 0.5
        
        if strict:
            is_stable = r < stability_limit
            margin = stability_limit - r
        else:
            is_stable = r <= stability_limit
            margin = stability_limit - r
        
        max_stable_dt = stability_limit * dx ** 2 / diffusion_coeff if diffusion_coeff > 0 else np.inf
        
        info = {
            'method': 'explicit_fdm',
            'fourier_number': r,
            'stability_limit': stability_limit,
            'is_stable': is_stable,
            'margin': margin,
            'max_stable_dt': max_stable_dt,
            'ratio': r / stability_limit,
        }
        
        return is_stable, info
    
    @staticmethod
    def check_crank_nicolson(
        diffusion_coeff: float,
        dx: float,
        dt: float,
        clearance: float = 0.0
    ) -> Tuple[bool, Dict]:
        """Verify Crank-Nicolson stability.
        
        Crank-Nicolson is unconditionally stable for linear problems,
        but we still compute the Fourier number for reference.
        
        Args:
            diffusion_coeff: Diffusion coefficient [m²/s]
            dx: Spatial step [m]
            dt: Time step [s]
            clearance: Clearance coefficient [1/s]
            
        Returns:
            Tuple:
                - is_stable: Always True (unconditionally stable)
                - info: Stability information dict
        """
        if dx <= 0 or dt <= 0:
            raise ValueError(f"dx and dt must be positive, got dx={dx}, dt={dt}")
        
        r = diffusion_coeff * dt / (dx ** 2)
        
        info = {
            'method': 'crank_nicolson',
            'stability': 'unconditionally_stable',
            'fourier_number': r,
            'note': 'Stable for any r, but large r may reduce accuracy',
            'diffusion_coeff': diffusion_coeff,
            'clearance': clearance,
        }
        
        return True, info
    
    @staticmethod
    def recommend_time_step(
        diffusion_coeff: float,
        dx: float,
        method: str = 'explicit_fdm',
        safety_factor: float = 0.4
    ) -> float:
        """Recommend safe time step for given grid and diffusion.
        
        Args:
            diffusion_coeff: Diffusion coefficient [m²/s]
            dx: Spatial step [m]
            method: 'explicit_fdm' or 'crank_nicolson'
            safety_factor: Fraction of stability limit (default 0.4 for explicit)
                          Ignored for CN (always returns 0 with note)
            
        Returns:
            Recommended time step [s]
        """
        if diffusion_coeff <= 0:
            raise ValueError(f"Diffusion must be positive for dt recommendation")
        if dx <= 0:
            raise ValueError(f"dx must be positive")
        if not (0 < safety_factor <= 1):
            raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
        
        if method == 'explicit_fdm':
            # r = 0.5 is stability limit, use safety factor
            dt = safety_factor * dx ** 2 / (2 * diffusion_coeff)
        elif method == 'crank_nicolson':
            # Unconditionally stable; recommend based on accuracy rather than stability
            # Use Courant number ~ 1 for reasonable accuracy
            dt = dx ** 2 / (2 * diffusion_coeff)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return dt


class StabilityReport:
    """Generate comprehensive stability verification report."""
    
    def __init__(self):
        """Initialize report."""
        self.checks = []
        self.recommendations = []
    
    def add_check(
        self,
        name: str,
        method: str,
        diffusion_coeff: float,
        dx: float,
        dt: float,
        clearance: float = 0.0,
        passed: bool = True
    ):
        """Add stability check result.
        
        Args:
            name: Check identifier
            method: Numerical method
            diffusion_coeff: Diffusion coefficient
            dx: Spatial step
            dt: Time step
            clearance: Clearance coefficient
            passed: Whether check passed
        """
        check = {
            'name': name,
            'method': method,
            'D': diffusion_coeff,
            'dx': dx,
            'dt': dt,
            'k': clearance,
            'passed': passed,
        }
        
        # Compute Fourier number
        r = diffusion_coeff * dt / (dx ** 2)
        check['fourier_number'] = r
        
        if method == 'explicit_fdm':
            check['stability_limit'] = 0.5
            check['is_within_limit'] = r <= 0.5
        elif method == 'crank_nicolson':
            check['stability'] = 'unconditionally_stable'
            check['is_within_limit'] = True
        
        self.checks.append(check)
    
    def add_recommendation(
        self,
        scenario: str,
        dx: float,
        recommended_dt: float,
        method: str,
        reason: str
    ):
        """Add time step recommendation.
        
        Args:
            scenario: Scenario description
            dx: Spatial step
            recommended_dt: Recommended time step
            method: Method name
            reason: Reason for recommendation
        """
        rec = {
            'scenario': scenario,
            'dx': dx,
            'recommended_dt': recommended_dt,
            'method': method,
            'reason': reason,
        }
        self.recommendations.append(rec)
    
    def summary(self) -> Dict:
        """Generate report summary.
        
        Returns:
            Dictionary with report statistics
        """
        n_checks = len(self.checks)
        n_passed = sum(1 for c in self.checks if c['passed'])
        n_failed = n_checks - n_passed
        
        summary = {
            'total_checks': n_checks,
            'passed': n_passed,
            'failed': n_failed,
            'all_passed': n_failed == 0,
            'checks': self.checks,
            'recommendations': self.recommendations,
        }
        
        return summary
    
    def print_summary(self):
        """Print human-readable summary."""
        summary = self.summary()
        
        print("\n" + "="*60)
        print("STABILITY VERIFICATION REPORT")
        print("="*60)
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print()
        
        for check in summary['checks']:
            status = "✓ PASS" if check['passed'] else "✗ FAIL"
            print(f"{status} | {check['name']}")
            print(f"      Method: {check['method']}, D={check['D']:.2e}, dx={check['dx']:.2e}, dt={check['dt']:.2e}")
            if 'fourier_number' in check:
                print(f"      Fourier number r = {check['fourier_number']:.6f}", end="")
                if 'stability_limit' in check:
                    print(f" (limit: {check['stability_limit']:.1f})")
                else:
                    print()
        
        print()
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"{i}. {rec['scenario']}")
            print(f"   Recommended dt = {rec['recommended_dt']:.2e} s")
            print(f"   Method: {rec['method']}")
            print(f"   Reason: {rec['reason']}")
        print("="*60 + "\n")
