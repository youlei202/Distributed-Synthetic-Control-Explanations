#!/usr/bin/env python3
"""Demo script showing complete DISCO pipeline."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False

def main():
    """Run complete DISCO demo."""
    print("🎯 DISCO Pipeline Demo")
    print("This script demonstrates the complete DISCO workflow.")

    # Set up paths
    base_dir = Path(__file__).parent.parent
    exp_dir = base_dir / "experiments"
    devices_dir = exp_dir / "demo_devices"
    results_dir = exp_dir / "results" / "demo_run"

    # Clean up previous runs
    if devices_dir.exists():
        shutil.rmtree(devices_dir)
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Step 1: Train devices
    success = run_command([
        sys.executable, "-m", "disco.cli.main", "train-devices",
        "--dataset", "adult",
        "--N", "5",
        "--non-iid", "label-skew",
        "--out", str(devices_dir),
        "--seed", "42"
    ], "Training heterogeneous devices")

    if not success:
        print("Failed to train devices, exiting...")
        return

    # Step 2: Compute probe trajectories
    probes_dir = results_dir / "probes"
    success = run_command([
        sys.executable, "-m", "disco.cli.main", "probe",
        "--devices", str(devices_dir),
        "--dataset", "adult",
        "--probes", "200",
        "--theta0", "[0.0,0.5,1.0]",
        "--interventions", str(exp_dir / "configs" / "tabular_interventions.yaml"),
        "--out", str(probes_dir),
        "--seed", "42"
    ], "Computing probe trajectories")

    if not success:
        print("Failed to compute probes, exiting...")
        return

    # Step 3: Run explanations
    success = run_command([
        sys.executable, "-m", "disco.cli.main", "explain",
        "--devices", str(devices_dir),
        "--probes", str(probes_dir),
        "--dataset", "adult",
        "--targets", "0,1",  # First two devices as targets
        "--queries", "10",   # Small number for demo
        "--interventions", str(exp_dir / "configs" / "tabular_interventions.yaml"),
        "--K", "30",
        "--tau", "0.5",
        "--lam", "0.01",
        "--mode", "p",
        "--baselines", "avg,lp",  # Just a few baselines for demo
        "--out", str(results_dir),
        "--seed", "42"
    ], "Running DISCO explanations")

    if not success:
        print("Failed to run explanations, exiting...")
        return

    # Step 4: Show results structure
    print(f"\n🎉 Demo completed successfully!")
    print(f"Results saved in: {results_dir}")
    print("\nResult structure:")

    # List result files
    if results_dir.exists():
        for item in sorted(results_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(results_dir)
                print(f"  {rel_path}")

    print(f"\nYou can explore the results by examining:")
    print(f"  - Probe trajectories: {probes_dir}")
    print(f"  - DISCO results: {results_dir / 'disco'}")
    print(f"  - Device models: {devices_dir}")

    print(f"\nTo run evaluation (not yet implemented):")
    print(f"  python -m disco.cli.main evaluate --results {results_dir}")

if __name__ == "__main__":
    main()