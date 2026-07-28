"""Smoke tests for the command-line interface."""

import subprocess
import sys
from pathlib import Path


def test_training_cli_displays_configuration_help() -> None:
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/run.py", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Maximum epochs" in result.stdout
    assert "Path to waveform data" in result.stdout
