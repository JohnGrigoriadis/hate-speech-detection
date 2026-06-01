"""Convenience script: run the in-domain experiment only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hate_speech.experiments.in_domain import main

if __name__ == "__main__":
    main()
