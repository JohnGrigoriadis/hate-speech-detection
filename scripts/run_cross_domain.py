"""Convenience script: run the cross-domain experiment only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hate_speech.experiments.cross_domain import main

if __name__ == "__main__":
    main()
