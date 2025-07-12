#!/usr/bin/env python3
"""
Simple data generation example

Usage:
    python simple_data_gen.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.data.generation import HawkesGenerator


def main() -> None:
    # Create simple Hawkes generator
    generator = HawkesGenerator(
        num_sequences=100, max_time=10.0, mu=0.5, alpha=0.8, beta=1.0
    )

    # Generate data
    sequences = generator.generate()

    # Save to file
    output_path = Path("./generated_hawkes_data.json")
    generator.save_sequences(sequences, output_path)
    print(f"Generated {len(sequences)} sequences, saved to {output_path}")


if __name__ == "__main__":
    main()
