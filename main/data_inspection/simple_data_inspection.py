#!/usr/bin/env python3
"""
Simple data inspection example

Usage:
    python simple_data_inspection.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.evaluation.distribution_analysis_helper import DistributionAnalyzer
from easy_tpp.data.preprocess import TPPDataset


def main() -> None:
    # Load dataset
    dataset = TPPDataset.load_from_file("path/to/your/data.json")
    
    # Create analyzer
    analyzer = DistributionAnalyzer(
        dataset=dataset,
        output_dir="./inspection_results"
    )
    
    # Run analysis
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_event_types()
    analyzer.generate_summary_report()
    
    print("Data inspection completed. Results saved to ./inspection_results")


if __name__ == "__main__":
    main()
