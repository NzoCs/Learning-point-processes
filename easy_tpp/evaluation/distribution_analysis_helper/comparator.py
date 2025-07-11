"""
Main Temporal Point Process Comparator
"""

from easy_tpp.utils import logger
from .interfaces import DataExtractor, PlotGenerator, MetricsCalculator
from .data_extractors import LabelDataExtractor, SimulationDataExtractor, TPPDatasetExtractor
from .plot_generators import (
    InterEventTimePlotGenerator, 
    EventTypePlotGenerator, 
    SequenceLengthPlotGenerator, 
    CrossCorrelationPlotGenerator
)

from .metrics_calculator import MetricsCalculatorImpl
from easy_tpp.data.preprocess.dataset import TPPDataset
from typing import Dict, List, Any, Union
import os


class TemporalPointProcessComparator:
    """
    Main orchestrator class that follows Dependency Inversion Principle.
    Depends on abstractions rather than concrete implementations.
    """
    
    def __init__(
        self,
        label_extractor: DataExtractor,
        simulation_extractor: DataExtractor,
        plot_generators: List[PlotGenerator],
        metrics_calculator: MetricsCalculator,
        output_dir: str,
        auto_run: bool = True
    ):
        """
        Initialize with dependency injection (DIP).
        
        Args:
            label_extractor: Ground truth data extractor
            simulation_extractor: Simulation data extractor  
            plot_generators: List of plot generators
            metrics_calculator: Metrics calculator
            output_dir: Output directory path
            auto_run: Whether to run evaluation automatically
        """
        
        self.label_extractor = label_extractor
        self.simulation_extractor = simulation_extractor
        self.plot_generators = plot_generators
        self.metrics_calculator = metrics_calculator
        self.output_dir = output_dir
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {str(e)}")
            raise
        
        if auto_run:
            self.run_comprehensive_evaluation()
    
    def run_comprehensive_evaluation(self) -> Dict[str, float]:
        """Execute comprehensive evaluation using injected dependencies."""
        logger.info("Starting comprehensive temporal point process evaluation...")
        
        try:
            # Extract data using injected extractors
            data = {
                'label_time_deltas': self.label_extractor.extract_time_deltas(),
                'label_event_types': self.label_extractor.extract_event_types(),
                'label_sequence_lengths': self.label_extractor.extract_sequence_lengths(),
                'simulated_time_deltas': self.simulation_extractor.extract_time_deltas(),
                'simulated_event_types': self.simulation_extractor.extract_event_types(),
                'simulated_sequence_lengths': self.simulation_extractor.extract_sequence_lengths(),
            }
            
            # Generate all plots using injected generators
            plot_filenames = [
                "comparison_inter_event_time_dist.png",
                "comparison_event_type_dist.png", 
                "comparison_sequence_length_dist.png",
                "comparison_cross_correlation_moments.png"
            ]
            
            for generator, filename in zip(self.plot_generators, plot_filenames):
                output_path = os.path.join(self.output_dir, filename)
                generator.generate_plot(data, output_path)
            
            # Calculate metrics using injected calculator
            metrics = self.metrics_calculator.calculate_metrics(data)
            
            logger.info("Comprehensive evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {str(e)}")
            return {}


class TemporalPointProcessComparatorFactory:
    """Factory for creating comparator with proper dependency injection.
    
    This factory now supports both TPPDataset objects and legacy DataLoader objects
    for improved performance and reduced conversion time.
    """
    
    @staticmethod
    def create_comparator(
        label_data: Union[TPPDataset, Any],
        simulation: List[Dict],
        num_event_types: int,
        output_dir: str,
        dataset_size: int = 10**5,
        auto_run: bool = True
    ) -> TemporalPointProcessComparator:
        """Create comparator with all dependencies properly injected.
        
        Args:
            label_data: Either a TPPDataset object (preferred for better performance) 
                       or a DataLoader (legacy support)
            simulation: List of simulation sequences
            num_event_types: Number of event types
            output_dir: Output directory for results
            dataset_size: Maximum number of events to process
            auto_run: Whether to run evaluation automatically
            
        Returns:
            TemporalPointProcessComparator: Configured comparator instance
            
        Note:
            When using TPPDataset, data extraction is significantly faster as it
            avoids DataLoader iteration and batch processing overhead.
        """
        
        # Create extractors - use TPPDatasetExtractor for better performance when possible
        if isinstance(label_data, TPPDataset):
            logger.info("Using TPPDatasetExtractor for optimized data extraction")
            label_extractor = TPPDatasetExtractor(label_data, dataset_size)
        else:
            logger.info("Using legacy LabelDataExtractor for DataLoader compatibility")
            # Fallback to DataLoader extractor for backward compatibility
            label_extractor = LabelDataExtractor(label_data, dataset_size)
            
        simulation_extractor = SimulationDataExtractor(simulation, dataset_size)
        
        # Create plot generators
        plot_generators = [
            InterEventTimePlotGenerator(),
            EventTypePlotGenerator(num_event_types),
            SequenceLengthPlotGenerator(),
            CrossCorrelationPlotGenerator()
        ]
        
        # Create metrics calculator
        metrics_calculator = MetricsCalculatorImpl()
        
        # Create and return comparator
        return TemporalPointProcessComparator(
            label_extractor=label_extractor,
            simulation_extractor=simulation_extractor,
            plot_generators=plot_generators,
            metrics_calculator=metrics_calculator,
            output_dir=output_dir,
            auto_run=auto_run
        )
