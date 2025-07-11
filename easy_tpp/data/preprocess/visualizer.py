from easy_tpp.data.preprocess.data_loader import TPPDataModule

from typing import Optional
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import json


class Visualizer:
    def __init__(
        self,
        data_module: TPPDataModule,
        split: Optional[str] = 'test',
        save_dir: Optional[str] = None,
        dataset_size: Optional[int] = 10**4,
        comparison_data_module: Optional[TPPDataModule] = None,
        comparison_split: Optional[str] = None,
    ):
        """
        Initialize the visualizer.
        
        Args:
            data_module: Data module for the main dataset
            split: Dataset split to use ('test', 'valid', or 'train')
            save_dir: Directory to save visualizations
            dataset_size: Maximum number of events to use
            comparison_data_module: Optional data module for comparison dataset
            comparison_split: Optional split for comparison dataset
        """
        self.data_module = data_module
        self.num_event_types = data_module.num_event_types
        self.save_dir = save_dir
        self.is_comparison_mode = comparison_data_module is not None

        # Validate the split parameter
        valid_splits = {'valid', 'test', 'train', None}
        if split not in valid_splits:
            raise ValueError(f"Split '{split}' is not valid. Choose from {valid_splits}")

        self.split = split
        data_dir = data_module.data_config.get_data_dir(split)
        data_format = data_module.data_config.data_format
        self.data = self.data_module.build_input(source_dir=data_dir, data_format=data_format, split=split)

        # Process main dataset
        self.all_event_types = []
        self.all_time_deltas = []
        # Preprocess the data to extract all event types and time deltas
        i=0
        for type_seq, time_seq in zip(self.data['type_seqs'], self.data['time_delta_seqs']):
            for event, time in zip(type_seq, time_seq):
                self.all_event_types.append(event)
                self.all_time_deltas.append(time)
                i += 1
                if i >= dataset_size:
                    break
                
        self.all_event_types = np.array(self.all_event_types)
        self.all_time_deltas = np.array(self.all_time_deltas)
        self.seq_lengths = np.array([len(seq) for seq in self.data["type_seqs"]])

        # Process comparison dataset if provided
        if self.is_comparison_mode:
            self.comparison_split = comparison_split or split
            self.comparison_data_module = comparison_data_module
            
            comp_data_dir = comparison_data_module.data_config.get_data_dir(self.comparison_split)
            comp_data_format = comparison_data_module.data_config.data_format
            self.comparison_data = self.comparison_data_module.build_input(
                source_dir=comp_data_dir, 
                data_format=comp_data_format, 
                split=self.comparison_split
            )
            
            self.comparison_event_types = []
            self.comparison_time_deltas = []
            
            i=0
            for type_seq, time_seq in zip(self.comparison_data['type_seqs'], self.comparison_data['time_delta_seqs']):
                for event, time in zip(type_seq, time_seq):
                    self.comparison_event_types.append(event)
                    self.comparison_time_deltas.append(time)
                    i += 1
                    if i >= dataset_size:
                        break
            
            self.comparison_event_types = np.array(self.comparison_event_types)
            self.comparison_time_deltas = np.array(self.comparison_time_deltas)
            self.comparison_seq_lengths = np.array([len(seq) for seq in self.comparison_data["type_seqs"]])

        if save_dir is None:
            parent_dir = os.path.dirname(data_dir)
            save_dir = os.path.join(parent_dir, 'visualizations')
            self.save_dir = save_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_metadata_to_json(self, metadata, filename_base):
        """
        Save metadata to a JSON file with the same base filename as the graph.

        Args:
            metadata (dict): Dictionary containing metadata to save
            filename_base (str): Base filename (without extension)
        """
        json_filename = f"{filename_base}.json"

        # Add timestamp to metadata
        metadata['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add data for graph recreation
        if 'plot_type' in metadata:
            # Include actual data points used for the graph (if not already included)
            if metadata['plot_type'] == 'delta_times_distribution' and 'data_points' not in metadata:
                # Include a sample of data points if dataset is large
                data_points = self.all_time_deltas
                if len(data_points) > 10000:
                    # Store statistics and sample if data is too large
                    metadata['data_points_sample'] = data_points[:10000].tolist()
                    metadata['data_points_is_sample'] = True
                else:
                    metadata['data_points'] = data_points.tolist() if isinstance(data_points, np.ndarray) else data_points
                    metadata['data_points_is_sample'] = False
                
            elif metadata['plot_type'] == 'marked_delta_times_distribution' and 'data_points' not in metadata:
                # For marked delta times, include data by event type
                metadata['data_by_event_type'] = {}
                for event_type in range(self.num_event_types):
                    data = [self.all_time_deltas[i] for i, t in enumerate(self.all_event_types) if t == event_type]
                    if len(data) > 5000:
                        metadata['data_by_event_type'][str(event_type)] = data[:5000]
                        metadata['data_by_event_type_is_sample'] = True
                    else:
                        metadata['data_by_event_type'][str(event_type)] = data
                        metadata['data_by_event_type_is_sample'] = False
                
            elif metadata['plot_type'] == 'event_type_distribution' and 'raw_counts' not in metadata:
                # Store raw counts of each event type
                event_counts = {str(k): int(v) for k, v in Counter(self.all_event_types).items()}
                metadata['raw_counts'] = event_counts
                
            elif metadata['plot_type'] == 'seq_len_distribution' and 'sequence_lengths' not in metadata:
                # Store all sequence lengths
                seq_lengths = [len(seq) for seq in self.data["type_seqs"]]
                metadata['sequence_lengths'] = seq_lengths
                
            elif metadata['plot_type'] == 'all_distributions':
                # For the combined visualization, add data for each component
                if 'raw_data' not in metadata:
                    metadata['raw_data'] = {
                        'delta_times_sample': self.all_time_deltas[:5000] if len(self.all_time_deltas) > 5000 else self.all_time_deltas,
                        'is_sample': len(self.all_time_deltas) > 5000,
                        'event_type_counts': {str(k): int(v) for k, v in Counter(self.all_event_types).items()},
                        'sequence_lengths': [len(seq) for seq in self.data["type_seqs"]][:1000] if len(self.data["type_seqs"]) > 1000 else [len(seq) for seq in self.data["type_seqs"]]
                    }
            
            # Add visualization parameters
            metadata['visualization_info'] = {
                'plot_type': metadata['plot_type'],
                'data_split': self.split,
                'num_event_types': self.num_event_types
            }

        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {json_filename}")

    def plot_inter_event_time_distribution(self, filename: str = "inter_event_time_dist.png"):
        """
        Plots and saves the distribution of inter-event times.
        In comparison mode, plots both datasets.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Main distribution plot
        plt.figure(figsize=(10, 6))
        
        # Get data
        times_data = self.all_time_deltas
        
        # Create histograms
        hist, bin_edges = np.histogram(times_data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Filter out bins with frequency <= 2
        mask = hist > 2
        filtered_hist = hist[mask]
        filtered_bin_centers = bin_centers[mask]
        
        # Plot histograms
        label = f'Main ({self.split})'
        color = 'royalblue'
        plt.bar(filtered_bin_centers, filtered_hist, 
                width=(bin_edges[1] - bin_edges[0]) * 0.8,
                label=label, alpha=0.6, color=color)
        
        # If in comparison mode, add comparison data
        if self.is_comparison_mode:
            comp_times_data = self.comparison_time_deltas
            comp_hist, comp_bin_edges = np.histogram(comp_times_data, bins=50)
            comp_bin_centers = (comp_bin_edges[:-1] + comp_bin_edges[1:]) / 2
            
            # Filter out bins with frequency <= 2
            comp_mask = comp_hist > 2
            filtered_comp_hist = comp_hist[comp_mask]
            filtered_comp_bin_centers = comp_bin_centers[comp_mask]
            
            # Plot comparison data
            comp_label = f'Comparison ({self.comparison_split})'
            comp_color = 'crimson'
            plt.bar(filtered_comp_bin_centers, filtered_comp_hist, 
                    width=(comp_bin_edges[1] - comp_bin_edges[0]) * 0.8,
                    label=comp_label, alpha=0.6, color=comp_color)
        
        plt.yscale('log')
        
        # Calculate and add regression lines if enough data points
        if len(filtered_hist) > 1:
            X = filtered_bin_centers.reshape(-1, 1)
            y = np.log10(filtered_hist)
            reg = np.polyfit(X.flatten(), y, 1)
            slope, intercept = reg
            
            x_line = np.linspace(min(filtered_bin_centers), max(filtered_bin_centers), 100)
            y_line = 10 ** (slope * x_line + intercept)
            
            plt.plot(x_line, y_line, '--', color='blue', 
                   label=f'Main slope: {slope:.4f}')
        
        if self.is_comparison_mode and len(filtered_comp_hist) > 1:
            comp_X = filtered_comp_bin_centers.reshape(-1, 1)
            comp_y = np.log10(filtered_comp_hist)
            comp_reg = np.polyfit(comp_X.flatten(), comp_y, 1)
            comp_slope, comp_intercept = comp_reg
            
            comp_x_line = np.linspace(min(filtered_comp_bin_centers), max(filtered_comp_bin_centers), 100)
            comp_y_line = 10 ** (comp_slope * comp_x_line + comp_intercept)
            
            plt.plot(comp_x_line, comp_y_line, '--', color='red', 
                   label=f'Comparison slope: {comp_slope:.4f}')
        
        # Calculate statistics
        mean = np.mean(times_data)
        median = np.median(times_data)
        std = np.std(times_data)
        
        stats = (f"Main Stats:\n"
               f"Mean: {mean:.4f}\n"
               f"Median: {median:.4f}\n"
               f"Std Dev: {std:.4f}")
        
        # Position text at top left
        plt.annotate(stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Add comparison statistics
        if self.is_comparison_mode:
            comp_mean = np.mean(comp_times_data)
            comp_median = np.median(comp_times_data)
            comp_std = np.std(comp_times_data)
            
            comp_stats = (f"Comparison Stats:\n"
                        f"Mean: {comp_mean:.4f}\n"
                        f"Median: {comp_median:.4f}\n"
                        f"Std Dev: {comp_std:.4f}")
            
            # Position text at top right
            plt.annotate(comp_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                       va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.title('Inter-Event Time Distribution (Log Scale)')
        plt.xlabel('Time Since Last Event')
        plt.ylabel('Frequency (Log Scale)')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Inter-event time distribution plot saved to {filepath}")
        
        # Add QQ plot if in comparison mode
        if self.is_comparison_mode:
            self._create_qq_plot(
                self.all_time_deltas, 
                self.comparison_time_deltas, 
                'QQ Plot: Inter-Event Times',
                os.path.join(self.save_dir, "qq_inter_event_times.png"),
                log_scale=True
            )

    def _create_qq_plot(self, data1: np.ndarray, data2: np.ndarray, title: str, save_path: str, log_scale: bool = False):
        """
        Creates and saves a QQ plot comparing two distributions.
        
        Args:
            data1: Data points from the first dataset
            data2: Data points from the second dataset
            title: Title for the plot
            save_path: Path to save the plot
            log_scale: Whether to use log scale for the axes
        """
        # Set the Seaborn style
        sns.set_theme(style="whitegrid")
        
        # Convert inputs to numpy arrays if they aren't already
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        
        # Filter out non-positive values if using log scale
        if log_scale:
            data1 = data1[data1 > 0]
            data2 = data2[data2 > 0]
        
        # Skip if empty arrays
        if len(data1) == 0 or len(data2) == 0:
            print(f"Warning: Empty arrays after filtering: data1={len(data1)}, data2={len(data2)}")
            return
            
        # Sort arrays for quantiles
        data1.sort()
        data2.sort()
        
        # Determine number of quantiles to use
        n_quantiles = min(len(data1), len(data2))
        
        if n_quantiles < 10:
            print(f"Warning: Not enough data points for QQ plot: {n_quantiles} points")
            return
        
        # Create evenly spaced quantiles
        quantiles = np.linspace(0.01, 0.99, min(100, n_quantiles))
        
        # Compute quantiles
        data1_quantiles = np.quantile(data1, quantiles)
        data2_quantiles = np.quantile(data2, quantiles)
        
        # Create QQ plot
        plt.figure(figsize=(8, 8))
        
        # Plot the quantiles
        if log_scale:
            plt.loglog(data1_quantiles, data2_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x) in log-log space
            min_val = max(min(data1_quantiles.min(), data2_quantiles.min()), 1e-10)  # Avoid log(0)
            max_val = max(data1_quantiles.max(), data2_quantiles.max())
            ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            plt.loglog(ref_line, ref_line, 'r--', alpha=0.7)
        else:
            plt.plot(data1_quantiles, data2_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x)
            min_val = min(data1_quantiles.min(), data2_quantiles.min())
            max_val = max(data1_quantiles.max(), data2_quantiles.max())
            ref_line = np.linspace(min_val, max_val, 100)
            plt.plot(ref_line, ref_line, 'r--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.title(title, fontsize=14)
        plt.xlabel(f'Main Quantiles ({self.split})', fontsize=12)
        plt.ylabel(f'Comparison Quantiles ({self.comparison_split})', fontsize=12)
        
        # Add annotation explaining interpretation
        annotation_text = ("Points along reference line indicate similar distributions.\n"
                         "Deviations suggest distributional differences.")
        plt.annotate(annotation_text, xy=(0.05, 0.05), xycoords='axes fraction',
                   va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"QQ plot saved to {save_path}")

    def plot_event_type_distribution(self, filename: str = "event_type_dist.png"):
        """
        Plots and saves the distribution of event types.
        In comparison mode, plots both datasets.
        """
        # Set the Seaborn style
        sns.set_theme(style="whitegrid")

        # Calculate event type distributions
        types_data = self.all_event_types
        all_event_types = np.arange(self.num_event_types)
        
        # Count occurrences
        types_int = types_data.astype(int)
        type_counts = np.bincount(types_int, minlength=self.num_event_types)
        
        # Calculate probabilities
        total = len(types_data)
        type_probs = type_counts / total if total > 0 else np.zeros_like(type_counts)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Width of bars - slightly offset for better visibility
        width = 0.35 if self.is_comparison_mode else 0.7
        x = np.arange(len(all_event_types))
        
        # Plot main data
        offset = -width/2 if self.is_comparison_mode else 0
        plt.bar(x + offset, type_probs, width, label=f'Main ({self.split})', 
              color='royalblue', alpha=0.7)
        
        # Add comparison data if in comparison mode
        if self.is_comparison_mode:
            comp_types_data = self.comparison_event_types
            comp_types_int = comp_types_data.astype(int)
            comp_type_counts = np.bincount(comp_types_int, minlength=self.num_event_types)
            
            comp_total = len(comp_types_data)
            comp_type_probs = comp_type_counts / comp_total if comp_total > 0 else np.zeros_like(comp_type_counts)
            
            plt.bar(x + width/2, comp_type_probs, width, label=f'Comparison ({self.comparison_split})', 
                  color='crimson', alpha=0.7)
        
        plt.title('Event Type Distribution', fontsize=14)
        plt.xlabel('Event Type', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(x, all_event_types)
        plt.legend(frameon=True)
        
        # Add statistics for top event types
        top3_indices = np.argsort(type_probs)[-3:][::-1]  # Top 3 in descending order
        main_stats = "Main Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {type_probs[i]:.3f}" 
                                                 for i in top3_indices])
        
        # Position text on left
        plt.annotate(main_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        # Add comparison statistics
        if self.is_comparison_mode:
            comp_top3_indices = np.argsort(comp_type_probs)[-3:][::-1]
            comp_stats = "Comparison Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {comp_type_probs[i]:.3f}" 
                                                         for i in comp_top3_indices])
            
            # Position text on right
            plt.annotate(comp_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                       va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                       fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Event type distribution plot saved to {filepath}")

    def plot_sequence_length_distribution(self, filename: str = "sequence_length_dist.png"):
        """
        Plots and saves the distribution of sequence lengths.
        In comparison mode, plots both datasets.
        """
        # Set the Seaborn style
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(10, 6))
        
        # Calculate bin width
        binwidth = max(1, int(np.std(self.seq_lengths) / 2)) if len(self.seq_lengths) > 1 else 1
        
        # Plot main dataset
        sns.histplot(self.seq_lengths, label=f'Main ({self.split})', kde=True, 
                  stat='density', binwidth=binwidth, color='royalblue', alpha=0.6)
        
        # Add comparison dataset if in comparison mode
        if self.is_comparison_mode:
            comp_binwidth = max(1, int(np.std(self.comparison_seq_lengths) / 2)) if len(self.comparison_seq_lengths) > 1 else 1
            binwidth = max(binwidth, comp_binwidth)
            
            sns.histplot(self.comparison_seq_lengths, label=f'Comparison ({self.comparison_split})', kde=True, 
                      stat='density', binwidth=binwidth, color='crimson', alpha=0.6)
        
        # Calculate statistics
        mean = np.mean(self.seq_lengths)
        median = np.median(self.seq_lengths)
        std = np.std(self.seq_lengths)
        
        main_stats = (f"Main Stats:\n"
                    f"Mean: {mean:.2f}\n"
                    f"Median: {median:.2f}\n"
                    f"Std Dev: {std:.2f}")
        
        # Position text at top left
        plt.annotate(main_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        # Add comparison statistics
        if self.is_comparison_mode:
            comp_mean = np.mean(self.comparison_seq_lengths)
            comp_median = np.median(self.comparison_seq_lengths)
            comp_std = np.std(self.comparison_seq_lengths)
            
            comp_stats = (f"Comparison Stats:\n"
                        f"Mean: {comp_mean:.2f}\n"
                        f"Median: {comp_median:.2f}\n"
                        f"Std Dev: {comp_std:.2f}")
            
            # Position text at top right
            plt.annotate(comp_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                       va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                       fontsize=10)
        
        plt.title('Sequence Length Distribution', fontsize=14)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sequence length distribution plot saved to {filepath}")
        
        # Add QQ plot if in comparison mode
        if self.is_comparison_mode:
            self._create_qq_plot(
                self.seq_lengths, 
                self.comparison_seq_lengths, 
                'QQ Plot: Sequence Lengths',
                os.path.join(self.save_dir, "qq_sequence_lengths.png"),
                log_scale=False
            )

    def run_visualization(self):
        """
        Run all visualization functions and save the plots.
        """
        print("Generating visualization plots...")
        try:
            self.plot_inter_event_time_distribution()
            self.plot_event_type_distribution()
            self.plot_sequence_length_distribution()
            print("All plots generated successfully!")
        except Exception as e:
            print(f"Error occurred during plot generation: {e}")
            
    def delta_times_distribution(
        self, plot=False, save_graph=False, bins=50, figsize=(10, 6), color='royalblue', log_scale=False,
    ) -> np.ndarray:
        """
        Legacy method preserved for backward compatibility.
        Now calls plot_inter_event_time_distribution instead.
        """
        if plot or save_graph:
            self.plot_inter_event_time_distribution()
        return self.all_time_deltas