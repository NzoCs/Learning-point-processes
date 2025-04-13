from easy_tpp.preprocess.data_loader import TPPDataModule

from typing import Optional
import numpy as np
import torch
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import json


class Visualizer:
    def __init__(
        self,
        data_module: TPPDataModule,
        split: Optional[str] = 'test',
        save_dir: Optional[str] = None
    ):
        self.data_module = data_module
        self.num_event_types = data_module.num_event_types
        self.save_dir = save_dir

        # Validate the split parameter
        valid_splits = {'valid', 'test', 'train', None}
        if split not in valid_splits:
            raise ValueError(f"Split '{split}' is not valid. Choose from {valid_splits}")

        self.split = split
        data_dir = data_module.data_config.get_data_dir(split)
        data_format = data_module.data_config.data_format
        self.data = self.data_module.build_input(source_dir=data_dir, data_format=data_format, split=split)

        self.all_event_types = [event for seq in self.data['type_seqs'] for event in seq]

        # Calculate time_delta_seqs statistics
        self.all_time_deltas = [delta for seq in self.data['time_delta_seqs'] for delta in seq]

        if save_dir is None:
            parent_dir = os.path.dirname(data_dir)
            save_dir = os.path.join(parent_dir, 'visualizations')
            self.save_dir = save_dir

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

    def delta_times_distribution(
        self, plot=False, save_graph=False, bins=50, figsize=(10, 6), color='royalblue', log_scale=False,
    ) -> np.ndarray:
        """
        Calculate and return the list of delta times, and optionally display
        a histogram of the distribution.

        Args:
            split (str): Dataset split to use ('test', 'valid', or 'train').
            plot (bool): Whether to display the histogram.
            save_graph (bool): Whether to save the histogram as an image.
            bins (int): Number of bins for the histogram.
            figsize (tuple): Figure size for the plot (width, height).
            color (str): Color for the histogram.
            log_scale (bool): Whether to use logarithmic scale for x-axis.

        Returns:
            numpy.ndarray: The array of delta times.
        """
        split = self.split

        # Combine all delta times
        all_time_deltas = self.all_time_deltas

        # Create visualization if requested
        if plot or save_graph:
            plt.figure(figsize=figsize)
            ax = sns.histplot(all_time_deltas, bins=bins, kde=True, color=color)

            plt.xlabel("Delta times")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of inter-event times ({split} set)")

            if log_scale:
                plt.xscale('log')
                plt.title(f"Distribution of inter-event times ({split} set) - Log Scale")

            # Add statistics to the plot
            stats_text = (f"Mean: {np.mean(self.all_time_deltas):.4f}\n"
                          f"Median: {np.median(self.all_time_deltas):.4f}\n"
                          f"Std Dev: {np.std(all_time_deltas):.4f}\n"
                          f"Min: {np.min(all_time_deltas):.4f}\n"
                          f"Max: {np.max(all_time_deltas):.4f}\n"
                          f"Count: {len(all_time_deltas)}")

            # Position text at top right
            plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                         va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

            # Save the figure if requested
            if save_graph:
                output_dir = self.save_dir
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f'delta_times_distribution_{split}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Graph saved to {filename}")

                # Save metadata
                metadata = {
                    'plot_type': 'delta_times_distribution',
                    'split': split,
                    'bins': bins,
                    'figsize': figsize,
                    'color': color,
                    'log_scale': log_scale,
                    'statistics': {
                        'mean': float(np.mean(all_time_deltas)),
                        'median': float(np.median(all_time_deltas)),
                        'std_dev': float(np.std(all_time_deltas)),
                        'min': float(np.min(all_time_deltas)),
                        'max': float(np.max(all_time_deltas)),
                        'count': int(len(all_time_deltas))
                    }
                }
                self._save_metadata_to_json(metadata, filename[:-4])  # Remove .png extension

            # Show the plot if requested
            if plot:
                plt.show()
            else:
                plt.close()

        return all_time_deltas

    def marked_delta_times_distribution(self, plot=False, save_graph=False,
                                        bins=30, figsize=(10, 6), cmap='tab10', alpha=0.6):
        """
        Calculate the distribution of inter-event times by mark and
        optionally display superimposed histograms using Seaborn.

        Args:
            split (str): Dataset split to use ('test', 'valid', or 'train').
            plot (bool): Whether to display the histogram.
            save_graph (bool): Whether to save the histogram as an image.
            bins (int): Number of bins for the histogram.
            figsize (tuple): Base figure size for the plot (width, height).
            cmap (str): Colormap for the different marks.
            alpha (float): Transparency level for the histograms.

        Returns:
            dict[int, numpy.ndarray]: Dictionary of inter-event times for each mark.
        """
        split = self.split

        flat_times = self.all_time_deltas
        flat_types = self.all_event_types

        # Calculate delta times per mark
        marked_all_times_delta = {
            m: flat_times[flat_types == m]
            for m in range(self.num_event_types)
        }

        # Create visualization if requested
        if plot or save_graph:
            # Set the Seaborn style and context
            sns.set_theme(style="whitegrid")

            colors = plt.cm.get_cmap(cmap, self.num_event_types)
            fig, axes = plt.subplots(self.num_event_types, 1,
                                      figsize=(figsize[0], figsize[1] * self.num_event_types),
                                      sharex=True)

            # If num_mark is 1, axes is not an array, so convert it to a list
            if self.num_event_types == 1:
                axes = [axes]

            # For storing statistics for metadata
            mark_statistics = {}

            for m in range(self.num_event_types):
                data = marked_all_times_delta[m]
                if len(data) > 0:  # Only plot if we have data for this mark
                    # Use Seaborn's histplot for better visual aesthetics
                    sns.histplot(data, bins=bins, color=colors(m),
                                 alpha=alpha, edgecolor="none", kde=True, ax=axes[m])

                    # Add statistics to the plot
                    stats_text = (f"Count: {len(data)}\n"
                                  f"Mean: {np.mean(data):.4f}\n"
                                  f"Median: {np.median(data):.4f}\n"
                                  f"Std Dev: {np.std(data):.4f}")

                    axes[m].annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                                     va='top', ha='right',
                                     bbox=dict(boxstyle='round', fc='white', alpha=0.7))

                    # Store statistics for metadata
                    mark_statistics[str(m)] = {
                        'count': int(len(data)),
                        'mean': float(np.mean(data)) if len(data) > 0 else 0,
                        'median': float(np.median(data)) if len(data) > 0 else 0,
                        'std_dev': float(np.std(data)) if len(data) > 0 else 0
                    }

                axes[m].set_ylabel("Density")
                axes[m].set_title(f"Distribution of inter-event times for type {m}")

            axes[-1].set_xlabel("Delta times")
            plt.tight_layout()

            # Save figure if requested
            if save_graph:
                output_dir = self.save_dir
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f'marked_delta_times_{split}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Graph saved to {filename}")

                # Save metadata
                metadata = {
                    'plot_type': 'marked_delta_times_distribution',
                    'split': split,
                    'bins': bins,
                    'figsize': figsize,
                    'cmap': cmap,
                    'alpha': alpha,
                    'num_event_types': self.num_event_types,
                    'mark_statistics': mark_statistics
                }
                self._save_metadata_to_json(metadata, filename[:-4])  # Remove .png extension

            # Show the plot if requested
            if plot:
                plt.show()
            else:
                plt.close(fig)

        return marked_all_times_delta

    def event_type_distribution(self, plot=False, save_graph=False,
                                 figsize=(10, 6), color='royalblue', palette="Blues_d"):
        """
        Calculate the distribution of event types and
        optionally display a histogram using Seaborn.

        Args:
            split (str): Dataset split to use ('test', 'valid', or 'train').
            plot (bool): Whether to display the histogram.
            save_graph (bool): Whether to save the histogram as an image.
            figsize (tuple): Figure size for the plot (width, height).
            color (str): Base color for the bars (used if palette is None).
            palette (str): Seaborn color palette for the bars.

        Returns:
            tuple:
                - list: Unique event types
                - list: Corresponding probabilities
        """
        split = self.split

        # Calculate event type distribution
        events_counts = Counter(self.all_event_types)
        events = list(events_counts.keys())
        counts = list(events_counts.values())

        # Calculate probability for each event
        total = sum(counts)
        probabilities = [c / total for c in counts] if total > 0 else []

        # Create ordered pairs and sort by event type
        event_prob_pairs = sorted(zip(events, probabilities))
        events = [pair[0] for pair in event_prob_pairs]
        probabilities = [pair[1] for pair in event_prob_pairs]

        # Create DataFrame for Seaborn
        df = pd.DataFrame({
            'Event Type': events,
            'Probability': probabilities
        })

        # Create visualization if requested
        if plot or save_graph:
            # Set the Seaborn style and context
            sns.set_theme(style="whitegrid")

            plt.figure(figsize=figsize)

            # Use Seaborn's barplot for better aesthetics
            ax = sns.barplot(x='Event Type', y='Probability', data=df,
                             palette=palette if palette else color)

            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom')

            plt.xlabel("Event types")
            plt.ylabel("Probabilities")
            plt.title(f"Distribution of event types ({split} set)")
            plt.ylim(0, max(probabilities) * 1.15)  # Add some space for the text

            # Save figure if requested
            if save_graph:
                output_dir = self.save_dir
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f'event_type_distribution_{split}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Graph saved to {filename}")

                # Save metadata
                event_probs = {str(event): float(prob) for event, prob in zip(events, probabilities)}
                event_counts = {str(event): int(count) for event, count in events_counts.items()}

                metadata = {
                    'plot_type': 'event_type_distribution',
                    'split': split,
                    'figsize': figsize,
                    'color': color,
                    'palette': palette,
                    'total_events': total,
                    'event_probabilities': event_probs,
                    'event_counts': event_counts
                }
                self._save_metadata_to_json(metadata, filename[:-4])  # Remove .png extension

            # Show the plot if requested
            if plot:
                plt.show()
            else:
                plt.close()

        return events, probabilities

    def seq_len_distribution(self, plot=False, save_graph=False,
                             bins=20, figsize=(10, 6), color='royalblue', kde=True):
        """
        Calculate the distribution of sequence lengths and
        optionally display a histogram using Seaborn.

        Args:
            split (str): Dataset split to use ('test', 'valid', or 'train').
            plot (bool): Whether to display the histogram.
            save_graph (bool): Whether to save the histogram as an image.
            bins (int): Number of bins for the histogram.
            figsize (tuple): Figure size for the plot (width, height).
            color (str): Color for the histogram.
            kde (bool): Whether to show a kernel density estimate.

        Returns:
            numpy.ndarray: Array of sequence lengths
        """
        split = self.split

        seq_lengths = [len(seq) for seq in self.data["type_seqs"]]

        # Create visualization if requested
        if plot or save_graph:
            # Set the Seaborn style and context
            sns.set_theme(style="whitegrid")

            plt.figure(figsize=figsize)

            # Use Seaborn's histplot for better aesthetics
            ax = sns.histplot(seq_lengths, bins=bins, color=color, kde=kde)

            # Add statistics to the plot
            stats_text = (f"Count: {len(seq_lengths)}\n"
                          f"Mean: {np.mean(seq_lengths):.2f}\n"
                          f"Median: {np.median(seq_lengths):.2f}\n"
                          f"Min: {np.min(seq_lengths):.2f}\n"
                          f"Max: {np.max(seq_lengths):.2f}\n"
                          f"Std Dev: {np.std(seq_lengths):.2f}")

            plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                         va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

            plt.xlabel("Sequence length")
            plt.ylabel("Count")
            plt.title(f"Distribution of sequence lengths ({split} set)")

            # Save figure if requested
            if save_graph:
                output_dir = self.save_dir
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f'seq_len_distribution_{split}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Graph saved to {filename}")

                # Save metadata
                metadata = {
                    'plot_type': 'seq_len_distribution',
                    'split': split,
                    'bins': bins,
                    'figsize': figsize,
                    'color': color,
                    'kde': kde,
                    'statistics': {
                        'count': int(len(seq_lengths)),
                        'mean': float(np.mean(seq_lengths)),
                        'median': float(np.median(seq_lengths)),
                        'min': int(np.min(seq_lengths)),
                        'max': int(np.max(seq_lengths)),
                        'std_dev': float(np.std(seq_lengths))
                    }
                }
                self._save_metadata_to_json(metadata, filename[:-4])  # Remove .png extension

            # Show the plot if requested
            if plot:
                plt.show()
            else:
                plt.close()

        return seq_lengths

    def show_all_distributions(self, log_scale=False, show_graph=False, save_graph=True,
                                figsize=(18, 6), palette="Blues_d", bins=(30, None, 20), kde=True):
        """
        Calculate and display, in a single figure, the following three distributions using Seaborn:
        - Distribution of delta times.
        - Distribution of event types (only if there is more than one event type).
        - Distribution of sequence lengths.

        Args:
            show_graph (bool): Whether to display the graph.
            save_graph (bool): Whether to save the graph as PNG.
            figsize (tuple): Figure size for the combined plot.
            palette (str or list): Seaborn color palette for the plots.
            bins (tuple): Number of bins for each histogram (delta_times, event_types, seq_lengths).
            kde (bool): Whether to show kernel density estimate for histograms.

        Returns:
            tuple:
                - numpy.ndarray: Delta times array
                - tuple: (list of event types, list of corresponding probabilities)
                - numpy.ndarray: Sequence lengths array
        """
        split = self.split

        # Get data without displaying separate graphs
        delta_times = self.delta_times_distribution()
        event_types, probabilities = self.event_type_distribution()
        seq_lengths = self.seq_len_distribution()

        # Check if there's more than one event type
        multiple_events = len(event_types) > 1

        # Create figure if requested
        if save_graph or show_graph:
            # Set the Seaborn style
            sns.set_theme(style="whitegrid")

            # Determine number of subplots based on event types
            num_plots = 3 if multiple_events else 2

            # Create figure with appropriate number of subplots
            fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] * num_plots / 3, figsize[1]),
                                     constrained_layout=True)

            # Get bin values
            delta_bins, _, seq_bins = bins

            # Plot index to track current subplot position
            plot_idx = 0

            # Graph 1: Delta times distribution
            sns.histplot(delta_times, bins=delta_bins, kde=kde, ax=axes[plot_idx],
                         color=sns.color_palette(palette)[0])

            if log_scale:
                axes[plot_idx].set_yscale('log')
                axes[plot_idx].set_title(f"Distribution of inter-event times (Log Scale)", fontsize=12)
            else:
                axes[plot_idx].set_title(f"Distribution of inter-event times", fontsize=12)

            # Add statistics for delta times
            stats_text = (f"Mean: {np.mean(delta_times):.4f}\n"
                          f"Median: {np.median(delta_times):.4f}\n"
                          f"Std Dev: {np.std(delta_times):.4f}")

            axes[plot_idx].annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                                    va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

            axes[plot_idx].set_xlabel("Delta times")
            axes[plot_idx].set_ylabel("Count")

            plot_idx += 1

            # Graph 2: Event types distribution (only if multiple event types)
            if multiple_events:
                # Create DataFrame for event types
                event_df = pd.DataFrame({
                    'Event Type': event_types,
                    'Probability': probabilities
                })

                bars = sns.barplot(x='Event Type', y='Probability', data=event_df, hue='Event Type', palette=palette,
                                   ax=axes[plot_idx], legend=False)

                # Add value labels on top of bars
                for i, p in enumerate(axes[plot_idx].patches):
                    height = p.get_height()
                    axes[plot_idx].text(p.get_x() + p.get_width() / 2., height + 0.005,
                                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

                axes[plot_idx].set_xlabel("Event types")
                axes[plot_idx].set_ylabel("Probabilities")
                axes[plot_idx].set_title("Distribution of event types", fontsize=12)
                axes[plot_idx].set_ylim(0, max(probabilities) * 1.15)  # Add some space for the text

                plot_idx += 1

            # Graph 3: Sequence lengths distribution
            sns.histplot(seq_lengths, bins=seq_bins, kde=kde, ax=axes[plot_idx],
                         color=sns.color_palette(palette)[2 if multiple_events else 1])

            # Add statistics for sequence lengths
            stats_text = (f"Mean: {np.mean(seq_lengths):.2f}\n"
                          f"Median: {np.median(seq_lengths):.2f}\n"
                          f"Std Dev: {np.std(seq_lengths):.2f}")

            axes[plot_idx].annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                                    va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

            axes[plot_idx].set_xlabel("Sequence length")
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_title("Distribution of sequence lengths", fontsize=12)

            # Add overall title with dataset split information
            fig.suptitle(f"Data distributions for {split.capitalize()} set",
                         fontsize=14, fontweight='bold', y=0.98)

            # Save figure if requested
            if save_graph:
                if log_scale:
                    filename = f"all_distributions_{split}_log.png"
                else:
                    filename = f"all_distributions_{split}.png"

                output_dir = self.save_dir
                os.makedirs(output_dir, exist_ok=True)

                # Use current datetime for filename
                filename = os.path.join(output_dir, filename)

                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {filename}")

                # Save metadata
                delta_times_stats = {
                    'mean': float(np.mean(delta_times)),
                    'median': float(np.median(delta_times)),
                    'std_dev': float(np.std(delta_times)),
                    'min': float(np.min(delta_times)),
                    'max': float(np.max(delta_times)),
                    'count': int(len(delta_times))
                }

                seq_lengths_stats = {
                    'mean': float(np.mean(seq_lengths)),
                    'median': float(np.median(seq_lengths)),
                    'min': int(np.min(seq_lengths)),
                    'max': int(np.max(seq_lengths)),
                    'std_dev': float(np.std(seq_lengths)),
                    'count': int(len(seq_lengths))
                }

                event_probs = {str(event): float(prob) for event, prob in zip(event_types, probabilities)}

                metadata = {
                    'plot_type': 'all_distributions',
                    'split': split,
                    'figsize': figsize,
                    'palette': palette,
                    'bins': bins,
                    'kde': kde,
                    'log_scale': log_scale,
                    'multiple_events': multiple_events,
                    'delta_times_statistics': delta_times_stats,
                    'event_type_probabilities': event_probs,
                    'seq_lengths_statistics': seq_lengths_stats
                }
                self._save_metadata_to_json(metadata, filename[:-4])  # Remove .png extension

            # Show figure if requested
            if show_graph:
                plt.show()
            else:
                plt.close(fig)

        return delta_times, (event_types, probabilities), seq_lengths