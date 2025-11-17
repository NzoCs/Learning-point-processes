"""
Event Tokenizer with Strategy-Based Architecture

This module implements an event tokenizer with a clean separation of padding and truncation
strategies. The architecture is organized as follows:

Architecture Overview:
---------------------
1. EventTokenizer: Main class that routes to appropriate strategy methods
2. Strategy Methods: Dedicated methods for each padding/truncation strategy
   - Padding strategies: DO_NOT_PAD, LONGEST, MAX_LENGTH
   - Truncation strategies: DO_NOT_TRUNCATE, LONGEST_FIRST

Key Components:
--------------
- Batch: Container for tokenized outputs (tensors, attention masks, etc.)
- EventTokenizer: Main tokenizer with strategy routing logic
  
Strategy Methods:
----------------
Truncation:
  - _truncate_do_not_truncate(): No truncation applied
  - _truncate_longest_first(): Truncate from left or right to max_length
  
Padding:
  - _pad_do_not_pad(): Convert to arrays without padding
  - _pad_to_longest(): Pad to longest sequence in batch
  - _pad_to_max_length(): Pad to fixed maximum length

Usage:
------
The tokenizer accepts PaddingStrategy and TruncationStrategy enums (or string equivalents)
and routes to the appropriate method automatically.

Example:
    tokenizer = EventTokenizer(config)
    batch = tokenizer.pad(
        encoded_inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=512
    )
"""
import copy
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np
import torch

from new_ltpp.configs import TokenizerConfig
from new_ltpp.shared_types import Batch, TPPSequence
from new_ltpp.utils.const import PaddingStrategy, TruncationStrategy


class EventTokenizer:
    """
    Event tokenizer with strategy-based padding and truncation.
    
    This tokenizer processes event sequences with configurable padding and truncation strategies.
    Each strategy has its own dedicated method for clean separation of concerns.
    """

    padding_side: Literal["left", "right"] = "left"
    truncation_side: Literal["left", "right"] = "right"

    def __init__(self, config: TokenizerConfig):
        """Initialize the EventTokenizer with configuration.
        
        Args:
            config: TokenizerConfig containing all tokenizer parameters
        """
        config = copy.deepcopy(config)
        self.num_event_types = config.num_event_types
        self.pad_token_id = config.pad_token_id

        # Convert and store the unified strategy from config
        if isinstance(config.strategy, str):
            # Try to convert as PaddingStrategy first, then TruncationStrategy
            try:
                self.strategy = PaddingStrategy(config.strategy)
            except ValueError:
                self.strategy = TruncationStrategy(config.strategy)
        else:
            self.strategy = config.strategy

        # Set padding and truncation sides
        self.padding_side = config.padding_side
        self.truncation_side = config.truncation_side

    # ============================================================================
    # TRUNCATION STRATEGIES
    # ============================================================================
    
    def _truncate_longest_first(
        self,
        sequences: List[TPPSequence],
        max_length: int,
    ) -> List[TPPSequence]:
        """Truncate sequences using longest-first strategy.
        
        Args:
            sequences: List of TPPSequence objects
            max_length: Maximum length after truncation
            
        Returns:
            List of truncated TPPSequence objects
        """
        truncated = []
        for seq in sequences:
            if self.truncation_side == "right":
                truncated.append(TPPSequence(
                    time_seqs=seq['time_seqs'][:max_length],
                    time_delta_seqs=seq['time_delta_seqs'][:max_length],
                    type_seqs=seq['type_seqs'][:max_length],
                ))
            else:
                truncated.append(TPPSequence(
                    time_seqs=seq['time_seqs'][-max_length:],
                    time_delta_seqs=seq['time_delta_seqs'][-max_length:],
                    type_seqs=seq['type_seqs'][-max_length:],
                ))
        return truncated
    

    def _truncate(
        self,
        sequences: List[TPPSequence],
        truncation_strategy: TruncationStrategy,
        max_length: Optional[int] = None,
    ) -> List[TPPSequence]:
        """Apply truncation based on strategy.
        
        Routes to the appropriate truncation method based on strategy.
        
        Args:
            sequences: List of TPPSequence objects
            truncation_strategy: Strategy to use for truncation
            max_length: Maximum length after truncation (not used if DO_NOT_TRUNCATE)
            
        Returns:
            List of truncated TPPSequence objects
        """
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            return sequences
        
        # All other truncation strategies use the same implementation
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            if max_length is None:
                raise ValueError("max_length is required for LONGEST_FIRST truncation")
            return self._truncate_longest_first(sequences, max_length)
        
        return sequences

    def pad(
        self,
        inputs: List[TPPSequence],
        max_length: Optional[int] = None,
    ) -> Batch:
        """Process TPPSequence inputs with the configured padding or truncation strategy.
        
        Args:
            inputs: List of TPPSequence objects from dataset
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Batch with processed sequences (torch tensors)
            
        Example:
            >>> sequences = [dataset[i] for i in range(batch_size)]
            >>> batch = tokenizer.pad(sequences)
        """
        
        # Determine padding and truncation strategies from self.strategy
        if isinstance(self.strategy, PaddingStrategy):
            padding_strategy = self.strategy
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
        elif isinstance(self.strategy, TruncationStrategy):
            padding_strategy = PaddingStrategy.DO_NOT_PAD
            truncation_strategy = self.strategy
        else:
            raise ValueError(
                f"strategy must be PaddingStrategy or TruncationStrategy, got {type(self.strategy)}"
            )
        
        # Apply truncation (no max_length needed for DO_NOT_TRUNCATE)
        sequences = self._truncate(
            inputs,
            truncation_strategy=truncation_strategy,
            max_length=max_length
        )

        # Apply padding (max_length will be calculated dynamically if needed)
        # Returns torch tensors directly
        batch_output = self._pad(
            sequences,
            padding_strategy=padding_strategy,
            max_length=max_length
        )
        
        return Batch.from_mapping(batch_output)

    # ============================================================================
    # PADDING STRATEGIES
    # ============================================================================
    
    def _pad_do_not_pad(
        self,
        sequences: List[TPPSequence],
    ) -> Dict[str, torch.Tensor]:
        """Do not pad sequences, just convert to tensors.
        
        Args:
            sequences: List of TPPSequence objects
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with torch tensors (no padding applied)
        """
        batch_output = {
            'time_seqs': torch.tensor(
                [seq['time_seqs'] for seq in sequences], dtype=torch.float32
            ),
            'time_delta_seqs': torch.tensor(
                [seq['time_delta_seqs'] for seq in sequences], dtype=torch.float32
            ),
            'type_seqs': torch.tensor(
                [seq['type_seqs'] for seq in sequences], dtype=torch.long
            ),
        }
        
        # Create sequence mask
        sequence_mask = batch_output['type_seqs'] != self.pad_token_id
        batch_output['seq_non_pad_mask'] = sequence_mask
        
        return batch_output
    
    def _pad_to_max_length(
        self,
        sequences: List[TPPSequence],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to a fixed maximum length.
        
        Args:
            sequences: List of TPPSequence objects
            max_length: Maximum length to pad to
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with padded tensors
        """
        batch_output = {
            'time_seqs': self.make_pad_sequence(
                [seq['time_seqs'] for seq in sequences],
                0.0,
                padding_side=self.padding_side,
                max_len=max_length,
                dtype=torch.float32,
            ),
            'time_delta_seqs': self.make_pad_sequence(
                [seq['time_delta_seqs'] for seq in sequences],
                0.0,
                padding_side=self.padding_side,
                max_len=max_length,
                dtype=torch.float32,
            ),
            'type_seqs': self.make_pad_sequence(
                [seq['type_seqs'] for seq in sequences],
                self.pad_token_id,
                padding_side=self.padding_side,
                max_len=max_length,
                dtype=torch.long,
            ),
        }
        
        # Create sequence mask
        sequence_mask = batch_output['type_seqs'] != self.pad_token_id
        batch_output['seq_non_pad_mask'] = sequence_mask
        
        return batch_output
    

    def _pad(
        self,
        sequences: List[TPPSequence],
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply padding based on strategy.
        
        Routes to the appropriate padding method based on strategy.
        
        Args:
            sequences: List of TPPSequence objects
            max_length: Maximum length for padding (required for MAX_LENGTH strategy)
            padding_strategy: Strategy to use for padding
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with padded tensors and masks
        """

        # Check if all sequences have the same length
        sequence_lengths = np.array([len(seq['time_seqs']) for seq in sequences])
        is_uniform_length = np.all(sequence_lengths == max_length)
        
        # Route to appropriate padding strategy
        if padding_strategy == PaddingStrategy.DO_NOT_PAD or is_uniform_length:
            return self._pad_do_not_pad(sequences)
        
        elif padding_strategy == PaddingStrategy.MAX_LENGTH:
            if max_length is None:
                raise ValueError("max_length must be specified for MAX_LENGTH padding strategy")
            return self._pad_to_max_length(sequences, max_length)
        
        else :
            max_length = max(len(seq['time_seqs']) for seq in sequences)
            return self._pad_to_max_length(
                sequences,
                max_length=max_length,
            )


    @staticmethod
    def make_pad_sequence(
        seqs,
        pad_token_id,
        padding_side,
        max_len,
        dtype=None,
        group_by_event_types=False,
    ):
        """Pad the sequence batch-wise.

        Args:
            seqs (list): list of sequences with variational length
            pad_token_id (int, float): optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad
            max_len (int): optional, the maximum length of the sequence after padding. If None, then the
            length is set to be the max length of all input sequences.
            pad_at_end (bool): optional, whether to pad the sequnce at the end. If False,
            the sequence is pad at the beginning

        Returns:
            a torch tensor of padded sequence


        Example:
        ```python
        seqs = [[0, 1], [3, 4, 5]]
        pad_sequence(seqs, 100)
        >>> [[0, 1, 100], [3, 4, 5]]

        pad_sequence(seqs, 100, max_len=5)
        >>> [[0, 1, 100, 100, 100], [3, 4, 5, 100, 100]]
        ```

        """
        if not group_by_event_types:
            if padding_side == "right":
                pad_seq = torch.tensor(
                    [seq + [pad_token_id] * (max_len - len(seq)) for seq in seqs],
                    dtype=dtype,
                )
            else:
                pad_seq = torch.tensor(
                    [[pad_token_id] * (max_len - len(seq)) + seq for seq in seqs],
                    dtype=dtype,
                )
        else:
            pad_seq = []
            for seq in seqs:
                if padding_side == "right":
                    pad_seq.append(
                        torch.tensor(
                            [s + [pad_token_id] * (max_len - len(s)) for s in seq],
                            dtype=dtype,
                        )
                    )
                else:
                    pad_seq.append(
                        torch.tensor(
                            [[pad_token_id] * (max_len - len(s)) + s for s in seqs],
                            dtype=dtype,
                        )
                    )

            pad_seq = torch.stack(pad_seq)
        return pad_seq

    # The attention and type mask helpers were removed because masks are now built lazily in the models.