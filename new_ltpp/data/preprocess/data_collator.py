from dataclasses import dataclass
from typing import Optional, Union, List

from new_ltpp.data.preprocess.event_tokenizer import EventTokenizer
from new_ltpp.utils.const import PaddingStrategy, TruncationStrategy
from new_ltpp.shared_types import TPPSequence
from new_ltpp.shared_types import Batch

@dataclass
class TPPDataCollator:
    """
    Data collator for temporal point process event sequences.
    
    This collator applies padding and/or truncation strategies to batches of sequences
    using the EventTokenizer's strategy-based architecture.

    Args:
        tokenizer: EventTokenizer instance for processing sequences
        strategy: Processing strategy - either a PaddingStrategy or TruncationStrategy.
                 If PaddingStrategy, sequences will be padded according to the strategy.
                 If TruncationStrategy, sequences will be truncated according to the strategy.
        max_length: Maximum sequence length (required for MAX_LENGTH padding or any truncation)
        
    Strategy Options:
        Padding:
            - PaddingStrategy.DO_NOT_PAD: No padding applied
            - PaddingStrategy.LONGEST: Pad to longest sequence in batch
            - PaddingStrategy.MAX_LENGTH: Pad to max_length (requires max_length parameter)
        
        Truncation:
            - TruncationStrategy.DO_NOT_TRUNCATE: No truncation applied
            - TruncationStrategy.LONGEST_FIRST: Truncate to max_length (requires max_length parameter)
    
    Example:
        # Pad to longest in batch
        collator = TPPDataCollator(tokenizer, strategy=PaddingStrategy.LONGEST)
        
        # Pad to fixed length
        collator = TPPDataCollator(tokenizer, strategy=PaddingStrategy.MAX_LENGTH, max_length=512)
        
        # Truncate sequences
        collator = TPPDataCollator(tokenizer, strategy=TruncationStrategy.LONGEST_FIRST, max_length=512)
    """

    tokenizer: EventTokenizer

    def __call__(self, features: List[TPPSequence]) -> Batch:
        """Process a batch of features with the configured strategy.
        
        Args:
            features: List of TPPSequence objects
            
        Returns:
            Batch with processed sequences (torch tensors)
        """
        batch = self.tokenizer.pad(
            inputs=features,
            return_attention_mask=True,
        )

        return batch
