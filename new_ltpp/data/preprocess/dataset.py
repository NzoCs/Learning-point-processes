from typing import Dict

from torch.utils.data import Dataset

from new_ltpp.shared_types import TPPSequence
from new_ltpp.utils import py_assert

from .sequence_identity import sequence_ids


class TPPDataset(Dataset):

    def __init__(self, data: Dict):

        self.time_seqs = data["time_seqs"]
        self.time_delta_seqs = data["time_delta_seqs"]
        self.type_seqs = data["type_seqs"]
        self.sequence_ids = data.get("sequence_ids")
        if self.sequence_ids is None:
            self.sequence_ids = sequence_ids(data)
        if len(self.sequence_ids) != len(self.time_seqs):
            raise ValueError("Sequence identifiers must match the dataset rows")

    def __len__(self):
        """

        Returns: length of the dataset

        """

        py_assert(
            len(self.time_seqs) == len(self.type_seqs)
            and len(self.time_delta_seqs) == len(self.type_seqs),
            ValueError,
            f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
            f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}",
        )

        return len(self.time_seqs)

    def __getitem__(self, index) -> TPPSequence:
        """

        Args:
            index: iteration index
        Returns:
            TPPSequence: A temporal point process sequence element

        """
        return TPPSequence(
            time_seqs=self.time_seqs[index],
            time_delta_seqs=self.time_delta_seqs[index],
            type_seqs=self.type_seqs[index],
            sequence_id=self.sequence_ids[index],
        )
