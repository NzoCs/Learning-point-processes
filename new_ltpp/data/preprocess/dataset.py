from typing import Dict

from torch.utils.data import Dataset

from new_ltpp.shared_types import TPPSequence
from new_ltpp.utils import py_assert


class TPPDataset(Dataset):

    def __init__(self, data: Dict):

        self.time_seqs = data["time_seqs"]
        self.time_delta_seqs = data["time_delta_seqs"]
        self.type_seqs = data["type_seqs"]

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

    def __getitem__(self, idx) -> TPPSequence:
        """

        Args:
            idx: iteration index

        Returns:
            TPPSequence: A temporal point process sequence element

        """
        return TPPSequence(
            time_seqs=self.time_seqs[idx],
            time_delta_seqs=self.time_delta_seqs[idx],
            type_seqs=self.type_seqs[idx],
        )
