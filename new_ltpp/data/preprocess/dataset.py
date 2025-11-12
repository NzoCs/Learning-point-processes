import math
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from new_ltpp.utils import py_assert
from .types import TPPSequence


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

    def get_dt_stats(self):
        x_bar, s_2_x, n = 0.0, 0.0, 0
        min_dt, max_dt = np.inf, -np.inf

        for dts, marks in zip(self.time_delta_seqs, self.type_seqs):
            dts = np.array(dts[1 : -1 if marks[-1] == -1 else None])
            min_dt = min(min_dt, dts.min())
            max_dt = max(max_dt, dts.max())
            y_bar = dts.mean()
            s_2_y = dts.var()
            m = dts.shape[0]
            n += m
            # Formula taken from https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            s_2_x = (((n - 1) * s_2_x + (m - 1) * s_2_y) / (n + m - 1)) + (
                (n * m * ((x_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1))
            )
            x_bar = (n * x_bar + m * y_bar) / (n + m)

        print(x_bar, (s_2_x**0.5))
        print(f"min_dt: {min_dt}")
        print(f"max_dt: {max_dt}")
        return x_bar, (s_2_x**0.5), min_dt, max_dt
