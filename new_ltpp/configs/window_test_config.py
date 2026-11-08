"""Fixed-exposure tests have an explicit contract separate from stopped MMD."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WindowTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    test_type: Literal["window_mmd", "relative_loss", "ksd"]
    horizon: float = Field(gt=0)
    num_event_types: int = Field(ge=1)
    n_samples: int = Field(default=499, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    representation: Literal["counting_grid", "intervals", "events"] = "counting_grid"
    point_process_kernel: Literal["signature", "count_rbf"] = "signature"
    space_kernel_type: Literal["rbf", "linear"] = "rbf"
    grid: int = Field(default=100, ge=2)
    count_scale: float = Field(default=10.0, gt=0)
    gap_scale: float = Field(default=0.02, gt=0)
    rank_scale: float = Field(default=32.0, gt=0)
    sigma: float = Field(default=0.3, gt=0)
    dyadic_order: int = Field(default=0, ge=0)
    max_batch: int = Field(default=16, ge=1)
    stein_bins: int = Field(default=20, ge=1)
    stein_scale: float = Field(default=10.0, gt=0)
    stein_sigma: float = Field(default=1.0, gt=0)
    quadrature_order: int = Field(default=8, ge=2)
    quadrature_rtol: float = Field(default=2e-5, gt=0, lt=1)
    max_states: int = Field(default=50000, ge=1)
