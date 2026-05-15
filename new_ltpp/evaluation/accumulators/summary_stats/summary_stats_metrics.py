from enum import StrEnum


class SummaryStatsMetric(StrEnum):
    """Names of the summary statistics metrics that can be computed."""

    TIME_HIST_L1 = "time_hist_l1"
    TIME_HIST_L2 = "time_hist_l2"
    TIME_HIST_KL = "time_hist_kl"
    EVENT_TYPE_HIST_L1 = "event_type_hist_l1"
    EVENT_TYPE_HIST_L2 = "event_type_hist_l2"
    EVENT_TYPE_HIST_KL = "event_type_hist_kl"
    SEQUENCE_LENGTH_HIST_L1 = "sequence_length_hist_l1"
    SEQUENCE_LENGTH_HIST_L2 = "sequence_length_hist_l2"
    SEQUENCE_LENGTH_HIST_KL = "sequence_length_hist_kl"
    ACF_HIST_L1 = "acf_hist_l1"
    ACF_HIST_L2 = "acf_hist_l2"
