from enum import StrEnum


class SummaryStatsMetric(StrEnum):
    """Names of the summary statistics metrics that can be computed."""

    TIME_HIST_L1 = "time_hist_l1"
    TIME_HIST_L2 = "time_hist_l2"
    TIME_HIST_KL = "time_hist_kl"
    EVENT_TYPE_HIST_L1 = "event_type_hist_l1"
    EVENT_TYPE_HIST_L2 = "event_type_hist_l2"
    EVENT_TYPE_HIST_KL = "event_type_hist_kl"
    SEQUENCE_LENGTH_MEAN_DIFF = "sequence_length_mean_diff"
    SEQUENCE_LENGTH_MEDIAN_DIFF = "sequence_length_median_diff"
    MOMENT_MEAN_DIFF = "moment_mean_diff"
    MOMENT_VARIANCE_DIFF = "moment_variance_diff"
    MOMENT_SKEWNESS_DIFF = "moment_skewness_diff"
    MOMENT_KURTOSIS_DIFF = "moment_kurtosis_diff"
