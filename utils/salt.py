from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PhotonTransferEstimate:
    """Diagnostics for an effective SUPPORT-image photon conversion.

    ``photons_per_unit`` is the value to pass to :func:`simulate_lower_power`.
    The remaining fields make the calibration auditable and allow callers to
    reject a dataset whose mean-variance relationship is not adequately linear.
    """

    photons_per_unit: float
    photons_per_raw_unit: float
    support_gain: float
    support_offset: float
    variance_slope: float
    variance_intercept: float
    effective_raw_pedestal: float
    effective_support_pedestal: float
    linear_r_squared: float
    quadratic_coefficient: float
    quadratic_r_squared: float
    delta_aicc: float
    scale_r_squared: float
    scale_quadratic_coefficient: float
    scale_delta_aicc: float
    quiet_photons_per_unit: Optional[float]
    quiet_photons_per_raw_unit: Optional[float]
    quiet_fraction: Optional[float]
    quiet_change_threshold: Optional[float]
    quiet_linear_r_squared: Optional[float]
    quiet_delta_aicc: Optional[float]
    pair_count: int
    coordinate_count: int
    eligible_sample_count: int
    fit_sample_count: int
    fit_intensity_range: Tuple[float, float]
    scale_raw_means: Tuple[float, ...]
    scale_support_means: Tuple[float, ...]
    bin_means: Tuple[float, ...]
    bin_variances: Tuple[float, ...]
    bin_standard_errors: Tuple[float, ...]
    bin_counts: Tuple[int, ...]
    warnings: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable diagnostic dictionary."""
        return asdict(self)


def _validate_quantiles(name, quantiles):
    try:
        lower, upper = quantiles
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two quantiles") from exc
    lower = float(lower)
    upper = float(upper)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= lower < upper <= 1")
    return lower, upper


def _aicc(residual_sum_squares, sample_count, parameter_count):
    residual_sum_squares = max(
        float(residual_sum_squares), np.finfo(np.float64).tiny
    )
    base = sample_count * np.log(residual_sum_squares / sample_count)
    base += 2.0 * parameter_count
    denominator = sample_count - parameter_count - 1
    if denominator <= 0:
        return np.inf
    return float(
        base
        + 2.0 * parameter_count * (parameter_count + 1) / denominator
    )


def _fit_line_and_quadratic(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 6:
        raise ValueError("At least six populated bins are required")

    slope, intercept = np.polyfit(x, y, 1)
    linear_prediction = intercept + slope * x

    center = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The selected intensities have no usable range")
    normalized = (x - center) / scale
    normalized_q2, normalized_q1, normalized_q0 = np.polyfit(
        normalized, y, 2
    )
    quadratic_prediction = (
        normalized_q0
        + normalized_q1 * normalized
        + normalized_q2 * normalized ** 2
    )
    q2 = normalized_q2 / scale ** 2
    q1 = normalized_q1 / scale - 2.0 * normalized_q2 * center / scale ** 2
    q0 = (
        normalized_q0
        - normalized_q1 * center / scale
        + normalized_q2 * center ** 2 / scale ** 2
    )

    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    linear_rss = float(np.sum((y - linear_prediction) ** 2))
    quadratic_rss = float(np.sum((y - quadratic_prediction) ** 2))
    linear_r_squared = (
        1.0 - linear_rss / total_sum if total_sum > 0 else np.nan
    )
    quadratic_r_squared = (
        1.0 - quadratic_rss / total_sum if total_sum > 0 else np.nan
    )
    linear_aicc = _aicc(linear_rss, x.size, 3)
    quadratic_aicc = _aicc(quadratic_rss, x.size, 4)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "linear_prediction": linear_prediction,
        "linear_r_squared": float(linear_r_squared),
        "q2": float(q2),
        "q1": float(q1),
        "q0": float(q0),
        "quadratic_prediction": quadratic_prediction,
        "quadratic_r_squared": float(quadratic_r_squared),
        "delta_aicc": float(linear_aicc - quadratic_aicc),
    }


def _equal_count_scale_fit(
    raw_means,
    support_means,
    n_bins,
    min_samples_per_bin,
    fit_quantiles,
):
    valid = np.isfinite(raw_means) & np.isfinite(support_means)
    if int(valid.sum()) < max(100, 5 * min_samples_per_bin):
        raise ValueError("Too few matched coordinates for the SUPPORT scale fit")
    raw_means = np.asarray(raw_means[valid], dtype=np.float64)
    support_means = np.asarray(support_means[valid], dtype=np.float64)
    lower, upper = np.quantile(support_means, fit_quantiles)
    central = (support_means >= lower) & (support_means <= upper)
    raw_means = raw_means[central]
    support_means = support_means[central]

    edges = np.unique(
        np.quantile(support_means, np.linspace(0.0, 1.0, n_bins + 1))
    )
    if edges.size < 6:
        raise ValueError("SUPPORT has too few distinct intensity levels")
    bin_ids = np.clip(
        np.searchsorted(edges, support_means, side="right") - 1,
        0,
        edges.size - 2,
    )
    binned_raw = []
    binned_support = []
    for bin_index in range(edges.size - 1):
        selected = bin_ids == bin_index
        if int(selected.sum()) < min_samples_per_bin:
            continue
        binned_raw.append(float(np.mean(raw_means[selected])))
        binned_support.append(float(np.mean(support_means[selected])))
    if len(binned_raw) < 6:
        raise ValueError("Fewer than six populated SUPPORT scale bins remain")
    binned_raw = np.asarray(binned_raw, dtype=np.float64)
    binned_support = np.asarray(binned_support, dtype=np.float64)
    models = _fit_line_and_quadratic(binned_raw, binned_support)
    if not np.isfinite(models["slope"]) or models["slope"] <= 0:
        raise ValueError("The fitted raw-to-SUPPORT gain is not positive")
    return models, binned_raw, binned_support


def _binned_pair_variance(
    predictor,
    difference,
    pair_ids,
    *,
    n_bins,
    min_samples_per_bin,
    max_fit_samples,
    fit_quantiles,
    min_mean_level,
    max_mean_level,
    rng,
):
    predictor = np.asarray(predictor).ravel()
    difference = np.asarray(difference).ravel()
    pair_ids = np.asarray(pair_ids).ravel()
    valid = np.isfinite(predictor) & np.isfinite(difference)
    if min_mean_level is not None:
        valid &= predictor >= min_mean_level
    if max_mean_level is not None:
        valid &= predictor <= max_mean_level
    indices = np.flatnonzero(valid)
    eligible_count = int(indices.size)
    if max_fit_samples is not None and indices.size > max_fit_samples:
        indices = rng.choice(indices, size=max_fit_samples, replace=False)
    x = predictor[indices].astype(np.float64, copy=False)
    d = difference[indices].astype(np.float64, copy=False)
    ids = pair_ids[indices]
    if x.size < max(1_000, 3 * min_samples_per_bin):
        raise ValueError(f"Too few eligible pair samples: {x.size:,}")

    lower, upper = np.quantile(x, fit_quantiles)
    central = (x >= lower) & (x <= upper)
    x = x[central]
    d = d[central]
    ids = ids[central]
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 6:
        raise ValueError("The pair predictor has too few distinct levels")
    bin_ids = np.clip(
        np.searchsorted(edges, x, side="right") - 1,
        0,
        edges.size - 2,
    )

    means = []
    variances = []
    standard_errors = []
    counts = []
    for bin_index in range(edges.size - 1):
        in_bin = bin_ids == bin_index
        if int(in_bin.sum()) < min_samples_per_bin:
            continue
        bin_x = x[in_bin]
        bin_difference = d[in_bin]
        bin_pairs = ids[in_bin]
        centered_parts = []
        used_x_parts = []
        degrees_of_freedom = 0
        for pair_id in np.unique(bin_pairs):
            in_pair = bin_pairs == pair_id
            count = int(in_pair.sum())
            if count < 2:
                continue
            pair_difference = bin_difference[in_pair]
            centered_parts.append(pair_difference - np.mean(pair_difference))
            used_x_parts.append(bin_x[in_pair])
            degrees_of_freedom += count - 1
        if degrees_of_freedom < 2:
            continue
        centered = np.concatenate(centered_parts)
        used_x = np.concatenate(used_x_parts)
        squared_noise = 0.5 * centered ** 2
        correction = centered.size / degrees_of_freedom
        variance = float(np.mean(squared_noise) * correction)
        standard_error = float(
            np.std(squared_noise, ddof=1)
            / np.sqrt(squared_noise.size)
            * correction
        )
        if np.isfinite(variance) and variance > 0:
            means.append(float(np.mean(used_x)))
            variances.append(variance)
            standard_errors.append(standard_error)
            counts.append(int(centered.size))
    if len(means) < 6:
        raise ValueError("Fewer than six populated photon-transfer bins remain")
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(standard_errors, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        eligible_count,
        int(x.size),
        (float(lower), float(upper)),
    )


def estimate_photons_per_unit(
    raw_video,
    support_video,
    *,
    frame_start=0,
    frame_stop=None,
    max_pairs=50,
    sample_count=40_000,
    n_bins=30,
    min_samples_per_bin=500,
    scale_min_samples_per_bin=50,
    max_fit_samples=2_000_000,
    fit_quantiles=(0.01, 0.99),
    scale_fit_quantiles=(0.01, 0.99),
    saturation_value=None,
    floor_value=None,
    min_mean_level=0.0,
    max_mean_level=None,
    random_seed=7,
    minimum_r_squared=0.98,
    minimum_scale_r_squared=0.99,
    maximum_quadratic_delta_aicc=6.0,
    quiet_change_quantile=0.5,
    maximum_quiet_relative_difference=0.15,
    minimum_recommended_pairs=10,
):
    """Estimate the effective photon scale from raw and SUPPORT time series.

    The raw frame-difference variance is binned by the matching denoised
    SUPPORT intensity. Temporal means over the selected frame window establish
    the affine conversion ``SUPPORT = offset + gain * raw``. The returned
    ``photons_per_unit`` is expressed in SUPPORT image units and can be passed
    directly to :func:`simulate_lower_power`.

    The primary estimate deliberately measures the unmodified raw frame-pair
    difference. A second fit on locations with the smallest SUPPORT-predicted
    frame-to-frame changes is returned as a sensitivity diagnostic. A material
    disagreement warns that motion or biological dynamics may be inflating the
    apparent shot noise; the quiet fit never silently replaces the primary.
    The linear-versus-quadratic AICc values are unweighted model diagnostics,
    not formal uncertainty intervals for heteroskedastic binned variances.

    Both inputs must be time-first ``(T, Y, X)`` or ``(T, Z, Y, X)`` arrays of
    the same acquisition. ``frame_stop`` is end-exclusive. When ``max_pairs``
    caps the available adjacent, non-overlapping pairs, pairs are distributed
    evenly across ``[frame_start, frame_stop)`` rather than taken only from the
    beginning. Only matched sampled voxels and selected pair frames are
    retained, so memory use remains bounded for TIFF memmaps.
    """
    raw_video = np.asanyarray(raw_video)
    support_video = np.asanyarray(support_video)
    if raw_video.shape != support_video.shape:
        raise ValueError(
            "raw_video and support_video must have the same time-first shape"
        )
    if raw_video.ndim not in (3, 4):
        raise ValueError(
            "raw_video and support_video must have shape (T,Y,X) or (T,Z,Y,X)"
        )
    if any(size <= 0 for size in raw_video.shape):
        raise ValueError("Input dimensions must be nonempty")
    if isinstance(frame_start, bool) or not isinstance(frame_start, (int, np.integer)):
        raise TypeError("frame_start must be an integer")
    frame_start = int(frame_start)
    if frame_start < 0 or frame_start >= raw_video.shape[0]:
        raise ValueError("frame_start is outside the available time range")
    if frame_stop is None:
        frame_stop = raw_video.shape[0]
    elif isinstance(frame_stop, bool) or not isinstance(
        frame_stop, (int, np.integer)
    ):
        raise TypeError("frame_stop must be an integer or None")
    frame_stop = int(frame_stop)
    if frame_stop <= frame_start or frame_stop > raw_video.shape[0]:
        raise ValueError(
            "frame_stop must be greater than frame_start and no greater than "
            "the available frame count"
        )
    if max_pairs is not None:
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)):
            raise TypeError("max_pairs must be a positive integer or None")
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
    if isinstance(sample_count, bool) or int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    if isinstance(n_bins, bool) or int(n_bins) < 6:
        raise ValueError("n_bins must be at least six")
    if int(min_samples_per_bin) < 2:
        raise ValueError("min_samples_per_bin must be at least two")
    if int(scale_min_samples_per_bin) < 2:
        raise ValueError("scale_min_samples_per_bin must be at least two")
    if max_fit_samples is not None and int(max_fit_samples) <= 0:
        raise ValueError("max_fit_samples must be positive or None")
    if (
        isinstance(minimum_recommended_pairs, bool)
        or int(minimum_recommended_pairs) < 2
    ):
        raise ValueError("minimum_recommended_pairs must be at least two")
    for name, value in (
        ("minimum_r_squared", minimum_r_squared),
        ("minimum_scale_r_squared", minimum_scale_r_squared),
        ("maximum_quadratic_delta_aicc", maximum_quadratic_delta_aicc),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    fit_quantiles = _validate_quantiles("fit_quantiles", fit_quantiles)
    scale_fit_quantiles = _validate_quantiles(
        "scale_fit_quantiles", scale_fit_quantiles
    )
    quiet_change_quantile = float(quiet_change_quantile)
    if not 0.0 < quiet_change_quantile < 1.0:
        raise ValueError("quiet_change_quantile must be between zero and one")
    maximum_quiet_relative_difference = float(
        maximum_quiet_relative_difference
    )
    if (
        not np.isfinite(maximum_quiet_relative_difference)
        or maximum_quiet_relative_difference < 0
    ):
        raise ValueError(
            "maximum_quiet_relative_difference must be finite and nonnegative"
        )

    available_pairs = (frame_stop - frame_start) // 2
    pair_count = (
        available_pairs
        if max_pairs is None
        else min(available_pairs, int(max_pairs))
    )
    if pair_count < 2:
        raise ValueError(
            "The selected frame window must contain at least two complete "
            "frame pairs"
        )

    candidate_pair_ids = np.arange(available_pairs, dtype=np.int64)
    if pair_count == available_pairs:
        selected_pair_ids = candidate_pair_ids
    else:
        selected_pair_ids = candidate_pair_ids[
            np.linspace(0, available_pairs - 1, pair_count, dtype=np.int64)
        ]
    pair_starts = frame_start + 2 * selected_pair_ids
    pair_frame_indices = np.column_stack(
        (pair_starts, pair_starts + 1)
    ).reshape(-1)

    spatial_shape = raw_video.shape[1:]
    spatial_count = int(np.prod(spatial_shape))
    coordinate_count = min(int(sample_count), spatial_count)
    rng = np.random.default_rng(random_seed)
    spatial_indices = np.sort(
        rng.choice(spatial_count, size=coordinate_count, replace=False)
    )

    raw_sum = np.zeros(coordinate_count, dtype=np.float64)
    support_sum = np.zeros(coordinate_count, dtype=np.float64)
    temporal_count = np.zeros(coordinate_count, dtype=np.int64)
    selected_frame_count = 2 * pair_count
    pair_raw = np.full(
        (selected_frame_count, coordinate_count), np.nan, dtype=np.float32
    )
    pair_support = np.full_like(pair_raw, np.nan)
    pair_positions = np.full(raw_video.shape[0], -1, dtype=np.int64)
    pair_positions[pair_frame_indices] = np.arange(
        selected_frame_count, dtype=np.int64
    )

    for frame_index in range(frame_start, frame_stop):
        raw_sample = np.take(
            np.asarray(raw_video[frame_index]), spatial_indices
        ).astype(np.float32, copy=False)
        support_sample = np.take(
            np.asarray(support_video[frame_index]), spatial_indices
        ).astype(np.float32, copy=False)
        valid = np.isfinite(raw_sample) & np.isfinite(support_sample)
        if saturation_value is not None:
            valid &= raw_sample < saturation_value
        if floor_value is not None:
            valid &= raw_sample > floor_value
        raw_sum[valid] += raw_sample[valid]
        support_sum[valid] += support_sample[valid]
        temporal_count[valid] += 1
        pair_position = pair_positions[frame_index]
        if pair_position >= 0:
            pair_raw[pair_position] = raw_sample
            pair_support[pair_position] = support_sample

    temporal_valid = temporal_count > 0
    raw_means = np.full(coordinate_count, np.nan, dtype=np.float64)
    support_means = np.full(coordinate_count, np.nan, dtype=np.float64)
    raw_means[temporal_valid] = (
        raw_sum[temporal_valid] / temporal_count[temporal_valid]
    )
    support_means[temporal_valid] = (
        support_sum[temporal_valid] / temporal_count[temporal_valid]
    )
    scale_models, scale_raw_means, scale_support_means = _equal_count_scale_fit(
        raw_means,
        support_means,
        int(n_bins),
        int(scale_min_samples_per_bin),
        scale_fit_quantiles,
    )
    support_gain = scale_models["slope"]
    support_offset = scale_models["intercept"]

    raw_a = pair_raw[0::2]
    raw_b = pair_raw[1::2]
    support_a = pair_support[0::2]
    support_b = pair_support[1::2]
    raw_difference = raw_a - raw_b
    support_level = 0.5 * (support_a + support_b)
    predictor = (support_level - support_offset) / support_gain
    valid_pairs = (
        np.isfinite(raw_a)
        & np.isfinite(raw_b)
        & np.isfinite(support_a)
        & np.isfinite(support_b)
    )
    if saturation_value is not None:
        valid_pairs &= (raw_a < saturation_value) & (raw_b < saturation_value)
    if floor_value is not None:
        valid_pairs &= (raw_a > floor_value) & (raw_b > floor_value)
    pair_ids = np.broadcast_to(
        np.arange(pair_count, dtype=np.int32)[:, None], raw_difference.shape
    )
    (
        bin_means,
        bin_variances,
        bin_standard_errors,
        bin_counts,
        eligible_sample_count,
        fit_sample_count,
        fit_intensity_range,
    ) = _binned_pair_variance(
        predictor[valid_pairs],
        raw_difference[valid_pairs],
        pair_ids[valid_pairs],
        n_bins=int(n_bins),
        min_samples_per_bin=int(min_samples_per_bin),
        max_fit_samples=(
            None if max_fit_samples is None else int(max_fit_samples)
        ),
        fit_quantiles=fit_quantiles,
        min_mean_level=min_mean_level,
        max_mean_level=max_mean_level,
        rng=rng,
    )
    variance_models = _fit_line_and_quadratic(bin_means, bin_variances)
    variance_slope = variance_models["slope"]
    if not np.isfinite(variance_slope) or variance_slope <= 0:
        raise ValueError("The fitted photon-transfer slope is not positive")

    photons_per_raw_unit = 1.0 / variance_slope
    photons_per_unit = photons_per_raw_unit / support_gain
    if not np.isfinite(photons_per_unit) or photons_per_unit <= 0:
        raise ValueError("The fitted photons_per_unit is not finite and positive")
    effective_raw_pedestal = -variance_models["intercept"] / variance_slope
    effective_support_pedestal = (
        support_offset + support_gain * effective_raw_pedestal
    )

    quality_warnings = []
    if pair_count < int(minimum_recommended_pairs):
        quality_warnings.append(
            f"only {pair_count} frame pairs were used; at least "
            f"{int(minimum_recommended_pairs)} are recommended"
        )
    if not np.isfinite(effective_support_pedestal):
        quality_warnings.append(
            "the extrapolated effective SUPPORT pedestal is not finite"
        )
    quiet_photons_per_raw_unit = None
    quiet_photons_per_unit = None
    quiet_fraction = None
    quiet_change_threshold = None
    quiet_linear_r_squared = None
    quiet_delta_aicc = None
    try:
        valid_predictor = predictor[valid_pairs]
        valid_difference = raw_difference[valid_pairs]
        valid_pair_ids = pair_ids[valid_pairs]
        valid_support_change = np.abs(
            (support_a[valid_pairs] - support_b[valid_pairs]) / support_gain
        )
        finite_change = np.isfinite(valid_support_change)
        if not np.any(finite_change):
            raise ValueError("no finite SUPPORT frame-to-frame changes")
        quiet_change_threshold = float(
            np.quantile(
                valid_support_change[finite_change], quiet_change_quantile
            )
        )
        quiet_selection = finite_change & (
            valid_support_change <= quiet_change_threshold
        )
        quiet_fraction = float(np.mean(quiet_selection))
        (
            quiet_bin_means,
            quiet_bin_variances,
            _,
            _,
            _,
            _,
            _,
        ) = _binned_pair_variance(
            valid_predictor[quiet_selection],
            valid_difference[quiet_selection],
            valid_pair_ids[quiet_selection],
            n_bins=int(n_bins),
            min_samples_per_bin=int(min_samples_per_bin),
            max_fit_samples=(
                None if max_fit_samples is None else int(max_fit_samples)
            ),
            fit_quantiles=fit_quantiles,
            min_mean_level=min_mean_level,
            max_mean_level=max_mean_level,
            rng=rng,
        )
        quiet_models = _fit_line_and_quadratic(
            quiet_bin_means, quiet_bin_variances
        )
        quiet_slope = quiet_models["slope"]
        if not np.isfinite(quiet_slope) or quiet_slope <= 0:
            raise ValueError("quiet-pair variance slope is not positive")
        quiet_photons_per_raw_unit = float(1.0 / quiet_slope)
        quiet_photons_per_unit = float(
            quiet_photons_per_raw_unit / support_gain
        )
        quiet_linear_r_squared = float(quiet_models["linear_r_squared"])
        quiet_delta_aicc = float(quiet_models["delta_aicc"])
        if not np.isfinite(quiet_linear_r_squared):
            raise ValueError("quiet-pair variance-fit R^2 is not finite")
        if not np.isfinite(quiet_delta_aicc):
            raise ValueError("quiet-pair variance-model delta AICc is not finite")
        quiet_relative_difference = abs(
            quiet_photons_per_unit / photons_per_unit - 1.0
        )
        if quiet_relative_difference > maximum_quiet_relative_difference:
            quality_warnings.append(
                "quiet-pair photon estimate differs from the primary by "
                f"{quiet_relative_difference:.1%} (limit "
                f"{maximum_quiet_relative_difference:.1%}); motion or "
                "biological dynamics may inflate frame-pair variance"
            )
    except (TypeError, ValueError, FloatingPointError) as exc:
        quiet_photons_per_raw_unit = None
        quiet_photons_per_unit = None
        quiet_fraction = None
        quiet_change_threshold = None
        quiet_linear_r_squared = None
        quiet_delta_aicc = None
        quality_warnings.append(
            f"quiet-pair sensitivity diagnostic unavailable: {exc}"
        )

    if not np.isfinite(variance_models["linear_r_squared"]):
        quality_warnings.append("variance fit R^2 is not finite")
    elif variance_models["linear_r_squared"] < minimum_r_squared:
        quality_warnings.append(
            f"variance fit R^2={variance_models['linear_r_squared']:.5f} "
            f"is below {minimum_r_squared:.5f}"
        )
    if not np.isfinite(variance_models["delta_aicc"]):
        quality_warnings.append("variance-model delta AICc is not finite")
    elif variance_models["delta_aicc"] > maximum_quadratic_delta_aicc:
        quality_warnings.append(
            "quadratic variance model is preferred "
            f"(delta AICc={variance_models['delta_aicc']:.3f})"
        )
    if not np.isfinite(scale_models["linear_r_squared"]):
        quality_warnings.append("raw-to-SUPPORT scale R^2 is not finite")
    elif scale_models["linear_r_squared"] < minimum_scale_r_squared:
        quality_warnings.append(
            f"raw-to-SUPPORT scale R^2={scale_models['linear_r_squared']:.5f} "
            f"is below {minimum_scale_r_squared:.5f}"
        )
    if not np.isfinite(scale_models["delta_aicc"]):
        quality_warnings.append("raw-to-SUPPORT scale delta AICc is not finite")
    elif scale_models["delta_aicc"] > maximum_quadratic_delta_aicc:
        quality_warnings.append(
            "quadratic raw-to-SUPPORT scale is preferred "
            f"(delta AICc={scale_models['delta_aicc']:.3f})"
        )

    return PhotonTransferEstimate(
        photons_per_unit=float(photons_per_unit),
        photons_per_raw_unit=float(photons_per_raw_unit),
        support_gain=float(support_gain),
        support_offset=float(support_offset),
        variance_slope=float(variance_slope),
        variance_intercept=float(variance_models["intercept"]),
        effective_raw_pedestal=float(effective_raw_pedestal),
        effective_support_pedestal=float(effective_support_pedestal),
        linear_r_squared=float(variance_models["linear_r_squared"]),
        quadratic_coefficient=float(variance_models["q2"]),
        quadratic_r_squared=float(variance_models["quadratic_r_squared"]),
        delta_aicc=float(variance_models["delta_aicc"]),
        scale_r_squared=float(scale_models["linear_r_squared"]),
        scale_quadratic_coefficient=float(scale_models["q2"]),
        scale_delta_aicc=float(scale_models["delta_aicc"]),
        quiet_photons_per_unit=quiet_photons_per_unit,
        quiet_photons_per_raw_unit=quiet_photons_per_raw_unit,
        quiet_fraction=quiet_fraction,
        quiet_change_threshold=quiet_change_threshold,
        quiet_linear_r_squared=quiet_linear_r_squared,
        quiet_delta_aicc=quiet_delta_aicc,
        pair_count=int(pair_count),
        coordinate_count=int(coordinate_count),
        eligible_sample_count=int(eligible_sample_count),
        fit_sample_count=int(fit_sample_count),
        fit_intensity_range=fit_intensity_range,
        scale_raw_means=tuple(float(value) for value in scale_raw_means),
        scale_support_means=tuple(float(value) for value in scale_support_means),
        bin_means=tuple(float(value) for value in bin_means),
        bin_variances=tuple(float(value) for value in bin_variances),
        bin_standard_errors=tuple(float(value) for value in bin_standard_errors),
        bin_counts=tuple(int(value) for value in bin_counts),
        warnings=tuple(quality_warnings),
    )


def simulate_lower_power(video,
                         power_ratio=0.5,
                         gamma=2.0,
                         photons_per_unit=1/100,
                         background_slices=None,
                         seed=1,
                         zero_photon_level=0.0):
    """
    Simulate lower laser power for a 2D+t or 3D+t two-photon imaging stack.
 
    Parameters
    ----------
    video (np.array): image stack, shape (t, y, x) or (t, z, y, x); arbitrary
        units. Values below ``zero_photon_level`` are treated as zero
        photon-bearing signal.
    power_ratio (float): Desired power ratio, e.g. 0.5 for half power.
    gamma (float): Excitation scaling exponent, ``signal ~ power**gamma``.
        The default 2.0 is the unsaturated two-photon fluorescence relationship.
    photons_per_unit (float): Effective conversion from SUPPORT image units to
        detected-count units before Poisson sampling. Use
        ``estimate_photons_per_unit`` on the matching raw and SUPPORT movies.
    background_slices (tuple of slices or None): Deprecated compatibility
        argument. Illumination-dependent background now scales with the signal;
        use ``zero_photon_level`` for a fixed detector/electronic offset.
    seed (int): random seed.
    zero_photon_level (float or array-like): Fixed additive, non-photon baseline
        in SUPPORT image units; this is not a detector-noise variance. The
        simulated count-bearing intensity is
        ``max(video - zero_photon_level, 0)``; the pedestal is added back after
        Poisson sampling. Keep this at zero unless a dark-frame calibration is
        available or explicitly opt into the estimator's effective pedestal.
 
    Returns
    -------
    noisy_video (np.array): Simulated video with lower power, same shape as input.
 
    """
    video = np.asarray(video, dtype=np.float32)
    assert video.ndim in (3, 4), "Input video must have shape (t, y, x) or (t, z, y, x)"
    assert all(size > 0 for size in video.shape), "Input video dimensions must be nonempty"
    assert power_ratio > 0 and power_ratio <= 1, "Power ratio must be between 0 and 1"
    assert gamma > 0, "Gamma must be positive"
    assert photons_per_unit > 0, "Poisson scale factor must be positive"
    if not np.all(np.isfinite(video)):
        raise ValueError("Input video contains non-finite values")

    try:
        zero_photon_level = np.asarray(
            zero_photon_level, dtype=np.float32
        )
        np.broadcast_shapes(video.shape, zero_photon_level.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "zero_photon_level must be finite and broadcastable to video"
        ) from exc
    if not np.all(np.isfinite(zero_photon_level)):
        raise ValueError("zero_photon_level must contain only finite values")

    # ``background_slices`` is retained so older callers do not break. A dark
    # image, not a dark image region, is required to identify a fixed detector
    # offset because out-of-focus fluorescence should scale with laser power.
    _ = background_slices
    rng = np.random.default_rng(seed)
    # Subtract before clipping: a calibrated detector pedestal may itself be
    # negative in SUPPORT units, so clipping the image at numeric zero first
    # would create photon-bearing signal that is not present in the data.
    signal = np.clip(video - zero_photon_level, 0, None)

    # Scale signal by the appropriate power ratio
    signal_scaling_factor = power_ratio ** gamma
    scaled_signal = signal_scaling_factor * signal
    noisy_video = scaled_signal

    # SUPPORT inference is unconstrained and can produce small negative values.
    # A Poisson rate cannot be negative, so treat those values as zero photons.
    if not np.all(np.isfinite(noisy_video)):
        raise ValueError("Poisson mean contains non-finite values")
    poisson_mean = np.clip(noisy_video, 0, None)

    # Convert to effective detected counts, sample, and convert back.
    noisy_video = rng.poisson(
        poisson_mean * photons_per_unit
    ).astype(np.float32) / photons_per_unit
    noisy_video += zero_photon_level

    return noisy_video
