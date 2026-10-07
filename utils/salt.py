import numpy as np


def _find_background_slices(video):
    """Find the darkest border tile in a ``(t, y, x)`` stack."""
    _, height, width = video.shape
    rows = min(8, height)
    columns = min(8, width)
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    column_edges = np.linspace(0, width, columns + 1, dtype=int)

    # A small temporal sample is enough to locate a stable background region
    # without constructing a full temporal projection of a large stack.
    sample_count = min(32, video.shape[0])
    sample_indices = np.linspace(
        0, video.shape[0] - 1, sample_count, dtype=int
    )
    temporal_median = np.median(video[sample_indices], axis=0)

    darkest = None
    for row in range(rows):
        for column in range(columns):
            if row not in (0, rows - 1) and column not in (0, columns - 1):
                continue
            y_slice = slice(row_edges[row], row_edges[row + 1])
            x_slice = slice(column_edges[column], column_edges[column + 1])
            level = float(np.median(temporal_median[y_slice, x_slice]))
            if darkest is None or level < darkest[0]:
                darkest = (level, y_slice, x_slice)

    return slice(None), darkest[1], darkest[2]


def simulate_lower_power(video,
                         power_ratio=0.5,
                         gamma=2.0,
                         photons_per_unit=1/100,
                         background_slices=None,
                         seed=1):
    """
    Simulate lower laser power for a 2D+t or 3D+t two-photon imaging stack.
 
    Parameters
    ----------
    video (np.array): image stack, shape (t, y, x) or (t, z, y, x); arbitrary units.
    power_ratio (float): Desired power ratio, e.g. 0.5 for half power.
    gamma (float): two-photon scaling exponent, signal ~ power^gamma. 2.0 should be roughly correct.
    photons_per_unit (float): Conversion from image units to photon units before Poisson noise; I estimated this visually and it's slightly off from the true value, but not horribly off.
    background_slices (tuple of slices or None): Optional background ROI in
        (t, y, x) coordinates. If None, the darkest border tile is selected
        automatically. For 4D input, selection and background estimation are
        performed independently for every z-slice.
    seed (int): random seed.
 
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

    rng = np.random.default_rng(seed)

    # Subtract out the background level so we can apply scaling to the signal alone
    if video.ndim == 3:
        slices = (
            _find_background_slices(video)
            if background_slices is None
            else background_slices
        )
        background_level = float(np.median(video[slices]))
    else:
        background_levels = []
        for z_index in range(video.shape[1]):
            z_video = video[:, z_index, :, :]
            slices = (
                _find_background_slices(z_video)
                if background_slices is None
                else background_slices
            )
            print(f'\t\t Background ROI for Z={z_index:02d}:\t', slices)
            background_levels.append(np.median(z_video[slices]))
        background_level = np.asarray(
            background_levels, dtype=np.float32
        )[None, :, None, None]

    signal = video - background_level

    # Scale signal by the appropriate power ratio
    signal_scaling_factor = power_ratio ** gamma
    scaled_signal = signal_scaling_factor * signal
    noisy_video = background_level + scaled_signal

    # Convert to pseudo-photons, apply Poisson noise, convert back - have to assume photons_per_unit
    noisy_video = rng.poisson(noisy_video * photons_per_unit).astype(np.float32) / photons_per_unit

    # noisy_video = noisy_video + background_level

    return noisy_video
