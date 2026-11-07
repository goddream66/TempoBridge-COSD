"""Raw-waveform diagnostics, not calibrated signal-quality probabilities."""

import numpy as np

DIAGNOSTIC_VERSION = "raw_diagnostics_v1"


def waveform_diagnostics(wave, fs):
    """Measure BEFORE interpolation, resampling, filtering or normalization.

    Extreme occupancy is a clipping proxy: repeated observed extrema need not be
    ADC saturation. Flatness counts adjacent finite differences below 1e-5 of
    the observed range. Spectral ratios are descriptive, not rejection criteria.
    Missing signals return missing_fraction=1 and zero measured coverage.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Diagnostic sample rate must be finite and positive")
    x = np.asarray([] if wave is None else wave, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Diagnostics require a one-dimensional waveform")
    finite = np.isfinite(x)
    result = {"missing_fraction": 1 - float(finite.mean()) if len(x) else 1.,
              "flatline_fraction": 0., "extreme_occupancy": 0.,
              "baseline_power_fraction": 0., "high_frequency_power_fraction": 0.,
              "finite_samples": int(finite.sum())}
    if not finite.any():
        return result
    values = x[finite]
    low, high = values.min(), values.max()
    tolerance = max((high - low) * 1e-5, 1e-12)
    result["extreme_occupancy"] = float(((values <= low + tolerance) | (values >= high - tolerance)).mean())
    adjacent = finite[1:] & finite[:-1]
    if adjacent.any():
        differences = x[1:][adjacent] - x[:-1][adjacent]
        result["flatline_fraction"] = float((abs(differences) <= tolerance).mean())
    # Interpolation is used only for diagnostic spectral ratios, never to hide missingness.
    if len(values) >= 4:
        filled = np.interp(np.arange(len(x)), np.flatnonzero(finite), values)
        power = abs(np.fft.rfft((filled - filled.mean()) * np.hanning(len(x)))) ** 2
        freq = np.fft.rfftfreq(len(x), 1 / fs)
        total = power.sum()
        if total > 0:
            result["baseline_power_fraction"] = float(power[freq < .5].sum() / total)
            result["high_frequency_power_fraction"] = float(power[freq > .4 * fs].sum() / total)
    return result
