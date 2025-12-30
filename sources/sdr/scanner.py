#!/usr/bin/python3

import application_killer
import datetime
import logging
import math
import matplotlib.mlab
import numpy as np
import os
import rtlsdr
import sdr.recorder
import sdr.tools


def __get_frequency_power(device, start, stop, **kwargs):
    samples = kwargs["samples"]
    fft = kwargs["fft"]

    logger = logging.getLogger("sdr")
    device.center_freq = (start + stop) // 2

    logger.debug(
        "PSD window: center=%s, start=%s, stop=%s, samples=%s, fft=%s, sample_rate=%s",
        sdr.tools.format_frequency(device.center_freq),
        sdr.tools.format_frequency(start),
        sdr.tools.format_frequency(stop),
        samples,
        fft,
        device.sample_rate,
    )

    # NOTE: this is the original behavior: log10 of PSD
    powers, frequencies = matplotlib.mlab.psd(
        device.read_samples(samples),
        NFFT=fft,
        Fs=device.sample_rate
    )
    return frequencies + device.center_freq, np.log10(powers)


def __is_frequency_ok(frequency, **kwargs):
    ignored_frequencies_ranges = kwargs["ignored_frequencies_ranges"]
    return not any(_range["start"] <= frequency <= _range["stop"]
                   for _range in ignored_frequencies_ranges)


def __filter_frequencies(frequencies, powers, **kwargs):
    print_best_frequencies = max(1, kwargs["print_best_frequencies"])
    sorted_frequencies_indexes = np.argsort(powers)[::-1]

    indexes = []
    total = 0
    for i in sorted_frequencies_indexes:
        if __is_frequency_ok(int(frequencies[i]), **kwargs):
            indexes.append(i)
            total += 1
        if print_best_frequencies <= total:
            break
    return frequencies[indexes], powers[indexes]


def __detect_best_signal(frequencies, powers, filtered_frequencies, filtered_powers, **kwargs):
    """
    Decide which frequency is "best" and whether it exceeds the noise threshold.

    Returns: (best_freq_hz: int, best_power: float, recording: bool)
    """
    logger = logging.getLogger("sdr")

    # Config noise level can be "auto" or a numeric string
    try:
        noise_level = float(kwargs["noise_level"])
        noise_mode = "config"
    except ValueError:
        # Auto mode: use the peak near center as noise
        noise_mode = "auto"
        if len(filtered_powers) == 0:
            noise_level = -100.0
        else:
            i = int(np.argmax(filtered_powers))
            if abs(filtered_frequencies[i] - frequencies[len(frequencies) // 2]) <= 1000:
                noise_level = float(filtered_powers[i])
            else:
                noise_level = -100.0

    if len(filtered_frequencies) == 0:
        # Nothing usable in this window
        logger.debug(
            "detect_best_signal: no filtered freqs; noise_mode=%s noise_level=%.2f",
            noise_mode,
            noise_level,
        )
        return 0, -100.0, False

    # Take the strongest filtered tone
    i = int(np.argmax(filtered_powers))
    best_freq = int(filtered_frequencies[i])
    best_power = float(filtered_powers[i])
    recording = noise_level < best_power

    logger.debug(
        "detect_best_signal: best_freq=%s, best_power=%.2f, noise_mode=%s, "
        "noise_level=%.2f, recording=%s",
        sdr.tools.format_frequency(best_freq),
        best_power,
        noise_mode,
        noise_level,
        recording,
    )

    return best_freq, best_power, recording


def __scan(device, **kwargs):
    logger = logging.getLogger("sdr")
    print_best_frequencies = kwargs["print_best_frequencies"]
    filter_best_frequencies = kwargs["filter_best_frequencies"]
    bandwidth = kwargs["bandwidth"]
    disable_recording = kwargs["disable_recording"]

    recording_any = False
    best_frequencies = np.zeros(shape=0, dtype=np.int64)
    best_powers = np.zeros(shape=0, dtype=np.float64)

    for _range in kwargs["frequencies_ranges"]:
        start = _range["start"]
        stop = _range["stop"]

        logger.debug(
            "scan range: %s -> %s (bandwidth=%s)",
            sdr.tools.format_frequency(start),
            sdr.tools.format_frequency(stop),
            sdr.tools.format_frequency(bandwidth),
        )

        for substart in range(start, stop, bandwidth):
            substop = substart + bandwidth

            freqs, powers = __get_frequency_power(
                device,
                substart,
                substop,
                **kwargs
            )

            # Raw stats for debug
            if len(powers) > 0:
                max_power = float(np.max(powers))
                min_power = float(np.min(powers))
            else:
                max_power = -999.0
                min_power = -999.0

            logger.debug(
                "PSD window stats: span=%s..%s, min_power=%.2f, max_power=%.2f",
                sdr.tools.format_frequency(int(freqs[0]) if len(freqs) else substart),
                sdr.tools.format_frequency(int(freqs[-1]) if len(freqs) else substop),
                min_power,
                max_power,
            )

            filtered_freqs, filtered_powers = __filter_frequencies(freqs, powers, **kwargs)
            frequency, power, recording = __detect_best_signal(
                freqs,
                powers,
                filtered_freqs,
                filtered_powers,
                **kwargs,
            )

            recording_any = recording_any or recording

            if len(filtered_freqs) > 0:
                best_frequencies = np.concatenate((best_frequencies, filtered_freqs.astype(np.int64)))
                best_powers = np.concatenate((best_powers, filtered_powers.astype(np.float64)))

            if recording and not disable_recording:
                logger.info(
                    "trigger record: freq=%s power=%.2f (recording=True)",
                    sdr.tools.format_frequency(frequency),
                    power,
                )
                sdr.recorder.record(device, frequency, power, _range, **kwargs)
            elif recording:
                logger.info(
                    "recording condition met at %s, but recording disabled",
                    sdr.tools.format_frequency(frequency),
                )

    # Print "best" frequencies summary
    if recording_any or not filter_best_frequencies:
        if len(best_powers) == 0:
            return

        idx = np.argsort(best_powers)[::-1][:print_best_frequencies]
        best_frequencies_sel = best_frequencies[idx]
        best_powers_sel = best_powers[idx]
        idx2 = np.argsort(best_frequencies_sel)
        best_frequencies_sel = best_frequencies_sel[idx2]
        best_powers_sel = best_powers_sel[idx2]

        for i in range(len(best_frequencies_sel)):
            logger.debug(
                "summary: %s power=%.2f",
                sdr.tools.format_frequency(int(best_frequencies_sel[i])),
                float(best_powers_sel[i]),
            )
        if 1 < print_best_frequencies:
            logger.debug("-" * 80)


def __filter_ranges(**kwargs):
    ranges = []
    logger = logging.getLogger("sdr")
    bandwidth = kwargs["bandwidth"]
    for _range in kwargs["frequencies_ranges"]:
        start = _range["start"]
        stop = _range["stop"]
        if (stop - start) % bandwidth != 0:
            _range["stop"] = start + (bandwidth * math.ceil((stop - start) / bandwidth))
            logger.warning(
                "frequency range: %s error! range not fit to bandwidth: %s! "
                "adjusting range end to %s!",
                sdr.tools.format_frequency_range(start, stop),
                sdr.tools.format_frequency(bandwidth),
                sdr.tools.format_frequency(_range["stop"]),
            )
        ranges.append(_range)
    if ranges:
        return ranges
    else:
        logger.error("empty frequency ranges! quitting!")
        exit(1)


def run(**kwargs):
    logger = logging.getLogger("sdr")

    sdr.tools.print_ignored_frequencies(kwargs["ignored_frequencies_ranges"])
    sdr.tools.print_frequencies_ranges(kwargs["frequencies_ranges"])
    sdr.tools.separator("scanning started")
    kwargs["frequencies_ranges"] = __filter_ranges(**kwargs)

    # --- Device selection (serial first, then index) ---

    # Count devices
    try:
        dev_count = rtlsdr.RtlSdr.get_device_count()
    except Exception as e:
        dev_count = -1
        logger.warning("Could not get RTL-SDR device count: %s", e)

    # Optional: device serial from env (set by scanner app via hub)
    dev_serial_env = os.getenv("SCANNER_DEVICE_SERIAL")

    # Fallback: device index from env
    dev_index_env = os.getenv("SCANNER_DEVICE_INDEX")
    try:
        dev_index = int(dev_index_env) if dev_index_env is not None else 0
    except ValueError:
        dev_index = 0

    # Expose to recorder
    kwargs["device_index"] = dev_index
    if dev_serial_env:
        kwargs["device_serial"] = dev_serial_env

    logger.info(
        "RTL-SDR devices=%s, requested_index=%d (SCANNER_DEVICE_INDEX=%r), "
        "requested_serial=%r (SCANNER_DEVICE_SERIAL)",
        dev_count,
        dev_index,
        dev_index_env,
        dev_serial_env,
    )

    try:
        # Prefer opening by serial if given
        if dev_serial_env:
            logger.info(
                "Opening RTL-SDR by serial %r (SCANNER_DEVICE_SERIAL)",
                dev_serial_env,
            )
            device = rtlsdr.RtlSdr(serial_number=dev_serial_env)
        else:
            logger.info(
                "Opening RTL-SDR by index %d (SCANNER_DEVICE_INDEX=%r)",
                dev_index,
                dev_index_env,
            )
            device = rtlsdr.RtlSdr(dev_index)

        # Try to log usb strings / actual index if available
        try:
            usb_info = device.get_usb_strings()
        except Exception:
            usb_info = None

        try:
            real_index = getattr(device, "device_index", None)
        except Exception:
            real_index = None

        logger.info(
            "Opened RTL-SDR: requested_index=%d, real_index=%r, usb_strings=%r",
            dev_index,
            real_index,
            usb_info,
        )

        device.ppm_error = kwargs["ppm_error"]
        device.gain = kwargs["tuner_gain"]
        device.sample_rate = kwargs["bandwidth"]

        logger.info(
            "Device params: ppm_error=%s, tuner_gain=%s, sample_rate=%s",
            device.ppm_error,
            device.gain,
            device.sample_rate,
        )

        killer = application_killer.ApplicationKiller()
        while killer.is_running:
            __scan(device, **kwargs)

    except rtlsdr.rtlsdr.LibUSBError as e:
        logger.critical(
            "Device error (index=%d, serial=%r), message: %s quitting!",
            dev_index,
            dev_serial_env,
            str(e),
        )
        exit(1)
