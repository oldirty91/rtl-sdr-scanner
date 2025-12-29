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

    device.center_freq = (start + stop) // 2
    [powers, frequencies] = matplotlib.mlab.psd(
        device.read_samples(samples),
        NFFT=fft,
        Fs=device.sample_rate,
    )
    return frequencies + device.center_freq, np.log10(powers)


def __is_frequency_ok(frequency, **kwargs):
    ignored_frequencies_ranges = kwargs["ignored_frequencies_ranges"]
    return not any(
        _range["start"] <= frequency <= _range["stop"]
        for _range in ignored_frequencies_ranges
    )


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
    try:
        noise_level = float(kwargs["noise_level"])
    except ValueError:
        i = np.argmax(filtered_powers)
        if abs(filtered_frequencies[i] - frequencies[len(frequencies) // 2]) <= 1000:
            noise_level = filtered_powers[i]
        else:
            noise_level = -100

    for i in range(len(filtered_frequencies)):
        # (freq, power, is_above_noise)
        return (int(filtered_frequencies[i]), float(filtered_powers[i]), noise_level < float(filtered_powers[i]))

    # Fallback (no signals)
    return (0, -100.0, False)


def __scan(device, **kwargs):
    logger = logging.getLogger("sdr")
    print_best_frequencies = kwargs["print_best_frequencies"]
    filter_best_frequencies = kwargs["filter_best_frequencies"]
    bandwidth = kwargs["bandwidth"]
    disable_recording = kwargs["disable_recording"]
    ignored_frequencies_ranges = kwargs["ignored_frequencies_ranges"]

    recording = False
    best_frequencies = np.zeros(shape=0, dtype=np.int)
    best_powers = np.zeros(shape=0, dtype=np.float)

    for _range in kwargs["frequencies_ranges"]:
        start = _range["start"]
        stop = _range["stop"]
        for substart in range(start, stop, bandwidth):
            frequencies, powers = __get_frequency_power(
                device,
                substart,
                substart + bandwidth,
                **kwargs,
            )
            filtered_frequencies, filtered_powers = __filter_frequencies(
                frequencies,
                powers,
                **kwargs,
            )
            (frequency, power, _recording) = __detect_best_signal(
                frequencies,
                powers,
                filtered_frequencies,
                filtered_powers,
                **kwargs,
            )

            recording = recording or _recording
            best_frequencies = np.concatenate((best_frequencies, filtered_frequencies))
            best_powers = np.concatenate((best_powers, filtered_powers))

            if _recording and not disable_recording:
                sdr.recorder.record(device, frequency, power, _range, **kwargs)

    if recording or not filter_best_frequencies:
        indexes = np.argsort(best_powers)[::-1][:print_best_frequencies]
        best_frequencies = best_frequencies[indexes]
        best_powers = best_powers[indexes]
        indexes = np.argsort(best_frequencies)
        best_frequencies = best_frequencies[indexes]
        best_powers = best_powers[indexes]
        for i in range(len(best_frequencies)):
            logger.debug(
                sdr.tools.format_frequency_power(
                    int(best_frequencies[i]),
                    float(best_powers[i]),
                )
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
                "frequency range: %s error! range not fit to bandwidth: %s! adjusting range end to %s!",
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

    # ------------------------------------------------------------------
    # Device selection: prefer SCANNER_DEVICE_SERIAL, fallback to index
    # ------------------------------------------------------------------
    dev_serial_env = os.getenv("SCANNER_DEVICE_SERIAL")
    dev_index_env = os.getenv("SCANNER_DEVICE_INDEX")

    try:
        dev_index = int(dev_index_env) if dev_index_env is not None else 1
    except ValueError:
        dev_index = 1

    # Log device count and serial addresses for debugging
    try:
        dev_count = rtlsdr.RtlSdr.get_device_count()
    except Exception as e:
        dev_count = -1
        logger.warning("Could not get RTL-SDR device count: %s", e)

    serial_addrs = []
    try:
        serial_addrs = rtlsdr.RtlSdr.get_device_serial_addresses()
    except Exception as e:
        logger.warning("Could not get RTL-SDR serial addresses: %s", e)
        serial_addrs = []

    # Normalize serial addresses into strings for logging
    serial_strs = []
    for s in serial_addrs:
        if isinstance(s, bytes):
            serial_strs.append(s.decode("utf-8", "ignore"))
        else:
            serial_strs.append(str(s))

    logger.info(
        "RTL-SDR device_count=%s, serial_addresses=%r, "
        "SCANNER_DEVICE_SERIAL=%r, SCANNER_DEVICE_INDEX=%r (parsed index=%d)",
        dev_count,
        serial_strs,
        dev_serial_env,
        dev_index_env,
        dev_index,
    )

    # Expose both to recorder via kwargs
    kwargs["device_index"] = dev_index
    if dev_serial_env:
        kwargs["device_serial"] = dev_serial_env

    # ------------------------------------------------------------------
    # Open the device
    # ------------------------------------------------------------------
    device = None
    opened_by = None

    try:
        if dev_serial_env:
            # HARD preference: open by serial, exactly like the docs example:
            #   RtlSdr(serial_number='00000001')
            logger.info(
                "Opening RTL-SDR via pyrtlsdr using serial_number=%r "
                "(SCANNER_DEVICE_SERIAL)",
                dev_serial_env,
            )
            device = rtlsdr.RtlSdr(serial_number=dev_serial_env)
            opened_by = f"serial:{dev_serial_env}"
        else:
            logger.info(
                "Opening RTL-SDR via pyrtlsdr using index=%d "
                "(SCANNER_DEVICE_INDEX=%r, default 0 if unset/invalid)",
                dev_index,
                dev_index_env,
            )
            device = rtlsdr.RtlSdr(dev_index)
            opened_by = f"index:{dev_index}"

        # Try to log actual USB info + internal index
        usb_info = None
        try:
            usb_info = device.get_usb_strings()
        except Exception:
            usb_info = None

        try:
            real_index = getattr(device, "device_index", None)
        except Exception:
            real_index = None

        logger.info(
            "RTL-SDR opened (opened_by=%s): device_index=%r, usb_strings=%r",
            opened_by,
            real_index,
            usb_info,
        )

        # Normal device config
        device.ppm_error = kwargs["ppm_error"]
        device.gain = kwargs["tuner_gain"]
        device.sample_rate = kwargs["bandwidth"]

        killer = application_killer.ApplicationKiller()
        while killer.is_running:
            __scan(device, **kwargs)

    except rtlsdr.rtlsdr.LibUSBError as e:
        logger.critical(
            "Device error (opened_by=%s, requested_index=%d, SCANNER_DEVICE_SERIAL=%r), message: %s quitting!",
            opened_by,
            dev_index,
            dev_serial_env,
            str(e),
        )
        exit(1)
    except Exception as e:
        logger.critical(
            "Unexpected error while opening/using RTL-SDR (opened_by=%s): %s",
            opened_by,
            e,
        )
        raise
