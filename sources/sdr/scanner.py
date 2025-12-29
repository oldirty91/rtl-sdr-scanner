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
        return (
            int(filtered_frequencies[i]),
            float(filtered_powers[i]),
            noise_level < float(filtered_powers[i]),
        )

    return (0, -100.0, 0, False)


def __scan(device, **kwargs):
    logger = logging.getLogger("sdr")
    print_best_frequencies = kwargs["print_best_frequencies"]
    filter_best_frequencies = kwargs["filter_best_frequencies"]
    bandwidth = kwargs["bandwidth"]
    disable_recording = kwargs["disable_recording"]

    recording = False
    # np.int / np.float are deprecated; use explicit dtypes
    best_frequencies = np.zeros(shape=0, dtype=np.int64)
    best_powers = np.zeros(shape=0, dtype=np.float64)

    for _range in kwargs["frequencies_ranges"]:
        start = _range["start"]
        stop = _range["stop"]
        for substart in range(start, stop, bandwidth):
            frequencies, powers = __get_frequency_power(
                device, substart, substart + bandwidth, **kwargs
            )
            filtered_frequencies, filtered_powers = __filter_frequencies(
                frequencies, powers, **kwargs
            )
            (frequency, power, _recording) = __detect_best_signal(
                frequencies,
                powers,
                filtered_frequencies,
                filtered_powers,
                **kwargs,
            )

            recording = recording or _recording
            best_frequencies = np.concatenate(
                (best_frequencies, filtered_frequencies)
            )
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
                    int(best_frequencies[i]), float(best_powers[i])
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
    """
    Main scan loop.

    Device selection logic:

      - Read SCANNER_DEVICE_SERIAL and SCANNER_DEVICE_INDEX from env.
      - If serial is set:
          * Ask pyrtlsdr for all device serials
          * Find the matching index
          * Open device by that index
      - Else:
          * Use SCANNER_DEVICE_INDEX (or 0)

    We still pass both device_index and device_serial into kwargs so
    recorder.py can use them for rtl_fm.
    """
    logger = logging.getLogger("sdr")

    sdr.tools.print_ignored_frequencies(kwargs["ignored_frequencies_ranges"])
    sdr.tools.print_frequencies_ranges(kwargs["frequencies_ranges"])
    sdr.tools.separator("scanning started")
    kwargs["frequencies_ranges"] = __filter_ranges(**kwargs)

    # Count devices
    try:
        dev_count = rtlsdr.RtlSdr.get_device_count()
    except Exception as e:
        dev_count = -1
        logger.warning("Could not get RTL-SDR device count: %s", e)

    # Env: preferred serial + index
    dev_serial_env = os.getenv("SCANNER_DEVICE_SERIAL")
    dev_index_env = os.getenv("SCANNER_DEVICE_INDEX")

    # Base index
    try:
        dev_index = int(dev_index_env) if dev_index_env is not None else 0
    except ValueError:
        dev_index = 0

    resolved_from_serial = False

    # If we got a serial from the outer tool, try to resolve it to an index
    if dev_serial_env:
        serial_str = str(dev_serial_env).strip()
        try:
            serial_list = rtlsdr.RtlSdr.get_device_serial_addresses()
        except Exception as e:
            serial_list = None
            logger.warning(
                "Could not get device serial list from pyrtlsdr: %s", e
            )

        if serial_list:
            # serial_list is typically a list of strings
            if serial_str in serial_list:
                dev_index = serial_list.index(serial_str)
                resolved_from_serial = True
                logger.info(
                    "Resolved SCANNER_DEVICE_SERIAL=%r to device_index=%d "
                    "via pyrtlsdr serial list %r",
                    serial_str,
                    dev_index,
                    serial_list,
                )
            else:
                logger.warning(
                    "SCANNER_DEVICE_SERIAL=%r not found in pyrtlsdr serial list %r; "
                    "falling back to device_index=%d",
                    serial_str,
                    serial_list,
                    dev_index,
                )
        else:
            logger.warning(
                "No serial list available from pyrtlsdr; using "
                "device_index=%d for SCANNER_DEVICE_SERIAL=%r",
                dev_index,
                serial_str,
            )

    # Pass through to recorder via kwargs
    kwargs["device_index"] = dev_index
    if dev_serial_env:
        kwargs["device_serial"] = dev_serial_env

    logger.info(
        "RTL-SDR device_count=%s, final device_index=%d "
        "(SCANNER_DEVICE_INDEX env=%r, resolved_from_serial=%s, "
        "SCANNER_DEVICE_SERIAL=%r)",
        dev_count,
        dev_index,
        dev_index_env,
        resolved_from_serial,
        dev_serial_env,
    )

    try:
        # Always open by index; we've already resolved serial → index if needed
        logger.info("Opening RTL-SDR by index %d", dev_index)
        device = rtlsdr.RtlSdr(dev_index)

        # Try to log usb strings / actual index if available
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
            "Opened RTL-SDR: requested_index=%d, real_index=%r, usb_strings=%r",
            dev_index,
            real_index,
            usb_info,
        )

        device.ppm_error = kwargs["ppm_error"]
        device.gain = kwargs["tuner_gain"]
        device.sample_rate = kwargs["bandwidth"]

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
