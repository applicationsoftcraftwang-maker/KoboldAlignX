"""Utility functions for data parsing and processing."""
import re
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_location(location: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """ Parse location string to extract UWI, township, and range."""
    try:
        # Extract the substring from the first digit to the last digit
        match = re.search(r"\d.*\d", location)
        if not match:
            logger.warning(f"No numeric pattern found in location: {location}")
            return None, None, None
        
        substring = match.group(0)
        
        # Replace the first underscore with a slash
        modified_location = substring.replace("_", "/", 1)
        
        # Find the index of the second-to-last hyphen
        hyphen_indices = [m.start() for m in re.finditer("-", modified_location)]
        if len(hyphen_indices) < 2:
            logger.warning(f"Insufficient hyphens in location: {location}")
            return None, None, None
        
        second_last_hyphen_index = hyphen_indices[-2]
        
        # Extract the part of the string after the second-to-last hyphen
        part_after_second_last_hyphen = modified_location[second_last_hyphen_index + 1:]
        
        # Extract numbers from this part
        numbers_after_second_last_hyphen = re.findall(r"\d+", part_after_second_last_hyphen)
        
        # Ensure we have at least two numbers after the second-to-last hyphen
        if len(numbers_after_second_last_hyphen) < 2:
            logger.warning(f"Insufficient numbers after second-to-last hyphen in location: {location}")
            return None, None, None
        
        twp = numbers_after_second_last_hyphen[0]  # Township number
        rng = numbers_after_second_last_hyphen[1]  # Range number
        
        # Modify township number if it's not three digits
        twp_3 = twp.zfill(3) if len(twp) < 3 else twp
        
        # Replace only the township number in the substring
        modified_location_parts = modified_location.split('-')
        modified_location_parts[2] = twp_3
        
        # Reconstruct the location with the modified township and range
        uwi = '-'.join(modified_location_parts)
        
        return uwi, twp_3, rng
        
    except (AttributeError, IndexError, ValueError) as e:
        logger.error(f"Error parsing location '{location}': {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error parsing location '{location}': {str(e)}")
        return None, None, None


def get_peak_indexes(data: pd.Series, threshold_offset: int = 500) -> Tuple[int, int]:
    """ Find the first and last peak indexes in pressure data. """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    
    mid_index = len(data) // 2
    second_half = data.iloc[mid_index:]
    
    second_max_value = second_half.max()
    lower_threshold = second_max_value * 0.85
    
    first_peak = -1
    last_peak = -1
    is_peak = False
    
    for idx in range(len(data) - 1, -1, -1):
        if data.iloc[idx] < lower_threshold:
            if is_peak:
                is_peak = False
        else:
            if not is_peak:
                if last_peak == -1:
                    last_peak = idx
                else:
                    first_peak = idx
                    break
                is_peak = True
    
    first_peak = first_peak - threshold_offset
    return first_peak, last_peak


def apply_low_pass_filter(data: np.ndarray, filter_length: int = 35) -> np.ndarray:
    """ Apply a low-pass filter using convolution. """
    low_freq = np.convolve(data, np.ones(filter_length) / filter_length, mode="same")
    
    # Fix the start and end of the low frequency data
    half_length = int(filter_length / 2)
    for i in range(half_length):
        low_freq[i] = low_freq[half_length]
        low_freq[-(i + 1)] = low_freq[-half_length - 1]
    
    return low_freq


def calculate_cross_correlation_shift(data1: np.ndarray, data2: np.ndarray, filter_length: int = 35) -> int:
    """ Calculate the shift between two datasets using cross-correlation. """
    # Create low and high frequency versions
    low_freq_1 = apply_low_pass_filter(data1, filter_length)
    low_freq_2 = apply_low_pass_filter(data2, filter_length)
    
    high_freq_1 = data1 - low_freq_1
    high_freq_2 = data2 - low_freq_2
    
    # Ensure even length for cross-correlation
    if len(high_freq_1) % 2 == 1:
        high_freq_1 = np.append(high_freq_1, [0])
    if len(high_freq_2) % 2 == 1:
        high_freq_2 = np.append(high_freq_2, [0])
    
    # Calculate cross-correlation
    x_corr = np.correlate(high_freq_1, high_freq_2, "same")
    mid_point = len(x_corr) / 2
    
    # Normalize
    x_max = np.max(x_corr)
    if x_max != 0:
        x_corr = x_corr / x_max
    
    max_location = np.argmax(x_corr)
    xc_shift = int(mid_point - max_location)
    
    logger.info(f"Cross-correlation: max={np.amax(x_corr):.4f}, "
                f"min={np.amin(x_corr):.4f}, shift={xc_shift}")
    
    return xc_shift


def make_datetime_unique(df: pd.DataFrame, datetime_col: str = 'Date/Time') -> pd.DataFrame:
    """ Make datetime values unique by adding nanosecond offsets to duplicates. """
    df = df.assign(
        counter=df.groupby(datetime_col).cumcount()
    ).assign(
        DateTimeUnique=lambda x: x[datetime_col] + pd.to_timedelta(x['counter'], unit='ns')
    ).drop(columns=['counter'])
    
    df[datetime_col] = df["DateTimeUnique"]
    df = df.drop(columns=['DateTimeUnique'])
    
    return df


def format_tundra_uwi(uwi: str) -> str:
    """ Format UWI for Tundra system (replace first slash with dot, add .00). """
    tundra_uwi = uwi + ".00"
    tundra_uwi = tundra_uwi.replace('/', '.', 1)
    return tundra_uwi