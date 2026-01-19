"""
Data processing classes for aligning and transforming well data.

This module provides robust data processing with:
- Comprehensive validation
- Progress tracking
"""
import logging
import datetime
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import (
    ATMOSPHERE_PRESSURE_KPA,
    THRESHOLD_PRESSURE,
    FILTER_LENGTH,
    GOOD_SAMPLE_LENGTH,
    PADDING,
)
from utils import (
    get_peak_indexes,
    calculate_cross_correlation_shift,
    make_datetime_unique,
)

from exceptions import (
    DataValidationError,
    EmptyDataError,
    DataAlignmentError,
    TimingError
)

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    threshold_pressure: float = THRESHOLD_PRESSURE
    filter_length: int = FILTER_LENGTH
    good_sample_length: int = GOOD_SAMPLE_LENGTH
    padding: int = PADDING
    skip_fr_jobs: bool = True
    

@dataclass
class DataQualityMetrics:
    """Metrics for tracking data quality."""
    total_records: int = 0
    valid_records: int = 0
    duplicates_removed: int = 0
    timing_issues_fixed: int = 0
    missing_values_filled: int = 0
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records

class DataProcessor:
    """
    Handles processing and alignment of GuideHawk and Reliance data.
    
    This class provides methods for:
    - Data preparation and cleaning
    - Timing issue correction
    - Cross-correlation alignment
    - Peak detection and trimming
    - Quality validation
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.metrics = DataQualityMetrics()
        self.guidehawk_top_df: Optional[pd.DataFrame] = None
        self.guidehawk_bottom_df: Optional[pd.DataFrame] = None
        self.reliance_df: Optional[pd.DataFrame] = None
    
    def prepare_guidehawk_data(
        self,
        guidehawk_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """ Prepare GuideHawk data by splitting into top and bottom dataframes."""
        try:
            # Validate input
            if not guidehawk_data:
                logger.error("GuideHawk data is empty")
                return None, None
            
            if not isinstance(guidehawk_data, list):
                raise DataValidationError(f"Expected list, got {type(guidehawk_data).__name__}")
            
            logger.info(f"Processing {len(guidehawk_data)} GuideHawk records")
            
            # Normalize JSON to DataFrame
            df = pd.json_normalize(guidehawk_data)
            self.metrics.total_records = len(df)
            
            # Validate required columns
            required_cols = ['kdatetime', 'jobid', 'bhaloc', 'annp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")
            
            # Create datetime column
            df["Date/Time"] = pd.to_datetime(df["kdatetime"], errors='coerce')
            
            # Check for datetime conversion issues
            null_dates = df["Date/Time"].isnull().sum()
            if null_dates > 0:
                logger.warning(f"Failed to parse {null_dates} datetime values")
            
            df = df.reset_index(drop=True).sort_values(by="Date/Time")
            
            # Check job name
            job_name_array = df["jobid"].str.lower()
            if len(job_name_array) == 0:
                logger.warning("Empty job name array")
                return None, None
            
            job_name = job_name_array.iloc[0]
            logger.info(f"Processing job: {job_name}")
            
            # Skip FR jobs if configured
            if self.config.skip_fr_jobs and "fr" in job_name:
                logger.info(f"Skipping FR job: {job_name}")
                return None, None
            
            # Split by location
            df["bhaloc"] = df["bhaloc"].str.lower()
            top_df = df[df["bhaloc"] == "a"].copy()
            bottom_df = df[df["bhaloc"] == "b"].copy()
            
            if top_df.empty:
                logger.warning("No top location (bhaloc='a') data found")
                return None, None
            
            # Process top dataframe
            top_df = self._clean_guidehawk_dataframe(top_df, "top")
            
            # Process bottom dataframe if exists
            if not bottom_df.empty:
                bottom_df = self._clean_guidehawk_dataframe(bottom_df, "bottom")
            else:
                logger.info("No bottom GuideHawk data found")
            
            self.guidehawk_top_df = top_df
            self.guidehawk_bottom_df = bottom_df
            
            logger.info(
                f"GuideHawk data prepared: top={len(top_df)} rows, "
                f"bottom={len(bottom_df) if not bottom_df.empty else 0} rows"
            )
            
            return top_df, bottom_df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error preparing GuideHawk data: {str(e)}")
            return None, None
    
    def _clean_guidehawk_dataframe(
        self,
        df: pd.DataFrame,
        location: str
    ) -> pd.DataFrame:
        """ Clean and deduplicate GuideHawk dataframe. """
        initial_count = len(df)
        
        # Remove duplicates based on datetime
        df = df.drop_duplicates(subset=["Date/Time"])
        duplicates = initial_count - len(df)
        
        if duplicates > 0:
            logger.info(f"Removed {duplicates} duplicate records from {location}")
            self.metrics.duplicates_removed += duplicates
        
        # Make datetime unique
        df = make_datetime_unique(df)
        
        # Validate pressure data
        if 'annp' in df.columns:
            null_pressure = df['annp'].isnull().sum()
            if null_pressure > 0:
                logger.warning(f"{location}: {null_pressure} null pressure values")
                df['annp'].fillna(0, inplace=True)
                self.metrics.missing_values_filled += null_pressure
        
        self.metrics.valid_records += len(df)
        
        return df
    
    def prepare_reliance_data(
        self,
        reliance_data: Dict[str, Any],
        columns: List[str]
    ) -> Optional[pd.DataFrame]:
        """ Prepare Reliance data and fix timing issues. """
        try:
            # Validate input structure
            if not isinstance(reliance_data, dict):
                raise DataValidationError(f"Expected dict, got {type(reliance_data).__name__}")
            
            if 'result' not in reliance_data:
                raise DataValidationError("Missing 'result' key in Reliance data")
            
            if not reliance_data['result'] or not reliance_data['result'][0]:
                raise DataValidationError("Empty result in Reliance data")
            
            if 'records' not in reliance_data['result'][0]:
                raise DataValidationError("Missing 'records' in Reliance result")
            
            logger.info("Extracting Reliance records")
            
            # Extract and normalize records
            job_df = pd.json_normalize(reliance_data["result"][0]["records"])
            
            if 'values' not in job_df.columns:
                raise DataValidationError("Missing 'values' column in records")
            
            df = pd.DataFrame(job_df["values"].tolist(), columns=columns)
            logger.info(f"Extracted {len(df)} Reliance records")
            
            # Validate required columns
            required_cols = ['Date', 'Time', 'Selected_Pressure (KPAg)']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")
            
            # Create datetime column
            df["Date/Time"] = pd.to_datetime(
                df["Date"] + " " + df["Time"],
                dayfirst=True,
                errors='coerce'
            )
            
            # Check for datetime parsing issues
            null_dates = df["Date/Time"].isnull().sum()
            if null_dates > 0:
                logger.warning(f"Failed to parse {null_dates} Reliance datetime values")
                df = df[df["Date/Time"].notnull()].copy()
            
            # Process pressure data
            df["Selected_Pressure (KPAg)"] = pd.to_numeric(
                df["Selected_Pressure (KPAg)"],
                errors="coerce"
            )
            
            # Skip first 5 rows (typically header/calibration)
            if len(df) > 5:
                df = df.iloc[5:].copy()
            
            df["Selected_Pressure (KPAg)"].fillna(0.0, inplace=True)
            df["Pressure"] = df["Selected_Pressure (KPAg)"] / 1000.0
            
            # Fill other numeric NaN values
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns
            null_counts = df[numeric_cols].isnull().sum()
            df[numeric_cols] = df[numeric_cols].fillna(0)
            self.metrics.missing_values_filled += null_counts.sum()
            
            # Fix timing issues
            df = self._fix_reliance_timing(df)
            
            self.reliance_df = df
            logger.info(f"Reliance data prepared: {len(df)} rows")
            
            return df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error preparing Reliance data: {str(e)}")
            return None
    
    def _fix_reliance_timing(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Fix timing irregularities in Reliance data. """
        logger.info("Fixing Reliance timing irregularities")
        initial_length = len(df)
        
        # Calculate time deltas in seconds
        df['deltaT'] = (
            (df["Date/Time"] - df["Date/Time"].shift(1)) / pd.Timedelta(seconds=1)
        ).fillna(1).astype('Int64')
        
        # Fix first sample
        df.loc[df.index[0], 'deltaT'] = 1
        
        # Identify good sample regions
        df['dtMax'] = df['deltaT'].rolling(
            self.config.good_sample_length,
            step=1,
            center=True,
            min_periods=1
        ).max()
        
        df['dtMin'] = df['deltaT'].rolling(
            self.config.good_sample_length,
            step=1,
            center=True,
            min_periods=1
        ).min()
        
        # Mark good bits (regions with consistent 1-second sampling)
        df['goodBit'] = np.where(
            (df['dtMax'] == 1) & (df['dtMin'] == 1),
            1,
            np.nan
        )
        
        # Expand good bit regions
        for _ in range(int(self.config.good_sample_length / 2)):
            df['goodBit'] = np.where(
                (df['goodBit'] == 1) | 
                ((df['deltaT'] == 1) & (df['goodBit'].shift(1) == 1)),
                1,
                np.nan
            )
            df['goodBit'] = np.where(
                (df['goodBit'] == 1) | 
                ((df['deltaT'] == 1) & (df['goodBit'].shift(-1) == 1)),
                1,
                np.nan
            )
        
        # Fix bad sections
        fixes_applied = 0
        zeros = 0
        last_bit = 1
        start = 0
        end = 0
        
        for i in range(1, len(df)):
            if pd.isna(df['goodBit'].iloc[i]) or pd.isna(df['deltaT'].iloc[i]):
                continue
            
            if (df['goodBit'].iloc[i] != 1) and (i > 1):
                if last_bit == 1:
                    start = i
                    end = i
                else:
                    end = i
                
                if df['deltaT'].iloc[i] == 0:
                    zeros += 1
            else:
                # Apply timing fix to bad section
                if zeros > 0 and start > 1:
                    times_to_fix = end - start + 1
                    for j in range(times_to_fix):
                        df.iloc[start + j, df.columns.get_loc("Date/Time")] = (
                            df["Date/Time"].iloc[start - 1] + 
                            datetime.timedelta(seconds=1 + j)
                        )
                    fixes_applied += 1
                    logger.debug(f"Fixed timing section: rows {start}-{end}")
                
                zeros = 0
            
            last_bit = df['goodBit'].iloc[i]
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} timing fixes")
            self.metrics.timing_issues_fixed = fixes_applied
        
        # Resample to regular 1-second grid
        df = df.sort_values(by="Date/Time").reset_index(drop=True)
        
        time_range = pd.date_range(
            start=df["Date/Time"].min(),
            end=df["Date/Time"].max(),
            freq="s"
        )
        
        logger.info(f"Resampling to regular grid: {len(time_range)} timestamps")
        
        # Remove datetime duplicates before reindexing
        df = df.drop_duplicates(subset="Date/Time", keep='first')
        
        # Reindex to regular time grid
        df = (df.set_index("Date/Time")
               .reindex(time_range)
               .reset_index())
        
        # Restore Date/Time column name
        df.rename(columns={'index': 'Date/Time'}, inplace=True)
        
        # Interpolate missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # Drop temporary columns
        df.drop(columns=['deltaT', 'dtMax', 'dtMin', 'goodBit'], inplace=True, errors='ignore')
        
        logger.info(f"Timing correction complete: {initial_length} -> {len(df)} rows")
        
        return df
    
    def align_guidehawk_datasets(
        self,
        top_df: pd.DataFrame,
        bottom_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Align top and bottom GuideHawk datasets using cross-correlation. """
        if bottom_df.empty:
            logger.info("No bottom dataset to align")
            return top_df, bottom_df
        
        try:
            logger.info("Aligning GuideHawk datasets using cross-correlation")
            
            # Validate pressure data exists
            if 'annp' not in top_df.columns or 'annp' not in bottom_df.columns:
                raise DataValidationError("Missing pressure column 'annp' for alignment")
            
            files = [
                top_df["annp"].to_numpy(),
                bottom_df["annp"].to_numpy()
            ]
            
            # Find pressure test boundaries
            start_of_ptest, end_of_ptest = self._find_pressure_test_boundaries(files)
            
            # Extract and pad pressure test regions
            files = self._extract_pressure_tests(files, start_of_ptest, end_of_ptest)
            
            # Calculate cross-correlation shift
            xc_shift = calculate_cross_correlation_shift(
                files[0],
                files[1],
                self.config.filter_length
            )
            
            logger.info(f"Cross-correlation shift: {xc_shift} samples")
            
            # Apply alignment
            top_df, bottom_df = self._apply_alignment_shift(
                [top_df, bottom_df],
                xc_shift
            )
            
            # Synchronize timestamps
            bottom_df["Date/Time"] = top_df["Date/Time"].iloc[:len(bottom_df)]
            
            logger.info("GuideHawk datasets aligned successfully")
            return top_df, bottom_df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error aligning GuideHawk datasets: {str(e)}")
            return top_df, bottom_df
    
    def _find_pressure_test_boundaries(
        self,
        files: List[np.ndarray]
    ) -> Tuple[List[int], List[int]]:
        """Find start and end of pressure tests in datasets."""
        start_of_ptest = [0, 0]
        end_of_ptest = [0, 0]
        
        for i in range(2):
            for j in range(len(files[i])):
                if files[i][j] > self.config.threshold_pressure:
                    if start_of_ptest[i] == 0:
                        start_of_ptest[i] = j
                else:
                    if start_of_ptest[i] > 0 and end_of_ptest[i] == 0:
                        end_of_ptest[i] = j
                        break
            
            logger.debug(
                f"Dataset {i}: pressure test from {start_of_ptest[i]} "
                f"to {end_of_ptest[i]}"
            )
        
        return start_of_ptest, end_of_ptest
    
    def _extract_pressure_tests(
        self,
        files: List[np.ndarray],
        start_of_ptest: List[int],
        end_of_ptest: List[int]
    ) -> List[np.ndarray]:
        """Extract and pad pressure test regions."""
        len_of_ptest = [
            end_of_ptest[0] - start_of_ptest[0],
            end_of_ptest[1] - start_of_ptest[1]
        ]
        
        max_length = max(len_of_ptest)
        logger.debug(f"Maximum pressure test length: {max_length}")
        
        extracted = []
        for i in range(2):
            start = max(0, start_of_ptest[i] - self.config.padding)
            end = start + max_length + 2 * self.config.padding
            
            # Ensure even length
            if (end - start) % 2 == 1:
                end += 1
            
            # Extract region
            extracted_data = files[i][start:min(end, len(files[i]))]
            extracted.append(extracted_data)
            
            logger.debug(f"Extracted {len(extracted_data)} samples from dataset {i}")
        
        return extracted
    
    def _apply_alignment_shift(
        self,
        dfs: List[pd.DataFrame],
        xc_shift: int
    ) -> List[pd.DataFrame]:
        """Apply shift to align dataframes."""
        start_delta = [0, 0]
        
        if xc_shift > 0:
            start_delta[1] = xc_shift
        else:
            start_delta[0] = -xc_shift
        
        aligned = []
        for i in range(2):
            df = dfs[i].copy()
            
            if start_delta[i] > 0:
                # Drop rows from beginning
                drop_list = list(range(min(start_delta[i], len(df))))
                df = df.drop(df.index[drop_list])
                df = df.reset_index(drop=True)
                logger.debug(f"Dropped {len(drop_list)} rows from dataset {i}")
            
            aligned.append(df)
        
        return aligned
    
    def align_guidehawk_with_reliance(
        self,
        guidehawk_top_df: pd.DataFrame,
        guidehawk_bottom_df: pd.DataFrame,
        reliance_df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """ Align GuideHawk data with Reliance data using cross-correlation. """
        try:
            logger.info("Aligning GuideHawk with Reliance data")
            
            # Validate required columns
            if 'annp' not in guidehawk_top_df.columns:
                raise DataValidationError("Missing 'annp' column in GuideHawk data")
            if 'Pressure' not in reliance_df.columns:
                raise DataValidationError("Missing 'Pressure' column in Reliance data")
            
            # Extract pressure arrays
            guidehawk_pressure = guidehawk_top_df["annp"].to_numpy()
            reliance_pressure = reliance_df["Pressure"].to_numpy()
            
            logger.debug(
                f"Pressure array lengths: GuideHawk={len(guidehawk_pressure)}, "
                f"Reliance={len(reliance_pressure)}"
            )
            
            # Pad to same length
            max_len = max(len(guidehawk_pressure), len(reliance_pressure))
            
            if len(guidehawk_pressure) < max_len:
                pad_length = max_len - len(guidehawk_pressure)
                guidehawk_pressure = np.pad(
                    guidehawk_pressure,
                    (0, pad_length),
                    'constant'
                )
                logger.debug(f"Padded GuideHawk with {pad_length} zeros")
            
            if len(reliance_pressure) < max_len:
                pad_length = max_len - len(reliance_pressure)
                reliance_pressure = np.pad(
                    reliance_pressure,
                    (0, pad_length),
                    'constant'
                )
                logger.debug(f"Padded Reliance with {pad_length} zeros")
            
            # Calculate shift
            xc_shift = calculate_cross_correlation_shift(
                guidehawk_pressure,
                reliance_pressure,
                self.config.filter_length
            )
            
            logger.info(f"Cross-correlation shift: {xc_shift} samples")
            
            # Apply shift
            drop_from_reliance = xc_shift >= 0
            xc_shift = abs(xc_shift)
            
            if xc_shift > 0:
                drop_list = list(range(xc_shift + 1))
                
                if drop_from_reliance:
                    reliance_df = reliance_df.drop(reliance_df.index[drop_list])
                    reliance_df = reliance_df.reset_index(drop=True)
                    logger.debug(f"Dropped {len(drop_list)} rows from Reliance")
                else:
                    guidehawk_top_df = guidehawk_top_df.drop(guidehawk_top_df.index[drop_list])
                    guidehawk_top_df = guidehawk_top_df.reset_index(drop=True)
                    logger.debug(f"Dropped {len(drop_list)} rows from GuideHawk top")
                    
                    if not guidehawk_bottom_df.empty:
                        guidehawk_bottom_df = guidehawk_bottom_df.drop(
                            guidehawk_bottom_df.index[drop_list]
                        )
                        guidehawk_bottom_df = guidehawk_bottom_df.reset_index(drop=True)
                        logger.debug(f"Dropped {len(drop_list)} rows from GuideHawk bottom")
            
            logger.info("GuideHawk and Reliance data aligned successfully")
            return guidehawk_top_df, guidehawk_bottom_df, reliance_df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error aligning GuideHawk with Reliance: {str(e)}")
            return None, None, None
    
    def trim_to_peaks(
        self,
        guidehawk_top_df: pd.DataFrame,
        guidehawk_bottom_df: pd.DataFrame,
        reliance_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Trim all dataframes to the peak pressure range."""
        try:
            logger.info("Trimming data to peak pressure range")
            
            # Validate pressure data exists
            if 'annp' not in guidehawk_top_df.columns:
                raise DataValidationError("Missing 'annp' column for peak detection")
            
            # Find peak indexes
            first_peak, last_peak = get_peak_indexes(guidehawk_top_df['annp'])
            
            if first_peak < 0 or last_peak < 0:
                raise DataValidationError(
                    f"Failed to detect pressure peaks: first={first_peak}, last={last_peak}"
                )
            
            if first_peak >= last_peak:
                raise DataValidationError(
                    f"Invalid peak range: first={first_peak}, last={last_peak}"
                )
            
            logger.info(f"Trimming to peaks: first={first_peak}, last={last_peak}")
            
            # Trim dataframes
            guidehawk_top_df = guidehawk_top_df.iloc[first_peak:last_peak + 1].reset_index(drop=True)
            reliance_df = reliance_df.iloc[first_peak:last_peak + 1].reset_index(drop=True)
            
            if not guidehawk_bottom_df.empty:
                guidehawk_bottom_df = guidehawk_bottom_df.iloc[first_peak:last_peak + 1].reset_index(drop=True)
            
            logger.info(
                f"Trimmed to {len(guidehawk_top_df)} rows "
                f"({len(guidehawk_top_df.columns)} columns)"
            )
            
            return guidehawk_top_df, guidehawk_bottom_df, reliance_df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error trimming to peaks: {str(e)}")
            return guidehawk_top_df, guidehawk_bottom_df, reliance_df
    
    def get_quality_report(self) -> Dict[str, Any]:
        """ Generate a quality report for the processed data. """
        return {
            'metrics': {
                'total_records': self.metrics.total_records,
                'valid_records': self.metrics.valid_records,
                'duplicates_removed': self.metrics.duplicates_removed,
                'timing_issues_fixed': self.metrics.timing_issues_fixed,
                'missing_values_filled': self.metrics.missing_values_filled,
                'quality_score': self.metrics.quality_score
            },
            'dataframes': {
                'guidehawk_top': {
                    'rows': len(self.guidehawk_top_df) if self.guidehawk_top_df is not None else 0,
                    'columns': len(self.guidehawk_top_df.columns) if self.guidehawk_top_df is not None else 0
                },
                'guidehawk_bottom': {
                    'rows': len(self.guidehawk_bottom_df) if self.guidehawk_bottom_df is not None else 0,
                    'columns': len(self.guidehawk_bottom_df.columns) if self.guidehawk_bottom_df is not None else 0
                },
                'reliance': {
                    'rows': len(self.reliance_df) if self.reliance_df is not None else 0,
                    'columns': len(self.reliance_df.columns) if self.reliance_df is not None else 0
                }
            }
        }