"""
Data transformation classes for stage calculations and final output preparation.

This module handles:
- Job time calculations
- Stage status tracking (injection, falloff)
- Stage time and falloff time calculations
- Advanced metrics (energy, impulse)
- Final data preparation and export
"""
import logging
import tempfile
import shutil
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import (
    ATMOSPHERE_PRESSURE_KPA,
    GUIDEHAWK_TOP_COLUMNS_TO_DROP,
    GUIDEHAWK_BOTTOM_COLUMNS_TO_DROP,
    RELIANCE_COLUMNS_TO_DROP,
    COLUMNS_TO_ROUND
)
from utils import format_tundra_uwi

logger = logging.getLogger(__name__)


@dataclass
class StageState:
    """Track state during stage status calculation."""
    previous_injection_flag: int = 0
    previous_falloff_flag: int = 0
    previous_stage_number: int = 0
    begin_stage_flag: bool = False
    begin_falloff_stage: bool = False
    stages_processed: List[int] = None
    
    def __post_init__(self):
        if self.stages_processed is None:
            self.stages_processed = []


@dataclass
class TimeState:
    """Track state during time calculations."""
    start_flag: bool = False
    skipped_stage_start_pos: int = -1
    skipped_stage_number: int = -1


class DataTransformer:
    """ Handles final data transformations and calculations. """
    
    @staticmethod
    def calculate_job_time(df: pd.DataFrame) -> pd.DataFrame:
        df["Job Time"] = 0.0
        
        # Ensure datetime format
        df['Date/Time'] = pd.to_datetime(
            df["Date/Time"],
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        
        # Calculate cumulative time
        for j in range(1, len(df)):
            if pd.isnull(df["Date/Time"].iloc[j]) or pd.isnull(df['Date/Time'].iloc[j - 1]):
                continue
            
            time_diff_seconds = (
                df['Date/Time'].iloc[j] - df['Date/Time'].iloc[j - 1]
            ).total_seconds()
            
            df.at[j, "Job Time"] = (
                time_diff_seconds / 3600 + df["Job Time"].iloc[j - 1]
            )
        
        # Round to 4 decimal places
        df['Job Time'] = df['Job Time'].astype(float).round(4)
        
        logger.debug(f"Calculated job time: {df['Job Time'].max():.2f} hours total")
        return df
    
    @staticmethod
    def calculate_stage_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stage time based on stage status transitions.
        
        Stage time tracks elapsed time within each stage, resetting
        at "BEGIN STAGE" markers and continuing through falloff.
        
        Args:
            df: DataFrame with Job Time and Stage Status columns
            
        Returns:
            DataFrame with Stage Time column added
        """
        df["Stage Time"] = None
        state = TimeState()
        
        for j in range(len(df)):
            if j == 0:
                df.at[j, "Stage Time"] = None
                continue
            
            stage_status = df.at[j, "Stage Status"]
            
            if stage_status is None:
                df.at[j, "Stage Time"] = DataTransformer._calculate_time_increment(
                    df, j, "Stage Time", state.start_flag
                )
            
            elif stage_status.startswith("BEGIN STAGE "):
                df.at[j, "Stage Time"] = 0.0
                state.start_flag = True
                
                # Handle previously skipped stage
                if state.skipped_stage_start_pos > 0 and state.skipped_stage_number > 0:
                    begin_stage_number = int(df["Stage_Number"].iloc[j])
                    if begin_stage_number == state.skipped_stage_number:
                        df.at[state.skipped_stage_start_pos, "Stage Status"] = None
                        state.skipped_stage_start_pos = -1
                        state.skipped_stage_number = -1
            
            elif stage_status.startswith("BEGIN FALLOFF STAGE"):
                state.start_flag = True
                df.at[j, "Stage Time"] = DataTransformer._calculate_time_increment(
                    df, j, "Stage Time", state.start_flag
                )
            
            elif stage_status.startswith("END"):
                df.at[j, "Stage Time"] = DataTransformer._calculate_time_increment(
                    df, j, "Stage Time", state.start_flag
                )
                state.start_flag = False
            
            elif stage_status.startswith("SKIPPED"):
                state.skipped_stage_start_pos = j
                state.skipped_stage_number = int(df["Stage_Number"].iloc[j])
        
        return df
    
    @staticmethod
    def calculate_falloff_time(df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate falloff time for each stage. """
        df["Falloff Time"] = None
        state = TimeState()
        
        for j in range(len(df)):
            if j == 0:
                df.at[j, "Falloff Time"] = None
                continue
            
            stage_status = df.at[j, "Stage Status"]
            
            if stage_status is None:
                df.at[j, "Falloff Time"] = DataTransformer._calculate_time_increment(
                    df, j, "Falloff Time", state.start_flag
                )
            
            elif stage_status.startswith("BEGIN FALLOFF STAGE"):
                df.at[j, "Falloff Time"] = 0.0
                state.start_flag = True
            
            elif stage_status.startswith("END"):
                df.at[j, "Falloff Time"] = DataTransformer._calculate_time_increment(
                    df, j, "Falloff Time", state.start_flag
                )
                state.start_flag = False
        
        return df
    
    @staticmethod
    def _calculate_time_increment(
        df: pd.DataFrame,
        index: int,
        time_column: str,
        is_active: bool
    ) -> Optional[float]:
        """ Calculate time increment for stage/falloff time. """
        if not is_active:
            return None
        
        previous_time = df[time_column].iloc[index - 1]
        if previous_time is None:
            previous_time = 0
        
        job_time_delta = (
            df["Job Time"].iloc[index] - df["Job Time"].iloc[index - 1]
        )
        
        return job_time_delta + previous_time
    
    @staticmethod
    def calculate_stage_status(df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate stage status based on injection and falloff flags."""
        df["Stage Status"] = None
        state = StageState()
        
        for j in range(len(df)):
            # Get current flags
            current_injection = DataTransformer._get_flag_value(df, j, "Injection_Flag")
            current_falloff = DataTransformer._get_flag_value(df, j, "Falloff_Flag")
            current_stage = DataTransformer._get_flag_value(df, j, "Stage_Number")
            
            if current_stage == 0:
                continue
            
            # Determine stage status
            df.at[j, "Stage Status"] = DataTransformer._determine_stage_status(
                df, j, current_injection, current_falloff, current_stage, state
            )
            
            # Update state
            if current_injection == 1 and current_falloff == 0:
                if state.previous_injection_flag != 1:
                    state.previous_injection_flag = current_injection
                    state.begin_stage_flag = True
            elif current_injection == 0 and current_falloff == 1:
                if state.previous_falloff_flag != 1:
                    state.previous_falloff_flag = current_falloff
                    state.begin_stage_flag = False
                    state.begin_falloff_stage = True
        
        return df
    
    @staticmethod
    def _get_flag_value(df: pd.DataFrame, index: int, column: str) -> int:
        """Get integer flag value, defaulting to 0 if null."""
        value = df[column].iloc[index]
        return int(value) if pd.notnull(value) else 0
    
    @staticmethod
    def _determine_stage_status(
        df: pd.DataFrame,
        index: int,
        current_injection: int,
        current_falloff: int,
        current_stage: int,
        state: StageState
    ) -> Optional[str]:
        """ Determine the stage status for a given row. """
        # Both flags are 0
        if current_injection == 0 and current_falloff == 0:
            return DataTransformer._handle_both_flags_off(
                df, index, current_stage, state
            )
        
        # Injection active
        elif current_injection == 1 and current_falloff == 0:
            return DataTransformer._handle_injection_active(
                df, index, current_stage, state
            )
        
        # Falloff active
        elif current_injection == 0 and current_falloff == 1:
            return DataTransformer._handle_falloff_active(
                df, index, current_stage, state
            )
        
        return None
    
    @staticmethod
    def _handle_both_flags_off(
        df: pd.DataFrame,
        index: int,
        current_stage: int,
        state: StageState
    ) -> Optional[str]:
        """Handle case where both injection and falloff flags are off."""
        if current_stage <= state.previous_stage_number:
            if state.previous_falloff_flag != 0 and state.begin_falloff_stage:
                state.previous_injection_flag = 0
                state.previous_falloff_flag = 0
                state.previous_stage_number = current_stage
                state.begin_falloff_stage = False
                
                if current_stage not in state.stages_processed:
                    state.stages_processed.append(current_stage)
                
                return f"END FALLOFF STAGE {df['Stage_Number'].iloc[index]}"
        else:
            if state.previous_falloff_flag == 0:
                state.previous_stage_number = current_stage
                return f"SKIPPED STAGE {df['Stage_Number'].iloc[index]}"
            elif state.begin_falloff_stage:
                state.previous_injection_flag = 0
                state.previous_falloff_flag = 0
                state.previous_stage_number = current_stage
                state.begin_falloff_stage = False
                
                if current_stage not in state.stages_processed:
                    state.stages_processed.append(current_stage)
                
                return f"END FALLOFF STAGE {df['Stage_Number'].iloc[index]}"
        
        return None
    
    @staticmethod
    def _handle_injection_active(
        df: pd.DataFrame,
        index: int,
        current_stage: int,
        state: StageState
    ) -> Optional[str]:
        """Handle case where injection is active."""
        if state.previous_injection_flag == 1:
            # Continue injection - check for stage number issues
            if state.stages_processed and current_stage < state.stages_processed[-1]:
                df.at[index, "Stage_Number"] = state.stages_processed[-1]
            return None
        
        # Start of injection
        if not state.begin_stage_flag and state.previous_stage_number <= current_stage:
            if current_stage not in state.stages_processed:
                state.previous_stage_number = current_stage
                return f"BEGIN STAGE {df['Stage_Number'].iloc[index]}"
            else:
                # Stage already processed - increment
                if pd.notnull(df["Stage_Number"].iloc[index]):
                    new_stage = int(df["Stage_Number"].iloc[index]) + 1
                    df.at[index, "Stage_Number"] = new_stage
                    state.stages_processed.append(new_stage)
                    state.previous_stage_number = new_stage
                    return f"BEGIN STAGE {new_stage}"
        
        return None
    
    @staticmethod
    def _handle_falloff_active(
        df: pd.DataFrame,
        index: int,
        current_stage: int,
        state: StageState
    ) -> Optional[str]:
        """Handle case where falloff is active."""
        if state.previous_falloff_flag != 1:
            if state.previous_stage_number <= current_stage and state.begin_stage_flag:
                state.previous_stage_number = current_stage
                return f"BEGIN FALLOFF STAGE {df['Stage_Number'].iloc[index]}"
        
        return None
    
    @staticmethod
    def clean_stage_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """ Clean up stage numbers to remove zeros and fill gaps. """
        # Step 1: Remove zeros and invalid entries
        df = DataTransformer._remove_invalid_stage_numbers(df)
        
        # Step 2: Fill gaps in stage sequences
        df = DataTransformer._fill_stage_number_gaps(df)
        
        return df
    
    @staticmethod
    def _remove_invalid_stage_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid stage numbers (zeros and where both flags are off)."""
        for j in range(len(df)):
            if pd.isnull(df["Stage_Number"].iloc[j]):
                continue
            
            current_stage = int(df["Stage_Number"].iloc[j])
            
            if current_stage == 0:
                df.at[j, "Stage_Number"] = None
            elif current_stage > 0:
                injection = DataTransformer._get_flag_value(df, j, "Injection_Flag")
                falloff = DataTransformer._get_flag_value(df, j, "Falloff_Flag")
                
                if falloff == 0 and injection == 0:
                    df.at[j, "Stage_Number"] = None
        
        return df
    
    @staticmethod
    def _fill_stage_number_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in stage number sequences."""
        starting_pos = -1
        previous_stage = None
        is_gap = False
        
        for j in range(len(df)):
            current_stage = df["Stage_Number"].iloc[j]
            
            if pd.isnull(current_stage):
                if previous_stage is not None:
                    is_gap = True
            else:
                if previous_stage is not None:
                    if previous_stage == current_stage:
                        if is_gap:
                            # Clear the gap
                            df.loc[starting_pos:(j - 1), "Stage_Number"] = None
                            starting_pos = j
                            previous_stage = current_stage
                            is_gap = False
                    elif previous_stage < current_stage:
                        previous_stage = current_stage
                        starting_pos = j
                        is_gap = False
                else:
                    previous_stage = current_stage
                    starting_pos = j
        
        return df
    
    @staticmethod
    def calculate_advanced_metrics(
        df: pd.DataFrame,
        pressure_above_packer: pd.Series
    ) -> pd.DataFrame:
        """ Calculate energy and impulse metrics for each stage. """
        # Setup flags and groupings
        df = DataTransformer._setup_advanced_metrics_flags(df)
        
        # Calculate time metrics
        df = DataTransformer._calculate_advanced_time_metrics(df)
        
        # Calculate volume metrics
        df = DataTransformer._calculate_volume_metrics(df)
        
        # Calculate energy and impulse
        df = DataTransformer._calculate_energy(df, pressure_above_packer)
        df = DataTransformer._calculate_impulse(df, pressure_above_packer)
        
        return df
    
    @staticmethod
    def _setup_advanced_metrics_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Setup flags for advanced metrics calculation."""
        df['Flag'] = df['Stage Status'].fillna(method='ffill')
        df['FALLOFF_2'] = df['Flag'].str.replace(
            r"BEGIN FALLOFF.*", "FALLOFF", regex=True
        )
        df['FALLOFF_2'] = df['FALLOFF_2'].where(
            df['FALLOFF_2'].str.contains("FALLOFF"), np.nan
        )
        return df
    
    @staticmethod
    def _calculate_advanced_time_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time metrics for advanced calculations."""
        # Falloff time
        df['FTimeset'] = df.groupby(['Stage Number', 'FALLOFF_2'])['Job Time'].transform('first')
        df['FALLOFF TIME 2'] = pd.to_numeric(df['Job Time']) - pd.to_numeric(df['FTimeset'])
        df.loc[df['FALLOFF_2'].isna(), 'FALLOFF TIME'] = np.nan
        
        # Stage time
        df['STimeset'] = df.groupby('Stage Number')['Job Time'].transform('first')
        df['STAGE TIME'] = pd.to_numeric(df['Job Time']) - pd.to_numeric(df['STimeset'])
        df.loc[df['Flag'].str.contains("END FALLOFF.*", na=False), 'STAGE TIME'] = np.nan
        
        # Time delta for calculations
        df['Timedelta_2'] = (
            pd.to_datetime(df['Date/Time']) - pd.to_datetime(df['Date/Time']).shift(1)
        ).dt.total_seconds()
        
        return df
    
    @staticmethod
    def _calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative volume metrics."""
        df['CumVolSet'] = df.groupby('Stage Number')['Surface Volume m3'].transform('first')
        df['CUM VOL m3'] = (
            pd.to_numeric(df['Surface Volume m3'], errors='coerce') - 
            pd.to_numeric(df['CumVolSet'], errors='coerce')
        )
        df['CUM VOL m3'] = df['CUM VOL m3'].fillna(method='ffill')
        df.loc[df['Stage Number'].isna(), ['CUM VOL m3', 'STAGE TIME']] = np.nan
        
        return df
    
    @staticmethod
    def _calculate_energy(
        df: pd.DataFrame,
        pressure_above_packer: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate energy during injection phase.
        
        Energy (kJ) = Pressure * Flow Rate * Time Delta / 1000 / 60
        """
        # Ensure numeric types and alignment
        pressure_above_packer = pd.to_numeric(pressure_above_packer, errors='coerce')
        df['Surface Water Rate Lmin'] = pd.to_numeric(
            df['Surface Water Rate Lmin'], errors='coerce'
        )
        df['Timedelta_2'] = pd.to_numeric(df['Timedelta_2'], errors='coerce')
        
        # Align pressure series with dataframe
        if len(pressure_above_packer) > len(df):
            pressure_above_packer = pressure_above_packer.iloc[:len(df)]
        elif len(pressure_above_packer) < len(df):
            pressure_above_packer = pressure_above_packer.reindex(df.index)
        
        # Calculate energy (only during injection, not falloff)
        df['Energy kJ'] = np.where(
            df['Stage Number'].notna() & df['FALLOFF_2'].isna(),
            pressure_above_packer * df['Surface Water Rate Lmin'] * df['Timedelta_2'] / 1000 / 60,
            np.nan
        )
        
        # Set zero energy to NaN
        df.loc[df['Energy kJ'] == 0, 'Energy kJ'] = np.nan
        
        return df
    
    @staticmethod
    def _calculate_impulse(
        df: pd.DataFrame,
        pressure_above_packer: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate impulse during falloff phase.
        
        Impulse (kNs) = 0.0081073 * (current_pressure + previous_pressure)
        """
        # Calculate impulse (only during falloff)
        shifted_pressure = pressure_above_packer.shift(1)
        
        df['Impulse kNs'] = np.where(
            df['Stage Number'].notna() & df['FALLOFF_2'].notna(),
            0.0081073 * (pressure_above_packer + shifted_pressure),
            np.nan
        )
        
        return df
    
    # ========================================================================
    # Final Data Preparation
    # ========================================================================
    
    @staticmethod
    def prepare_final_dataframes(
        guidehawk_top_df: pd.DataFrame,
        guidehawk_bottom_df: pd.DataFrame,
        reliance_df: pd.DataFrame,
        uwi: str,
        twp: str,
        rng: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """
        Prepare final dataframes with all calculations and formatting.
        
        This method:
        1. Processes GuideHawk top/bottom data (pressure, temperature)
        2. Processes Reliance data (stages, times, metrics)
        3. Calculates all advanced metrics
        4. Rounds and formats columns
        
        """
        # Process GuideHawk data
        guidehawk_top_df, pressure_kpag = DataTransformer._prepare_guidehawk_top(
            guidehawk_top_df
        )
        
        if not guidehawk_bottom_df.empty:
            guidehawk_bottom_df = DataTransformer._prepare_guidehawk_bottom(
                guidehawk_bottom_df
            )
        
        # Process Reliance data
        reliance_df, tundra_uwi = DataTransformer._prepare_reliance_data(
            reliance_df, uwi, twp, rng, pressure_kpag
        )
        
        return guidehawk_top_df, guidehawk_bottom_df, reliance_df, tundra_uwi
    
    @staticmethod
    def _prepare_guidehawk_top(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare GuideHawk top data with pressure and temperature."""
        # Convert pressure from MPa to kPa
        top_pressure_kpa = df["annp"] * 1000
        
        # Add metadata columns
        df["Well Name"] = df.get("wellname", np.nan)
        df["Formation"] = df.get("formation", np.nan)
        
        # Add pressure columns
        df["Pressure Above Packer kPaa"] = top_pressure_kpa + ATMOSPHERE_PRESSURE_KPA
        df["Pressure Above Packer kPag"] = top_pressure_kpa
        
        # Add temperature
        df["Temp Above Packer C"] = df["temp"]
        
        # Store pressure for advanced metrics
        pressure_kpag = df["Pressure Above Packer kPag"].copy()
        
        # Drop unnecessary columns
        df = df.drop(
            columns=[col for col in GUIDEHAWK_TOP_COLUMNS_TO_DROP if col in df.columns]
        )
        
        logger.debug(f"Prepared GuideHawk top: {len(df)} rows, {len(df.columns)} columns")
        return df, pressure_kpag
    
    @staticmethod
    def _prepare_guidehawk_bottom(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare GuideHawk bottom data with pressure and temperature."""
        # Convert pressure from MPa to kPa
        btm_pressure_kpa = df["annp"] * 1000
        
        # Add pressure columns
        df["Pressure Below Packer kPag"] = btm_pressure_kpa
        df["Pressure Below Packer kPaa"] = btm_pressure_kpa + ATMOSPHERE_PRESSURE_KPA
        
        # Add temperature
        df["Temp Below Packer C"] = df["temp"]
        
        # Drop unnecessary columns
        df = df.drop(
            columns=[col for col in GUIDEHAWK_BOTTOM_COLUMNS_TO_DROP if col in df.columns]
        )
        
        logger.debug(f"Prepared GuideHawk bottom: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    @staticmethod
    def _prepare_reliance_data(
        df: pd.DataFrame,
        uwi: str,
        twp: str,
        rng: str,
        pressure_kpag: pd.Series
    ) -> Tuple[pd.DataFrame, str]:
        """Prepare Reliance data with all calculations."""
        # Add well identifiers
        tundra_uwi = format_tundra_uwi(uwi)
        df["UWI"] = tundra_uwi
        df["UWI2"] = uwi + "/00"
        df["TWP"] = twp
        df["RNG"] = rng
        
        # Calculate time metrics
        logger.debug("Calculating time metrics...")
        df = DataTransformer.calculate_job_time(df)
        df = DataTransformer.calculate_stage_status(df)
        df = DataTransformer.calculate_stage_time(df)
        df = DataTransformer.calculate_falloff_time(df)
        
        # Add pressure and time delta
        df["Surface Casing Pressure kPaa"] = (
            df["Selected_Pressure (KPAg)"] + ATMOSPHERE_PRESSURE_KPA
        )
        df["Timedelta"] = df['Date/Time'].diff().dt.total_seconds()
        
        # Clean stage numbers
        df = DataTransformer.clean_stage_numbers(df)
        
        # Map columns to final names
        df = DataTransformer._map_reliance_columns(df)
        
        # Calculate advanced metrics
        logger.debug("Calculating advanced metrics...")
        df = DataTransformer.calculate_advanced_metrics(df, pressure_kpag)
        
        # Round numeric columns
        df = DataTransformer._round_numeric_columns(df)
        
        # Drop unnecessary columns
        df = df.drop(
            columns=[col for col in RELIANCE_COLUMNS_TO_DROP if col in df.columns]
        )
        
        logger.debug(f"Prepared Reliance data: {len(df)} rows, {len(df.columns)} columns")
        return df, tundra_uwi
    
    @staticmethod
    def _map_reliance_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map Reliance columns to final column names."""
        df["Stage Number"] = df["Stage_Number"]
        df["Surface Casing Temp C"] = df["CASING_Temp (Celcius)"]
        df["Surface Casing Pressure kPag"] = df["Selected_Pressure (KPAg)"]
        
        # Falloff intermediate flag
        df['Falloff_Intermediate_Flag'] = (
            df['Falloff_Intermediate_Flag'].fillna(0).astype(int)
        )
        df["Falloff"] = df['Falloff_Intermediate_Flag'].apply(
            lambda x: 'FALLOFF' if x == 1 else ''
        )
        
        # Flow and volume
        df["Surface Water Rate Lmin"] = df["Selected_Flow_Rate (L/min)"]
        df["Surface Volume m3"] = df["Turbine_1_Accum"]
        
        # DateTime
        df['Date Time'] = df['Date/Time']
        
        return df
    
    @staticmethod
    def _round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Round specified numeric columns to 4 decimal places."""
        for col in COLUMNS_TO_ROUND:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(4)
        
        return df
    
    @staticmethod
    def merge_and_export(
        guidehawk_top_df: pd.DataFrame,
        guidehawk_bottom_df: pd.DataFrame,
        reliance_df: pd.DataFrame,
        tundra_uwi: str
    ) -> str:
        """ Merge dataframes horizontally and export to CSV. """
        # Align all dataframes to same length
        target_length = len(reliance_df)
        dfs = [guidehawk_top_df, guidehawk_bottom_df, reliance_df]
        
        aligned_dfs = []
        for df in dfs:
            aligned_df = DataTransformer._align_dataframe_length(df, target_length)
            aligned_dfs.append(aligned_df)
        
        # Merge horizontally
        merged_df = pd.concat(aligned_dfs, axis=1)
        
        # Export to file
        filename = f"FallOff_{tundra_uwi}.csv"
        DataTransformer._export_to_csv(merged_df, filename)
        
        logger.info(f"Exported merged data to {filename}: {len(merged_df)} rows")
        return filename
    
    @staticmethod
    def _align_dataframe_length(
        df: pd.DataFrame,
        target_length: int
    ) -> pd.DataFrame:
        """ Align dataframe to target length by padding or truncating."""
        current_length = len(df)
        
        if current_length < target_length:
            # Pad with NaN rows
            padding = pd.DataFrame(
                np.nan,
                index=range(target_length - current_length),
                columns=df.columns
            )
            return pd.concat([df, padding], ignore_index=True)
        else:
            # Truncate to target length
            return df.head(target_length)
    
    @staticmethod
    def _export_to_csv(df: pd.DataFrame, filename: str) -> None:
        """ Export dataframe to CSV via temporary file."""
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            temp_filename = tmp_file.name
            df.to_csv(temp_filename, index=False)
        
        # Move to final location
        shutil.move(temp_filename, filename)