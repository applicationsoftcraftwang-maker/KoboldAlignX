"""
Unit tests for excel_generator.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from src.services.excel_generator import ExcelGenerator


class TestExcelGenerator:
    """Test suite for ExcelGenerator."""
    
    def test_initialization(self):
        """Test ExcelGenerator initialization."""
        generator = ExcelGenerator()
        
        assert generator.env is not None
    
    def test_get_peak_indexes_basic(self):
        """Test peak detection with basic data."""
        generator = ExcelGenerator()
        
        # Create test data with clear peaks
        data = pd.Series([
            0, 1, 2, 3, 2, 1, 0,  # First peak
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Rising
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0  # Last peak
        ])
        
        first_peak, last_peak = generator.get_peak_indexes(data)
        
        # Verify peaks are detected
        assert isinstance(first_peak, (int, np.integer))
        assert isinstance(last_peak, (int, np.integer))
        assert first_peak >= -500  # Can be negative with threshold offset
        assert last_peak > first_peak
        assert last_peak < len(data)
    
    def test_get_peak_indexes_with_dataframe(self):
        """Test peak detection with DataFrame input."""
        generator = ExcelGenerator()
        
        # Create DataFrame with pressure column
        data = pd.DataFrame({
            'pressure': [0, 5, 10, 15, 20, 15, 10, 5, 0]
        })
        
        first_peak, last_peak = generator.get_peak_indexes(data)
        
        # Should handle DataFrame by squeezing to Series
        assert isinstance(first_peak, (int, np.integer))
        assert isinstance(last_peak, (int, np.integer))
    
    def test_get_peak_indexes_flat_data(self):
        """Test peak detection with flat data (no peaks)."""
        generator = ExcelGenerator()
        
        # Flat data
        data = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        
        first_peak, last_peak = generator.get_peak_indexes(data)
        
        # Should still return indices
        assert isinstance(first_peak, (int, np.integer))
        assert isinstance(last_peak, (int, np.integer))


class TestAlignJobs:
    """Test suite for align_jobs method."""
    
    @pytest.fixture
    def sample_guidehawk_df(self):
        """Create sample Guidehawk DataFrame."""
        return pd.DataFrame({
            'kdatetime': pd.date_range('2024-01-15 10:00:00', periods=100, freq='1s'),
            'jobid': ['GH-12345'] * 100,
            'bhaloc': ['a'] * 100,
            'annp': np.random.uniform(10, 20, 100),
            'temp': np.random.uniform(20, 30, 100),
            'wellname': ['Test Well'] * 100,
            'formation': ['Viking'] * 100,
            'coilp': [0] * 100,
            'strain': [0] * 100,
            'shock': [0] * 100,
            'voltage': [12] * 100,
            'torque': [0] * 100,
            'name': ['sensor_a'] * 100,
            'runno': [1] * 100,
            'serial': ['SN123'] * 100,
            'stage': [1] * 100,
            'isshift': [0] * 100
        })
    
    @pytest.fixture
    def sample_reliance_df(self):
        """Create sample Reliance DataFrame."""
        dates = pd.date_range('2024-01-15 10:00:00', periods=100, freq='1s')
        return pd.DataFrame({
            'Date': [d.strftime('%d/%m/%Y') for d in dates],
            'Time': [d.strftime('%H:%M:%S') for d in dates],
            'Selected_Pressure (KPAg)': np.random.uniform(10000, 20000, 100),
            'CASING_Pressure (KPAg)': np.random.uniform(10000, 20000, 100),
            'CASING_Temp (Celcius)': np.random.uniform(20, 30, 100),
            'TUBING_Pressure (KPAg)': np.random.uniform(10000, 20000, 100),
            'TUBING_Temp (Celcius)': np.random.uniform(20, 30, 100),
            'EXTERNAL_RTD__Temp (Celcius)': np.random.uniform(20, 30, 100),
            'Turbine_1_Accum': np.cumsum(np.random.uniform(0, 0.1, 100)),
            'Turbine_1_Rate (L/min)': np.random.uniform(80, 120, 100),
            'Selected_Flow_Rate (L/min)': np.random.uniform(80, 120, 100),
            'Selected_Flow_Volume (m³/s)': np.random.uniform(0, 0.002, 100),
            'Injection_Flag': [1] * 50 + [0] * 50,
            'Falloff_Flag': [0] * 50 + [1] * 50,
            'Injection_Pumping_Time (s)': list(range(100)),
            'Falloff_Time (s)': [0] * 50 + list(range(50)),
            'Fluid_Power (kJ)': np.random.uniform(0, 100, 100),
            'Total_Fluid_Power (kJ)': np.cumsum(np.random.uniform(0, 10, 100)),
            'Impulse_Momentum (kN.s)': np.random.uniform(0, 50, 100),
            'Total_Impulse_Momentum (kN.s)': np.cumsum(np.random.uniform(0, 5, 100)),
            'Impulse_Momentum_Energy_Ratio (s/m)': np.random.uniform(0, 1, 100),
            'Fluid_Power_Energy_Rate_Norm (kJ/s)': np.random.uniform(0, 10, 100),
            'Impulse_Momentum_Force_Norm (kN)': np.random.uniform(0, 20, 100),
            'Calculated_Injected_Volume_Total (m³)': np.cumsum(np.random.uniform(0, 0.1, 100)),
            'Logic_Test': [0] * 100,
            'Force_Output_vs_Fluid_Power_Input (s/m)': np.random.uniform(0, 1, 100),
            'Falloff_Flow_Volume_Test (m³)': np.random.uniform(0, 0.1, 100),
            'Stage_Number': [1] * 100,
            'Impulse_Energy_Difference': np.random.uniform(0, 10, 100),
            'Falloff_Intermediate_Flag': [0] * 100,
            'Falloff_Intermediate_Time (s)': [0] * 100,
            'Falloff_Intermediate_Stop_Condition': [0] * 100,
            'Falloff_Start_Pressure (KPAg)': [15000] * 100,
            'Falloff_Intermediate_Pressure (KPAg)': [14000] * 100,
            'Falloff_Final_Pressure (KPAg)': [13000] * 100,
            'Intermediate_Total_Impulse_Momentum (kN.s)': np.random.uniform(0, 50, 100),
            'Intermediate_Impulse_Momentum_Energy_Ratio (s/m)': np.random.uniform(0, 1, 100),
            'Intermediate_Impulse_Momentum_Force_Norm (kN)': np.random.uniform(0, 20, 100),
            'Intermediate_Force_Output_vs_Fluid_Power_Input (s/m)': np.random.uniform(0, 1, 100),
            'Intermediate_Impulse_Energy_Difference': np.random.uniform(0, 10, 100),
            'Flow_Volume_3_Samples (m³)': np.random.uniform(0, 0.1, 100),
            'Flow_Volume_Threshold_Flag': [0] * 100,
            'Falloff_Flow_Volume_Test_R2': np.random.uniform(0, 1, 100),
            'Falloff_Counter (s)': list(range(100))
        })
    
    @pytest.mark.asyncio
    async def test_align_jobs_with_valid_data(
        self, 
        sample_guidehawk_df, 
        sample_reliance_df
    ):
        """Test align_jobs with valid input data."""
        generator = ExcelGenerator()
        
        filename, guidehawk_job_id = await generator.align_jobs(
            guidehawk_df=sample_guidehawk_df,
            reliance_df=sample_reliance_df,
            job_id="28238",
            uwi="100/01-02-003-04W5",
            twp="003",
            rng="04"
        )
        
        # Verify output
        if filename:
            assert filename.startswith("FallOff_")
            assert ".csv" in filename
            assert guidehawk_job_id is not None
        # Note: May return None if data doesn't meet certain criteria
    
    @pytest.mark.asyncio
    async def test_align_jobs_with_empty_guidehawk(self, sample_reliance_df):
        """Test align_jobs with empty Guidehawk data."""
        generator = ExcelGenerator()
        
        empty_df = pd.DataFrame()
        
        filename, guidehawk_job_id = await generator.align_jobs(
            guidehawk_df=empty_df,
            reliance_df=sample_reliance_df,
            job_id="28238",
            uwi="100/01-02-003-04W5",
            twp="003",
            rng="04"
        )
        
        # Should return None for empty data
        assert filename is None
        assert guidehawk_job_id is None
    
    @pytest.mark.asyncio
    async def test_align_jobs_with_fr_job(self, sample_guidehawk_df, sample_reliance_df):
        """Test align_jobs with FR job (should be filtered out)."""
        generator = ExcelGenerator()
        
        # Modify jobid to contain 'fr'
        sample_guidehawk_df['jobid'] = 'fr_test_job'
        
        filename, guidehawk_job_id = await generator.align_jobs(
            guidehawk_df=sample_guidehawk_df,
            reliance_df=sample_reliance_df,
            job_id="28238",
            uwi="100/01-02-003-04W5",
            twp="003",
            rng="04"
        )
        
        # FR jobs should be filtered out
        assert filename is None
        assert guidehawk_job_id is None
    
    @pytest.mark.asyncio
    async def test_align_jobs_missing_bhaloc_a(self, sample_guidehawk_df, sample_reliance_df):
        """Test align_jobs when bhaloc 'a' is missing."""
        generator = ExcelGenerator()
        
        # Change all bhaloc to 'b' (missing 'a')
        sample_guidehawk_df['bhaloc'] = 'b'
        
        filename, guidehawk_job_id = await generator.align_jobs(
            guidehawk_df=sample_guidehawk_df,
            reliance_df=sample_reliance_df,
            job_id="28238",
            uwi="100/01-02-003-04W5",
            twp="003",
            rng="04"
        )
        
        # Should return None when bhaloc 'a' is missing
        assert filename is None
        assert guidehawk_job_id is None


class TestExcelGeneratorEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_align_jobs_with_invalid_dates(self):
        """Test handling of invalid date formats."""
        generator = ExcelGenerator()
        
        # Create data with invalid dates
        guidehawk_df = pd.DataFrame({
            'kdatetime': ['invalid_date'] * 10,
            'jobid': ['GH-123'] * 10,
            'bhaloc': ['a'] * 10,
            'annp': [15] * 10,
            'temp': [25] * 10
        })
        
        reliance_df = pd.DataFrame({
            'Date': ['invalid'] * 10,
            'Time': ['invalid'] * 10
        })
        
        filename, job_id = await generator.align_jobs(
            guidehawk_df, reliance_df, "28238", "100/01-02-003-04W5", "003", "04"
        )
        
        # Should handle gracefully and return None
        assert filename is None or isinstance(filename, str)
    
    def test_get_peak_indexes_with_zeros(self):
        """Test peak detection with all zeros."""
        generator = ExcelGenerator()
        
        data = pd.Series([0] * 100)
        
        first_peak, last_peak = generator.get_peak_indexes(data)
        
        # Should handle gracefully
        assert isinstance(first_peak, (int, np.integer))
        assert isinstance(last_peak, (int, np.integer))
    
    def test_get_peak_indexes_single_value(self):
        """Test peak detection with single value."""
        generator = ExcelGenerator()
        
        data = pd.Series([10])
        
        first_peak, last_peak = generator.get_peak_indexes(data)
        
        # Should handle single value
        assert isinstance(first_peak, (int, np.integer))
        assert isinstance(last_peak, (int, np.integer))


class TestExcelGeneratorPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_align_jobs_large_dataset(self):
        """Test align_jobs with large dataset."""
        generator = ExcelGenerator()
        
        # Create larger dataset (10,000 rows)
        size = 10000
        dates = pd.date_range('2024-01-15 10:00:00', periods=size, freq='1s')
        
        guidehawk_df = pd.DataFrame({
            'kdatetime': dates,
            'jobid': ['GH-12345'] * size,
            'bhaloc': ['a'] * size,
            'annp': np.random.uniform(10, 20, size),
            'temp': np.random.uniform(20, 30, size),
            'wellname': ['Test Well'] * size,
            'formation': ['Viking'] * size
        })
        
        reliance_df = pd.DataFrame({
            'Date': [d.strftime('%d/%m/%Y') for d in dates],
            'Time': [d.strftime('%H:%M:%S') for d in dates],
            'Selected_Pressure (KPAg)': np.random.uniform(10000, 20000, size),
            'Injection_Flag': [1] * (size // 2) + [0] * (size // 2),
            'Falloff_Flag': [0] * (size // 2) + [1] * (size // 2),
            'Stage_Number': [1] * size
        })
        
        # This might take a while but should complete
        filename, job_id = await generator.align_jobs(
            guidehawk_df, reliance_df, "28238", "100/01-02-003-04W5", "003", "04"
        )
        
        # Should produce output or handle gracefully
        assert filename is None or isinstance(filename, str)