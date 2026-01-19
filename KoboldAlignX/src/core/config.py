"""Configuration settings for the job processing application."""
import os
from dotenv import load_dotenv

load_dotenv(".env")

# Environment
ENV = os.getenv('ENV', 'production')

# API Configuration
API_BASE_URL = "https://test.net"
RELIANCE_BASE_URL = "https://test.mrlsolutions.com"
RELIANCE_OAUTH_URL = f"{RELIANCE_BASE_URL}/oauth/token"
RELIANCE_JOBS_URL = f"{RELIANCE_BASE_URL}/api/v2/jobs"

# OAuth Credentials
OAUTH_CLIENT_ID = "*****"
OAUTH_CLIENT_SECRET = "********"

# Celery Configuration
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq/%2F")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
CELERY_TIMEZONE = "America/Edmonton"
CELERY_TASK_TIME_LIMIT = 3600
CELERY_TASK_SOFT_TIME_LIMIT = 3000
CELERY_BEAT_MAX_LOOP_INTERVAL = 1500
CELERY_BEAT_SCHEDULE_INTERVAL_MINUTES = 15

# Email Configuration
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587
OUTLOOK_FROM_EMAIL = "test@koboldinc.com"
OUTLOOK_EMAIL = os.environ.get("OUTLOOK_EMAIL", "testdev@koboldinc.com")
OUTLOOK_PASSWORD = os.environ.get("OUTLOOK_PASSWORD", "********")
DEFAULT_TO_EMAIL = "test@koboldinc.com"

# Processing Constants
ATMOSPHERE_PRESSURE_KPA = 101.325
THRESHOLD_PRESSURE = 10
FILTER_LENGTH = 35
GOOD_SAMPLE_LENGTH = 20
PEAK_THRESHOLD_OFFSET = 500
PADDING = 500
SECOND_MAX_THRESHOLD = 0.85

# Column Definitions
GUIDEHAWK_TOP_COLUMNS_TO_DROP = [
    "isshift", "shock", "voltage", "coilp", "annp", "strain", "temp",
    "name", "torque", "jobid", "kdatetime", "runno", "serial", "bhaloc",
    "stage", "Date/Time", "wellname", "formation"
]

GUIDEHAWK_BOTTOM_COLUMNS_TO_DROP = [
    "isshift", "shock", "voltage", "coilp", "annp", "strain", "temp",
    "name", "torque", "jobid", "kdatetime", "runno", "serial", "bhaloc",
    "stage", "Date/Time", "wellname", "formation"
]

RELIANCE_COLUMNS_TO_DROP = [
    "Date", "Time", 'CASING_Pressure (KPAg)', 'CASING_Temp (Celcius)',
    'TUBING_Pressure (KPAg)', 'TUBING_Temp (Celcius)',
    'EXTERNAL_RTD__Temp (Celcius)', 'Turbine_1_Accum',
    'Turbine_1_Rate (L/min)', 'Selected_Pressure (KPAg)',
    'Selected_Flow_Rate (L/min)', 'Selected_Flow_Volume (m³/s)',
    'Injection_Flag', 'Falloff_Flag', 'Injection_Pumping_Time (s)',
    'Falloff_Time (s)', 'Fluid_Power (kJ)', 'Total_Fluid_Power (kJ)',
    'Impulse_Momentum (kN.s)', 'Total_Impulse_Momentum (kN.s)',
    'Impulse_Momentum_Energy_Ratio (s/m)', 'Fluid_Power_Energy_Rate_Norm (kJ/s)',
    'Impulse_Momentum_Force_Norm (kN)', 'Calculated_Injected_Volume_Total (m³)',
    'Logic_Test', 'Force_Output_vs_Fluid_Power_Input (s/m)',
    'Falloff_Flow_Volume_Test (m³)', 'Stage_Number', 'Impulse_Energy_Difference',
    'Falloff_Intermediate_Flag', 'Falloff_Intermediate_Time (s)',
    'Falloff_Intermediate_Stop_Condition', 'Falloff_Start_Pressure (KPAg)',
    'Falloff_Intermediate_Pressure (KPAg)', 'Falloff_Final_Pressure (KPAg)',
    'Intermediate_Total_Impulse_Momentum (kN.s)',
    'Intermediate_Impulse_Momentum_Energy_Ratio (s/m)',
    'Intermediate_Impulse_Momentum_Force_Norm (kN)',
    'Intermediate_Force_Output_vs_Fluid_Power_Input (s/m)',
    'Intermediate_Impulse_Energy_Difference', 'Flow_Volume_3_Samples (m³)',
    'Flow_Volume_Threshold_Flag', 'Falloff_Flow_Volume_Test_R2',
    'Falloff_Counter (s)', 'Pressure', 'FTimeset', 'STimeset', 'Flag',
    'CumVolSet', 'FALLOFF TIME 2', 'FALLOFF_2', 'STAGE TIME',
    'FALLOFF TIME', 'Timedelta_2', 'Date/Time'
]

COLUMNS_TO_ROUND = [
    "Pressure Above Packer kPaa", "Pressure Above Packer kPag",
    "Temp Above Packer C", "Pressure Below Packer kPag",
    "Temp Below Packer C", "Pressure Below Packer kPaa",
    "Energy kJ", "Impulse kNs", "Surface Casing Pressure kPaa"
]