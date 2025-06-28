"""
Configuration for anomaly detection pipeline.

# -----------------------------------------------------------------------------
# DATABASE CONNECTION INSTRUCTIONS
# -----------------------------------------------------------------------------
# You can set the following environment variables to specify database locations:
#   - LIMS_CONNECTION_STRING
#   - MES_CONNECTION_STRING
#   - QMS_CONNECTION_STRING
#
# These can be set in your shell, or in a .env file at the project root:
#
# Example .env:
#   LIMS_CONNECTION_STRING=postgresql://user@localhost:5432/lims_db
#   MES_CONNECTION_STRING=postgresql://user@localhost:5432/mes_db
#   QMS_CONNECTION_STRING=postgresql://user@localhost:5432/qms_db
#
# The system supports various database types: PostgreSQL, MySQL, SQLite, etc.
# -----------------------------------------------------------------------------
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import os
import logging
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ColumnConfig:
    """Configuration for individual columns."""
    dtype: str = "auto"  # "numeric", "categorical", "timestamp", "auto"
    category: str = "auto"  # "CPP", "CQA", "auto"
    is_identifier: bool = False
    is_required: bool = False
    is_grouping: bool = False
    is_timestamp: bool = False
    description: str = ""

@dataclass
class SystemConfig:
    name: str
    connection_string: str
    tables: List[str]
    anomaly_threshold: float = 0.95
    column_configs: Dict[str, ColumnConfig] = None

@dataclass
class DetectionConfig:
    window_size: int = 10
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    trend_threshold: float = 0.1
    deviation_threshold: float = 2.0
    min_samples: int = 5
    contamination: float = 0.1
    pattern_threshold: float = 0.05

@dataclass
class MLConfig:
    model_type: str = 'isolation_forest'
    random_state: int = 42
    n_estimators: int = 100
    max_samples: str = 'auto'
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1
    # Enhanced isolation forest parameters
    contamination: float = 0.1
    warm_start: bool = False
    verbose: int = 0

@dataclass
class DatabaseConfig:
    """Database configuration for different systems."""
    connection_type: str = "postgresql"  # postgresql, mysql, sqlite, etc.
    timeout: int = 30
    max_connections: int = 10

# Database configuration
DB_CONFIG = DatabaseConfig()

# System-specific configurations with column configs
SYSTEMS_CONFIG = {
    "LIMS": SystemConfig(
        name="LIMS",
        connection_string=os.getenv("LIMS_CONNECTION_STRING"),
        tables=["lims_data"],
        anomaly_threshold=0.95,
        column_configs = {
            "Lab_Sample_ID": ColumnConfig(dtype="categorical", category="CQA", is_identifier=True),
            "Batch_ID": ColumnConfig(dtype="categorical", category="CQA", is_identifier=True),
            "Product_Name": ColumnConfig(dtype="categorical", category="CQA", is_grouping=True),
            "Sample_Type": ColumnConfig(dtype="categorical", category="CQA"),
            "Sample_Login_Timestamp": ColumnConfig(dtype="timestamp", category="CQA", is_timestamp=True),
            "Sample_Status": ColumnConfig(dtype="categorical", category="CQA"),
            "Test_Name": ColumnConfig(dtype="categorical", category="CQA"),
            "Test_Method_ID": ColumnConfig(dtype="categorical", category="CQA"),
            "Instrument_ID": ColumnConfig(dtype="categorical", category="CQA"),
            "Spec_Limit_Min": ColumnConfig(dtype="numeric", category="CQA"),
            "Spec_Limit_Max": ColumnConfig(dtype="numeric", category="CQA"),
            "Result_Value": ColumnConfig(dtype="numeric", category="CQA", is_required=True),
            "Result_UOM": ColumnConfig(dtype="categorical", category="CQA"),
            "Result_Status": ColumnConfig(dtype="categorical", category="CQA"),
            "Analyst_ID": ColumnConfig(dtype="categorical", category="CQA"),
            "Result_Entry_Timestamp": ColumnConfig(dtype="timestamp", category="CQA", is_timestamp=True),
            "Reviewed_By": ColumnConfig(dtype="categorical", category="CQA"),
            "Approved_By": ColumnConfig(dtype="categorical", category="CQA")
        }

    ),
    "MES": SystemConfig(
        name="MES",
        connection_string=os.getenv("MES_CONNECTION_STRING"),
        tables=["mes_data"],
        anomaly_threshold=0.95,
        column_configs = {
            "Work_Order_ID": ColumnConfig(dtype="categorical", category="CPP", is_identifier=True),
            "Batch_ID": ColumnConfig(dtype="categorical", category="CPP", is_identifier=True),
            "Product_Code": ColumnConfig(dtype="categorical", category="CPP", is_grouping=True),
            "Master_Recipe_ID": ColumnConfig(dtype="categorical", category="CPP"),
            "Batch_Phase": ColumnConfig(dtype="categorical", category="CPP"),
            "Phase_Step": ColumnConfig(dtype="categorical", category="CPP"),
            "Parameter_Name": ColumnConfig(dtype="categorical", category="CPP"),
            "Parameter_Value": ColumnConfig(dtype="numeric", category="CPP", is_required=True),
            "Performed_By": ColumnConfig(dtype="categorical", category="CPP"),
            "Verified_By": ColumnConfig(dtype="categorical", category="CPP"),
            "Execution_Timestamp": ColumnConfig(dtype="timestamp", category="CPP", is_timestamp=True),
            "Execution_Status": ColumnConfig(dtype="categorical", category="CPP"),
            "Equipment_ID": ColumnConfig(dtype="categorical", category="CPP",is_grouping=True)
        }

    ),
    "QMS": SystemConfig(
        name="QMS",
        connection_string=os.getenv("QMS_CONNECTION_STRING"),
        tables=["qms_data"],
        anomaly_threshold=0.95,
        column_configs = {
            "Record_ID": ColumnConfig(dtype="categorical", category="CQA", is_identifier=True),
            "Record_Type": ColumnConfig(dtype="categorical", category="CQA"),
            "Title": ColumnConfig(dtype="text", category="CQA"),
            "Description": ColumnConfig(dtype="text", category="CQA"),
            "Batch_ID": ColumnConfig(dtype="categorical", category="CQA", is_identifier=True),
            "Product_Code": ColumnConfig(dtype="categorical", category="CQA",is_grouping=True),
            "Status_Workflow": ColumnConfig(dtype="categorical", category="CQA"),
            "Owner_Name": ColumnConfig(dtype="categorical", category="CQA"),
            "Open_Date": ColumnConfig(dtype="timestamp", category="CQA", is_timestamp=True),
            "Due_Date": ColumnConfig(dtype="timestamp", category="CQA", is_timestamp=True),
            "Closure_Date": ColumnConfig(dtype="timestamp", category="CQA", is_timestamp=True),
            "Source_Event_ID": ColumnConfig(dtype="categorical", category="CQA")
        }

    )
}

# Detection configuration
DETECTION_CONFIG = DetectionConfig()

# ML model configuration
ML_CONFIG = MLConfig()

# Derived columns configuration - columns that need to be computed
DERIVED_COLUMNS = {
    'mes_data': {
    },
    'lims_data': {
        'Test_Duration': {
            'formula': 'Result_Entry_Timestamp - Sample_Login_Timestamp',
            'type': 'duration',
            'description': 'Test duration from sample login to result entry',
            'dtype': 'numeric',
            'category': 'CQA'
        },
        'Result_Entry_Delay': {
            'formula': 'Result_Entry_Timestamp - Sample_Login_Timestamp',
            'type': 'duration',
            'description': 'Delay between sample login and result entry',
            'dtype': 'numeric',
            'category': 'CQA'
        }
    },
    'qms_data': {
        'Record_Lifecycle_Duration': {
            'formula': 'Closure_Date - Open_Date',
            'type': 'duration',
            'description': 'Total lifecycle time of QMS record',
            'dtype': 'numeric',
            'category': 'QMS'
        },
        'SLA_Breach_Time': {
            'formula': 'Closure_Date - Due_Date',
            'type': 'duration',
            'description': 'Time taken beyond due date, if any',
            'dtype': 'numeric',
            'category': 'QMS'
        }
    }
}



# Multivariate feature groups for isolation forest
MULTIVARIATE_FEATURES = {
    'MES': {
        'manufacturing_process': [
            'Parameter_Value'  # Valid
        ]
    },
    'LIMS': {
        'test_metrics': [
            'Result_Value',
            'Test_Duration',
            'Result_Entry_Delay'
        ],
        'deviation_metrics': [
            'Result_Value',
            'Spec_Limit_Min',
            'Spec_Limit_Max'
        ]
    },
    'QMS': {
        'record_closure_metrics': [
            'Record_Lifecycle_Duration',
            'SLA_Breach_Time'
        ]
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'output_dir': os.getenv('ANOMALY_REPORT_DIR', 'data_quality_reports/anomaly'),
    'timestamp_format': '%Y%m%d_%H%M%S',
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'save_intermediate': False,
    'intermediate_dir': os.getenv('INTERMEDIATE_DIR', 'data_quality_reports/anomaly/intermediate')
}

# Categorical anomaly detection configuration
CATEGORICAL_CONFIG = {
    'min_freq': 0.05  # Minimum frequency threshold for rare category detection
}

def get_column_config(system_name: str, column_name: str) -> Optional[ColumnConfig]:
    """Get column configuration for a specific system and column."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs and column_name in system_config.column_configs:
            return system_config.column_configs[column_name]
    return None

def get_identifier_columns(system_name: str) -> List[str]:
    """Get identifier columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.is_identifier]
    return []

def get_grouping_columns(system_name: str) -> List[str]:
    """Get grouping columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.is_grouping]
    return []

def get_timestamp_columns(system_name: str) -> List[str]:
    """Get timestamp columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.is_timestamp]
    return []

def get_numeric_columns(system_name: str) -> List[str]:
    """Get numeric columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.dtype == "numeric"]
    return []

def get_cpp_columns(system_name: str) -> List[str]:
    """Get CPP columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.category == "CPP"]
    return []

def get_cqa_columns(system_name: str) -> List[str]:
    """Get CQA columns for a system."""
    if system_name in SYSTEMS_CONFIG:
        system_config = SYSTEMS_CONFIG[system_name]
        if system_config.column_configs:
            return [col for col, config in system_config.column_configs.items() 
                   if config.category == "CQA"]
    return []

def detect_field_category(field_name: str, system_name: str = None) -> str:
    """Detect field category using only column config (no auto-detection)."""
    # Only use column config - no auto-detection
    if system_name:
        column_config = get_column_config(system_name, field_name)
        if column_config and column_config.category != "auto":
            return column_config.category
    
    # If no configuration found, default to CQA
    logger.warning(f"No category configuration found for field '{field_name}' in system '{system_name}'. Defaulting to CQA.")
    return "CQA"

# Generate anomaly rules dynamically based on column configs
def generate_anomaly_rules():
    """Generate anomaly detection rules based on column configurations."""
    rules = []
    for system_name, system_config in SYSTEMS_CONFIG.items():
        if not system_config.column_configs:
            continue
        # Get numeric columns for statistical detection
        numeric_columns = get_numeric_columns(system_name)
        if numeric_columns:
            rules.append({
                'system': system_name,
                'table': f'{system_name.lower()}_data',
                'field': numeric_columns,
                'window': DETECTION_CONFIG.window_size,
                'group_by': get_grouping_columns(system_name),
                'detection_method': 'univariate',
                'detection_types': ['iqr', 'zscore', 'trend', 'isolation_forest']
            })
        # Add categorical rules for frequency-based anomaly detection
        categorical_columns = [col for col, config in system_config.column_configs.items() if config.dtype == "categorical"]
        if categorical_columns:
            rules.append({
                'system': system_name,
                'table': f'{system_name.lower()}_data',
                'field': categorical_columns,
                'window': DETECTION_CONFIG.window_size,
                'group_by': get_grouping_columns(system_name),
                'detection_method': 'univariate',
                'detection_types': ['category_frequency']
            })
        # Add multivariate rules for feature groups
        if system_name in MULTIVARIATE_FEATURES:
            for feature_name, feature_group in MULTIVARIATE_FEATURES[system_name].items():
                available_features = [f for f in feature_group if f in system_config.column_configs]
                if len(available_features) >= 2:
                    rules.append({
                        'system': system_name,
                        'table': f'{system_name.lower()}_data',
                        'field': available_features,
                        'window': DETECTION_CONFIG.window_size,
                        'group_by': get_grouping_columns(system_name),
                        'detection_method': 'multivariate',
                        'detection_types': ['isolation_forest'],
                        'feature_group': feature_name
                    })
    return rules

# Generate the anomaly rules
ANOMALY_RULES = generate_anomaly_rules()

# Update ML_CONFIG to match DETECTION_CONFIG contamination
ML_CONFIG.contamination = DETECTION_CONFIG.contamination 