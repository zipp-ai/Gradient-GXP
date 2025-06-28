from dataclasses import dataclass
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ColumnConfig:
    is_identifier: bool = False
    is_required: bool = False

@dataclass
class SystemConfig:
    name: str
    connection_string: str
    tables: List[str]
    consistency_threshold: float = 0.90
    column_configs: Dict[str, ColumnConfig] = None

@dataclass
class PipelineConfig:
    systems: Dict[str, SystemConfig]
    max_workers: int = 4

# System-specific configurations (only identifier columns listed)
SYSTEMS_CONFIG = {
    "LIMS": SystemConfig(
        name="LIMS",
        connection_string=os.getenv("LIMS_CONNECTION_STRING"),
        tables=["lims_data"],
        consistency_threshold=0.90,
        column_configs={
            "Lab_Sample_ID": ColumnConfig(is_identifier=True),
            "Batch_ID": ColumnConfig(is_identifier=True),
            "Product_Name": ColumnConfig(is_identifier=True)
        }
    ),
    "MES": SystemConfig(
        name="MES",
        connection_string=os.getenv("MES_CONNECTION_STRING"),
        tables=["mes_data"],
        consistency_threshold=0.90,
        column_configs={
            "Work_Order_ID": ColumnConfig(is_identifier=True),
            "Batch_ID": ColumnConfig(is_identifier=True),
            "Product_Code": ColumnConfig(is_identifier=True)
        }
    ),
    "QMS": SystemConfig(
        name="QMS",
        connection_string=os.getenv("QMS_CONNECTION_STRING"),
        tables=["qms_data"],
        consistency_threshold=0.90,
        column_configs={
            "Record_ID": ColumnConfig(is_identifier=True),
            "Batch_ID": ColumnConfig(is_identifier=True),
            "Product_Code": ColumnConfig(is_identifier=True)
        }
    )
}

# Field mappings for cross-system comparison
FIELD_MAPPINGS = [
    {
        "source_system": "QMS",
        "target_system": "LIMS",
        "source_table": "qms_data",
        "target_table": "lims_data",
        "field_map": {
             "Owner_Name": "Reviewed_By",
             "Owner_Name": "Approved_By"
        }
    }
    # {
    #     "source_system": "MES",
    #     "target_system": "LIMS",
    #     "source_table": "mes_data",
    #     "target_table": "lims_data",
    #     "field_map": {
    #         "CPP1_Name": "Test_Name",
    #         "CPP1_Value": "Test_Result_Value",
    #         "CPP1_UOM": "Test_Result_UOM",
    #         "CPP1_Status": "Test_Status",
    #         "CPP2_Name": "Test_Name",
    #         "CPP2_Value": "Test_Result_Value",
    #         "CPP2_UOM": "Test_Result_UOM",
    #         "CPP2_Status": "Test_Status"
    #     }
    # }
    # Add more mappings for other system pairs as needed
]

# Inconsistency-specific configurations
INCONSISTENCY_CONFIG = {
    "output_dir": "data_quality_reports/inconsistency",
    "report_format": "csv",
    "severity_weights": {
        "High": 1.0,
        "Medium": 0.7,
        "Low": 0.3
    }
}

def get_identifier_columns(system_config) -> list:
    """
    Return a list of column names that are marked as identifier columns in the given SystemConfig.
    """
    if hasattr(system_config, 'column_configs') and system_config.column_configs:
        return [col for col, config in system_config.column_configs.items() if getattr(config, 'is_identifier', False)]
    return [] 