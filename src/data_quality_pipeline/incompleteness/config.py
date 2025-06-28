from dataclasses import dataclass
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ColumnConfig:
    severity: str  # High, Medium, Low
    is_identifier: bool = False
    is_required: bool = False

@dataclass
class SystemConfig:
    name: str
    connection_string: str
    tables: List[str]
    completeness_threshold: float = 0.95
    column_configs: Dict[str, ColumnConfig] = None

@dataclass
class PipelineConfig:
    systems: Dict[str, SystemConfig]
    max_workers: int = 4

# System-specific configurations
SYSTEMS_CONFIG = {
    "LIMS": SystemConfig(
        name="LIMS",
        connection_string=os.getenv("LIMS_CONNECTION_STRING"),
        tables=["lims_data"],
        completeness_threshold=0.98,
        column_configs={
            "Lab_Sample_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Batch_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Product_Name": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Sample_Type": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Sample_Login_Timestamp": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Sample_Status": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Test_Name": ColumnConfig(severity="High", is_identifier=False, is_required=True),
            "Test_Method_ID": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Instrument_ID": ColumnConfig(severity="Medium", is_identifier=False, is_required=False),
            "Spec_Limit_Min": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Spec_Limit_Max": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Result_Value": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Result_UOM": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Result_Status": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Analyst_ID": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Result_Entry_Timestamp": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Reviewed_By": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Approved_By": ColumnConfig(severity="Low", is_identifier=False, is_required=True)
        }

    ),
    "MES": SystemConfig(
        name="MES",
        connection_string=os.getenv("MES_CONNECTION_STRING"),
        tables=["mes_data"],
        completeness_threshold=0.95,
        column_configs={
            "Work_Order_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Batch_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Product_Code": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Master_Recipe_ID": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Batch_Phase": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Phase_Step": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Parameter_Name": ColumnConfig(severity="High", is_identifier=False, is_required=True),
            "Parameter_Value": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Performed_By": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Verified_By": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Execution_Timestamp": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Execution_Status": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Equipment_ID": ColumnConfig(severity="Medium", is_identifier=False, is_required=True)
        }
    ),
    "QMS": SystemConfig(
        name="QMS",
        connection_string=os.getenv("QMS_CONNECTION_STRING"),
        tables=["qms_data"],
        completeness_threshold=0.97,
       column_configs = {
            "Record_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Record_Type": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Title": ColumnConfig(severity="High", is_identifier=False, is_required=True),
            "Description": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Batch_ID": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Product_Code": ColumnConfig(severity="High", is_identifier=True, is_required=True),
            "Status_Workflow": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Owner_Name": ColumnConfig(severity="Low", is_identifier=False, is_required=True),
            "Open_Date": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Due_Date": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Closure_Date": ColumnConfig(severity="Medium", is_identifier=False, is_required=True),
            "Source_Event_ID": ColumnConfig(severity="High", is_identifier=False, is_required=True)
        }

    )
}

# Incompleteness-specific configurations
INCOMPLETENESS_CONFIG = {
    "output_dir": "data_quality_reports/incompleteness",
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