import pandas as pd
from typing import List, Dict
from datetime import datetime
from .config import get_identifier_columns

class IncompletenessChecker:
    def __init__(self, system_name: str, config: Dict):
        self.system_name = system_name
        self.config = config
        
    def _is_null_value(self, value) -> bool:
        """Check if a value is null or null-like"""
        # First check for pandas null values (np.nan, None, pd.NA)
        if pd.isna(value):
            return True
            
        # For string values, check for null-like strings
        if isinstance(value, str):
            # Convert to lowercase and strip whitespace for string comparison
            value = value.lower().strip()
            
            # Check for null-like strings
            null_like_values = {
                # Standard null representations
                "n/a", "na", "null", "none", "nil",
                # Empty or whitespace
                "", " ",
                # Common placeholders
                "-", "--", ".",
                # Additional variations
                "n.a.", "n.a", "n/a.", "na.", "null.", "none.",
                # String representations of NaN
                "nan"
            }
            
            # Check if the value is in our null set
            if value in null_like_values or value.isspace():
                return True
                
        return False
    
    def check_completeness(self, df: pd.DataFrame, table_name: str, batch_id: str) -> List[Dict]:
        findings = []
        identifier_columns = get_identifier_columns(self.config)
        
        for column in df.columns:
            column_config = self.config.column_configs.get(column)
            if not column_config or not column_config.is_required:
                continue
            # Work on a copy, do not modify df in place!
            col_values = df[column].copy()
            null_mask = col_values.apply(self._is_null_value)
            missing_rows = df[null_mask]

            # # Debug: print null counts and unique null values
            # actual_null_count = col_values.isna().sum()
            # detected_null_count = null_mask.sum()
            # print(f"[DEBUG] {table_name}.{column}: pandas isna() count = {actual_null_count}, custom null detection count = {detected_null_count}")
            # if detected_null_count != actual_null_count:
            #     print(f"[DEBUG] {table_name}.{column}: Values detected as null by custom logic but not by pandas:")
            #     print(col_values[null_mask & ~col_values.isna()].unique())

            for idx, row in missing_rows.iterrows():
                severity = column_config.severity
                row_identifiers = {id_col: str(row[id_col]) for id_col in identifier_columns if id_col in df.columns}
                findings.append({
                    # 'finding_id' will be assigned later in the pipeline
                    "type": "Data Incompleteness",
                    "system": self.system_name,
                    "description": f"Missing or null-like value in {column}",
                    "severity": severity,
                    "table": table_name,
                    "column": column,
                    "row_identifiers": row_identifiers,
                    "timestamp": datetime.now().isoformat()
                })
        return findings 