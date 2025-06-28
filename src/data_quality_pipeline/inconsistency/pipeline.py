import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from .checker import CrossSystemConsistencyChecker
from .config import FIELD_MAPPINGS, PipelineConfig
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InconsistencyPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.findings = []
        self.output_dir = "data_quality_reports/inconsistency"
        self.systems_data = {}  # Initialize systems_data as instance variable
        self.finding_index = 1  # Initialize finding index starting from 1
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_finding_id(self, batch_id: str) -> str:
        """Generate a finding ID using batch ID and index"""
        finding_id = f"INS-{batch_id}-{self.finding_index:04d}"
        self.finding_index += 1
        return finding_id
        
    def _load_system_data(self, system_name: str) -> Dict[str, pd.DataFrame]:
        """Load data from a specific system's tables"""
        system_config = self.config.systems[system_name]
        data = {}
        
        for table in system_config.tables:
            try:
                # In a real implementation, this would connect to the actual database
                # For now, we'll use a placeholder
                df = pd.read_sql(f"SELECT * FROM {table}", system_config.connection_string)
                data[table] = df
            except Exception as e:
                logger.error(f"Error loading data from {system_name}.{table}: {str(e)}")
                # Return empty DataFrame instead of None
                data[table] = pd.DataFrame()
                
        return data
    
    def _save_findings_to_csv(self, findings_df):
        """Save findings to a CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"inconsistency_{timestamp}.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Select and reorder columns (remove 'severity' and 'inconsistent_values')
        columns_to_save = [
            'finding_id', 'system', 'description', 'timestamp'
        ]
        columns_to_save += [col for col in ['row_identifiers', 'identifiers', 'table_name', 'column_name', 'source_field', 'target_field', 'source_value', 'target_value'] if col in findings_df.columns]
        findings_df = findings_df[columns_to_save]
        findings_df.to_csv(output_file, index=False)
        logger.info(f"Saved inconsistency findings to {output_file}")
        logger.info(f"Total inconsistency findings: {len(findings_df)}")
    
    def run_pipeline(self) -> List[Dict]:
        """Execute the inconsistency pipeline"""
        start_time = datetime.now()
        batch_id = start_time.strftime("%Y%m%d%H%M%S")  # Use timestamp as batch ID
        
        self.systems_data = {}  # Reset systems_data
        for system_name in self.config.systems:
            self.systems_data[system_name] = self._load_system_data(system_name)
        
        # Run cross-system consistency checks
        if any(not df.empty for tables in self.systems_data.values() for df in tables.values()):
            consistency_checker = CrossSystemConsistencyChecker(
                self.systems_data, 
                self.config.systems, 
                FIELD_MAPPINGS,
                batch_id,
                self._generate_finding_id
            )
            self.findings.extend(consistency_checker.check_inconsistencies())
        
        findings_df = pd.DataFrame(self.findings)
        # Convert dict columns to string for deduplication
        if 'row_identifiers' in findings_df.columns:
            findings_df['row_identifiers'] = findings_df['row_identifiers'].apply(str)
        if 'identifiers' in findings_df.columns:
            findings_df['identifiers'] = findings_df['identifiers'].apply(str)
        findings_df = findings_df.drop_duplicates()
        if not findings_df.empty:
            self._save_findings_to_csv(findings_df)
        
        return self.findings
    
    def get_findings_summary(self) -> pd.DataFrame:
        """Return a summary of findings in a DataFrame format"""
        if not self.findings:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.findings)
        # Sort by system and timestamp
        return df.sort_values(['system', 'timestamp'], ascending=[True, False]) 