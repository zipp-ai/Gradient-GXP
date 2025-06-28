import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from .checker import IncompletenessChecker
from .config import SYSTEMS_CONFIG, PipelineConfig, get_identifier_columns
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncompletenessPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.findings = []
        self.output_dir = "data_quality_reports/incompleteness"
        self.systems_data = {}  # Initialize systems_data as instance variable
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_finding_id(self, batch_id: str, index: int) -> str:
        """Generate a finding ID using batch ID and index"""
        return f"INC-{batch_id}-{index:04d}"
        
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
    
    def _save_findings_to_csv(self, findings_df: pd.DataFrame):
        """Save findings to a CSV file with timestamp"""
        if findings_df.empty:
            logger.info("No incompleteness findings to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incompleteness_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure consistent columns
        findings_df = findings_df[[
            'finding_id', 'system', 'table', 'column', 
            'description', 'severity', 'timestamp', 'row_identifiers'
        ]]
        
        findings_df.to_csv(filepath, index=False)
        logger.info(f"Saved incompleteness findings to {filepath}")
        logger.info(f"Total incompleteness findings: {len(findings_df)}")
    
    def _cross_system_identifier_check(self, batch_id: str) -> List[Dict]:
        findings = []
        system_names = list(self.config.systems.keys())
        # For each pair of systems
        for i in range(len(system_names)):
            for j in range(i + 1, len(system_names)):
                sys1 = system_names[i]
                sys2 = system_names[j]
                config1 = self.config.systems[sys1]
                config2 = self.config.systems[sys2]
                df1 = next(iter(self.systems_data[sys1].values())) if self.systems_data[sys1] else None
                df2 = next(iter(self.systems_data[sys2].values())) if self.systems_data[sys2] else None
                if df1 is None or df2 is None or df1.empty or df2.empty:
                    continue
                ids1 = set(get_identifier_columns(config1))
                ids2 = set(get_identifier_columns(config2))
                common_ids = ids1 & ids2
                for id_col in common_ids:
                    vals1 = set(df1[id_col].dropna().unique())
                    vals2 = set(df2[id_col].dropna().unique())
                    missing_in_2 = vals1 - vals2
                    missing_in_1 = vals2 - vals1
                    for val in missing_in_2:
                        findings.append({
                            'type': 'Identifier Incompleteness',
                            'system': f'{sys1} → {sys2}',
                            'description': f'{id_col} {val} present in {sys1} but missing in {sys2}',
                            'severity': 'High',
                            'table': '',
                            'column': id_col,
                            'row_identifiers': {id_col: str(val)},
                            'timestamp': datetime.now().isoformat()
                        })
                    for val in missing_in_1:
                        findings.append({
                            'type': 'Identifier Incompleteness',
                            'system': f'{sys2} → {sys1}',
                            'description': f'{id_col} {val} present in {sys2} but missing in {sys1}',
                            'severity': 'High',
                            'table': '',
                            'column': id_col,
                            'row_identifiers': {id_col: str(val)},
                            'timestamp': datetime.now().isoformat()
                        })
        return findings

    def run_pipeline(self) -> List[Dict]:
        """Execute the incompleteness pipeline"""
        start_time = datetime.now()
        batch_id = start_time.strftime("%Y%m%d%H%M%S")  # Use timestamp as batch ID
        
        # Load data from all systems
        self.systems_data = {}  # Reset systems_data
        for system_name in self.config.systems:
            self.systems_data[system_name] = self._load_system_data(system_name)
        
        # Run completeness checks in parallel
        all_findings = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            quality_futures = []
            for system_name, tables_data in self.systems_data.items():
                checker = IncompletenessChecker(system_name, self.config.systems[system_name])
                for table_name, df in tables_data.items():
                    if not df.empty:  # Only check non-empty DataFrames
                        # Run completeness check
                        future = executor.submit(checker.check_completeness, df, table_name, batch_id)
                        quality_futures.append(future)
            # Collect all findings
            for future in quality_futures:
                all_findings.extend(future.result())
        # Add cross-system identifier presence findings
        all_findings.extend(self._cross_system_identifier_check(batch_id))
        # Assign finding IDs in a single thread-safe loop
        for idx, finding in enumerate(all_findings, start=1):
            finding['finding_id'] = self._generate_finding_id(batch_id, idx)
        self.findings = all_findings
        # Save findings to CSV
        findings_df = pd.DataFrame(self.findings)
        if not findings_df.empty:
            self._save_findings_to_csv(findings_df)
        return self.findings
    
    def get_findings_summary(self) -> pd.DataFrame:
        """Return a summary of findings in a DataFrame format"""
        if not self.findings:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.findings)
        return df.sort_values("severity", ascending=False) 