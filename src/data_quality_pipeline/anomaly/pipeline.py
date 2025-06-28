"""
Anomaly detection pipeline module.
"""
import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from .checker import AnomalyChecker
from .config import (
    SYSTEMS_CONFIG, ANOMALY_RULES, OUTPUT_CONFIG, 
    DERIVED_COLUMNS, get_grouping_columns, get_timestamp_columns,
    get_column_config, detect_field_category
)
from .checker import (
    auto_detect_column_type, is_numeric_field, is_categorical_field, is_timestamp_field
)

logger = logging.getLogger(__name__)

class AnomalyPipeline:
    def __init__(self):
        self.checker = AnomalyChecker()
        self.output_dir = OUTPUT_CONFIG['output_dir']
        self.timestamp = datetime.now().strftime(OUTPUT_CONFIG['timestamp_format'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize finding ID counter
        self.finding_id_counter = 0

    def _generate_finding_id(self) -> str:
        """Generate a unique finding ID."""
        self.finding_id_counter += 1
        return f"ANOM_{self.timestamp}_{self.finding_id_counter:06d}"

    def _standardize_finding(self, finding: Dict) -> Dict:
        """Standardize a finding to ensure consistent format."""
        # Extract technical details that might be directly in the finding
        technical_details = {}
        for key in ['deviation', 'method', 'anomaly_score', 'feature_group', 'iqr_multiplier', 
                   'threshold_lower', 'threshold_upper', 'q1', 'q3', 'iqr', 'z_score', 'threshold', 
                   'mean', 'std', 'window_size', 'trend_threshold', 'pattern_threshold', 
                   'contamination', 'multivariate_fields']:
            if key in finding:
                technical_details[key] = finding.pop(key)
        
        # Build row_identifier string from identifier columns
        row_identifier = ''
        if 'system' in finding:
            system_name = finding['system']
            if system_name in SYSTEMS_CONFIG:
                system_config = SYSTEMS_CONFIG[system_name]
                if system_config.column_configs:
                    id_cols = [col for col, config in system_config.column_configs.items() if getattr(config, 'is_identifier', False)]
                    id_parts = []
                    for col in id_cols:
                        if col in finding:
                            id_parts.append(f"{col}={finding[col]}")
                    row_identifier = ','.join(id_parts)
        
        standardized = {
            'finding_id': self._generate_finding_id(),
            'system': finding.get('system', ''),
            'anomaly_type': finding.get('anomaly_type', 'outlier'),
            'field': finding.get('field', ''),
            'value': finding.get('value', ''),
            'timestamp': finding.get('timestamp', datetime.now()),
            'group': finding.get('group', ''),
            'row_identifier': row_identifier,
            'category': finding.get('category', 'CQA'),
            'technical_details': technical_details
        }
        
        # Ensure technical_details is a dictionary
        if not isinstance(standardized['technical_details'], dict):
            standardized['technical_details'] = {}
            
        return standardized

    def _load_data_from_database(self, system_config) -> Dict[str, pd.DataFrame]:
        """Load data from database for a specific system."""
        data = {}
        
        for table in system_config.tables:
            try:
                # Use the same pattern as inconsistency module
                df = pd.read_sql(f"SELECT * FROM {table}", system_config.connection_string)
                logger.info(f"Loaded {len(df)} rows from {table}")
                logger.info(f"Columns in {table}: {list(df.columns)}")
                
                # Auto-detect and convert column types based on actual data
                df = self._auto_detect_and_convert_columns(df, table, system_config.name)
                
                # Compute derived columns
                df = self._compute_derived_columns(df, table)
                
                data[table] = df
                
            except Exception as e:
                logger.error(f"Error loading data from {system_config.name}.{table}: {str(e)}")
                # Return empty DataFrame instead of None (same as inconsistency module)
                data[table] = pd.DataFrame()
                
        return data

    def _auto_detect_and_convert_columns(self, df: pd.DataFrame, table: str, system_name: str) -> pd.DataFrame:
        """Auto-detect and convert column types based on actual data and column configs."""
        df_copy = df.copy()
        
        for column in df_copy.columns:
            # Get column configuration if available
            column_config = get_column_config(system_name, column)
            
            if column_config and column_config.dtype != "auto":
                # Use explicit configuration
                target_dtype = column_config.dtype
            else:
                # Auto-detect from actual data
                target_dtype = auto_detect_column_type(df_copy[column])
                logger.info(f"Auto-detected {column} as {target_dtype} in {table}")
            
            # Convert column based on detected/configured type
            try:
                if target_dtype == "numeric":
                    df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
                elif target_dtype == "timestamp":
                    df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
                elif target_dtype == "categorical":
                    df_copy[column] = df_copy[column].astype(str)
                
                logger.info(f"Converted {column} to {target_dtype} in {table}")
                
            except Exception as e:
                logger.warning(f"Failed to convert {column} to {target_dtype}: {str(e)}")
                    
        return df_copy

    def _compute_derived_columns(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Compute derived columns based on configuration."""
        if table not in DERIVED_COLUMNS:
            return df
            
        df_copy = df.copy()
        derived_config = DERIVED_COLUMNS[table]
        
        for col_name, config in derived_config.items():
            try:
                formula = config['formula']
                logger.info(f"Computing derived column: {col_name} = {formula}")
                
                # Handle different types of derived columns
                if config['type'] == 'duration':
                    df_copy[col_name] = self._compute_duration(df_copy, formula)
                elif config['type'] == 'ratio':
                    df_copy[col_name] = self._compute_ratio(df_copy, formula)
                elif config['type'] == 'efficiency':
                    df_copy[col_name] = self._compute_efficiency(df_copy, formula)
                else:
                    df_copy[col_name] = self._evaluate_formula(df_copy, formula)
                    
                # Convert to appropriate type based on config
                if 'dtype' in config:
                    if config['dtype'] == 'numeric':
                        df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce')
                    elif config['dtype'] == 'timestamp':
                        df_copy[col_name] = pd.to_datetime(df_copy[col_name], errors='coerce')
                
                logger.info(f"Successfully computed {col_name}")
                
            except Exception as e:
                logger.error(f"Error computing derived column {col_name}: {str(e)}")
                continue
                
        return df_copy

    def _compute_duration(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute duration between two timestamp columns."""
        parts = formula.split(' - ')
        if len(parts) != 2:
            raise ValueError(f"Invalid duration formula: {formula}")
            
        end_col = parts[0].strip()
        start_col = parts[1].strip()
        
        if end_col not in df.columns or start_col not in df.columns:
            logger.warning(f"Duration columns not found: {start_col}, {end_col}")
            return pd.Series([pd.NaT] * len(df))
            
        end_times = pd.to_datetime(df[end_col], errors='coerce')
        start_times = pd.to_datetime(df[start_col], errors='coerce')
        
        duration = (end_times - start_times).dt.total_seconds() / 3600  # Convert to hours
        return duration

    def _compute_ratio(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute ratio between two numeric columns."""
        parts = formula.split(' / ')
        if len(parts) != 2:
            raise ValueError(f"Invalid ratio formula: {formula}")
            
        numerator_col = parts[0].strip()
        denominator_col = parts[1].strip()
        
        if numerator_col not in df.columns or denominator_col not in df.columns:
            logger.warning(f"Ratio columns not found: {numerator_col}, {denominator_col}")
            return pd.Series([pd.NA] * len(df))
            
        numerator = pd.to_numeric(df[numerator_col], errors='coerce')
        denominator = pd.to_numeric(df[denominator_col], errors='coerce')
        
        ratio = numerator / denominator.replace(0, pd.NA)
        return ratio

    def _compute_efficiency(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute efficiency ratio."""
        return self._compute_ratio(df, formula)

    def _evaluate_formula(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Evaluate a generic formula using pandas eval."""
        try:
            result = df.eval(formula)
            return result
        except Exception as e:
            logger.error(f"Error evaluating formula {formula}: {str(e)}")
            return pd.Series([pd.NA] * len(df))

    def _create_row_identifier(self, row: pd.Series, system_name: str) -> Dict[str, str]:
        """Create a row identifier dictionary using configuration."""
        # Get identifier columns for the system
        identifier_columns = []
        if system_name in SYSTEMS_CONFIG:
            system_config = SYSTEMS_CONFIG[system_name]
            if system_config.column_configs:
                identifier_columns = [col for col, config in system_config.column_configs.items() 
                                   if config.is_identifier]
        
        row_id = {}
        for col in identifier_columns:
            if col in row:
                row_id[col] = str(row[col])
            else:
                row_id[col] = ''
                
        return row_id

    def _infer_category(self, field: str, system_name: str) -> str:
        """Infer the category of a field using column config or auto-detection."""
        return detect_field_category(field, system_name)

    def run(self) -> List[Dict]:
        """Run the anomaly detection pipeline."""
        all_findings = []
        
        # Reset cache at the beginning of each pipeline run to prevent duplicates
        self.checker.reset_cache()
        
        logger.info("Starting anomaly detection pipeline")
        logger.info(f"Processing {len(SYSTEMS_CONFIG)} systems: {list(SYSTEMS_CONFIG.keys())}")
        
        for system_name, system_config in SYSTEMS_CONFIG.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing system: {system_name}")
            logger.info(f"{'='*50}")
            
            # Load data
            data = self._load_data_from_database(system_config)
            if not data:
                logger.warning(f"No data loaded for {system_name}")
                continue
                
            # Get rules for this system
            system_rules = [rule for rule in ANOMALY_RULES if rule['system'] == system_name]
            if not system_rules:
                logger.warning(f"No rules defined for {system_name}")
                continue
                
            logger.info(f"Found {len(system_rules)} rules for {system_name}")
            
            # Process each rule
            for rule_idx, rule in enumerate(system_rules):
                table = rule['table']
                if table not in data:
                    logger.warning(f"Table {table} not found in data for {system_name}")
                    continue
                    
                logger.info(f"\nProcessing rule {rule_idx + 1}/{len(system_rules)}:")
                logger.info(f"  Table: {table}")
                logger.info(f"  Fields: {rule['field']}")
                logger.info(f"  Detection method: {rule.get('detection_method', 'univariate')}")
                logger.info(f"  Detection types: {rule.get('detection_types', [])}")
                
                # Check for anomalies
                findings = self.checker.check_anomalies(data[table], rule)
                
                # Process findings
                for finding in findings:
                    finding['system'] = system_name
                    # Add row identifier
                    if 'row_index' in finding:
                        try:
                            row = data[table].iloc[finding['row_index']]
                            row_id = self._create_row_identifier(row, system_name)
                            # Add all identifier columns
                            for col, value in row_id.items():
                                finding[col] = value
                        except IndexError:
                            logger.warning(f"Row index {finding['row_index']} out of bounds for table {table}")
                            continue
                    # Add category using column config or auto-detection
                    finding['category'] = self._infer_category(finding['field'], system_name)
                    # Standardize the finding
                    standardized_finding = self._standardize_finding(finding)
                    all_findings.append(standardized_finding)
                
                logger.info(f"  Found {len(findings)} anomalies for this rule")
            
            logger.info(f"Total findings for {system_name}: {len([f for f in all_findings if f['system'] == system_name])}")
        
        # Save findings
        self.save_findings(all_findings)
        
        logger.info(f"\n{'='*50}")
        logger.info("Pipeline Summary")
        logger.info(f"{'='*50}")
        logger.info(f"Total findings: {len(all_findings)}")
        
        # Group by system
        system_counts = pd.Series([f['system'] for f in all_findings]).value_counts()
        logger.info("\nFindings by System:")
        for system, count in system_counts.items():
            logger.info(f"  {system}: {count}")
            
        # Group by detection method
        method_counts = pd.Series([f.get('method', 'unknown') for f in all_findings]).value_counts()
        logger.info("\nFindings by Detection Method:")
        for method, count in method_counts.items():
            logger.info(f"  {method}: {count}")
            
        # Group by anomaly type
        type_counts = pd.Series([f['anomaly_type'] for f in all_findings]).value_counts()
        logger.info("\nFindings by Anomaly Type:")
        for anomaly_type, count in type_counts.items():
            logger.info(f"  {anomaly_type}: {count}")
        
        return all_findings

    def save_findings(self, findings: List[Dict]) -> None:
        """Save findings to CSV files with standardized format, including deviation column."""
        if not findings:
            logger.info("No findings to save")
            return
        
        logger.info(f"\nSaving {len(findings)} findings to CSV files...")
        
        # Define the standard column order
        base_columns = [
            'finding_id',
            'system',
            'anomaly_type',
            'field',
            'value',
            'timestamp',
            'group',
            'row_identifier',
            'category'
        ]
        
        # Flatten technical details to include deviation and other key metrics
        for f in findings:
            tech = f.pop('technical_details', {})
            if isinstance(tech, dict):
                f['deviation'] = tech.get('deviation', None)
                f['method'] = tech.get('method', None)
                f['anomaly_score'] = tech.get('anomaly_score', None)
                f['feature_group'] = tech.get('feature_group', None)
        
        # Final column order
        columns = base_columns + ['deviation', 'method', 'anomaly_score', 'feature_group']
        
        # Separate findings by category
        cpp_findings = [f for f in findings if f.get('category') == 'CPP']
        cqa_findings = [f for f in findings if f.get('category') == 'CQA']
        
        # Export CPP findings
        if cpp_findings:
            cpp_df = pd.DataFrame(cpp_findings)
            cpp_df = cpp_df.reindex(columns=columns)  # Ensure consistent column order
            cpp_output_path = os.path.join(self.output_dir, f'anomaly_findings_cpp_{self.timestamp}.csv')
            cpp_df.to_csv(cpp_output_path, index=False)
            logger.info(f"CPP findings exported to: {cpp_output_path}")
            logger.info(f"CPP findings columns: {list(cpp_df.columns)}")
            logger.info(f"CPP findings shape: {cpp_df.shape}")
        
        # Export CQA findings
        if cqa_findings:
            cqa_df = pd.DataFrame(cqa_findings)
            cqa_df = cqa_df.reindex(columns=columns)  # Ensure consistent column order
            cqa_output_path = os.path.join(self.output_dir, f'anomaly_findings_cqa_{self.timestamp}.csv')
            cqa_df.to_csv(cqa_output_path, index=False)
            logger.info(f"CQA findings exported to: {cqa_output_path}")
            logger.info(f"CQA findings columns: {list(cqa_df.columns)}")
            logger.info(f"CQA findings shape: {cqa_df.shape}")
            
        # Export combined findings
        combined_df = pd.DataFrame(findings)
        combined_df = combined_df.reindex(columns=columns)
        combined_output_path = os.path.join(self.output_dir, f'anomaly_findings_combined_{self.timestamp}.csv')
        combined_df.to_csv(combined_output_path, index=False)
        logger.info(f"Combined findings exported to: {combined_output_path}")
        
        logger.info("All findings saved successfully")

if __name__ == "__main__":
    pipeline = AnomalyPipeline()
    findings = pipeline.run()
    print("\nPipeline completed successfully") 