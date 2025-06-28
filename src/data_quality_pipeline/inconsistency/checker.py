import pandas as pd
from typing import List, Dict, Callable
from datetime import datetime
import logging
from .config import FIELD_MAPPINGS, SystemConfig, get_identifier_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_identifier_columns(system_config) -> list:
    if hasattr(system_config, 'column_configs') and system_config.column_configs:
        return [col for col, config in system_config.column_configs.items() if getattr(config, 'is_identifier', False)]
    return []

class CrossSystemConsistencyChecker:
    def __init__(self, systems_data: Dict[str, Dict[str, pd.DataFrame]], 
                 systems_config: Dict[str, SystemConfig],
                 field_mappings: List[Dict],
                 batch_id: str,
                 generate_finding_id: Callable[[str], str]):
        self.systems_data = systems_data
        self.systems_config = systems_config
        self.field_mappings = field_mappings
        self.batch_id = batch_id
        self.generate_finding_id = generate_finding_id

    def check_inconsistencies(self) -> List[Dict]:
        """
        Comprehensive method to check for inconsistencies between systems, handling both mapped fields
        and common fields across systems.
        """
        findings = []
        
        # Process each system pair
        for mapping in FIELD_MAPPINGS:
            source_system = mapping['source_system']
            target_system = mapping['target_system']
            source_table = mapping['source_table']
            target_table = mapping['target_table']
            field_map = mapping['field_map']
            
            # Get data
            source_data = self.systems_data.get(source_system, {}).get(source_table)
            target_data = self.systems_data.get(target_system, {}).get(target_table)
            
            print(f"[DEBUG] Source: {source_system}.{source_table} shape: {None if source_data is None else source_data.shape}")
            print(f"[DEBUG] Target: {target_system}.{target_table} shape: {None if target_data is None else target_data.shape}")
            
            if source_data is None or target_data is None or source_data.empty or target_data.empty:
                print(f"[DEBUG] Skipping {source_system}-{target_system} due to missing/empty data.")
                continue
                
            # Get identifier columns common to both systems
            id_cols_1 = set(get_identifier_columns(self.systems_config[source_system]))
            id_cols_2 = set(get_identifier_columns(self.systems_config[target_system]))
            common_id_cols = list(id_cols_1 & id_cols_2)
            
            print(f"[DEBUG] Common identifier columns for {source_system}-{target_system}: {common_id_cols}")
            
            if not common_id_cols:
                print(f"[DEBUG] No common identifier columns for {source_system}-{target_system}.")
                continue
                
            # Merge on identifiers
            merged = pd.merge(
                source_data,
                target_data,
                left_on=common_id_cols,
                right_on=common_id_cols,
                suffixes=(f'_{source_system}', f'_{target_system}')
            )
            
            print(f"[DEBUG] Merged rows for {source_system}-{target_system}: {merged.shape[0]}")
            
            # Check mapped fields
            for _, row in merged.iterrows():
                for source_field, target_field in field_map.items():
                    # Try suffixed, then unsuffixed
                    source_col_suffixed = f'{source_field}_{source_system}'
                    target_col_suffixed = f'{target_field}_{target_system}'
                    val1_raw = row.get(source_col_suffixed) if source_col_suffixed in row else row.get(source_field)
                    val2_raw = row.get(target_col_suffixed) if target_col_suffixed in row else row.get(target_field)
                    
                    # Normalize: convert to string, strip, lower
                    val1 = str(val1_raw).strip().lower() if val1_raw is not None else ''
                    val2 = str(val2_raw).strip().lower() if val2_raw is not None else ''
                    
                    if pd.isna(val1_raw) and pd.isna(val2_raw):
                        continue
                        
                    if (pd.isna(val1_raw) != pd.isna(val2_raw)) or (val1 != val2):
                        row_identifiers = {id_col: row[id_col] for id_col in common_id_cols}
                        findings.append({
                            'finding_id': self.generate_finding_id(self.batch_id),
                            'type': 'Data Inconsistency',
                            'system': f'{source_system} ↔ {target_system}',
                            'description': f"Value mismatch for mapped fields '{source_field}' ↔ '{target_field}': {source_system}='{val1_raw}', {target_system}='{val2_raw}'",
                            'table_name': f'{source_table} ↔ {target_table}',
                            'column_name': f'{source_field} ↔ {target_field}',
                            'row_identifiers': row_identifiers,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Check common fields (fields with same name in both systems)
            common_fields = set(source_data.columns) & set(target_data.columns)
            common_fields = [f for f in common_fields if f not in common_id_cols]  # Exclude identifier columns
            
            for field in common_fields:
                source_col = f'{field}_{source_system}'
                target_col = f'{field}_{target_system}'
                
                # Get unique combinations of identifiers and their values
                value_pairs = merged[common_id_cols + [source_col, target_col]].drop_duplicates()
                
                # Find rows where values don't match
                mismatches = value_pairs[value_pairs[source_col] != value_pairs[target_col]]
                
                for _, row in mismatches.iterrows():
                    val1_raw = row[source_col]
                    val2_raw = row[target_col]
                    
                    if pd.isna(val1_raw) and pd.isna(val2_raw):
                        continue
                        
                    row_identifiers = {id_col: row[id_col] for id_col in common_id_cols}
                    findings.append({
                        'finding_id': self.generate_finding_id(self.batch_id),
                        'type': 'Data Inconsistency',
                        'system': f'{source_system} ↔ {target_system}',
                        'description': f"Value mismatch for common field '{field}': {source_system}='{val1_raw}', {target_system}='{val2_raw}'",
                        'table_name': f'{source_table} ↔ {target_table}',
                        'column_name': field,
                        'row_identifiers': row_identifiers,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return findings 