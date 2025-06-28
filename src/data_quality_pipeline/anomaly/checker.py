"""
Anomaly detection checker module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
import re
from .config import DETECTION_CONFIG, ML_CONFIG, DERIVED_COLUMNS, get_column_config

logger = logging.getLogger(__name__)

def auto_detect_column_type(series: pd.Series) -> str:
    """Auto-detect column data type from pandas series."""
    try:
        # Check if it's numeric
        pd.to_numeric(series, errors='raise')
        return "numeric"
    except (ValueError, TypeError):
        pass
    
    try:
        # Check if it's datetime
        pd.to_datetime(series, errors='raise')
        return "timestamp"
    except (ValueError, TypeError):
        pass
    
    # Default to categorical
    return "categorical"

def is_numeric_field(series: pd.Series, field_name: str, system_name: str = None) -> bool:
    """
    Check if a field is numeric using column config if available, then validate with pandas dtype.
    """
    # First check config if available
    if system_name:
        column_config = get_column_config(system_name, field_name)
        if column_config and getattr(column_config, 'dtype', 'auto') != "auto":
            config_dtype = column_config.dtype
            if config_dtype == "numeric":
                # Validate with pandas that the actual data is numeric
                return pd.api.types.is_numeric_dtype(series)
            else:
                # Config says it's not numeric, so it's not numeric
                return False
    
    # Fallback to pandas dtype inference if no config or config is "auto"
    return pd.api.types.is_numeric_dtype(series)

def is_timestamp_field(series: pd.Series, field_name: str, system_name: str = None) -> bool:
    """
    Check if a field is a timestamp using column config if available, then validate with pandas dtype.
    """
    # First check config if available
    if system_name:
        column_config = get_column_config(system_name, field_name)
        if column_config and getattr(column_config, 'dtype', 'auto') != "auto":
            config_dtype = column_config.dtype
            if config_dtype == "timestamp":
                # Validate with pandas that the actual data is datetime-like
                return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64_ns_dtype(series)
            else:
                # Config says it's not timestamp, so it's not timestamp
                return False
    
    # Fallback to pandas dtype inference if no config or config is "auto"
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64_ns_dtype(series)

class AnomalyChecker:
    def __init__(self):
        self.ml_models = {}
        self.anomaly_cache = {}  # Cache to track detected anomalies
        
    def reset_cache(self):
        """Reset the anomaly cache to prevent duplicates across different processing runs."""
        self.anomaly_cache.clear()
        logger.info("Anomaly cache reset")
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the current cache."""
        return {
            'cache_size': len(self.anomaly_cache),
            'unique_keys': len(set(self.anomaly_cache.keys()))
        }
    
    def _get_ml_model(self, key: str) -> IsolationForest:
        """Get or create ML model for a specific key."""
        if key not in self.ml_models:
            self.ml_models[key] = IsolationForest(
                n_estimators=ML_CONFIG.n_estimators,
                max_samples=ML_CONFIG.max_samples,
                random_state=ML_CONFIG.random_state,
                contamination=ML_CONFIG.contamination,
                max_features=ML_CONFIG.max_features,
                bootstrap=ML_CONFIG.bootstrap,
                n_jobs=ML_CONFIG.n_jobs,
                warm_start=ML_CONFIG.warm_start,
                verbose=ML_CONFIG.verbose
            )
        return self.ml_models[key]
    
    def _format_group_name(self, group_name: Optional[Union[str, tuple]]) -> str:
        """Format group name to handle tuples."""
        if isinstance(group_name, tuple):
            return '_'.join(str(x) for x in group_name)
        return str(group_name) if group_name is not None else ''
    
    def _is_duplicate_anomaly(self, finding: Dict) -> bool:
        """Check if an anomaly is a duplicate based on key attributes."""
        # Include method in the key to allow same row to have different detection methods
        method = finding.get('method', 'unknown')
        anomaly_type = finding.get('anomaly_type', 'outlier')
        key = (finding['system'], finding['field'], finding['group'], finding['row_index'], method, anomaly_type)
        if key in self.anomaly_cache:
            logger.debug(f"Duplicate anomaly detected: {key}")
            return True
        self.anomaly_cache[key] = True
        logger.debug(f"New anomaly added to cache: {key}")
        return False
    
    def _compute_derived_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Compute derived columns based on configuration."""
        if table_name not in DERIVED_COLUMNS:
            return df
            
        derived_config = DERIVED_COLUMNS[table_name]
        df_copy = df.copy()
        
        for col_name, config in derived_config.items():
            try:
                formula = config['formula']
                logger.info(f"Computing derived column: {col_name} = {formula}")
                
                # Handle different types of derived columns
                if config['type'] == 'duration':
                    # Duration calculation (timestamp subtraction)
                    df_copy[col_name] = self._compute_duration(df_copy, formula)
                elif config['type'] == 'ratio':
                    # Ratio calculation
                    df_copy[col_name] = self._compute_ratio(df_copy, formula)
                elif config['type'] == 'efficiency':
                    # Efficiency calculation
                    df_copy[col_name] = self._compute_efficiency(df_copy, formula)
                else:
                    # Generic formula evaluation
                    df_copy[col_name] = self._evaluate_formula(df_copy, formula)
                    
                # Convert to numeric if possible
                df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce')
                logger.info(f"Successfully computed {col_name}, data type: {df_copy[col_name].dtype}")
                
            except Exception as e:
                logger.error(f"Error computing derived column {col_name}: {str(e)}")
                continue
                
        return df_copy
    
    def _compute_duration(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute duration between two timestamp columns."""
        # Extract column names from formula like "End_Timestamp - Start_Timestamp"
        parts = formula.split(' - ')
        if len(parts) != 2:
            raise ValueError(f"Invalid duration formula: {formula}")
            
        end_col = parts[0].strip()
        start_col = parts[1].strip()
        
        if end_col not in df.columns or start_col not in df.columns:
            logger.warning(f"Duration columns not found: {start_col}, {end_col}")
            return pd.Series([np.nan] * len(df))
            
        # Convert to datetime and compute difference
        end_times = pd.to_datetime(df[end_col], errors='coerce')
        start_times = pd.to_datetime(df[start_col], errors='coerce')
        
        duration = (end_times - start_times).dt.total_seconds() / 3600  # Convert to hours
        return duration
    
    def _compute_ratio(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute ratio between two numeric columns."""
        # Extract column names from formula like "A / B"
        parts = formula.split(' / ')
        if len(parts) != 2:
            raise ValueError(f"Invalid ratio formula: {formula}")
            
        numerator_col = parts[0].strip()
        denominator_col = parts[1].strip()
        
        if numerator_col not in df.columns or denominator_col not in df.columns:
            logger.warning(f"Ratio columns not found: {numerator_col}, {denominator_col}")
            return pd.Series([np.nan] * len(df))
            
        # Convert to numeric and compute ratio
        numerator = pd.to_numeric(df[numerator_col], errors='coerce')
        denominator = pd.to_numeric(df[denominator_col], errors='coerce')
        
        # Avoid division by zero
        ratio = numerator / denominator.replace(0, np.nan)
        return ratio
    
    def _compute_efficiency(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Compute efficiency ratio."""
        return self._compute_ratio(df, formula)
    
    def _evaluate_formula(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Evaluate a generic formula using pandas eval."""
        try:
            # Replace column names with df references for safe evaluation
            safe_formula = formula
            for col in df.columns:
                if col in formula:
                    safe_formula = safe_formula.replace(col, f"df['{col}']")
            
            # Use pandas eval for safe evaluation
            result = df.eval(safe_formula)
            return result
        except Exception as e:
            logger.error(f"Error evaluating formula {formula}: {str(e)}")
            return pd.Series([np.nan] * len(df))
    
    def _create_anomaly_finding(self, df: pd.DataFrame, idx: int, field: str, system: str, 
                              group_name: Optional[str] = None, anomaly_type: str = 'outlier',
                              technical_details: Optional[Dict] = None) -> Dict:
        """Create a standardized anomaly finding."""
        try:
            row = df.loc[idx]
            finding = {
                'anomaly_type': anomaly_type,
                'system': system,
                'field': field,
                'value': float(row[field]) if pd.api.types.is_numeric_dtype(row[field]) else str(row[field]),
                'timestamp': row.get('Timestamp', datetime.now()),
                'group': self._format_group_name(group_name),
                'row_index': idx
            }
            
            # Add technical details if provided
            if technical_details:
                finding.update(technical_details)
                
            return finding
        except Exception as e:
            logger.error(f"Error creating finding for field {field} at index {idx}: {str(e)}")
            return None
    
    def check_anomalies(self, df: pd.DataFrame, rule: Dict) -> List[Dict]:
        """Check for anomalies based on the specified rule."""
        if df.empty:
            return []
            
        findings = []
        field = rule['field']
        detection_method = rule.get('detection_method', 'univariate')
        
        logger.info(f"Checking anomalies for {detection_method} detection on fields: {field}")
        
        # Handle single field
        if not isinstance(field, list):
            field = [field]
            
        for single_field in field:
            if single_field not in df.columns:
                logger.warning(f"Field {single_field} not found in dataframe columns: {list(df.columns)}")
                continue
                
            # Convert field to numeric if needed
            if not pd.api.types.is_numeric_dtype(df[single_field]):
                try:
                    df[single_field] = pd.to_numeric(df[single_field], errors='coerce')
                except Exception as e:
                    logger.error(f"Error converting field {single_field} to numeric: {str(e)}")
                    continue
                    
            # Group data if specified
            if 'group_by' in rule and rule['group_by']:
                for group_name, group_df in df.groupby(rule['group_by']):
                    if len(group_df) >= DETECTION_CONFIG.min_samples:
                        logger.info(f"Processing group {group_name} with {len(group_df)} samples")
                        group_findings = self._check_group_anomalies(group_df, rule, group_name)
                        findings.extend([f for f in group_findings if f is not None])
            else:
                if len(df) >= DETECTION_CONFIG.min_samples:
                    logger.info(f"Processing ungrouped data with {len(df)} samples")
                    findings.extend([f for f in self._check_group_anomalies(df, rule) if f is not None])
            
        logger.info(f"Found {len(findings)} anomalies for fields: {field}")
        return findings
    
    def _auto_tune_thresholds(self, df: pd.DataFrame, field: str) -> Dict[str, float]:
        """Auto-tune thresholds using IsolationForest's contamination parameter."""
        try:
            data = df[field].dropna().values.reshape(-1, 1)
            if len(data) < 10:
                return {'z_score': DETECTION_CONFIG.z_score_threshold, 'iqr_multiplier': DETECTION_CONFIG.iqr_multiplier}
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(data)
            predictions = model.predict(data)
            contamination = (predictions == -1).mean()
            # Adjust thresholds based on contamination, ensuring they are not too low
            z_score = max(0.5, DETECTION_CONFIG.z_score_threshold * (1 - contamination))
            iqr_multiplier = max(0.5, DETECTION_CONFIG.iqr_multiplier * (1 - contamination))
            return {'z_score': z_score, 'iqr_multiplier': iqr_multiplier}
        except Exception as e:
            logger.error(f"Error auto-tuning thresholds for field {field}: {str(e)}")
            return {'z_score': DETECTION_CONFIG.z_score_threshold, 'iqr_multiplier': DETECTION_CONFIG.iqr_multiplier}

    def _check_mixed_type_anomalies(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None) -> List[Dict]:
        """Detect anomalies in mixed-type columns using trend and pattern detection."""
        findings = []
        try:
            # Check for trend anomalies
            trend_findings = self._check_trend_anomalies(df, field, system, group_name)
            findings.extend(trend_findings)
            # Check for pattern anomalies
            pattern_findings = self._check_pattern_anomalies(df, field, system, group_name)
            findings.extend(pattern_findings)
        except Exception as e:
            logger.error(f"Error in mixed-type anomaly detection for field {field}: {str(e)}")
        return findings

    def _check_group_anomalies(self, df: pd.DataFrame, rule: Dict, group_name: Optional[Union[str, tuple]] = None) -> List[Dict]:
        """Check anomalies for a specific group."""
        findings = []
        field = rule['field']
        detection_method = rule.get('detection_method', 'univariate')
        
        if len(df) < DETECTION_CONFIG.min_samples:
            logger.warning(f"Insufficient samples ({len(df)}) for group {group_name}")
            return []
            
        # Handle single field
        if not isinstance(field, list):
            field = [field]
        
        # For multivariate detection, process all fields together
        if detection_method == 'multivariate':
            logger.info(f"Processing multivariate detection for fields: {field}")
            for detection_type in rule.get('detection_types', ['isolation_forest']):
                try:
                    if detection_type == 'isolation_forest':
                        findings.extend(self._check_multivariate_isolation_forest(df, field, rule['system'], group_name, rule.get('feature_group', 'default')))
                except Exception as e:
                    logger.error(f"Error in multivariate {detection_type} detection for group {group_name}: {str(e)}")
                    continue
        else:
            # For univariate detection, process each field individually
            for single_field in field:
                if single_field not in df.columns:
                    logger.warning(f"Field {single_field} not found in dataframe")
                    continue
                
                # Convert field to numeric if needed
                if not pd.api.types.is_numeric_dtype(df[single_field]):
                    try:
                        df[single_field] = pd.to_numeric(df[single_field], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error converting field {single_field} to numeric: {str(e)}")
                        continue
                
                # Only process numeric fields
                if pd.api.types.is_numeric_dtype(df[single_field]):
                    logger.info(f"Processing numeric field: {single_field}")
                    thresholds = self._auto_tune_thresholds(df, single_field)
                    for detection_type in rule.get('detection_types', ['iqr']):
                        try:
                            if detection_type == 'iqr':
                                findings.extend(self._check_iqr_anomalies(df, single_field, rule['system'], group_name, thresholds['iqr_multiplier']))
                            elif detection_type == 'zscore':
                                findings.extend(self._check_zscore_anomalies(df, single_field, rule['system'], group_name, thresholds['z_score']))
                            elif detection_type == 'isolation_forest':
                                findings.extend(self._check_isolation_forest(df, single_field, rule['system'], group_name))
                            elif detection_type == 'trend':
                                findings.extend(self._check_trend_anomalies(df, single_field, rule['system'], group_name))
                            elif detection_type == 'pattern':
                                findings.extend(self._check_pattern_anomalies(df, single_field, rule['system'], group_name))
                        except Exception as e:
                            logger.error(f"Error in {detection_type} detection for group {group_name}: {str(e)}")
                            continue
                else:
                    # Skip non-numeric fields
                    logger.info(f"Skipping non-numeric field: {single_field}")
        return findings
    
    def _check_iqr_anomalies(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None, iqr_multiplier: Optional[float] = None) -> List[Dict]:
        """Check for anomalies using IQR method."""
        try:
            if iqr_multiplier is None:
                iqr_multiplier = DETECTION_CONFIG.iqr_multiplier
            q1 = df[field].quantile(0.25)
            q3 = df[field].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            anomalies = df[(df[field] < lower_bound) | (df[field] > upper_bound)]
            findings = []
            for idx, row in anomalies.iterrows():
                # Calculate deviation as distance from median normalized by IQR
                median = df[field].median()
                deviation = abs(row[field] - median) / iqr if iqr > 0 else 0
                
                technical_details = {
                    'iqr_multiplier': float(iqr_multiplier),
                    'threshold_lower': float(lower_bound),
                    'threshold_upper': float(upper_bound),
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr),
                    'method': 'iqr',
                    'deviation': float(deviation)
                }
                finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'outlier', technical_details)
                if finding and not self._is_duplicate_anomaly(finding):
                    findings.append(finding)
            return findings
        except Exception as e:
            logger.error(f"Error in IQR detection for field {field}: {str(e)}")
            return []
    
    def _check_zscore_anomalies(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None, z_score_threshold: Optional[float] = None) -> List[Dict]:
        """Check for anomalies using Z-score method."""
        try:
            if z_score_threshold is None:
                z_score_threshold = DETECTION_CONFIG.z_score_threshold
            mean = df[field].mean()
            std = df[field].std()
            if std == 0:
                return []
            z_scores = np.abs((df[field] - mean) / std)
            mask = (z_scores > z_score_threshold).fillna(False)
            anomalies = df[mask]
            findings = []
            for idx, row in anomalies.iterrows():
                z_score = z_scores[idx]
                technical_details = {
                    'z_score': float(z_score),
                    'threshold': float(z_score_threshold),
                    'mean': float(mean),
                    'std': float(std),
                    'method': 'z_score',
                    'deviation': float(z_score)  # Deviation is the z-score itself
                }
                finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'outlier', technical_details)
                if finding and not self._is_duplicate_anomaly(finding):
                    findings.append(finding)
            return findings
        except Exception as e:
            logger.error(f"Error in Z-score detection for field {field}: {str(e)}")
            return []
    
    def _check_isolation_forest(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None) -> List[Dict]:
        """Check for anomalies using Isolation Forest (univariate)."""
        try:
            logger.info(f"Running univariate isolation forest on field: {field}")
            data = df[field].dropna().values.reshape(-1, 1)
            if len(data) < 5:
                logger.warning(f"Insufficient data for univariate isolation forest: {len(data)} samples")
                return []
                
            logger.info(f"Univariate isolation forest data shape: {data.shape}")
            logger.info(f"Univariate isolation forest data range: {data.min():.2f} to {data.max():.2f}")
                
            model = self._get_ml_model(f"{system}_{field}_{self._format_group_name(group_name)}")
            
            # Fit and predict
            model.fit(data)
            predictions = model.predict(data)
            anomaly_scores = model.decision_function(data)
            
            # Find anomalies (predictions == -1)
            anomaly_indices = np.where(predictions == -1)[0]
            original_indices = df[field].dropna().index[anomaly_indices]
            
            logger.info(f"Univariate isolation forest found {len(anomaly_indices)} anomalies")
            
            findings = []
            for i, idx in enumerate(original_indices):
                # Normalize anomaly score to 0-1 range for deviation
                score = anomaly_scores[anomaly_indices[i]]
                normalized_deviation = (score - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) if anomaly_scores.max() != anomaly_scores.min() else 0
                
                technical_details = {
                    'anomaly_score': float(score),
                    'method': 'isolation_forest',
                    'deviation': float(normalized_deviation),
                    'contamination': float(ML_CONFIG.contamination)
                }
                finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'outlier', technical_details)
                logger.debug(f"Created finding: {finding}")
                if finding and not self._is_duplicate_anomaly(finding):
                    findings.append(finding)
                    logger.debug(f"Added finding to results")
                else:
                    logger.debug(f"Finding was duplicate or None")
            
            logger.info(f"Univariate isolation forest returning {len(findings)} findings")
            return findings
            
        except Exception as e:
            logger.error(f"Error in isolation_forest detection for field {field}: {str(e)}")
            return []
    
    def _check_multivariate_isolation_forest(self, df: pd.DataFrame, fields: List[str], system: str, group_name: Optional[Union[str, tuple]] = None, feature_group: str = 'default') -> List[Dict]:
        """Check for anomalies using Isolation Forest (multivariate)."""
        try:
            # Filter fields that exist in the dataframe
            available_fields = [f for f in fields if f in df.columns]
            if len(available_fields) < 2:
                logger.warning(f"Insufficient fields for multivariate detection: {available_fields}")
                return []
                
            logger.info(f"Running multivariate isolation forest on fields: {available_fields}")
            
            # Prepare data
            data = df[available_fields].dropna()
            if len(data) < 5:
                logger.warning(f"Insufficient data for multivariate detection: {len(data)} samples")
                return []
                
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create model key
            model_key = f"{system}_{feature_group}_{self._format_group_name(group_name)}"
            model = self._get_ml_model(model_key)
            
            # Fit and predict
            model.fit(scaled_data)
            predictions = model.predict(scaled_data)
            anomaly_scores = model.decision_function(scaled_data)
            
            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            original_indices = data.index[anomaly_indices]
            
            findings = []
            for i, idx in enumerate(original_indices):
                score = anomaly_scores[anomaly_indices[i]]
                # Normalize anomaly score for deviation
                normalized_deviation = (score - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) if anomaly_scores.max() != anomaly_scores.min() else 0
                
                # Create finding for each field in the multivariate group
                for field in available_fields:
                    technical_details = {
                        'anomaly_score': float(score),
                        'method': 'multivariate_isolation_forest',
                        'deviation': float(normalized_deviation),
                        'contamination': float(ML_CONFIG.contamination),
                        'feature_group': feature_group,
                        'multivariate_fields': available_fields
                    }
                    finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'multivariate_outlier', technical_details)
                    if finding and not self._is_duplicate_anomaly(finding):
                        findings.append(finding)
            
            logger.info(f"Found {len(findings)} multivariate anomalies for feature group {feature_group}")
            return findings
            
        except Exception as e:
            logger.error(f"Error in multivariate isolation_forest detection: {str(e)}")
            return []
    
    def _check_trend_anomalies(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None) -> List[Dict]:
        """Check for trend anomalies."""
        try:
            window = DETECTION_CONFIG.window_size
            
            # Calculate rolling mean and std
            rolling_mean = df[field].rolling(window=window, min_periods=1).mean()
            rolling_std = df[field].rolling(window=window, min_periods=1).std()
            
            # Calculate percentage change
            pct_change = df[field].pct_change(fill_method=None)
            
            # Identify trend shifts
            trend_shifts = df[
                (np.abs(pct_change) > DETECTION_CONFIG.trend_threshold) &
                (np.abs(df[field] - rolling_mean) > DETECTION_CONFIG.trend_threshold * rolling_std)
            ]
            
            findings = []
            for idx, row in trend_shifts.iterrows():
                # Calculate deviation as percentage change
                deviation = abs(pct_change[idx]) if not pd.isna(pct_change[idx]) else 0
                
                technical_details = {
                    'method': 'trend',
                    'deviation': float(deviation),
                    'window_size': window,
                    'trend_threshold': DETECTION_CONFIG.trend_threshold
                }
                finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'trend_shift', technical_details)
                if finding and not self._is_duplicate_anomaly(finding):
                    findings.append(finding)
            
            return findings
            
        except Exception as e:
            logger.error(f"Error in trend detection for field {field}: {str(e)}")
            return []
    
    def _check_pattern_anomalies(self, df: pd.DataFrame, field: str, system: str, group_name: Optional[Union[str, tuple]] = None) -> List[Dict]:
        """Check for pattern anomalies."""
        try:
            # Calculate rolling statistics
            window = DETECTION_CONFIG.window_size
            rolling_mean = df[field].rolling(window=window, min_periods=1).mean()
            rolling_std = df[field].rolling(window=window, min_periods=1).std()
            
            # Identify pattern deviations
            pattern_deviations = df[
                (np.abs(df[field] - rolling_mean) > DETECTION_CONFIG.pattern_threshold * rolling_std)
            ]
            
            findings = []
            for idx, row in pattern_deviations.iterrows():
                # Calculate deviation as distance from rolling mean normalized by rolling std
                deviation = abs(row[field] - rolling_mean[idx]) / rolling_std[idx] if rolling_std[idx] > 0 else 0
                
                technical_details = {
                    'method': 'pattern',
                    'deviation': float(deviation),
                    'window_size': window,
                    'pattern_threshold': DETECTION_CONFIG.pattern_threshold
                }
                finding = self._create_anomaly_finding(df, idx, field, system, group_name, 'pattern_deviation', technical_details)
                if finding and not self._is_duplicate_anomaly(finding):
                    findings.append(finding)
            
            return findings
            
        except Exception as e:
            logger.error(f"Error in pattern detection for field {field}: {str(e)}")
            return [] 