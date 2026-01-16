import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

classifiers_label_dict={
    '合格':0,#//
    'Pass':0,
    '没问题':0,
    '不确定':0,
    'OK':0,
    'ok':0,
    '正常':0,
    '不稳定':0,
    '其他':0,
    0:'合格',

    '压痕太深':1,
    '位置偏压坑深有气孔':1,
    '压坑深减薄率小':1,
    '压坑深减薄率低':1,
    '压坑太深':1,
    '大毛刺 压痕深':1,
    '毛刺 压痕深':1,
    '压痕深':1,#//
    '压痕浅':1,
    1:'压坑过深',

    '焊点毛刺':2,
    '毛刺':2,#//
    '大毛刺':2,
    '轻微毛刺':2,
    '过烧':2,
    '鼓包':2,
    '表面状态不佳':2,
    '过烧/毛刺':2,
    '小毛刺': 2,
    '过烧、焊接不良':2,
    '过烧、不确定':2,
    '过烧、表面状态不佳':2,
    2:'毛刺',

    '焊点有气孔':3,
    '气孔':3,
    '有气孔':3,
    '压坑深有气孔':3,
    '焊点表面穿孔':3,
    '过烧、针孔':3,
    '过烧、烧穿':3,
    '烧穿':3,
    '烧穿/过烧':3,
    '过烧/烧穿':3,
    '针孔':3,#//
    3:'气孔',

    '位置偏打R角上':4,
    '偏焊':4,#//
    '焊点靠内':4,
    '焊点位置偏移':4,
    4:'偏焊',

    '焊核小位置偏':5,
    '焊点熔核过小':5,
    '焊点直径太小有气孔':5,
    '焊点直径太小':5,
    '小焊核':5,#//
    '正常，焊核偏小':5,
    '脱焊，正常，焊核小，满足要求':5,
    '小焊核、虚焊':5,
    '脱焊、小焊核':5,
    5:'小焊核',

    '位置偏没焊核':6,
    '开焊':6,#//
    '单边开焊':6,
    '脱焊':6,
    '脱焊，不确定':6,
    '脱焊、正常':6,
    '右侧虚焊':6,
    '左侧虚焊': 6,
    '虚焊':6,
    6:'开焊',

    '减薄率不合格':7,#//
    '减薄率低':7,
    '焊点减薄率不合格':7,
    7:'减薄率不合格',

    '扭曲':8,#//
    # 8:'扭曲',
    8:'不平整',

    '裂纹/过烧': 9,
    '过烧/裂纹': 9,
    '裂纹':9,#//
    9:'裂纹',
    10:'值异常'
}
# cols_to_keep = ['wear', 'wearpercent', 'ilsts', 'idemand1', 'idemand3', 'idemand2', 'iactual2', 'phastd',
#        'pha2', 'tactualstd', 'tipdresscounter', 'voltageactualvalue',
#        'currentactualvalue', 'weldtimeactualvalue', 'energyactualvalue',
#        'poweractualvalue', 'resistanceactualvalue', 'pulsewidthactualvalue',
#        'uirexpulsiontime', 'uirqstopprefcntvalue', 'umax', 'imax', 'energie',
#        'pmax', 'splash_time', 'wear_counter', 'milling_counter', 'u_curve', 'i_curve', #'Diameter 1', 'Indentation 1', 'Decision',
#        'u_curve_mean', 'u_curve_std', 'u_curve_median',
#        'u_curve_q25', 'u_curve_q75', 'u_curve_iqr', 'u_curve_rms',
#        'u_curve_abs_mean', 'u_curve_pos_mean', 'u_curve_pos_peaks',
#        'u_curve_neg_peaks', 'u_curve_mean_diff', 'u_curve_std_diff',
#        'i_curve_mean', 'i_curve_std', 'i_curve_max', 'i_curve_median',
#        'i_curve_range', 'i_curve_q25', 'i_curve_q75', 'i_curve_iqr',
#        'i_curve_rms', 'i_curve_abs_mean', 'i_curve_max_abs',
#        'i_curve_pos_mean', 'i_curve_pos_peaks', 'i_curve_neg_peaks',
#        'i_curve_mean_diff', 'i_curve_std_diff']

cols_to_keep = [
    'wear', 'wearpercent', 'idemandstd', 'ilsts', 'idemand1', 'iactual1', 'idemand2', 'iactual2', 'idemand3',
    'iactual3',
    'phastd', 'pha1', 'pha2', 'pha3', 'tactualstd', 'tipdresscounter', 'voltageactualvalue', 'currentactualvalue',
    'weldtimeactualvalue', 'energyactualvalue', 'poweractualvalue', 'resistanceactualvalue',
    'pulsewidthactualvalue', 'uirexpulsiontime',
    'energie', 'splash_time', 'wear_counter', 'milling_counter',
    'umax', 'imax', 'pmax', 'zmax',
    'u_curve', 'i_curve',
    'u_curve_mean', 'u_curve_std', 'u_curve_median', 'u_curve_q25', 'u_curve_q75', 'u_curve_iqr', 'u_curve_rms',
    'u_curve_abs_mean', 'u_curve_pos_mean', 'u_curve_pos_peaks',
    'u_curve_neg_peaks', 'u_curve_mean_diff', 'u_curve_std_diff',
    'i_curve_mean', 'i_curve_std', 'i_curve_max', 'i_curve_median', 'i_curve_range', 'i_curve_q25', 'i_curve_q75',
    'i_curve_iqr', 'i_curve_rms', 'i_curve_abs_mean', 'i_curve_max_abs',
    'i_curve_pos_mean', 'i_curve_pos_peaks', 'i_curve_neg_peaks', 'i_curve_mean_diff', 'i_curve_std_diff',
    'Total Decision',
    'Diameter 1',
    'Indentation 1',
    'Front Thickness 1',
    'Stack Thickness 1',


    'spot_tag',
    'id',
    'rui_time'
]
cols_to_keep_classifiers = [
    'wear', 'wearpercent', 'idemandstd', 'ilsts', 'idemand1', 'iactual1', 'idemand2', 'iactual2', 'idemand3',
    'iactual3',
    'phastd', 'pha1', 'pha2', 'pha3', 'tactualstd', 'tipdresscounter', 'voltageactualvalue', 'currentactualvalue',
    'weldtimeactualvalue', 'energyactualvalue', 'poweractualvalue', 'resistanceactualvalue',
    'pulsewidthactualvalue', 'uirexpulsiontime',
    'energie', 'splash_time', 'wear_counter', 'milling_counter',
    'umax', 'imax', 'pmax', 'zmax',
    'u_curve', 'i_curve',
    'u_curve_mean', 'u_curve_std', 'u_curve_median', 'u_curve_q25', 'u_curve_q75', 'u_curve_iqr', 'u_curve_rms',
    'u_curve_abs_mean', 'u_curve_pos_mean', 'u_curve_pos_peaks',
    'u_curve_neg_peaks', 'u_curve_mean_diff', 'u_curve_std_diff',
    'i_curve_mean', 'i_curve_std', 'i_curve_max', 'i_curve_median', 'i_curve_range', 'i_curve_q25', 'i_curve_q75',
    'i_curve_iqr', 'i_curve_rms', 'i_curve_abs_mean', 'i_curve_max_abs',
    'i_curve_pos_mean', 'i_curve_pos_peaks', 'i_curve_neg_peaks', 'i_curve_mean_diff', 'i_curve_std_diff',
    'Total Decision',
    'Diameter 1',
    'Indentation 1',
    'Front Thickness 1',
    'Stack Thickness 1',
    'Classifiers',
]




class WeldingDataset_Old(torch.utils.data.Dataset):
    """
    Dataset class for welding data that handles both scalar features and curve features.
    """
    def __init__(self, data, scalar_columns, curve_configs, train=True):
        self.data = data
        self.scalar_columns = scalar_columns
        self.curve_configs = curve_configs
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract scalar features
        scalar_features = torch.tensor(row[self.scalar_columns].values, dtype=torch.float32)
        
        # Extract curve features
        curve_features = {}
        for curve_name, config in self.curve_configs.items():
            if 'column' in config and config['column'] in row:
                curve_data = row[config['column']]
                
                # Convert to tensor (handle different data formats)
                if isinstance(curve_data, list) or isinstance(curve_data, (torch.Tensor, np.ndarray)):
                    curve_tensor = torch.tensor(curve_data, dtype=torch.float32)
                elif isinstance(curve_data, str):
                    try:
                        # Try to parse string representation of list
                        curve_list = eval(curve_data)
                        curve_tensor = torch.tensor(curve_list, dtype=torch.float32)
                    except:
                        # Fallback to zeros if parsing fails
                        curve_tensor = torch.zeros(config['length'], dtype=torch.float32)
                else:
                    # Fallback to zeros
                    curve_tensor = torch.zeros(config['length'], dtype=torch.float32)
                
                # Ensure correct length
                if len(curve_tensor) != config['length']:
                    if len(curve_tensor) > config['length']:
                        # Downsample
                        indices = torch.linspace(0, len(curve_tensor) - 1, config['length']).long()
                        curve_tensor = curve_tensor[indices]
                    else:
                        # Pad with zeros
                        padding = torch.zeros(config['length'] - len(curve_tensor), dtype=torch.float32)
                        curve_tensor = torch.cat([curve_tensor, padding])
                
                curve_features[curve_name] = curve_tensor
        
        return {
            'scalar_features': scalar_features,
            'curve_features': curve_features
        }

def drop_columns_robust(df, cols_to_drop):
    """
    Drops specified columns from a pandas DataFrame, handling duplicate column names
    and cases where some column names might not exist in the DataFrame.

    Args:
        df: pandas DataFrame to drop columns from.
        cols_to_drop: A list of column names to drop. This list may contain
                      duplicate names or names not present in the DataFrame.

    Returns:
        pandas DataFrame with the specified columns dropped (if they existed).
    """

    # Convert the list of columns to a set to remove duplicates
    unique_cols_to_drop = set(cols_to_drop)

    # Convert the set back to a list for use with pandas drop (optional, but can be useful for order if it mattered)
    cols_to_drop_list = list(unique_cols_to_drop)

    # Use pandas drop method with errors='ignore' to handle non-existent columns
    df_dropped = df.drop(columns=cols_to_drop_list, errors='ignore')

    return df_dropped

def compute_sequence_features(sequence):
    """Compute statistical features for sequence data like torque measurements."""
    try:
        # Convert set to list if needed
        if isinstance(sequence, set):
            sequence = list(sequence)
            
        # Convert sequence to numpy array and ensure it's float
        sequence = np.array(sequence, dtype=np.float64)
        
        # Check if sequence is constant
        is_constant = len(np.unique(sequence)) == 1
        
        # Check if sequence has no negative values
        has_negatives = np.any(sequence < 0)
        
        # New set of features with error handling
        def safe_compute(func, *args, **kwargs):
            """Safely compute a function, returning NaN if result is non-numeric or NaN."""
            try:
                result = func(*args, **kwargs)
                # Check if result is numeric and not NaN
                if not np.isreal(result) or np.isnan(result) or np.isinf(result):
                    return np.nan
                return result
            except:
                return np.nan

        features = {
            # Basic statistics
            'mean': safe_compute(np.mean, sequence),
            'std': safe_compute(np.std, sequence),
            'max': safe_compute(np.max, sequence),
            'min': safe_compute(np.min, sequence),
            'median': safe_compute(np.median, sequence),
            'range': safe_compute(np.ptp, sequence),
            
            # Quartiles and IQR
            'q25': safe_compute(np.percentile, sequence, 25),
            'q75': safe_compute(np.percentile, sequence, 75),
            'iqr': safe_compute(lambda x: np.percentile(x, 75) - np.percentile(x, 25), sequence),
            
            # Signal characteristics
            'rms': safe_compute(lambda x: np.sqrt(np.mean(np.square(x))), sequence),
            'zero_crossings': safe_compute(lambda x: len(np.where(np.diff(np.signbit(x)))[0]), sequence),
            
            # Additional features
            'abs_mean': safe_compute(lambda x: np.mean(np.abs(x)), sequence),
            'max_abs': safe_compute(lambda x: np.max(np.abs(x)), sequence),
            'pos_mean': safe_compute(lambda x: np.mean(x[x > 0]) if np.any(x > 0) else 0.0, sequence),
            'neg_mean': safe_compute(lambda x: np.mean(x[x < 0]) if np.any(x < 0) else 0.0, sequence),
            'pos_peaks': safe_compute(lambda x: len(np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]), sequence),
            'neg_peaks': safe_compute(lambda x: len(np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]), sequence),
            'mean_diff': safe_compute(lambda x: np.mean(np.abs(np.diff(x))), sequence),
            'std_diff': safe_compute(lambda x: np.std(np.diff(x)), sequence)
        }

        return features
        
    except Exception as e:
        # Safe error reporting that won't fail on sets
        sequence_start = list(sequence)[:5] if isinstance(sequence, set) else sequence[:5] if hasattr(sequence, '__getitem__') else []
        print(f"Error computing features: {str(e)} for sequence type: {type(sequence)}, start: {sequence_start}")
        return {k: np.nan for k in ['mean', 'std', 'max', 'min', 'median', 'range', 
                                   'q25', 'q75', 'iqr',
                                   'rms', 'zero_crossings', 'abs_mean', 'max_abs', 
                                   'pos_mean', 'neg_mean', 'pos_peaks', 'neg_peaks', 
                                   'mean_diff', 'std_diff']}

def get_sequence(data):
    """Safely extract sequence data with improved type handling."""
    try:
        # Handle None case
        if data is None:
            return None
            
        # If it's already a list or numpy array, return as is
        if isinstance(data, (list, np.ndarray)):
            return data
            
        # If it's a set, convert to list
        if isinstance(data, set):
            return list(data)
            
        # Convert string representation of list/set to actual list if needed
        if isinstance(data, str):
            # Try to evaluate the string
            evaluated_data = eval(data)
            # Handle case where eval returns a set
            if isinstance(evaluated_data, set):
                return list(evaluated_data)
            # If data is a list of lists, take first list
            if isinstance(evaluated_data, list) and len(evaluated_data) > 0 and isinstance(evaluated_data[0], list):
                return evaluated_data[0]
            return evaluated_data
            
        return data
    except:
        return None

def process_curves_and_torque(result_df):
    print("Processing curves and torque data...")
    total_rows = len(result_df)
    
    # Initialize result DataFrame
    processed_df = result_df.copy()
    
    for idx in tqdm.tqdm(range(total_rows)):
        # Process curves
        curves = ['u_curve', 'i_curve']
        for curve in curves:
            if curve in result_df.columns:
                sequence = get_sequence(result_df.iloc[idx][curve])
                if sequence is not None:
                    features = compute_sequence_features(sequence)
                    for feat_name, feat_value in features.items():
                        processed_df.at[idx, f'{curve}_{feat_name}'] = feat_value
                    
        # Process torque data
        for i in range(1, 7):
            torque_col = f'torqueAxis_{i}'
            if torque_col in result_df.columns:
                sequence = get_sequence(result_df.iloc[idx][torque_col])
                if sequence is not None:
                    features = compute_sequence_features(sequence)
                    for feat_name, feat_value in features.items():
                        processed_df.at[idx, f'robot_torque_data_{torque_col}_{feat_name}'] = feat_value
               
    return processed_df

def process_data(input_file, output_file=None):
    # Load data
    print("Loading data...")
    data = pd.read_csv(input_file)
    
    # Drop columns
    print("Dropping columns...")
    data = drop_columns_robust(data, cols_to_drop)
    
    # Process curves and torque data
    data = process_curves_and_torque(data)
    
    # Save processed data
    if output_file:
        print("Saving processed data...")
        data.to_csv(output_file, index=False)
    
    print("Data processing complete!")

    return data

class WeldingDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 scalar_columns: List[str],
                 curve_configs: Dict,
                 additional_task_columns: Dict[str, str] = None,
                 scalar_scaler: StandardScaler = None,
                 curve_scalers: Dict[str, StandardScaler] = None,
                 task_scalers: Dict[str, StandardScaler] = None,
                 train: bool = True):
        
        # # Check if the old style calling convention is being used
        # if isinstance(additional_task_columns_or_scalar_scaler, StandardScaler):
        #     # Old style - move parameters
        #     task_scalers = None
        #     train = curve_scalers_or_train if isinstance(curve_scalers_or_train, bool) else train
        #     curve_scalers = scalar_scaler_or_curve_scalers
        #     scalar_scaler = additional_task_columns_or_scalar_scaler
        #     additional_task_columns = None
        # else:
        #     # New style
        #     additional_task_columns = additional_task_columns_or_scalar_scaler
        #     scalar_scaler = scalar_scaler_or_curve_scalers
        #     curve_scalers = curve_scalers_or_train
        
        self.df = df.copy()

        # Critical fix: Add missing target columns for both training AND inference
        # These columns are being incorrectly treated as features by the scaler
        if 'Decision' not in self.df.columns:
            self.df['Total Decision'] = 0  # Placeholder class
            print("Added placeholder 'Total Decision' column for scaling compatibility")
        
        if 'Diameter 1' not in self.df.columns:
            self.df['Diameter 1'] = 0.0  # Placeholder value
            print("Added placeholder 'Diameter 1' column for scaling compatibility")
            
        if 'Indentation 1' not in self.df.columns:
            self.df['Indentation 1'] = 0.0  # Placeholder value
            print("Added placeholder 'Indentation 1' column for scaling compatibility")

        if 'Front Thickness 1' not in self.df.columns:
            self.df['Front Thickness 1'] = 0.0  # Placeholder value
            print("Added placeholder 'Front Thickness 1' column for scaling compatibility")
            
        if 'Stack Thickness 1' not in self.df.columns:
            self.df['Stack Thickness 1'] = 0.0  # Placeholder value
            print("Added placeholder 'Stack Thickness 1' column for scaling compatibility")
        
        # Make sure scalar_columns includes these columns if they're needed by the scaler
        self.scalar_columns = list(scalar_columns)
        for col in ['Diameter 1', 'Indentation 1', 'Front Thickness 1', 'Stack Thickness 1', 'Total Decision']:  # ADD new columns
            if col not in self.scalar_columns and col in self.df.columns:
                self.scalar_columns.append(col)
                print(f"Added {col} to scalar_columns for scaling compatibility")
                
        self.scalar_columns = scalar_columns
        self.curve_configs = curve_configs
        self.train = train
        
        # Handle scalar features
        scalar_features_list = []
        for col in scalar_columns:
            col_data = []
            for val in df[col].values:
                if isinstance(val, str):
                    # Check if it's a string representation of a list or dict
                    if val.startswith('[') and val.endswith(']'):
                        # Take the first value if it's a list
                        try:
                            val = float(val.strip('[]').split(',')[0])
                        except (IndexError, ValueError):
                            val = 0.0
                    else:
                        # Try to convert string to float
                        try:
                            val = float(val)
                        except ValueError:
                            val = 0.0
                col_data.append(val)
            scalar_features_list.append(col_data)
        
        scalar_features = np.array(scalar_features_list).T
        # Add after line: scalar_features = np.array(scalar_features_list).T
        print(f"Checking scalar features before scaling...")
        self.check_for_invalid_values(scalar_features, "scalar_features_before_scaling")
        
        if train:
            self.scalar_scaler = StandardScaler().fit(scalar_features)
        else:
            self.scalar_scaler = scalar_scaler
        self.scaled_scalar_features = self.scalar_scaler.transform(scalar_features)
        # Add after line: self.scaled_scalar_features = self.scalar_scaler.transform(scalar_features)
        print(f"Checking scaled scalar features...")
        self.check_for_invalid_values(self.scaled_scalar_features, "scaled_scalar_features")
        
        # Handle curve features
        # self.curve_scalers = {}
        # self.scaled_curve_features = {}
        
#        for curve_name in curve_configs:
#            # Convert string representation of lists/dicts to numpy arrays
#            curve_data = []
#            for curve_str in df[curve_name].values:
#                if isinstance(curve_str, str):
#                    # Remove any whitespace and handle both [] and {} formats
#                    cleaned_str = curve_str.strip()
#                    
#                    # Handle list format [...]
#                    if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
#                        values = cleaned_str[1:-1].split(',')
#                    # Handle dict format {...}
#                    elif cleaned_str.startswith('{') and cleaned_str.endswith('}'):
#                        # Remove braces and split by comma
#                        items = cleaned_str[1:-1].split(',')
#                        
#                        # Check if this is a key:value format or just a list of values in curly braces
#                        has_key_value_format = any(':' in item for item in items[:5])  # Check first few items
#    
#                        if has_key_value_format:
#                          # Extract only the values (assuming key:value format)
#                          values = []
#                          for item in items:
#                              try:
#                                  # Handle cases where there might be no colon
#                                  if ':' in item:
#                                      value = item.split(':')[1].strip()
#                                  else:
#                                      value = item.strip()
#                                  values.append(value)
#                              except Exception as e:
#                                  print(f"Warning: Error processing item '{item}': {str(e)}")
#                                  values.append('0')
#                    else:
#                        # Handle single value case
#                        # values = [cleaned_str]
#                        values = items
#                    
#                    # Convert to float array, replacing errors with 0.0
#                    try:
#                        curve_values = np.array([
#                            float(x.strip().strip("'").strip('"')) 
#                            if x.strip().strip("'").strip('"').replace('.','',1).isdigit() 
#                            else 0.0 
#                            for x in values
#                        ])
#                    except Exception as e:
#                        print(f"Warning: Error converting values to float: {str(e)}")
#                        print(f"Problematic values: {values}")
#                        curve_values = np.zeros(curve_configs[curve_name]['length'])
#                else:
#                    curve_values = curve_str
#                
#                # Ensure correct length
#                if len(curve_values) != curve_configs[curve_name]['length']:
#                    print(f"Warning: Curve length mismatch. Expected {curve_configs[curve_name]['length']}, got {len(curve_values)}")
#                    # Pad or truncate to match expected length
#                    if len(curve_values) < curve_configs[curve_name]['length']:
#                        curve_values = np.pad(
#                            curve_values, 
#                            (0, curve_configs[curve_name]['length'] - len(curve_values)),
#                            'constant'
#                        )
#                    else:
#                        curve_values = curve_values[:curve_configs[curve_name]['length']]
#                
#                curve_data.append(curve_values)

        # Handle curve features
        self.curve_scalers = {}
        self.scaled_curve_features = {}
        
        for curve_name in curve_configs:
            # Convert string representation of lists/dicts to numpy arrays
            curve_data = []
            for curve_str in df[curve_name].values:
                if isinstance(curve_str, str):
                    # Remove any whitespace and handle both [] and {} formats
                    cleaned_str = curve_str.strip()
                    
                    # Handle list format [...]
                    if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                        values = cleaned_str[1:-1].split(',')
                    # Handle dict format {...}
                    elif cleaned_str.startswith('{') and cleaned_str.endswith('}'):
                        # Remove braces and split by comma
                        items = cleaned_str[1:-1].split(',')
                        
                        # Check if this is a key:value format or just a list of values in curly braces
                        has_key_value_format = any(':' in item for item in items[:min(5, len(items))])  # Check first few items
        
                        if has_key_value_format:
                            # Extract only the values (assuming key:value format)
                            values = []
                            for item in items:
                                try:
                                    # Handle cases where there might be no colon
                                    if ':' in item:
                                        value = item.split(':')[1].strip()
                                    else:
                                        value = item.strip()
                                    values.append(value)
                                except Exception as e:
                                    print(f"Warning: Error processing item '{item}': {str(e)}")
                                    values.append('0')
                        else:
                            # Simple list of values in curly braces (no key:value format)
                            values = items
                    else:
                        # Handle single value case
                        values = [cleaned_str]
                    
                    # Debug info to help diagnose the issue
                    print(f"Found {len(values)} values in {curve_name}")
                    
                    # Convert to float array, replacing errors with 0.0
                    try:
                        # Improved conversion - better handling of negative numbers and diagnostic info
                        float_values = []
                        for i, x in enumerate(values):
                            x_clean = x.strip().strip("'").strip('"')
                            try:
                                float_values.append(float(x_clean))
                            except ValueError:
                                # Only print for the first few errors to avoid flooding logs
                                if i < 5:
                                    print(f"Warning: Could not convert '{x_clean}' to float, using 0.0")
                                float_values.append(0.0)
                        
                        curve_values = np.array(float_values)
                    except Exception as e:
                        print(f"Warning: Error converting values to float: {str(e)}")
                        print(f"Problematic values (first 10): {values[:10]}")
                        curve_values = np.zeros(curve_configs[curve_name]['length'])
                else:
                    curve_values = curve_str
                
                # Ensure correct length
                if len(curve_values) != curve_configs[curve_name]['length']:
                    print(f"Warning: Curve length mismatch for {curve_name}. Expected {curve_configs[curve_name]['length']}, got {len(curve_values)}")
                    # Pad or truncate to match expected length
                    if len(curve_values) < curve_configs[curve_name]['length']:
                        curve_values = np.pad(
                            curve_values, 
                            (0, curve_configs[curve_name]['length'] - len(curve_values)),
                            'constant'
                        )
                    else:
                        curve_values = curve_values[:curve_configs[curve_name]['length']]
                
                curve_data.append(curve_values)
            
            curve_data = np.stack(curve_data)

            # Add inside the curve processing loop, after creating curve_data
            print(f"Checking curve data for {curve_name} before scaling...")
            self.check_for_invalid_values(curve_data, f"curve_data_{curve_name}")
            
            if train:
                self.curve_scalers[curve_name] = StandardScaler().fit(curve_data)
            else:
                self.curve_scalers[curve_name] = curve_scalers[curve_name]
            self.scaled_curve_features[curve_name] = self.curve_scalers[curve_name].transform(curve_data)

             # And after scaling:
            print(f"Checking scaled curve data for {curve_name}...")
            self.check_for_invalid_values(self.scaled_curve_features[curve_name], f"scaled_curve_{curve_name}")

        # Handle additional task targets
        self.additional_task_columns = additional_task_columns or {}
        self.task_targets = {}
        self.task_scalers = {}
        
        for col_name, task_type in self.additional_task_columns.items():
            if col_name in df.columns:
                # Get the target values
                values = df[col_name].values
                
                if task_type == 'regression':
                    # For regression tasks, normalize the values
                    values = values.astype(np.float32).reshape(-1, 1)
                    
                    if train:
                        self.task_scalers[col_name] = StandardScaler().fit(values)
                    else:
                        self.task_scalers[col_name] = task_scalers[col_name]
                    
                    scaled_values = self.task_scalers[col_name].transform(values).squeeze()
                    self.task_targets[col_name] = torch.tensor(scaled_values, dtype=torch.float32)

                    # Add after creating self.task_targets[col_name]:
                    print(f"Checking task targets for {col_name}...")
                    self.check_for_invalid_values(self.task_targets[col_name], f"task_targets_{col_name}")
                
                else:
                    # For classification tasks, convert to class indices
                    if task_type.startswith('classification_'):
                        # Convert labels to 0-based indices
                        unique_values = sorted(df[col_name].unique())
                        value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
                        class_indices = [value_to_idx[val] for val in values]
                        self.task_targets[col_name] = torch.tensor(class_indices, dtype=torch.long)
        
        # Convert labels to tensor
        # self.labels = torch.tensor(df['Decision'].values, dtype=torch.long)

        # With this block:
        # Convert labels to 0-based indices
        unique_labels = sorted(df['Total Decision'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        converted_labels = [label_to_idx[label] for label in df['Total Decision'].values]
        
        print("Label conversion info:")
        print(f"Original unique labels: {unique_labels}")
        print(f"Label to index mapping: {label_to_idx}")
        
        # Convert labels to tensor
        self.labels = torch.tensor(converted_labels, dtype=torch.long)

    # Add this function to your WeldingDataset class
    def check_for_invalid_values(self, tensor_or_array, name="unnamed tensor"):
        """Check for NaN and Inf values in tensor or numpy array"""
        if isinstance(tensor_or_array, torch.Tensor):
            has_nan = torch.isnan(tensor_or_array).any().item()
            has_inf = torch.isinf(tensor_or_array).any().item()
            if has_nan:
                print(f"WARNING: NaN values detected in {name}")
                # Optionally print where the NaNs are
                if tensor_or_array.numel() < 100:
                    nan_indices = torch.nonzero(torch.isnan(tensor_or_array))
                    print(f"NaN indices: {nan_indices}")
            if has_inf:
                print(f"WARNING: Inf values detected in {name}")
                if tensor_or_array.numel() < 100:
                    inf_indices = torch.nonzero(torch.isinf(tensor_or_array))
                    print(f"Inf indices: {inf_indices}")
        elif isinstance(tensor_or_array, np.ndarray):
            # has_nan = np.isnan(tensor_or_array).any()
            # has_inf = np.isinf(tensor_or_array).any()
            # if has_nan:
            #     print(f"WARNING: NaN values detected in {name}")
            #     if tensor_or_array.size < 100:
            #         nan_indices = np.where(np.isnan(tensor_or_array))
            #         print(f"NaN indices: {nan_indices}")
            # if has_inf:
            #     print(f"WARNING: Inf values detected in {name}")
            #     if tensor_or_array.size < 100:
            #         inf_indices = np.where(np.isinf(tensor_or_array))
            #         print(f"Inf indices: {inf_indices}")
            # Check dtype before applying isnan/isinf
            if np.issubdtype(tensor_or_array.dtype, np.number):
                has_nan = np.isnan(tensor_or_array).any()
                has_inf = np.isinf(tensor_or_array).any()
                if has_nan:
                    print(f"WARNING: NaN values detected in {name}")
                if has_inf:
                    print(f"WARNING: Inf values detected in {name}")
            else:
                print(f"WARNING: Non-numeric array in {name}, skipping NaN/Inf check")
        else:
            print(f"WARNING: Unsupported type {type(tensor_or_array)} for {name}, skipping NaN/Inf check")
    # except Exception as e:
    #     print(f"WARNING: Error checking values in {name}: {str(e)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        scalar_features = torch.tensor(self.scaled_scalar_features[idx], dtype=torch.float32)
        curve_features = {
            name: torch.tensor(self.scaled_curve_features[name][idx], dtype=torch.float32)
            for name in self.curve_configs
        }
        label = self.labels[idx]

        # Add task targets
        task_targets = {
            name: values[idx] 
            for name, values in self.task_targets.items()
        }

        # Add checks for getitem return values
        self.check_for_invalid_values(scalar_features, f"scalar_features[{idx}]")
        for name, tensor in curve_features.items():
            self.check_for_invalid_values(tensor, f"curve_features[{name}][{idx}]")
        for name, tensor in task_targets.items():
            self.check_for_invalid_values(tensor, f"task_targets[{name}][{idx}]")
        
        return scalar_features, curve_features, label, task_targets