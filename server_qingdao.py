from contextlib import asynccontextmanager
import json
from fastapi import FastAPI
from clickhouse_driver import Client
from datetime import datetime
import pandas as pd
import asyncio
import yaml
import uvicorn
import torch
import time
import traceback
import pymysql
import ast
from zoneinfo import ZoneInfo
import numpy as np
import os
# ===================== Safe rule-model loader (no need to edit qdyq_peakrules_model.py) =====================
import types
import math

def load_peakrules_model_safely(module_path: str = "qdyq_peakrules_model.py"):
    """
    Load peakrules_model without modifying the source file on disk.

    1) Try normal import.
    2) If import fails (e.g., circular import / SyntaxError), read the source file and patch
       known problematic lines *in memory*, then exec into a fresh module namespace.

    Returns: (peakrules_model_callable, source_tag)
    """
    # 1) normal import
    try:
        from qdyq_peakrules_model import peakrules_model as _pm
        return _pm, "import"
    except Exception as e1:
        # 2) in-memory patched exec
        mp = os.path.abspath(module_path)
        try:
            with open(mp, "r", encoding="utf-8") as f:
                code = f.read()

            # Patch A: remove self-import (circular import)
            code = re.sub(r'^\s*from\s+qdyq_peakrules_model\s+import\s+peakrules_model.*$\n?', '', code, flags=re.M)

            # Patch B: remove invalid global line(s) that can break parsing
            code = re.sub(r'^\s*global\s+peakrules_model\s*,\s*USE_RULE_MODEL\s*$\n?', '', code, flags=re.M)

            compiled = compile(code, mp, "exec")
            mod = types.ModuleType("qdyq_peakrules_model_patched")
            mod.__file__ = mp

            # provide minimal globals commonly used
            mod.os = os
            mod.re = re
            mod.math = math

            exec(compiled, mod.__dict__)
            _pm = getattr(mod, "peakrules_model", None)
            if _pm is None:
                raise AttributeError("peakrules_model not found after patched load")
            return _pm, "patched_exec"
        except Exception as e2:
            raise RuntimeError(f"Failed to load peakrules_model. import_error={e1} patched_error={e2}")
# ==========================================================================================================

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F

import math

def _to_jsonable(x):
    """Convert numpy/pandas scalars (e.g., np.float32) into JSON-serializable Python types.
    Also converts NaN/Inf to None to avoid invalid JSON."""
    if x is None:
        return None
    # pandas Timestamp / datetime
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    # numpy scalars
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (not math.isfinite(v)) else v
    # python floats
    if isinstance(x, float):
        return None if (not math.isfinite(x)) else x
    if isinstance(x, (np.str_,)):
        return str(x)
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode('utf-8', errors='ignore')
        except Exception:
            return str(x)

    # numpy arrays
    if isinstance(x, np.ndarray):
        return [_to_jsonable(i) for i in x.tolist()]
    # lists / tuples
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(i) for i in x]
    # dicts
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


from cols_to_drop_v2 import cols_to_keep

# from abnormal_spikes_curve2 import data_funcAEQ, ob_func, strategy_1_funcAEQ

# ==================== NEW UIP Predictor (adapted to current UIPRegreator export) ====================
# This replaces: from large_predictor_Qingdao import Predictor
# It loads:
#   - <MODEL_EXPORT_DIR>/global_model.pth   (required)
#   - <MODEL_EXPORT_DIR>/inference_config.json (optional; per-part routing & thresholds)
#
# Model input: 4x1000 curve features [U, I, R=U/I, P=U*I] with normalization coefficients.
# Model output keys: qrk2_decision (logit), diameter_1, indentation_1, stack_thickness_1, front_thickness_1.

NORM_COEFFS_DEFAULT = {"R": 1, "I": 5, "U": 1, "P": 2}
TARGET_LEN_DEFAULT = 1000


def _extract_state_dict(obj: Any, model_path: str) -> Dict[str, torch.Tensor]:
    """Compatible loader for state_dict formats."""
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f"Unrecognized model format: {model_path}")


class UIPRegreator(nn.Module):
    def __init__(self, in_channels: int = 4, dropout_prob: float = 0.2):
        super().__init__()
        self.feature_extractors = nn.ModuleList([self._create_feature_extractor() for _ in range(in_channels)])
        self.map1 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout_prob)
        conv_output_size = 128 * in_channels

        self.regression_heads = nn.ModuleDict()
        for col in ["diameter_1", "indentation_1", "stack_thickness_1", "front_thickness_1"]:
            self.regression_heads[col] = nn.Sequential(
                nn.Linear(conv_output_size, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1),
            )
        self.decision_head = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def _create_feature_extractor(self):
        def d_block(in_f, out_f, norm=True):
            layers = [nn.Conv1d(in_f, out_f, 5, stride=2, padding=2)]
            if norm:
                layers.append(nn.InstanceNorm1d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        return nn.Sequential(
            *d_block(1, 1024, False),
            *d_block(1024, 512, False),
            *d_block(512, 256),
            *d_block(256, 128),
            *d_block(128, 64),
            nn.Conv1d(64, 16, 5, padding=2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = [ext(x[:, i: i + 1, :]) for i, ext in enumerate(self.feature_extractors)]
        x_cat = torch.cat(feats, dim=1)
        x_flat = self.dropout(F.leaky_relu(self.map1(x_cat), 0.2).view(x_cat.size(0), -1))
        return {
            "qrk2_decision": self.decision_head(x_flat),
            "diameter_1": self.regression_heads["diameter_1"](x_flat),
            "indentation_1": self.regression_heads["indentation_1"](x_flat),
            "stack_thickness_1": self.regression_heads["stack_thickness_1"](x_flat),
            "front_thickness_1": self.regression_heads["front_thickness_1"](x_flat),
        }


@dataclass
class _RouteInfo:
    model_path: str
    threshold: float
    name: str
    routed_by: str


def _load_inference_config(export_dir: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    cfg_path = os.path.join(export_dir, "inference_config.json")
    if not os.path.exists(cfg_path):
        return False, {}, {"path": "global_model.pth", "threshold": 0.5}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        models_cfg = cfg.get("models", {}) or {}
        default_info = models_cfg.get("default", {"path": "global_model.pth", "threshold": 0.5})
        default_info.setdefault("path", "global_model.pth")
        default_info.setdefault("threshold", 0.5)
        return True, models_cfg, default_info
    except Exception as e:
        logging.warning(f"[UIP Predictor] Failed to parse inference_config.json: {e}. Fallback to global_model.pth")
        return False, {}, {"path": "global_model.pth", "threshold": 0.5}


def _route_model(part_all: str, use_json: bool, models_cfg: Dict[str, Any], default_info: Dict[str, Any],
                 export_dir: str) -> _RouteInfo:
    # Inference config format:
    # { "models": { "default": {"path": "...", "threshold": 0.5}, "part_xxx": {"path": "...", "threshold": ...}, ... } }
    if not use_json:
        p = os.path.join(export_dir, default_info.get("path", "global_model.pth"))
        return _RouteInfo(p, float(default_info.get("threshold", 0.5)), "default", "fallback")

    key = str(part_all) if part_all is not None else "unknown"
    info = models_cfg.get(key) or models_cfg.get("default") or default_info
    p = os.path.join(export_dir, info.get("path", "global_model.pth"))
    thr = float(info.get("threshold", default_info.get("threshold", 0.5)))
    name = info.get("path", "global_model.pth")
    return _RouteInfo(p, thr, name, "inference_config.json")


class Predictor:
    """Deployment predictor wrapper for UIPRegreator."""

    def __init__(self, model_export_dir: str, device: Optional[str] = None,
                 norm_coeffs: Optional[Dict[str, float]] = None, target_len: int = TARGET_LEN_DEFAULT):
        # Allow passing a direct .pth file as model_export_dir
        if os.path.isfile(model_export_dir):
            # model_export_dir can be a direct .pth file; export_dir should be its parent folder
            self.export_dir = os.path.dirname(os.path.abspath(model_export_dir))
            self._single_model_file = os.path.basename(model_export_dir)
        else:
            # model_export_dir is a directory
            self.export_dir = os.path.abspath(model_export_dir)
            self._single_model_file = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_len = int(target_len)
        self.norm = dict(NORM_COEFFS_DEFAULT)
        if norm_coeffs:
            self.norm.update({k: float(v) for k, v in norm_coeffs.items()})
        self.use_json, self.models_cfg, self.default_info = _load_inference_config(self.export_dir)
        if self._single_model_file is not None:
            self.use_json = False
            self.models_cfg = {}
            self.default_info = {"path": self._single_model_file, "threshold": 0.5}
        self._model_cache: Dict[str, UIPRegreator] = {}

        # Pre-load default model for faster cold-start
        default_route = _route_model("default", self.use_json, self.models_cfg, self.default_info, self.export_dir)
        self._get_model(default_route.model_path)

        logging.info(f"[UIP Predictor] export_dir={self.export_dir} device={self.device} use_json={self.use_json}")

    def _get_model(self, model_path: str) -> UIPRegreator:
        model_path = os.path.abspath(model_path)
        if model_path in self._model_cache:
            return self._model_cache[model_path]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        obj = torch.load(model_path, map_location="cpu")
        sd = _extract_state_dict(obj, model_path)
        model = UIPRegreator(in_channels=4, dropout_prob=0.2)
        model.load_state_dict(sd, strict=True)
        model.to(self.device)
        model.eval()
        self._model_cache[model_path] = model
        logging.info(f"[UIP Predictor] Loaded model: {model_path}")
        return model

    def _parse_curve_to_np(self, x: Any) -> np.ndarray:
        if x is None:
            return np.asarray([], dtype=np.float32)
        if isinstance(x, np.ndarray):
            arr = x.astype(np.float32, copy=False).reshape(-1)
            return arr
        if isinstance(x, (list, tuple)):
            try:
                return np.asarray(x, dtype=np.float32).reshape(-1)
            except Exception:
                return np.asarray([], dtype=np.float32)
        if isinstance(x, (bytes, bytearray)):
            try:
                x = x.decode("utf-8", errors="ignore")
            except Exception:
                return np.asarray([], dtype=np.float32)
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() == "none":
                return np.asarray([], dtype=np.float32)
            try:
                v = ast.literal_eval(s)
                return self._parse_curve_to_np(v)
            except Exception:
                return np.asarray([], dtype=np.float32)
        return np.asarray([], dtype=np.float32)

    def _pad_or_truncate_np(self, arr: np.ndarray) -> np.ndarray:
        L = self.target_len
        if arr.size >= L:
            return arr[:L].astype(np.float32, copy=False)
        out = np.zeros((L,), dtype=np.float32)
        out[: arr.size] = arr.astype(np.float32, copy=False)
        return out

    def _build_features(self, i_curve: Any, u_curve: Any) -> Optional[np.ndarray]:
        i = self._parse_curve_to_np(i_curve)
        u = self._parse_curve_to_np(u_curve)
        if i.size == 0 or u.size == 0:
            return None
        mlen = min(i.size, u.size)
        i = i[:mlen] / float(self.norm["I"])
        u = u[:mlen] / float(self.norm["U"])
        i_nonzero = np.clip(i, 1e-10, None)
        r = (u / i_nonzero) / float(self.norm["R"])
        p = (u * i) / float(self.norm["P"])

        u = self._pad_or_truncate_np(u)
        i = self._pad_or_truncate_np(i)
        r = self._pad_or_truncate_np(r)
        p = self._pad_or_truncate_np(p)
        return np.stack([u, i, r, p], axis=0).astype(np.float32, copy=False)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with predicted_prob + (optional) regression outputs."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["predicted_prob", "predicted_label", "pred_diameter_1", "pred_indentation_1",
                                         "pred_stack_thickness_1", "pred_front_thickness_1"])

        # Build per-row route
        part_all_series = df["part_all"] if "part_all" in df.columns else pd.Series(["default"] * len(df))
        routes = []
        for v in part_all_series.tolist():
            routes.append(_route_model(v, self.use_json, self.models_cfg, self.default_info, self.export_dir))

        # Prepare output arrays
        n = len(df)
        prob_out = np.full((n,), np.nan, dtype=np.float32)
        d_out = np.full((n,), np.nan, dtype=np.float32)
        ind_out = np.full((n,), np.nan, dtype=np.float32)
        st_out = np.full((n,), np.nan, dtype=np.float32)
        ft_out = np.full((n,), np.nan, dtype=np.float32)

        # Group by model_path
        by_model: Dict[str, list] = {}
        for idx, rinfo in enumerate(routes):
            by_model.setdefault(rinfo.model_path, []).append(idx)

        # Fetch curve columns
        if "i_curve" not in df.columns or "u_curve" not in df.columns:
            # If no curves, return NaNs
            out = pd.DataFrame({"predicted_prob": prob_out, "predicted_label": ["OK"] * n})
            out["pred_diameter_1"] = d_out
            out["pred_indentation_1"] = ind_out
            out["pred_stack_thickness_1"] = st_out
            out["pred_front_thickness_1"] = ft_out
            return out

        i_list = df["i_curve"].tolist()
        u_list = df["u_curve"].tolist()

        for model_path, indices in by_model.items():
            model = self._get_model(model_path)

            # Build batch features
            feats = []
            keep_idx = []
            for j in indices:
                f = self._build_features(i_list[j], u_list[j])
                if f is None:
                    continue
                feats.append(f)
                keep_idx.append(j)

            if not feats:
                continue

            x = torch.from_numpy(np.stack(feats, axis=0)).to(self.device, non_blocking=True)
            out = model(x)
            prob = torch.sigmoid(out["qrk2_decision"].view(-1)).detach().cpu().numpy().astype(np.float32, copy=False)
            pred_d = out["diameter_1"].view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            pred_ind = out["indentation_1"].view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            pred_st = out["stack_thickness_1"].view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            pred_ft = out["front_thickness_1"].view(-1).detach().cpu().numpy().astype(np.float32, copy=False)

            for k, row_idx in enumerate(keep_idx):
                prob_out[row_idx] = prob[k]
                d_out[row_idx] = pred_d[k]
                ind_out[row_idx] = pred_ind[k]
                st_out[row_idx] = pred_st[k]
                ft_out[row_idx] = pred_ft[k]

        # Provide default label @0.5 (will be re-thresholded upstream if needed)
        labels = np.where(prob_out > 0.5, "NOK", "OK").astype(object)
        labels[pd.isna(prob_out)] = "OK"
        out_df = pd.DataFrame(
            {
                "predicted_prob": prob_out,
                "predicted_label": labels,
                "pred_diameter_1": d_out,
                "pred_indentation_1": ind_out,
                "pred_stack_thickness_1": st_out,
                "pred_front_thickness_1": ft_out,
            }
        )
        return out_df


# =================================================================================================

import logging

peakrules_model = None  # lazy-imported in startup_event

# FastAPI instance
app = FastAPI()
semaphore = asyncio.Semaphore(5)

# Read config file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


def _to_bool(v, default: bool = False) -> bool:
    """Robust bool parser for YAML/env values."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


# Database configs
clickhouse_config = config["clickhouse"]
mysql_config = config["mysql"]
tables = config.get("tables", {}) or {}
# Default source_table -> yiqi.weld_detail_pz (can override in config.yaml)
tables.setdefault("source_table", "yiqi.weld_detail_pz")
# Default MySQL result_table key (keep your config.yaml if already set)
tables.setdefault("result_table", tables.get("result_table", "weld_result"))
polling_interval = config.get("polling_interval", 10)
save_dir = config["save_dir"]
stat_dict_path = config["stat_dict_path"]
problem_class = config["problem_class"]

# ==================== DYNAMIC CLASSIFICATION THRESHOLD (calibrated JSON) ====================
# Optional config.yaml keys:
#   classification_threshold_default: 0.5
#   calibrated_threshold_path: /root/laizhongyuan/qingdao/calibrated_threshold_0.99.json

CLASSIFICATION_THRESHOLD = float(config.get("classification_threshold_default", 0.5))
CALIBRATED_THRESHOLD_PATH = config.get(
    "calibrated_threshold_path",
    "calibrated_threshold_0.99.json"  # fallback to your saved file
)


def load_calibrated_threshold(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        t = float(payload["threshold"])
        logging.info(f"[Calibration] Loaded threshold T={t:.6e} from {path}")
        return t
    except Exception as e:
        logging.warning(f"[Calibration] Could not load threshold from {path}: {e}")
        return None


# Try to load the calibrated threshold, otherwise keep the default
_loaded_T = load_calibrated_threshold(CALIBRATED_THRESHOLD_PATH)
if _loaded_T is not None:
    CLASSIFICATION_THRESHOLD = _loaded_T

# === Add a tightening scale for the calibrated threshold ===
# (Values >1.0 make the threshold stricter → fewer NOKs.)
# THRESHOLD_SCALE = float(config.get("threshold_scale", 1.0))
THRESHOLD_SCALE = float(config.get("threshold_scale", 1.0))
CLASSIFICATION_THRESHOLD *= THRESHOLD_SCALE
logging.info(f"[Calibration] Final threshold after scaling: T={CLASSIFICATION_THRESHOLD:.6e} (scale={THRESHOLD_SCALE})")
# ===========================================================
# ============================================================================================

# ==================== MODEL TOGGLES ====================
# Control which models are active (config.yaml overridable)
#   use_ml_model: true/false
#   use_rule_model: true/false
USE_ML_MODEL = _to_bool(config.get("use_ml_model", True), True)
USE_RULE_MODEL = _to_bool(config.get("use_rule_model", True), True)
# =======================================================

# ClickHouse client
clickhouse_client = Client(
    host=clickhouse_config["host"],
    port=clickhouse_config["port"],
    user=clickhouse_config["user"],
    password=clickhouse_config["password"],
    database=clickhouse_config["database"],
    connect_timeout=clickhouse_config["connect_timeout"]
)

# Global variable for MySQL connection, initialized once at startup
mysql_connection = None


def get_mysql_connection():
    """Establish and return a MySQL connection."""
    return pymysql.connect(
        host=mysql_config["host"],
        user=mysql_config["user"],
        password=mysql_config["password"],
        database=mysql_config["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


print('Database connection established.')

# curve_s_file = r"AEQ20250303_板材标准曲线_V1.0.xlsx"
# curve_s_data = None  # To be loaded on startup

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def safe_parse_standard_curve(curve_string):
    """Safely parse standard curve data with multiple fallback methods"""
    try:
        return ast.literal_eval(curve_string)
    except (ValueError, SyntaxError):
        try:
            return json.loads(curve_string)
        except (ValueError, json.JSONDecodeError):
            try:
                result = eval(curve_string)
                if isinstance(result, list):
                    return result
                else:
                    raise ValueError("Not a list")
            except Exception as e:
                print(f"All parsing methods failed: {e}")
                return None


def data_dispose_energy(data_in=None):
    """Data processing function with validation, from Aeq_q_rules2.py"""
    if not isinstance(data_in, dict):
        return [], [], []

    data = data_in
    try:
        umax = float(data.get('umax', '0'))
        imax = float(data.get('imax', '0'))
        u_curve_data = data.get('u_curve', [])
        i_curve_data = data.get('i_curve', [])
    except (ValueError, TypeError):
        return [], [], []

    if umax <= 0 or imax <= 0:
        return [], [], []

    u_list = _parse_curve(u_curve_data)
    i_list = _parse_curve(i_curve_data)

    if not u_list or not i_list or len(u_list) < 100 or len(i_list) < 100:
        return [], [], []

    min_len = min(len(u_list), len(i_list), 1000)
    u_list = u_list[:min_len]
    i_list = i_list[:min_len]

    u_c = [round(i * umax * 0.01, 2) for i in u_list]
    i_c = [round(i * imax * 0.0001, 2) for i in i_list]
    r_c = [round(u / i, 4) if i != 0 else 0 for u, i in zip(u_c, i_c)]

    return r_c, u_c, i_c


def _parse_curve(curve_data):
    """Helper function to parse curve data, supports string and list formats."""
    if isinstance(curve_data, str):
        try:
            return [float(x.strip()) for x in curve_data.strip('[]').split(',') if x.strip()]
        except (ValueError, IndexError):
            return []
    elif isinstance(curve_data, list):
        return [float(x) for x in curve_data if isinstance(x, (int, float))]
    return []


def energy_rule_analysis(data_in=None):
    """Energy rule analysis function, optimized for speed and clarity."""
    global curve_s_data, peakrules_model, USE_RULE_MODEL
    if curve_s_data is None:
        return 'NoStandardCurve', None, None

    spot_tag = None
    try:
        spot_tag = data_in.get('spot_tag') or data_in.get('spotTag')
        if not spot_tag:
            return 'NoSpotTag', None, None

        rui_r, rui_u, rui_i = data_dispose_energy(data_in=data_in)
        if not all(isinstance(arr, list) and len(arr) >= 100 for arr in [rui_r, rui_u, rui_i]):
            print(f"Insufficient or invalid curve data for spot {spot_tag}")
            return 'InsufficientData', None, None

        standard_curve_data = curve_s_data[curve_s_data['spot_tag'] == spot_tag]
        if standard_curve_data.empty:
            station_prefix = spot_tag.split('_')[0] if '_' in spot_tag else spot_tag[:6]
            fallback_data = curve_s_data[curve_s_data['spot_tag'].str.startswith(station_prefix)]
            if not fallback_data.empty:
                print(f"Using fallback standard curve for spot {spot_tag}")
                standard_curve_data = fallback_data.iloc[[0]]
            else:
                print(f"No standard curve found for spot {spot_tag}")
                return 'NoStandardCurve', None, None

        curve_row = standard_curve_data.iloc[0]
        rui_r_s = safe_parse_standard_curve(curve_row['标准曲线'])
        if rui_r_s is None or not isinstance(rui_r_s, list):
            return 'StandardCurveParseError', None, None

        # Pad or truncate standard curve to fixed length
        rui_r_s = (rui_r_s + [rui_r_s[-1] if rui_r_s else 0] * (1000 - len(rui_r_s)))[:1000]

        # Convert to NumPy arrays for faster computation
        rui_i_np = np.array(rui_i)
        rui_u_np = np.array(rui_u)
        rui_r_s_np = np.array(rui_r_s)

        # Ensure arrays have compatible lengths
        min_len = min(len(rui_i_np), len(rui_r_s_np))
        if min_len == 0:
            return 'ZeroLength', None, None

        rui_i_np = rui_i_np[:min_len]
        rui_u_np = rui_u_np[:min_len]
        rui_r_s_np = rui_r_s_np[:min_len]

        # Vectorized calculations for energy values
        q_i = round(np.dot(rui_i_np, rui_r_s_np), 3)

        with np.errstate(divide='ignore', invalid='ignore'):
            q_u_calc = np.square(rui_u_np) / rui_r_s_np
            q_u_calc[np.isinf(q_u_calc)] = 0
            q_u = round(np.sum(q_u_calc), 3)

        q_s_i = curve_row['part_q_i_min']
        q_s_u = curve_row['part_q_u_min']
        part_clc_line_i = curve_row['part_clc_line_i']
        part_clc_line_u = curve_row['part_clc_line_u']

        q_result = 'nok' if q_i < q_s_i * part_clc_line_i or q_u < q_s_u * part_clc_line_u else 'ok'
        return q_result, q_i, q_u
    except Exception as e:
        print(f"Energy analysis error for spot {spot_tag}: {e}")
        traceback.print_exc()
        return 'Error', None, None


def compute_sequence_features(sequence, name=None):
    """Compute statistical features for sequence data."""
    sequence = np.array(sequence, dtype=np.float64)
    if sequence.size == 0:
        return {}

    def safe_compute(func, *args, **kwargs):
        """Safely compute a function, returning NaN if result is non-numeric or NaN."""
        try:
            result = func(*args, **kwargs)
            if not np.isreal(result) or np.isnan(result) or np.isinf(result):
                return np.nan
            return result
        except:
            return np.nan

    features = {
        'mean': np.mean, 'std': np.std, 'max': np.max, 'min': np.min,
        'median': np.median, 'range': np.ptp, 'q25': lambda x: np.percentile(x, 25),
        'q75': lambda x: np.percentile(x, 75),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'rms': lambda x: np.sqrt(np.mean(np.square(x))),
        'zero_crossings': lambda x: len(np.where(np.diff(np.signbit(x)))[0]),
        'abs_mean': lambda x: np.mean(np.abs(x)),
        'max_abs': lambda x: np.max(np.abs(x)),
        'pos_mean': lambda x: np.mean(x[x > 0]) if np.any(x > 0) else 0.0,
        'neg_mean': lambda x: np.mean(x[x < 0]) if np.any(x > 0) else 0.0,
        'pos_peaks': lambda x: len(np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]),
        'neg_peaks': lambda x: len(np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]),
        'mean_diff': lambda x: np.mean(np.abs(np.diff(x))),
        'std_diff': lambda x: np.std(np.diff(x))
    }

    result = {f'{name}_{feat}': round(safe_compute(func, sequence), 4) for feat, func in features.items()}
    return result


def get_sequence(data):
    """Safely extract sequence data with improved type handling."""
    if data is None:
        return None
    if isinstance(data, (list, np.ndarray)):
        return data
    if isinstance(data, set):
        return list(data)
    if isinstance(data, str):
        try:
            evaluated_data = ast.literal_eval(data)
            if isinstance(evaluated_data, (list, np.ndarray)):
                return evaluated_data
            if isinstance(evaluated_data, set):
                return list(evaluated_data)
        except (ValueError, SyntaxError):
            return None
    return None


def process_curves_and_torque(result_df):
    """Vectorized processing of curves and torque data to compute features."""
    processed_df = result_df.copy()

    # Process curves
    for curve in ['u_curve', 'i_curve']:
        if curve in result_df.columns:
            # Apply feature calculation to each row using a vectorized approach if possible
            processed_df = pd.concat([processed_df, processed_df[curve].apply(
                lambda x: pd.Series(compute_sequence_features(get_sequence(x), name=curve))
            )], axis=1)

    # Process torque data
    for i in range(1, 7):
        torque_col = f'torqueAxis_{i}'
        if torque_col in result_df.columns:
            processed_df = pd.concat([processed_df, processed_df[torque_col].apply(
                lambda x: pd.Series(compute_sequence_features(get_sequence(x), name=f'robot_torque_data_{torque_col}'))
            )], axis=1)

    return processed_df


def validate_curve_data(data):
    """Validate curve data quality before processing, using vectorized operations."""
    if data.empty:
        return data

    data['is_valid'] = (
            data['u_curve'].apply(lambda x: isinstance(x, (list, str)) and len(str(x)) > 10) &
            data['i_curve'].apply(lambda x: isinstance(x, (list, str)) and len(str(x)) > 10) &
            pd.to_numeric(data['umax'], errors='coerce').gt(0) &
            pd.to_numeric(data['imax'], errors='coerce').gt(0)
    )

    invalid_rows_count = len(data) - data['is_valid'].sum()
    if invalid_rows_count > 0:
        print(f"Removing {invalid_rows_count} rows with invalid curve data.")

    return data[data['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)


def process_data(input_df):
    """Main data processing pipeline."""
    processed_df = input_df.copy()

    # Unified column mapping
    column_mappings = {
        'Decision': 'Total Decision',
        '判断': 'Total Decision',
        'Reason': 'Classifiers',
        '原因': 'Classifiers',
        '实测直径': 'Diameter 1',
        '实测压坑': 'Indentation 1',
        '前板厚度': 'Front Thickness 1',
        '剩余板材': 'Stack Thickness 1'
    }
    processed_df.rename(columns=column_mappings, inplace=True)

    # Convert 'Total Decision' and fill NaNs
    if 'Total Decision' in processed_df.columns:
        processed_df['Total Decision'] = processed_df['Total Decision'].apply(lambda x: 1 if x != 'Pass' else 0)
    else:
        processed_df['Total Decision'] = 0

    if 'Classifiers' in processed_df.columns:
        processed_df['Classifiers'].fillna('合格', inplace=True)
    else:
        processed_df['Classifiers'] = '合格'

    # Ensure required columns exist
    required_cols = ['Diameter 1', 'Indentation 1', 'Front Thickness 1', 'Stack Thickness 1']
    for col in required_cols:
        if col not in processed_df.columns:
            processed_df[col] = 'not available'

    # Calculate curve features and concat
    # curve_features = processed_df.apply(calculate_curve_features_for_row, axis=1)
    # processed_df = pd.concat([processed_df, curve_features], axis=1)

    # Calculate curve features for all rows (optimized)
    print(f"Computing curve features for {len(processed_df)} rows...")
    feature_start = time.time()

    all_curve_features = []
    for idx in range(len(processed_df)):
        all_curve_features.append(calculate_curve_features_for_row(processed_df.iloc[idx]))

    curve_features = pd.DataFrame(all_curve_features, index=processed_df.index)
    processed_df = pd.concat([processed_df, curve_features], axis=1)

    print(f"Curve features computed in {time.time() - feature_start:.4f}s")

    processed_df.fillna(0, inplace=True)
    processed_df = keep_only_columns(processed_df, cols_to_keep)

    return processed_df


def calculate_curve_features_for_row(row):
    """Calculate all curve-related features for a single DataFrame row."""
    try:
        umax = float(row.get('umax', 0))
        imax = float(row.get('imax', 0))

        u_curve_raw = row.get('u_curve', [])
        i_curve_raw = row.get('i_curve', [])

        u_curve_raw = get_sequence(u_curve_raw)
        i_curve_raw = get_sequence(i_curve_raw)

        if not u_curve_raw or not i_curve_raw:
            return pd.Series(dtype='float64')

        u_c = [val * umax * 0.01 for val in u_curve_raw[:1000]]
        i_c = [val * imax * 0.0001 for val in i_curve_raw[:1000]]
        r_c = [u / i if i != 0 else 0 for u, i in zip(u_c, i_c)]

        feature_dict = {}
        feature_dict.update(compute_sequence_features(u_c, 'u_curve'))
        feature_dict.update(compute_sequence_features(i_c, 'i_curve'))
        feature_dict.update(compute_sequence_features(r_c, 'r_curve'))

        feature_dict['u_curve_processed'] = u_c
        feature_dict['i_curve_processed'] = i_c
        feature_dict['r_curve_processed'] = r_c

        return pd.Series(feature_dict)

    except Exception as e:
        print(f"Error processing row: {e}. Row data: {row.get('id', 'N/A')}")
        return pd.Series(dtype='float64')


def keep_only_columns(df, columns_to_keep):
    """Ensure DataFrame has a specific set of columns, adding missing ones with default values."""
    current_columns = set(df.columns)
    missing_columns = [col for col in columns_to_keep if col not in current_columns]

    if missing_columns:
        print(
            f"Warning: {len(missing_columns)} columns in keep list don't exist in DataFrame: {missing_columns}. Adding them with default value 0.")
        for col in missing_columns:
            df[col] = 0

    return df[columns_to_keep].copy()


def mock_new_model_inference(data):
    """Mock model inference function."""
    aeq_results = {}
    h_batch_thr = 0.04
    h_serious_thr = 0.045
    w_thr = 30

    results = []
    for i, row in data.iterrows():
        defect_type = ''
        # energy_result, q_i_value, q_u_value = 'Unknown', None, None

        # try:
        #     energy_result, q_i_value, q_u_value = energy_rule_analysis(row.to_dict())
        #     energy_oknok = 1 if energy_result == 'nok' else 0
        # except Exception as e:
        #     print(f"Energy analysis exception for row {i}: {e}")
        #     energy_oknok = 0

        # Instead, just set defaults:
        energy_result, q_i_value, q_u_value = 'Disabled', None, None
        energy_oknok = 0

        # query_spot_data = row.to_dict()
        # data_in = {'id': row['id'], 'data': query_spot_data}
        # data_out = data_funcAEQ(data_q=data_in)
        # stt_1_q = ob_func(ob_q=data_out)
        # plot_q = strategy_1_funcAEQ(
        #     stt_1_q=stt_1_q,
        #     h_batch_thr=h_batch_thr,
        #     h_serious_thr=h_serious_thr,
        #     w_thr=w_thr,
        #     batch_num=len(data),
        #     stat_dict_path=stat_dict_path
        # )

        # aeq_result = {'ob_code': 0, 'result': 'OK', 'batch_serious': 'normal'}
        # if plot_q is not None:
        #     aeq_result = {
        #         'ob_code': plot_q['ob_code'],
        #         'result': plot_q['result'],
        #         'batch_serious': plot_q['batch_serious']
        #     }

        # surface_oknok = 1 if aeq_result['ob_code'] > 0 else 0
        surface_oknok = 0

        diameter_oknok = 0
        indentation_oknok = 0
        original_oknok = 0

        exist_nok = int(any([original_oknok, diameter_oknok, indentation_oknok, surface_oknok, energy_oknok]))
        oknok_str = 'NOK' if exist_nok else 'OK'
        if energy_oknok:
            defect_type += "小焊核/虚焊"
            # print(defect_type)
        if surface_oknok:
            if defect_type: defect_type += ","
            defect_type += "表面缺陷"
            # print(defect_type)
        if not defect_type:
            defect_type = "合格"
            # print(defect_type)

        finish_time = datetime.now(ZoneInfo("Asia/Shanghai"))

        raw_result = {
            "diameter": diameter_val,
            "indentation": indentation_val,
            'Front_Thickness': front_thickness_val,
            'Stack_Thickness': stack_thickness_val,
            "defect_type": defect_type,
            "oknok": oknok_str,
            "diameter_oknok": diameter_oknok,
            "indentation_oknok": indentation_oknok,
            "is_yahen": row.get('is_yahen'),
            "is_xiaohanhe": row.get('is_xiaohanhe'),
            "is_qikong": row.get('is_qikong'),
            "surface_oknok": surface_oknok,
            "total_oknok": int(exist_nok),
            "energy_result": energy_result,
            "energy_q_i": q_i_value,
            "energy_q_u": q_u_value,
            "energy_oknok": energy_oknok,
            "exist_nok": exist_nok
        }

        result_row = {
            "weld_result": json.dumps(_to_jsonable(raw_result), ensure_ascii=False, allow_nan=False),
            "defect_type": defect_type,
            "oknok": oknok_str,
            "alg_complete_time": finish_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "diameter": diameter_val,
            "indentation": indentation_val,
            'spot_tag': row.get('spot_tag'),
            'Front_Thickness': front_thickness_val,
            'Stack_Thickness': stack_thickness_val,
            "diameter_oknok": diameter_oknok,
            "indentation_oknok": indentation_oknok,
            "surface_oknok": surface_oknok,
            "total_oknok": int(exist_nok),
        }

        results.append(result_row)

    return pd.DataFrame(results)


def new_process_inference(data, predictor=None, threshold=CLASSIFICATION_THRESHOLD, use_ml=USE_ML_MODEL,
                          use_rules=USE_RULE_MODEL):
    """
    Execute model inference with adjustable threshold and return processed data.

    Args:
        data: Input DataFrame
        predictor: Model predictor instance
        threshold: Classification threshold (default from config). Higher = fewer false positives
        use_ml: Whether to use ML model (default from config)
        use_rules: Whether to use rule-based model (default from config)
    """
    start_time = time.time()
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Input data is not a valid DataFrame or is empty.")
        return pd.DataFrame()

    # Data type conversion
    cols_to_trans = ['wear', 'energie', 'tactualstd', 'idemand1', 'idemand2', 'idemand3',
                     'iactual1', 'iactual2', 'wearpercent', 'idemandstd', 'ilsts']
    for col in cols_to_trans:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].replace('None', np.nan), errors='coerce')

    data = data.dropna(subset=['wear', 'energie', 'tactualstd']).reset_index(drop=True)

    # Process data and run inference with selected models
    data_derived = process_data(data)

    # Initialize prediction results
    ml_predictions = None

    # ==================== ML MODEL (if enabled) ====================
    if use_ml and predictor is not None:
        print("Running ML model prediction...")
        result_2 = predictor.predict(data)

        # Check if predictor returns probabilities
        if 'predicted_prob' in result_2.columns:
            # Apply custom threshold to probabilities
            print(f"Using ML custom threshold: {threshold}")
            result_2['predicted_label'] = result_2['predicted_prob'].apply(
                lambda prob: 'NOK' if prob > threshold else 'OK'
            )

            # Log threshold effectiveness
            orig_nok_rate = (result_2['predicted_prob'] > 0.5).mean() * 100
            new_nok_rate = (result_2['predicted_prob'] > threshold).mean() * 100
            print(
                f"ML NOK rate change: {orig_nok_rate:.2f}% @ 0.5 threshold -> {new_nok_rate:.2f}% @ {threshold} threshold")
        else:
            print("WARNING: ML Predictor does not return probabilities. Using hard predictions only.")

        ml_predictions = result_2
    # ===============================================================

    # Build result dataframe from ML predictions only
    finish_time = datetime.now(ZoneInfo("Asia/Shanghai"))

    # Build result dataframe combining both models
    finish_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    results = []

    # ========== PRE-COMPUTE ALL RULE PREDICTIONS (before loop) ==========
    rule_predictions = []
    if use_rules:
        print(f"Running rule-based predictions for {len(data)} samples...")
        rule_start = time.time()
        for idx in range(len(data)):
            try:
                rule_data = data.iloc[idx].to_dict()
                rule_output = peakrules_model(
                    data_ins=rule_data,
                    on_off=True,
                    data_model='ck',
                    save_flag=False,
                    save_file=None,
                    save_file_dict=None
                )
                rule_predictions.append(rule_output.get('result', 'ok'))
            except Exception as e:
                print(f"Rule-based model error for row {idx}: {e}")
                rule_predictions.append('ok')
        print(f"Rule predictions completed in {time.time() - rule_start:.4f}s")
    else:
        rule_predictions = ['ok'] * len(data)
    # ====================================================================

    for idx, row in data_derived.iterrows():
        # Initialize default values
        ml_label = 'OK'
        ml_prob = None
        rule_result = 'ok'

        # Get ML prediction if enabled
        if use_ml and ml_predictions is not None:
            ml_label = ml_predictions.iloc[idx]['predicted_label']
            ml_prob = ml_predictions.iloc[idx].get('predicted_prob', None)

            # ---- Regression outputs from current UIPRegreator (if provided) ----
            ml_pred_diameter = ml_predictions.iloc[idx].get('pred_diameter_1', None)
            ml_pred_indentation = ml_predictions.iloc[idx].get('pred_indentation_1', None)
            ml_pred_stack_thickness = ml_predictions.iloc[idx].get('pred_stack_thickness_1', None)
            ml_pred_front_thickness = ml_predictions.iloc[idx].get('pred_front_thickness_1', None)
        else:
            ml_pred_diameter = None
            ml_pred_indentation = None
            ml_pred_stack_thickness = None
            ml_pred_front_thickness = None

        # Final numeric outputs (prefer ML regression when available)
        diameter_val = ml_pred_diameter if (ml_pred_diameter is not None and pd.notna(ml_pred_diameter)) else row.get(
            'Diameter 1')
        indentation_val = ml_pred_indentation if (
                    ml_pred_indentation is not None and pd.notna(ml_pred_indentation)) else row.get('Indentation 1')
        front_thickness_val = ml_pred_front_thickness if (
                    ml_pred_front_thickness is not None and pd.notna(ml_pred_front_thickness)) else row.get(
            'Front Thickness 1')
        stack_thickness_val = ml_pred_stack_thickness if (
                    ml_pred_stack_thickness is not None and pd.notna(ml_pred_stack_thickness)) else row.get(
            'Stack Thickness 1')

        # Get rule-based prediction if enabled
        # if use_rules:
        #     try:
        #         # Prepare data for rule-based model
        #         rule_data = data.iloc[idx].to_dict()
        #         rule_output = peakrules_model(
        #             data_ins=rule_data,
        #             on_off=True,
        #             data_model='ck',  # Data from ClickHouse
        #             save_flag=False,  # Don't save plots
        #             save_file=None,
        #             save_file_dict=None
        #         )
        #         rule_result = rule_output.get('result', 'ok')
        #     except Exception as e:
        #         print(f"Rule-based model error for row {idx}: {e}")
        #         rule_result = 'ok'  # Default to OK on error
        if use_rules:
            rule_result = rule_predictions[idx]

        # # ==================== COMBINE MODEL RESULTS (AND gating) ====================
        # # Only return NOK when BOTH models are NOK. If one model is disabled, fall back to the enabled one.
        # is_ml_nok = (use_ml and ml_label == 'NOK')
        # is_rule_nok = (use_rules and rule_result == 'nok')

        # if use_ml and use_rules:
        #     is_final_nok = (is_ml_nok and is_rule_nok)   # strict AND
        # elif use_ml:
        #     is_final_nok = is_ml_nok                     # only ML available
        # elif use_rules:
        #     is_final_nok = is_rule_nok                   # only rules available
        # else:
        #     is_final_nok = False

        # final_oknok = 'NOK' if is_final_nok else 'OK'

        # # Defect type: only name defects when final is NOK (i.e., both fired)
        # if is_final_nok:
        #     defect_type_parts = []
        #     if is_ml_nok:
        #         defect_type_parts.append('小焊核')
        #     if is_rule_nok:
        #         defect_type_parts.append('表面缺陷')
        #     defect_type = ','.join(defect_type_parts) if defect_type_parts else '不明缺陷'
        # else:
        #     defect_type = '合格'
        # # ===========================================================================

        # is_ml_nok = (use_ml and ml_label == 'NOK')
        # is_rule_nok = (use_rules and rule_result == 'nok')

        # # Soft-AND: AND by default, but let very high ML score override.
        # ML_OVERRIDE_MULT = 50.0   # tune: 20–100
        # strong_ml = False
        # if use_ml and (ml_prob is not None):
        #     strong_ml = (ml_prob > (threshold * ML_OVERRIDE_MULT))

        # is_final_nok = (is_ml_nok and is_rule_nok) or strong_ml
        # final_oknok = 'NOK' if is_final_nok else 'OK'

        # ==================== COMBINE MODEL RESULTS (Soft-AND + defect type) ====================
        is_ml_nok = (use_ml and ml_label == 'NOK')
        is_rule_nok = (use_rules and rule_result == 'nok')

        # Soft-AND: AND by default, but let very high ML score override.
        # ML_OVERRIDE_MULT = 50.0   # tune: 20–100
        ML_OVERRIDE_MULT = 1000.0  # tune: 20–100
        strong_ml = bool(use_ml and (ml_prob is not None) and (ml_prob > (threshold * ML_OVERRIDE_MULT)))

        is_final_nok = (is_ml_nok and is_rule_nok) or strong_ml
        final_oknok = 'NOK' if is_final_nok else 'OK'

        # ---- RESTORE defect_type (label who triggered) ----
        if is_final_nok:
            parts = []
            # Count ML as contributing if it fired normally OR via the override
            if is_ml_nok or strong_ml:
                # Annotate when override is the reason (rules didn't fire)
                parts.append('小焊核' + ('(高置信度)' if strong_ml and not is_rule_nok else ''))
            if is_rule_nok:
                parts.append('表面缺陷')
            defect_type = ','.join(parts) if parts else '不明缺陷'
        else:
            defect_type = '合格'
        # =====================================================================

        raw_result = {
            "diameter": diameter_val,
            "indentation": indentation_val,
            'Front_Thickness': front_thickness_val,
            'Stack_Thickness': stack_thickness_val,
            "defect_type": defect_type,
            "oknok": final_oknok,
            "diameter_oknok": 0,
            "indentation_oknok": 0,
            "surface_oknok": 1 if (use_rules and rule_result == 'nok') else 0,
            "total_oknok": 1 if final_oknok == 'NOK' else 0,
            "energy_oknok": 0,
            "exist_nok": 1 if final_oknok == 'NOK' else 0,
            "ml_confidence": float(ml_prob) if ml_prob is not None else None,

            "ml_pred_diameter": float(ml_pred_diameter) if (
                        ml_pred_diameter is not None and pd.notna(ml_pred_diameter)) else None,
            "ml_pred_indentation": float(ml_pred_indentation) if (
                        ml_pred_indentation is not None and pd.notna(ml_pred_indentation)) else None,
            "ml_pred_front_thickness": float(ml_pred_front_thickness) if (
                        ml_pred_front_thickness is not None and pd.notna(ml_pred_front_thickness)) else None,
            "ml_pred_stack_thickness": float(ml_pred_stack_thickness) if (
                        ml_pred_stack_thickness is not None and pd.notna(ml_pred_stack_thickness)) else None,
            "ml_prediction": ml_label if use_ml else None,
            "rule_prediction": rule_result if use_rules else None
        }

        result_row = {
            "weld_result": json.dumps(_to_jsonable(raw_result), ensure_ascii=False, allow_nan=False),
            "defect_type": defect_type,
            "oknok": final_oknok,
            "alg_complete_time": finish_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "diameter": diameter_val,
            "indentation": indentation_val,
            'spot_tag': row.get('spot_tag'),
            'Front_Thickness': front_thickness_val,
            'Stack_Thickness': stack_thickness_val,
            "diameter_oknok": 0,
            "indentation_oknok": 0,
            "surface_oknok": 1 if (use_rules and rule_result == 'nok') else 0,
            "total_oknok": 1 if final_oknok == 'NOK' else 0,
        }
        results.append(result_row)

    result = pd.DataFrame(results)

    # 把result_2['predicted_label']跟result['oknok']合并，
    # Assuming both dataframes have the same row index
    # result.loc[result_2['predicted_label'] == 'NOK', 'oknok'] = 'NOK'
    # result.loc[result_2['predicted_label'] == 'NOK',
    # 'defect_type'] = '小焊核'
    # result_aeq = data.assign(
    #     weld_result=result['weld_result'],
    #     oknok=result['oknok'],
    #     alg_complete_time=result['alg_complete_time'],
    #     diameter=result['diameter'],
    #     indentation=result['indentation'],
    #     Front_Thickness=result['Front_Thickness'],
    #     Stack_Thickness=result['Stack_Thickness'],
    #     defect_type=result['defect_type']
    # )
    result_aeq = data.assign(
        weld_result=result['weld_result'],
        oknok=result['oknok'],
        alg_complete_time=result['alg_complete_time'],
        defect_type=result['defect_type'],
        diameter=result['diameter'],
        indentation=result['indentation'],
        Front_Thickness=result['Front_Thickness'],
        Stack_Thickness=result['Stack_Thickness'],
    )

    def extract(str, cols, row):
        a = json.loads(str)
        for col in cols:
            a[col] = row[col]
        return a

    cols_to_merge = ['diameter', 'indentation', 'Front_Thickness', 'Stack_Thickness']
    result_aeq["weld_result"] = result_aeq.apply(
        lambda row: json.dumps(extract(row["weld_result"], cols_to_merge, row), ensure_ascii=False),
        axis=1)
    end_time = time.time()
    print(f"new_process_inference finished in {end_time - start_time:.4f} seconds.")
    return result_aeq


def write_to_clickhouse_with_curve_data(data):
    """Batch insert data into ClickHouse, with timer."""
    start_time = time.time()
    if data.empty:
        print("No data to insert into ClickHouse.")
        return

    # List of columns to drop that are not in the ClickHouse schema for insertion
    cols_to_drop = [
        'diameter', 'indentation', 'Front_Thickness', 'Stack_Thickness',
        'u_curve_processed', 'i_curve_processed', 'r_curve_processed',
        'u_curve_mean', 'u_curve_std', 'u_curve_max', 'u_curve_min', 'u_curve_median', 'u_curve_range',
        'u_curve_q25', 'u_curve_q75', 'u_curve_iqr', 'u_curve_rms', 'u_curve_zero_crossings',
        'u_curve_abs_mean', 'u_curve_max_abs', 'u_curve_pos_mean', 'u_curve_neg_mean',
        'u_curve_pos_peaks', 'u_curve_neg_peaks', 'u_curve_mean_diff', 'u_curve_std_diff',
        'i_curve_mean', 'i_curve_std', 'i_curve_max', 'i_curve_min', 'i_curve_median', 'i_curve_range',
        'i_curve_q25', 'i_curve_q75', 'i_curve_iqr', 'i_curve_rms', 'i_curve_zero_crossings',
        'i_curve_abs_mean', 'i_curve_max_abs', 'i_curve_pos_mean', 'i_curve_neg_mean',
        'i_curve_pos_peaks', 'i_curve_neg_peaks', 'i_curve_mean_diff', 'i_curve_std_diff',
        'r_curve_mean', 'r_curve_std', 'r_curve_max', 'r_curve_min', 'r_curve_median', 'r_curve_range',
        'r_curve_q25', 'r_curve_q75', 'r_curve_iqr', 'r_curve_rms', 'r_curve_zero_crossings',
        'r_curve_abs_mean', 'r_curve_max_abs', 'r_curve_pos_mean', 'r_curve_neg_mean',
        'r_curve_pos_peaks', 'r_curve_neg_peaks', 'r_curve_mean_diff', 'r_curve_std_diff',
    ]

    # Drop the columns that are not part of the destination table schema
    data_to_insert = data.drop(columns=cols_to_drop, errors='ignore')

    records = [tuple(x) for x in data_to_insert.to_numpy()]

    clickhouse_client.execute(
        f"INSERT INTO {tables['source_table']} ({','.join(data_to_insert.columns)}) VALUES",
        records
    )

    end_time = time.time()
    print(f"Inserted {len(data_to_insert)} rows to ClickHouse in {end_time - start_time:.4f} seconds.")


def save_local_copy(data: pd.DataFrame, save_dir: str):
    """Save a light local copy of inference results (for audit/debug)."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cols = [c for c in [
            "ck_local_time", "weld_time", "spot_tag", "prog_no",
            "oknok", "defect_type", "alg_complete_time",
            "diameter", "indentation", "Front_Thickness", "Stack_Thickness",
            "weld_result"
        ] if c in data.columns]
        out = data[cols].copy() if cols else data.copy()
        out_path = os.path.join(save_dir, f"inference_{ts}_{len(out)}.csv.gz")
        out.to_csv(out_path, index=False, encoding="utf-8", compression="gzip")
        print(f"[LocalSave] Saved batch result -> {out_path}")
    except Exception as e:
        print(f"[LocalSave] Failed: {e}")


def filter_mysql_records(data):
    """Filter records for MySQL insertion and add necessary columns."""
    filtered_data = data[(data['oknok'] == 'NOK') | (data['oknok'] == '0')]
    if filtered_data.empty:
        # print("No records to insert into MySQL.")
        return filtered_data

    return filtered_data.assign(
        workshop_section=data['workshop_section'],
        area=data['area'],
        workshop_position=filtered_data["spot_tag"].str.split("_").str[0],
        defect_type_alg=filtered_data["defect_type"],
        kpi='[]',
        rec_cause='[]',
        atlas_info='Not Available',
        diameter=pd.to_numeric(filtered_data['diameter'], errors='coerce').round(4),
        indentation=pd.to_numeric(filtered_data['indentation'], errors='coerce').round(4),
        Front_Thickness=pd.to_numeric(filtered_data['Front_Thickness'], errors='coerce').round(4),
        Stack_Thickness=pd.to_numeric(filtered_data['Stack_Thickness'], errors='coerce').round(4),
    )


def write_to_mysql(data):
    """Batch insert records into MySQL, with timer."""
    start_time = time.time()
    global mysql_connection
    if data.empty:
        print("No records to insert into MySQL.")
        return

    try:
        # Ping the connection to ensure it is still alive. Reconnect if necessary.
        # This is a robust way to handle long-running connections.
        mysql_connection.ping(reconnect=True)

        with mysql_connection.cursor() as cursor:
            insert_columns = [
                'workshop_section', 'area', 'workshop_position',
                'defect_type_alg', 'spot_tag', 'rfid', 'weld_time',
                'kpi', 'rec_cause', 'atlas_info', 'diameter', 'indentation', 'Front_Thickness', 'Stack_Thickness'
            ]

            missing_cols = [col for col in insert_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required fields: {missing_cols}")

            records = data[insert_columns].fillna('not available').values.tolist()
            cursor.executemany(
                f"""INSERT INTO {tables['result_table']}
                    ({','.join(insert_columns)})
                    VALUES ({','.join(['%s'] * len(insert_columns))})""",
                records
            )
            mysql_connection.commit()

            end_time = time.time()
            print(f"Inserted {len(data)} rows to MySQL in {end_time - start_time:.4f} seconds.")

    except Exception as e:
        if mysql_connection:
            mysql_connection.rollback()
        print(f"Failed to insert into MySQL: {str(e)}")
        traceback.print_exc()


writetime = None


async def fetch_and_process_data(predictor=None):
    """
    生产模式：轮询读取“当天 data_date”的未处理数据（alg_complete_time IS NULL），做推理后写回 ClickHouse + MySQL（仅 NOK）并本地落盘。
    - 不会扫全库 10G：每次只取 batch_size 行，且带增量游标 ck_local_time > writetime。
    - 若当天没有新数据：sleep(loop_interval_seconds) 后继续轮询。
    """
    global writetime, mysql_connection

    WAIT_TIME = int(config.get("loop_interval_seconds", config.get("polling_interval", 5)))
    BATCH_SIZE = int(config.get("batch_size", 50))

    # 初始化 MySQL 连接（长跑任务需要 ping）
    try:
        mysql_connection = get_mysql_connection()
        print("Global MySQL connection initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize global MySQL connection: {e}")

    data_date_override = config.get("data_date_override", None)

    while True:
        try:
            batch_start_time = time.time()

            if mysql_connection:
                mysql_connection.ping(reconnect=True)

            # 当天 data_date（YYYYMMDD）
            if data_date_override:
                data_time = str(data_date_override)
            else:
                data_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")

            base_where = f"""
                SELECT * FROM {tables['source_table']} FINAL
                WHERE weld_time IS NOT NULL AND prog_no NOT IN ('41', '51')
                AND spot_tag IS NOT NULL AND spot_tag != '0'
                AND length(u_curve) != 0 AND length(i_curve) != 0
                AND umax != '0' AND data_date = '{data_time}'
            """

            params = {}
            if writetime is None:
                query = f"""{base_where} AND alg_complete_time IS NULL
                ORDER BY ck_local_time ASC
                LIMIT {BATCH_SIZE}"""
            else:
                params = {'writetime': writetime}
                query = f"""{base_where} AND alg_complete_time IS NULL AND ck_local_time > %(writetime)s
                ORDER BY ck_local_time ASC
                LIMIT {BATCH_SIZE}"""

            result = clickhouse_client.execute(query, params)
            columns = [desc[0] for desc in clickhouse_client.execute(f"DESCRIBE TABLE {tables['source_table']}")]
            data = pd.DataFrame(result, columns=columns)

            if data.empty:
                print(f"No new data for data_date={data_time}. Waiting {WAIT_TIME}s...")
                time.sleep(WAIT_TIME)
                continue

            # 更新游标（下一轮从最后一条之后继续）
            writetime = data.iloc[-1]['ck_local_time']

            valid_data = validate_curve_data(data.copy())
            if valid_data.empty:
                print("All rows in this batch are invalid (curve/umax/imax). Skipping.")
                time.sleep(0.2)
                continue

            # 推理：ML + Rule（取决于开关）
            new_processed = new_process_inference(
                valid_data.copy(),
                predictor,
                threshold=CLASSIFICATION_THRESHOLD,
                use_ml=USE_ML_MODEL,
                use_rules=USE_RULE_MODEL
            )

            if 'weld_result' in new_processed.columns:
                new_processed['weld_result'] = new_processed['weld_result'].astype(str)

            # 回填关键字段到 valid_data（用于写库）
            valid_data['weld_result'] = new_processed['weld_result']
            valid_data['oknok'] = new_processed['oknok']
            valid_data['alg_complete_time'] = new_processed['alg_complete_time']
            valid_data['defect_type'] = new_processed['defect_type']
            valid_data['diameter'] = new_processed['diameter']
            valid_data['indentation'] = new_processed['indentation']
            valid_data['Front_Thickness'] = new_processed['Front_Thickness']
            valid_data['Stack_Thickness'] = new_processed['Stack_Thickness']

            # 写回 ClickHouse + MySQL（只写 NOK）
            write_to_clickhouse_with_curve_data(valid_data)
            mysql_records = filter_mysql_records(valid_data)
            write_to_mysql(mysql_records)

            # 本地落盘
            save_local_copy(valid_data, save_dir)

            batch_end_time = time.time()
            print(
                f"Batch done: data_date={data_time} rows={len(valid_data)} "
                f"ck_local_time_last={writetime} "
                f"elapsed={batch_end_time - batch_start_time:.4f}s"
            )

        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()
            time.sleep(WAIT_TIME)
@app.on_event("startup")
async def startup_event():
    """Startup event handler to load data and start the background task."""
    global curve_s_data, peakrules_model, USE_RULE_MODEL

    # try:
    #     curve_s_data = pd.read_excel(curve_s_file, sheet_name='spot_tag_标准曲线')
    #     print(f"Loaded standard curves: {len(curve_s_data)} records")
    # except Exception as e:
    #     print(f"Failed to load standard curves: {e}")
    #     curve_s_data = None
    model_weights_path = os.environ.get('MODEL_EXPORT_DIR', config.get('model_export_dir', 'model_export'))
    # model_weights_path can be a directory (recommended) or a direct .pth file

    try:
        norm_coeffs = config.get("norm_coeffs", None)
        target_len = int(config.get("target_len", TARGET_LEN_DEFAULT))
        predictor = Predictor(model_weights_path, norm_coeffs=norm_coeffs, target_len=target_len)
        print("UIP Predictor initialized successfully.")
        print(f"Using classification threshold: {CLASSIFICATION_THRESHOLD:.6e}")
        print(f"Threshold source: {CALIBRATED_THRESHOLD_PATH if _loaded_T is not None else 'default/config'}")
        print(f"ML Model: {'ENABLED' if USE_ML_MODEL else 'DISABLED'}")
        print(f"Rule Model: {'ENABLED' if USE_RULE_MODEL else 'DISABLED'}")
    except FileNotFoundError as e:
        print(f"Failed to load UIP Predictor model: {e}")
        predictor = None
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        predictor = None


    # ---------- Rule Model lazy-import (避免规则文件异常拖死服务) ----------
    peakrules_model = None
    if USE_RULE_MODEL:
        # 把 stat_dict_path 传给规则模块（如果它需要）
        stat_path = str(config.get("stat_dict_path", "") or "")
        if stat_path:
            os.environ.setdefault("STAT_DICT_PATH", stat_path)
        try:

            _pm, _src = load_peakrules_model_safely(config.get('rule_module_path', 'qdyq_peakrules_model.py'))

            peakrules_model = _pm

            print(f"[RuleModel] Loaded peakrules_model via {_src} (no file changes)")
        except Exception as e:
            print(f"[RuleModel] Import failed, disabling rule model: {e}")
            USE_RULE_MODEL = False
            peakrules_model = None
    # ----------------------------------------------------------------------

    asyncio.create_task(fetch_and_process_data(predictor))


@app.get("/")
async def root():
    """
    Test endpoint.
    """
    return {"message": "Batch processing service for weld_confirm is running!",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "ml_model_enabled": USE_ML_MODEL,
            "rule_model_enabled": USE_RULE_MODEL}


if __name__ == "__main__":
    # Start FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)