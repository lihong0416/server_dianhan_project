import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn.functional as F
import ast
import re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import json
import hashlib
import copy
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay, confusion_matrix, mean_squared_error, \
    r2_score
import mlflow


# ----------------- MLflow helpers -----------------
_MLFLOW_ALLOWED_PATTERN = re.compile(r"[^0-9A-Za-z_\-\. :/]")

def mlflow_safe_name(name: str) -> str:
    """Make a string safe for MLflow metric keys.
    MLflow metric names may only contain alphanumerics, underscores, dashes (-),
    periods (.), spaces, colon(:) and slashes (/). Characters like '+' are invalid.
    """
    if name is None:
        return "Unknown"
    s = str(name).strip()
    # Common problematic chars
    s = s.replace("+", "plus")
    # Normalize whitespace to underscore to avoid trailing spaces
    s = re.sub(r"\s+", "_", s)
    # Replace any remaining illegal chars with underscore
    s = _MLFLOW_ALLOWED_PATTERN.sub("_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "Unknown"


import mlflow.pytorch
import glob
import traceback

warnings.filterwarnings("ignore")

MLFLOW_ENABLED = False
mlflow_tracking_uri = "http://10.6.0.181:30081/"
mlflow_experiment_name = "ZL_Training_Improved"
CONSISTENCY_LOSS_WEIGHT = 200
# 默认值（保持你现在的写法）
DEFAULT_AEQ_INPUT_ROOT = "/mlm/Qingdao/V2"
DEFAULT_OUTPUT_ROOT    = "/mlm/Qingdao/V2"

# 环境变量优先：支持你 export 的方式
AEQ_INPUT_ROOT = os.getenv("AEQ_INPUT_ROOT", DEFAULT_AEQ_INPUT_ROOT)
OUTPUT_ROOT    = os.getenv("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)

# 关键：DATA_DIR / WEIGHTS_PATH 也允许 env 覆盖
DATA_DIR    = os.getenv("DATA_DIR", os.path.join(AEQ_INPUT_ROOT, "feather_data_v2"))
CACHE_FILE  = os.getenv("CACHE_FILE", os.path.join(OUTPUT_ROOT, "V2_combined_sampled_data_cache.feather"))
SAMPLE_FRACTION = 1.0  # 训练用全量数据（不再随机采样）

try:
    print(f"MLflow tracking URI set to {mlflow_tracking_uri}.")
except Exception as e:
    print(f"Warning: MLflow configuration failed. Proceeding without MLflow. Error: {e}")
    MLFLOW_ENABLED = False

# --- 权重调整 ---
# 确认 Decision Loss 权重，若需提高其在总Loss中的影响，可再次调整此值
TESS_DECISION_WEIGHT = 4.0
REG_LOSS_WEIGHT = 3.0
LABEL_COLUMN = "qrk2_decision"

# --- Configuration (配置部分) ---
CONFIG = {
    "global_epochs": 20,
    "finetune_epochs": 1,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "finetune_lr": 1e-4,
    "weights": {
        "tess_decision": TESS_DECISION_WEIGHT,  # 使用 TESS_DECISION_WEIGHT 常量
        "reg_loss": REG_LOSS_WEIGHT,
        "consistency": CONSISTENCY_LOSS_WEIGHT
    },
    "max_allowed_fpr": 0.02,
    "dirs": {
        "log": "logs",
        "checkpoint": "checkpoints",
        "plots": "roc_plots",
        "export": "model_export"
    }
}

# 将所有输出目录统一放到 OUTPUT_ROOT 下，避免无权限写入当前目录
CONFIG["dirs"] = {k: os.path.join(OUTPUT_ROOT, v) for k, v in CONFIG["dirs"].items()}

for d in CONFIG["dirs"].values():

    os.makedirs(d, exist_ok=True)

# ----------------- DataLoader acceleration helpers -----------------
# 解决“单线程主进程喂数据，GPU等CPU”的核心：多进程 DataLoader + pinned memory + 预取
# Windows 下多进程 DataLoader 常见兼容性问题：自动回退到 num_workers=0
def _default_num_workers():
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    # 经验值：4~8 通常就够了，过大反而争抢CPU/IO
    return max(2, min(8, n // 2))

_DATALOADER_NUM_WORKERS = 0 if os.name == "nt" else _default_num_workers()
_DATALOADER_PIN_MEMORY = torch.cuda.is_available()
_DATALOADER_PERSISTENT = (_DATALOADER_NUM_WORKERS > 0)
_DATALOADER_PREFETCH = 2  # 每个 worker 预取 batch 数，太大可能占内存


def _sanitize_key(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s))
    return s[:120] if len(s) > 120 else s

def make_loader(df, material_map, batch_size, shuffle, collate_fn, cache_key: str, rebuild_cache: bool = False):
    """统一创建 DataLoader（多进程+pin_memory）并启用曲线特征缓存。"""
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=_DATALOADER_NUM_WORKERS,
        pin_memory=_DATALOADER_PIN_MEMORY,
    )
    # 只有 num_workers>0 时才能使用这些参数
    if _DATALOADER_NUM_WORKERS > 0:
        kwargs.update(dict(
            persistent_workers=_DATALOADER_PERSISTENT,
            prefetch_factor=_DATALOADER_PREFETCH,
        ))

    ds = CustomDataset(
        df,
        material_map=material_map,
        cache_dir=CURVE_CACHE_DIR,
        cache_key=_sanitize_key(cache_key),
        use_cache=True,
        rebuild_cache=rebuild_cache,
        target_len=1000,
    )
    return DataLoader(ds, **kwargs)

# 给用户一个可见提示（不影响训练）
print(f"[DataLoader] num_workers={_DATALOADER_NUM_WORKERS}, pin_memory={_DATALOADER_PIN_MEMORY}, "
      f"persistent_workers={_DATALOADER_PERSISTENT}")


# ----------------- Curve feature cache (pre-parse once, reuse many epochs) -----------------
# 把 __getitem__ 里的 ast.literal_eval + pad/cat 重活挪到这里：首次运行构建缓存，后续直接读取
CURVE_CACHE_DIR = os.path.join(OUTPUT_ROOT, "curve_feature_cache")
os.makedirs(CURVE_CACHE_DIR, exist_ok=True)
CURVE_CACHE_VERSION = 1
logging.basicConfig(
    filename=os.path.join(CONFIG["dirs"]["log"], "training_enhanced.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def pre_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)

    df_cleaned = df.dropna(subset=['i_curve', 'u_curve'])
    numerical_cols = ['imax', 'umax']
    for col in numerical_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    df_cleaned = df_cleaned.dropna(subset=numerical_cols)

    def is_valid_curve_string(s):
        if not isinstance(s, str):
            return True
        try:
            result = ast.literal_eval(s)
            return isinstance(result, (list, tuple)) and len(result) > 0
        except:
            return False

    valid_i = df_cleaned['i_curve'].apply(is_valid_curve_string)
    valid_u = df_cleaned['u_curve'].apply(is_valid_curve_string)

    df_cleaned = df_cleaned[valid_i & valid_u]

    final_count = len(df_cleaned)
    print(
        f"数据预清洗：原始数据 {initial_count} 条，清洗后 {final_count} 条。共过滤掉 {initial_count - final_count} 条异常数据。")

    return df_cleaned


def plot_and_save_roc_curve(y_true, y_pred_prob, epoch, mode_name="Global", threshold=0.5):
    ml_mode = mlflow_safe_name(mode_name)
    """绘制并保存ROC曲线和分类报告"""
    try:
        # 1. Prepare data (Filter NaN labels)
        mask = ~np.isnan(y_true)
        y_true_filtered = y_true[mask]
        y_pred_prob_filtered = y_pred_prob[mask]

        if len(np.unique(y_true_filtered)) < 2:
            logging.warning(f"Epoch {epoch + 1}: Not enough unique classes for AUC/Report in {mode_name}.")
            return

        plt.figure(figsize=(8, 8))
        RocCurveDisplay.from_predictions(y_true_filtered, y_pred_prob_filtered, ax=plt.gca())
        auc_score = roc_auc_score(y_true_filtered, y_pred_prob_filtered)
        plt.title(f'ROC Curve - {mode_name} - Epoch {epoch + 1} (AUC: {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
        plt.legend()
        plt.grid(True)
        filename = f"roc_{mode_name}_epoch_{epoch + 1}.png"
        plot_path = os.path.join(CONFIG["dirs"]["plots"], filename)
        plt.savefig(plot_path)
        plt.close()

        y_pred_class = (y_pred_prob_filtered > threshold).astype(int)
        report = classification_report(y_true_filtered, y_pred_class, output_dict=True, zero_division=0)

        report_str = f"\n--- Classification Report ({mode_name} Epoch {epoch + 1}, Thresh={threshold:.2f}) ---\n"
        report_str += f"AUC: {auc_score:.4f}\n"
        report_str += classification_report(y_true_filtered, y_pred_class, zero_division=0)
        report_str += "----------------------------------------------------------------\n"

        print(report_str)
        logging.info(report_str)

        if MLFLOW_ENABLED:
            mlflow.log_metric(f"{ml_mode}_AUC", auc_score, step=epoch)
            mlflow.log_metrics({f"{ml_mode}_{k}_precision": v['precision'] for k, v in report.items() if k.isdigit()},
                               step=epoch)
            mlflow.log_metrics({f"{ml_mode}_accuracy": report['accuracy']}, step=epoch)

    except Exception as e:
        # 避免冗余的异常处理：这里只记录错误，不使用模拟数据或假设逻辑
        logging.error(f"Failed to plot ROC or print report: {e}")


def plot_loss_curve(train_losses, val_losses, mode_name="Global"):
    """绘制 Loss 曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {mode_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG["dirs"]["plots"], f"loss_{mode_name}.png"))
    plt.close()


class ConsistencyLoss(nn.Module):
    def __init__(self, nok_diameter_threshold=3.0, ok_diameter_threshold=2.0, reduction='mean'):
        super(ConsistencyLoss, self).__init__()
        self.nok_diameter_threshold = nok_diameter_threshold
        self.ok_diameter_threshold = ok_diameter_threshold
        self.reduction = reduction

    def forward(self, decision_logits, pred_diameter):
        # Keep batch dimension even when batch_size==1 (avoid 0-d tensors)
        decision_logits = decision_logits.reshape(-1)
        pred_diameter = pred_diameter.reshape(-1)

        # If one of them is a scalar, expand to match the other (rare edge case)
        if pred_diameter.numel() != decision_logits.numel():
            if pred_diameter.numel() == 1:
                pred_diameter = pred_diameter.expand_as(decision_logits)
            elif decision_logits.numel() == 1:
                decision_logits = decision_logits.expand_as(pred_diameter)
            else:
                raise ValueError(f"Shape mismatch in ConsistencyLoss: decision_logits={decision_logits.shape}, pred_diameter={pred_diameter.shape}")
        prob_nok = torch.sigmoid(decision_logits)
        prob_ok = 1.0 - prob_nok
        diameter_excess = F.relu(pred_diameter - self.nok_diameter_threshold)
        penalty1 = prob_nok * (diameter_excess ** 2)
        diameter_deficit = F.relu(self.ok_diameter_threshold - pred_diameter)
        penalty2 = prob_ok * (diameter_deficit ** 2)
        loss = penalty1 + penalty2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # none
            return loss


class DiameterAwareLoss(nn.Module):
    def __init__(self, threshold=2.0, reduction='mean'):
        super(DiameterAwareLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.dim() == 0: pred = pred.unsqueeze(0)
        if target.dim() == 0: target = target.unsqueeze(0)

        loss = torch.zeros_like(target)
        zero_mask = (target == 0)
        nonzero_mask = (target > 0)

        if nonzero_mask.any():
            loss[nonzero_mask] = self.mse(pred[nonzero_mask], target[nonzero_mask])
        if zero_mask.any():
            zero_preds = pred[zero_mask]
            diff = F.relu(zero_preds - self.threshold)
            loss[zero_mask] = diff ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # none
            return loss



class CustomDataset(Dataset):
    """
    加速版 Dataset：把 __getitem__ 中最耗 CPU 的
      - ast.literal_eval / 曲线字符串解析
      - u/i/r/p 计算
      - pad/truncate + torch.cat/stack
    挪到 Dataset 初始化阶段一次性预处理，并落盘缓存（.pt）。
    训练时 __getitem__ 只做 O(1) 索引返回，GPU 不再等 CPU 喂数据。
    """

    def __init__(
        self,
        dataframe,
        norm_coeffs=None,
        reg_cols=None,
        material_map=None,
        cache_dir=None,
        cache_key="dataset",
        use_cache=True,
        rebuild_cache=False,
        target_len=1000,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.reg_cols = ['diameter_1', 'indentation_1', 'front_thickness_1', 'stack_thickness_1']
        default_coeffs = {'R': 1, 'I': 5, 'U': 1, 'P': 2, 'Pulse': 50}
        if norm_coeffs:
            default_coeffs.update(norm_coeffs)
        self.norm_coeffs = default_coeffs
        self.material_map = material_map
        self.target_len = int(target_len)

        self.use_cache = bool(use_cache and cache_dir)
        self.cache_dir = cache_dir
        self.cache_key = str(cache_key)
        self.rebuild_cache = bool(rebuild_cache)

        # 缓存容器（CPU tensor）
        self._features = None          # [N, 4, L]
        self._decision = None          # [N]
        self._reg_targets = None       # [N, 4]
        self._is_tessonic = None       # [N]
        self._mat_id = None            # [N]
        self._valid = None             # [N] bool

        if self.use_cache:
            self._init_or_build_cache()

    def __len__(self):
        return len(self.data)

    # ----------------- cache helpers -----------------
    def _material_map_hash(self):
        if not self.material_map:
            return "nomap"
        try:
            s = json.dumps(self.material_map, sort_keys=True, ensure_ascii=False)
        except Exception:
            s = str(sorted(self.material_map.items()))
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

    def _df_signature(self):
        # 轻量签名：只取头部若干行，避免对大列全量 hash 过慢
        cols = [c for c in ["spot_tag", "part_all", "qrk2_decision", "diameter_1", "is_tessonic"] if c in self.data.columns]
        try:
            sample = self.data[cols].head(64).to_json()
        except Exception:
            sample = str(self.data[cols].head(64).values.tolist()) if cols else str(len(self.data))
        return hashlib.md5(sample.encode("utf-8")).hexdigest()[:10]

    def _cache_path(self):
        n = len(self.data)
        sig = self._df_signature()
        maph = self._material_map_hash()
        fname = f"curve_cache_{self.cache_key}_n{n}_{sig}_{maph}_v{CURVE_CACHE_VERSION}.pt"
        return os.path.join(self.cache_dir, fname)

    def _init_or_build_cache(self):
        path = self._cache_path()
        if (not self.rebuild_cache) and os.path.exists(path):
            try:
                obj = torch.load(path, map_location="cpu")
                self._features = obj["features"]
                self._decision = obj["decision"]
                self._reg_targets = obj["reg_targets"]
                self._is_tessonic = obj["is_tessonic"]
                self._mat_id = obj["mat_id"]
                self._valid = obj["valid"]
                return
            except Exception as e:
                print(f"[Cache] 读取缓存失败，将重建: {path} ({e})")

        print(f"[Cache] Building curve feature cache: {path}")
        self._build_cache()
        try:
            torch.save(
                {
                    "features": self._features,
                    "decision": self._decision,
                    "reg_targets": self._reg_targets,
                    "is_tessonic": self._is_tessonic,
                    "mat_id": self._mat_id,
                    "valid": self._valid,
                },
                path,
            )
        except Exception as e:
            print(f"[Cache] 写缓存失败（不影响训练，仅本次无法复用）: {e}")

    @staticmethod
    def _parse_curve_to_np(x):
        """
        更快的曲线解析：
        - 优先 np.fromstring 解析类似 "[1,2,3]"（最快）
        - 失败则 json.loads
        - 再失败才用 ast.literal_eval（最慢）
        """
        if x is None:
            raise ValueError("curve is None")
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=np.float32)

        if isinstance(x, str):
            s = x.strip()
            if not s:
                raise ValueError("empty curve string")
            # 1) np.fromstring
            try:
                core = s.strip("[]")
                arr = np.fromstring(core, sep=",", dtype=np.float32)
                if arr.size > 0:
                    return arr
            except Exception:
                pass
            # 2) json
            try:
                arr = np.asarray(json.loads(s), dtype=np.float32)
                return arr
            except Exception:
                pass
            # 3) ast fallback
            arr = np.asarray(ast.literal_eval(s), dtype=np.float32)
            return arr

        raise ValueError(f"unsupported curve type: {type(x)}")

    def _build_cache(self):
        n = len(self.data)
        L = self.target_len

        # 预先把标量列向量化（快）
        # decision
        if "qrk2_decision" in self.data.columns:
            decision_np = pd.to_numeric(self.data["qrk2_decision"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        else:
            decision_np = np.full((n,), np.nan, dtype=np.float32)

        # is_tessonic
        if "is_tessonic" in self.data.columns:
            is_tess_np = pd.to_numeric(self.data["is_tessonic"], errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
        else:
            is_tess_np = np.zeros((n,), dtype=np.float32)

        # reg_targets
        reg_np = np.full((n, len(self.reg_cols)), np.nan, dtype=np.float32)
        for j, c in enumerate(self.reg_cols):
            if c in self.data.columns:
                reg_np[:, j] = pd.to_numeric(self.data[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)

        # mat_id
        if self.material_map and "part_all" in self.data.columns:
            mats = self.data["part_all"].fillna("unknown").astype(str).tolist()
            mat_id_np = np.array([self.material_map.get(m, -1) for m in mats], dtype=np.int64)
        else:
            mat_id_np = np.full((n,), -1, dtype=np.int64)

        # curves
        i_list = self.data["i_curve"].tolist() if "i_curve" in self.data.columns else [None] * n
        u_list = self.data["u_curve"].tolist() if "u_curve" in self.data.columns else [None] * n

        # features：预先置 0，后面只填前 nlen 部分
        features_np = np.zeros((n, 4, L), dtype=np.float32)
        valid_np = np.ones((n,), dtype=bool)

        coeff_I = float(self.norm_coeffs.get("I", 1.0))
        coeff_U = float(self.norm_coeffs.get("U", 1.0))
        coeff_R = float(self.norm_coeffs.get("R", 1.0))
        coeff_P = float(self.norm_coeffs.get("P", 1.0))

        for idx in tqdm(range(n), desc=f"[Cache] Parsing curves ({self.cache_key})", leave=False):
            try:
                i_arr = self._parse_curve_to_np(i_list[idx]) / coeff_I
                u_arr = self._parse_curve_to_np(u_list[idx]) / coeff_U

                if i_arr.size == 0 or u_arr.size == 0:
                    valid_np[idx] = False
                    continue

                # 长度不一致：取最短对齐（避免异常）
                mlen = min(i_arr.size, u_arr.size)
                i_arr = i_arr[:mlen]
                u_arr = u_arr[:mlen]

                # r / p
                i_nonzero = np.where(i_arr == 0.0, 1e-10, i_arr)
                r_arr = (u_arr / i_nonzero) / coeff_R
                p_arr = (u_arr * i_arr) / coeff_P

                # pad / truncate 到 L
                nlen = min(mlen, L)
                features_np[idx, 0, :nlen] = u_arr[:nlen]
                features_np[idx, 1, :nlen] = i_arr[:nlen]
                features_np[idx, 2, :nlen] = r_arr[:nlen]
                features_np[idx, 3, :nlen] = p_arr[:nlen]
            except Exception:
                valid_np[idx] = False

        # 转 torch tensor（CPU）
        self._features = torch.from_numpy(features_np)                    # float32
        self._decision = torch.from_numpy(decision_np)                    # float32
        self._reg_targets = torch.from_numpy(reg_np)                      # float32
        self._is_tessonic = torch.from_numpy(is_tess_np)                  # float32
        self._mat_id = torch.from_numpy(mat_id_np)                        # int64
        self._valid = torch.from_numpy(valid_np.astype(np.bool_))         # bool

    # ----------------- standard dataset api -----------------
    def __getitem__(self, idx):
        # 如果启用了缓存：O(1) 返回
        if self.use_cache and self._features is not None:
            if not bool(self._valid[idx]):
                return None, None, None, None, None
            features = self._features[idx]
            decision = self._decision[idx]
            reg_targets = self._reg_targets[idx]
            is_tessonic = self._is_tessonic[idx]
            mat_id = self._mat_id[idx]
            return features, decision, reg_targets, is_tessonic, mat_id

        # 兜底：不启用缓存时用原逻辑（保留兼容）
        row = self.data.iloc[idx]
        if 'i_curve' not in row or 'u_curve' not in row:
            return None, None, None, None, None

        try:
            i_curve = ast.literal_eval(row["i_curve"]) if isinstance(row['i_curve'], str) else row['i_curve']
            u_curve = ast.literal_eval(row["u_curve"]) if isinstance(row['u_curve'], str) else row['u_curve']
            i_curve = torch.tensor(i_curve, dtype=torch.float32)
            u_curve = torch.tensor(u_curve, dtype=torch.float32)
            i = (i_curve) / self.norm_coeffs['I']
            u = (u_curve) / self.norm_coeffs['U']
            i_nonzero = i.clamp(min=1e-10)
            r = (u / i_nonzero) / self.norm_coeffs['R']
            p = (u * i) / self.norm_coeffs['P']
            u = self._pad_or_truncate(u, 1000)
            i = self._pad_or_truncate(i, 1000)
            r = self._pad_or_truncate(r, 1000)
            p = self._pad_or_truncate(p, 1000)
            features = torch.stack([u, i, r, p], dim=0)
        except Exception:
            raise ValueError(f"Data parsing error at index {idx}")
        reg_targets = torch.tensor([float(row.get(c, np.nan)) for c in self.reg_cols], dtype=torch.float32)
        decision = row.get('qrk2_decision', float('nan'))
        decision = torch.tensor(decision, dtype=torch.float32) if pd.notna(decision) else torch.tensor(float('nan'))
        is_tessonic = torch.tensor(row.get('is_tessonic', 0), dtype=torch.float32)
        mat_name = row.get('part_all', 'unknown')
        mat_id = self.material_map.get(mat_name, -1) if self.material_map else -1
        mat_id = torch.tensor(mat_id, dtype=torch.long)
        return features, decision, reg_targets, is_tessonic, mat_id

    def _pad_or_truncate(self, tensor, target_length):
        current_len = tensor.size(0)
        if current_len < target_length:
            return torch.cat([tensor, torch.zeros(target_length - current_len, dtype=tensor.dtype)])
        else:
            return tensor[:target_length]


def custom_collate_fn(batch):
    """把 batch 里无效样本过滤掉，并把各字段 stack 成 batch tensor。
    Dataset 约定返回：(features, decision, reg_targets, is_tessonic_flag, mat_id)
    """
    # 过滤无效样本
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None, None
    features, decisions, reg_targets, is_tessonic_flags, mat_ids = zip(*batch)
    return (torch.stack(features), torch.stack(decisions), torch.stack(reg_targets),
            torch.stack(is_tessonic_flags), torch.stack(mat_ids))


class UIPRegreator(nn.Module):
    def __init__(self, in_channels=4, dropout_prob=0.2):
        super(UIPRegreator, self).__init__()
        self.feature_extractors = nn.ModuleList([self._create_feature_extractor() for _ in range(in_channels)])
        self.map1 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout_prob)
        conv_output_size = 128 * in_channels

        self.regression_heads = nn.ModuleDict()
        for col in ['diameter_1', 'indentation_1', 'stack_thickness_1', 'front_thickness_1']:
            self.regression_heads[col] = nn.Sequential(
                nn.Linear(conv_output_size, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1)
            )
        self.decision_head = nn.Sequential(nn.Linear(conv_output_size, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def _create_feature_extractor(self):
        def d_block(in_f, out_f, norm=True):
            layers = [nn.Conv1d(in_f, out_f, 5, stride=2, padding=2)]
            if norm: layers.append(nn.InstanceNorm1d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        return nn.Sequential(
            *d_block(1, 1024, False), *d_block(1024, 512, False), *d_block(512, 256),
            *d_block(256, 128), *d_block(128, 64), nn.Conv1d(64, 16, 5, padding=2, bias=False)
        )

    def forward(self, x):
        feats = [ext(x[:, i:i + 1, :]) for i, ext in enumerate(self.feature_extractors)]
        x_cat = torch.cat(feats, dim=1)
        x_flat = self.dropout(F.leaky_relu(self.map1(x_cat), 0.2).view(x_cat.size(0), -1))
        return {'qrk2_decision': self.decision_head(x_flat), 'diameter_1': self.regression_heads['diameter_1'](x_flat),
                'indentation_1': self.regression_heads['indentation_1'](x_flat),
                'stack_thickness_1': self.regression_heads['stack_thickness_1'](x_flat),
                'front_thickness_1': self.regression_heads['front_thickness_1'](x_flat)
                }


# =================================================================
# =================================================================

def find_optimal_threshold(y_true, y_probs, max_fpr=0.02):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_thresh = 0.95
    best_recall = -1.0
    fallback_thresh = 0.5
    best_f1 = -1.0

    mask = ~np.isnan(y_true)
    y_true, y_probs = y_true[mask], y_probs[mask]
    if len(y_true) == 0: return 0.5

    total_neg = (y_true == 0).sum()
    total_pos = (y_true == 1).sum()
    if total_neg == 0: return 0.5

    for t in thresholds:
        y_pred = (y_probs > t).astype(int)

        # 使用 sklearn.metrics.confusion_matrix
        try:
            # 确保标签和预测只有 0 或 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            # 如果某一方只有一类，confusion_matrix.ravel() 会失败，跳过
            continue

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            fallback_thresh = t

        if fpr <= max_fpr:
            if recall > best_recall:
                best_recall = recall
                best_thresh = t

    if best_recall == -1.0:
        return float(fallback_thresh)
    return float(best_thresh)




def compute_losses(outputs, decision, reg_targets, criteria):
    """
    统一计算三类 loss，并返回：
    total_loss, d_loss, r_loss, c_loss （均为标量 tensor）

    outputs: model 输出 dict，至少包含 'qrk2_decision' 与 'diameter_1'，
             可能还包含 'indentation_1' / 'front_thickness_1' / 'stack_thickness_1' 等
    decision: (B,) 0/1 label，允许 NaN
    reg_targets: (B,4) [diameter_1, indentation_1, front_thickness_1, stack_thickness_1]，允许 NaN
    criteria: dict，包含
        - 'qrk2_decision': BCEWithLogitsLoss(reduction='none')
        - 'diameter_1': DiameterAwareLoss(reduction='none')
        - 'regression': MSELoss(reduction='none')
        - 'consistency': ConsistencyLoss(reduction='none')
    """
    device = decision.device

    # ---------- 1) Decision (BCE) ----------
    decision_logits = outputs.get('qrk2_decision', None)
    if decision_logits is None:
        raise KeyError("Model outputs missing key 'qrk2_decision'. Please check model.forward return dict.")
    decision_logits = decision_logits.view(-1)
    decision = decision.view(-1)

    d_loss_raw = criteria['qrk2_decision'](decision_logits, decision)  # (B,)
    d_loss_raw = torch.nan_to_num(d_loss_raw, nan=0.0)

    mask_d = ~torch.isnan(decision)
    if mask_d.any():
        d_loss_val = d_loss_raw[mask_d].mean()
    else:
        d_loss_val = torch.tensor(0.0, device=device, requires_grad=True)

    # ---------- 2) Regression ----------
    # reg_targets: (B,4)
    if reg_targets is None:
        r_loss_val = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        # Diameter
        pred_diam = outputs.get('diameter_1', None)
        if pred_diam is None:
            raise KeyError("Model outputs missing key 'diameter_1'. Please check model.forward return dict.")
        pred_diam = torch.atleast_1d(pred_diam.squeeze())
        target_diam = reg_targets[:, 0].view(-1)

        diam_raw = criteria['diameter_1'](pred_diam, target_diam)  # (B,)
        diam_raw = torch.nan_to_num(diam_raw, nan=0.0)
        mask_diam = ~torch.isnan(target_diam)
        diam_loss = diam_raw[mask_diam].mean() if mask_diam.any() else torch.tensor(0.0, device=device, requires_grad=True)

        # Optional other regressions
        other_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # idx mapping based on CustomDataset.reg_cols
        other_items = [
            (1, 'indentation_1'),
            (2, 'front_thickness_1'),
            (3, 'stack_thickness_1'),
        ]
        for col_idx, out_key in other_items:
            if out_key in outputs:
                pred = outputs[out_key].view(-1)
                tgt = reg_targets[:, col_idx].view(-1)
                l_raw = criteria['regression'](pred, tgt)
                l_raw = torch.nan_to_num(l_raw, nan=0.0)
                m = ~torch.isnan(tgt)
                if m.any():
                    other_loss = other_loss + l_raw[m].mean()

        r_loss_val = diam_loss + other_loss

    # ---------- 3) Consistency ----------
    pred_diam_for_c = outputs.get('diameter_1', None)
    if pred_diam_for_c is None:
        raise KeyError("Model outputs missing key 'diameter_1' (needed for consistency loss).")
    c_loss_raw = criteria['consistency'](decision_logits, pred_diam_for_c)  # (B,) if reduction='none'
    c_loss_raw = torch.nan_to_num(c_loss_raw, nan=0.0)
    c_loss_raw = c_loss_raw.view(-1)  # ensure 1D even for last batch size=1
    # 对齐 mask：用 diameter 的有效性（更稳），也可用 decision mask
    if 'target_diam' in locals():
        mask_c = ~torch.isnan(target_diam)
        mask_c = mask_c.view(-1)
    else:
        mask_c = mask_d
    if mask_c.any():
        c_loss_val = c_loss_raw[mask_c].mean()
    else:
        c_loss_val = torch.tensor(0.0, device=device, requires_grad=True)

    # ---------- 4) Weighted sum ----------
    w = CONFIG.get("weights", {})
    total_loss = (d_loss_val * float(w.get("tess_decision", 1.0))
                  + r_loss_val * float(w.get("reg_loss", 1.0))
                  + c_loss_val * float(w.get("consistency", 1.0)))

    # 防御 NaN
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

    return total_loss, d_loss_val, r_loss_val, c_loss_val
def train_model(model, train_loader, val_loader, epochs, device, criteria, optimizer, mode="Global", resume=True):
    ml_mode = mlflow_safe_name(mode)
    """
    训练入口（支持断点续训 + 从历史最优权重继续训练）

    目标：
    1) 训练中每个 epoch 评估一次 val_loss，若优于历史 best，则保存 best 权重/ckpt；
    2) 中途断掉后再次运行：epoch 计数从 latest_ckpt 恢复（例如从120继续），
       但模型初始化权重优先使用 best_ckpt（历史最优），避免“盲目接着最后一轮继续”。
    """
    mode_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(mode))

    # checkpoint paths
    ckpt_dir = CONFIG["dirs"]["checkpoint"]
    path_latest = os.path.join(ckpt_dir, f"latest_ckpt_{mode_safe}.pth")
    path_best_weights = os.path.join(ckpt_dir, f"best_model_{mode_safe}.pth")     # 仅保存 state_dict
    path_best_ckpt = os.path.join(ckpt_dir, f"best_ckpt_{mode_safe}.pth")         # 保存完整 ckpt（含 epoch/optimizer）

    # training state
    start_epoch = 0
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1

    # 先用“当前模型参数”初始化 best_weights（后续会被 best_ckpt 覆盖）
    best_weights = copy.deepcopy(model.state_dict())

    def _load_checkpoint(pth):
        """safe torch.load with map_location"""
        return torch.load(pth, map_location=device)

    def _move_optimizer_state_to_device(opt, dev):
        """确保 optimizer state tensor 在正确 device 上（避免 resume 后 device mismatch）"""
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)

    # ======================================================================
    # 1) 恢复“训练进度（从第几轮开始、loss曲线等）” —— 来自 latest_ckpt
    # ======================================================================
    if resume and os.path.exists(path_latest):
        print(f"[{mode}] Found checkpoint. Resuming epoch counter & history from latest...")
        latest_ckpt = _load_checkpoint(path_latest)

        try:
            model.load_state_dict(latest_ckpt["model_state_dict"])
        except Exception as e:
            print(f"[{mode}] ⚠️ Failed to load model_state_dict from latest_ckpt: {e}")

        try:
            optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        except Exception as e:
            print(f"[{mode}] ⚠️ Failed to load optimizer_state_dict from latest_ckpt: {e}")

        start_epoch = int(latest_ckpt.get("epoch", -1)) + 1
        train_losses = list(latest_ckpt.get("train_losses", []))
        val_losses = list(latest_ckpt.get("val_losses", []))
        best_val_loss = float(latest_ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(latest_ckpt.get("best_epoch", best_epoch))

    # ======================================================================
    # 2) 恢复“历史最优权重” —— 优先 best_ckpt，其次 best_model(state_dict)
    #    这样：即使从 epoch=120 继续，模型权重也从 0~120 中最好的那一轮开始。
    # ======================================================================
    best_loaded = False
    if resume:
        if os.path.exists(path_best_ckpt):
            print(f"[{mode}] Loading BEST checkpoint to continue training (recommended)...")
            best_ckpt = _load_checkpoint(path_best_ckpt)
            try:
                best_weights = best_ckpt["model_state_dict"]
                best_val_loss = float(best_ckpt.get("best_val_loss", best_val_loss))
                best_epoch = int(best_ckpt.get("epoch", best_epoch))
                best_loaded = True
            except Exception as e:
                print(f"[{mode}] ⚠️ Failed to read best_ckpt content: {e}")

            # optimizer 与 best 权重保持一致（更稳）
            try:
                optimizer.load_state_dict(best_ckpt["optimizer_state_dict"])
                _move_optimizer_state_to_device(optimizer, device)
            except Exception as e:
                print(f"[{mode}] ⚠️ Failed to load optimizer_state_dict from best_ckpt: {e}")

        elif os.path.exists(path_best_weights):
            print(f"[{mode}] Loading BEST weights (state_dict) to continue training...")
            try:
                best_weights = torch.load(path_best_weights, map_location=device)
                best_loaded = True
            except Exception as e:
                print(f"[{mode}] ⚠️ Failed to load best_model weights: {e}")

    if best_loaded:
        try:
            model.load_state_dict(best_weights)
            if best_epoch >= 0:
                print(f"[{mode}] ✅ Using BEST weights from epoch {best_epoch} (best_val_loss={best_val_loss:.6f})")
            else:
                print(f"[{mode}] ✅ Using BEST weights (best_val_loss={best_val_loss:.6f})")
        except Exception as e:
            print(f"[{mode}] ⚠️ Failed to load best_weights into model: {e}")

    # 如果已经训练到 epochs（比如 global_epochs=10 且断点=10），直接返回 best_weights
    if start_epoch >= epochs:
        print(f"[{mode}] ✅ Training already finished: start_epoch({start_epoch}) >= epochs({epochs}).")
        plot_loss_curve(train_losses, val_losses, mode_name=mode)
        return best_weights

    # ======================================================================
    # 3) 正式训练
    # ======================================================================
    # --------------------------------------------------------------
    # Compatibility: some models (e.g., UIPRegreator) do NOT take material_id in forward().
    # We detect the forward() signature once and call the model accordingly.
    # --------------------------------------------------------------
    _accepts_material_id = False
    try:
        import inspect as _inspect
        _sig = _inspect.signature(model.forward)
        _params = list(_sig.parameters.values())
        # Drop `self` if present
        if _params and _params[0].name == 'self':
            _params = _params[1:]
        if any(p.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD) for p in _params):
            _accepts_material_id = True
        elif len(_params) >= 2:
            _accepts_material_id = True
    except Exception:
        _accepts_material_id = False

    if _accepts_material_id:
        print(f"[{mode}] ✅ Detected model.forward supports material_id; calling model(inputs, material_id).")
    else:
        print(f"[{mode}] ✅ Detected model.forward does NOT support material_id; calling model(inputs) only.")

    prev_val_loss = val_losses[-1] if len(val_losses) > 0 else None

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_losses = []

        pbar = tqdm(train_loader, desc=f"[{mode}] Epoch {epoch + 1}/{epochs} (Training)", ncols=120)
        for batch in pbar:
            optimizer.zero_grad()

            inputs, decision, reg_targets, is_tessonic, material_id = batch
            inputs = inputs.to(device, non_blocking=True)
            # Align with legacy stability: clean NaN/Inf in inputs
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            decision = decision.to(device, non_blocking=True)
            reg_targets = reg_targets.to(device, non_blocking=True)
            is_tessonic = is_tessonic.to(device, non_blocking=True)
            material_id = material_id.to(device, non_blocking=True)

            if _accepts_material_id:
                outputs = model(inputs, material_id)
            else:
                outputs = model(inputs)
            total_loss, d_loss, r_loss, c_loss = compute_losses(outputs, decision, reg_targets, criteria)

            total_loss.backward()
            # Align with legacy stability: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_losses.append(total_loss.item())
            # 进度条频繁计算 np.mean(list) 会越跑越慢；改为间隔刷新 + 使用当前 running 均值
            if (pbar.n + 1) % 10 == 0:
                pbar.set_postfix({
                    "Loss": f"{(sum(epoch_train_losses) / len(epoch_train_losses)):.4f}",
                    "D_Loss": f"{d_loss.item():.4f}",
                    "R_Loss": f"{r_loss.item():.4f}",
                    "C_Loss": f"{c_loss.item():.4f}"
                })

        avg_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("inf")
        train_losses.append(avg_train_loss)

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        epoch_val_losses = []
        # Collect diameter predictions for per-epoch RMSE/R2 reporting
        y_true_diam_all = []
        y_pred_diam_all = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, decision, reg_targets, is_tessonic, material_id = batch
                inputs = inputs.to(device, non_blocking=True)
                # Align with legacy stability: clean NaN/Inf in inputs
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
                decision = decision.to(device, non_blocking=True)
                reg_targets = reg_targets.to(device, non_blocking=True)
                is_tessonic = is_tessonic.to(device, non_blocking=True)
                material_id = material_id.to(device, non_blocking=True)

                if _accepts_material_id:
                    outputs = model(inputs, material_id)
                else:
                    outputs = model(inputs)
                total_loss, _, _, _ = compute_losses(outputs, decision, reg_targets, criteria)
                # Diameter regression metrics (ignore NaN targets)
                try:
                    y_true_d = reg_targets[:, 0]
                    y_pred_d = outputs.get('diameter_1', None)
                    if y_pred_d is not None:
                        y_pred_d = y_pred_d.view(-1)
                        mask = ~torch.isnan(y_true_d)
                        if mask.any():
                            y_true_diam_all.append(y_true_d[mask].detach().cpu())
                            y_pred_diam_all.append(y_pred_d[mask].detach().cpu())
                except Exception:
                    pass
                epoch_val_losses.append(total_loss.item())

        avg_val_loss = float(np.mean(epoch_val_losses)) if epoch_val_losses else float("inf")
        val_losses.append(avg_val_loss)

        # Per-epoch regression report (Diameter RMSE/R2)
        diam_rmse_epoch = float('nan')
        diam_r2_epoch = float('nan')
        if len(y_true_diam_all) > 0 and len(y_pred_diam_all) > 0:
            y_true_cat = torch.cat(y_true_diam_all).numpy()
            y_pred_cat = torch.cat(y_pred_diam_all).numpy()
            if y_true_cat.size >= 2:
                try:
                    diam_rmse_epoch = float(np.sqrt(mean_squared_error(y_true_cat, y_pred_cat)))
                except Exception:
                    diam_rmse_epoch = float('nan')
                # R2 needs variance in y_true
                try:
                    if np.unique(y_true_cat).size >= 2:
                        diam_r2_epoch = float(r2_score(y_true_cat, y_pred_cat))
                except Exception:
                    diam_r2_epoch = float('nan')

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  [Regression] Diameter: RMSE: {diam_rmse_epoch:.4f} | R2: {diam_r2_epoch:.4f}")

        # 与上一轮对比（更直观）
        if prev_val_loss is not None:
            delta = prev_val_loss - avg_val_loss
            if delta > 0:
                print(f"  ✅ Val Loss improved vs prev epoch by {delta:.6f}")
            else:
                print(f"  ⚠️ Val Loss worse vs prev epoch by {abs(delta):.6f}")
        prev_val_loss = avg_val_loss

        # =================================================================
        # Loss Logging & Checkpointing
        # =================================================================
        if MLFLOW_ENABLED:
            mlflow.log_metric(f"{ml_mode}_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric(f"{ml_mode}_val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric(f"{ml_mode}_diam_rmse", diam_rmse_epoch, step=epoch)
            mlflow.log_metric(f"{ml_mode}_diam_r2", diam_r2_epoch, step=epoch)

        # 若优于历史 best，保存 best（权重 + 完整 ckpt）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

            torch.save(best_weights, path_best_weights)
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_weights,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            }, path_best_ckpt)

            print(f"[{mode}] ⭐ New BEST at epoch {epoch + 1}: best_val_loss={best_val_loss:.6f}")

        # 每个 epoch 都保存 latest，保证断点续训可用
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }, path_latest)

    plot_loss_curve(train_losses, val_losses, mode_name=mode)
    return best_weights


def evaluate_per_material(model, val_loader, device, material_map_rev, max_fpr_constraint):
    """详细评估：按板材组合统计 FPR, Recall, 并添加直径回归RMSE和R2"""
    model.eval()
    results = {mat: {'y_true': [], 'y_prob': [], 'y_target_diam': [], 'y_pred_diam': []} for mat in
               material_map_rev.values()}
    results['GLOBAL'] = {'y_true': [], 'y_prob': [], 'y_target_diam': [], 'y_pred_diam': []}

    with torch.no_grad():
        for features, labels, reg_targets, _, mat_ids in val_loader:  # 接收 reg_targets
            if features is None: continue
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            reg_targets = reg_targets.to(device, non_blocking=True)

            # === Evaluation Input Defense ===
            features = torch.nan_to_num(features, nan=0.0)

            outputs = model(features)
            mat_ids = mat_ids.cpu().numpy()
            probs = torch.sigmoid(outputs['qrk2_decision'].view(-1)).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # --- 回归结果 ---
            pred_diam_np = outputs['diameter_1'].squeeze().cpu().numpy()
            target_diam_np = reg_targets[:, 0].cpu().numpy()
            if pred_diam_np.ndim == 0:
                pred_diam_np = np.array([pred_diam_np])
            if target_diam_np.ndim == 0:
                target_diam_np = np.array([target_diam_np])

            for i in range(len(labels_np)):
                if np.isnan(labels_np[i]): continue
                mat_name = material_map_rev.get(mat_ids[i], 'unknown')

                is_reg_valid = not np.isnan(target_diam_np[i])

                current_data = results.get(mat_name)

                if current_data:
                    current_data['y_true'].append(labels_np[i])
                    current_data['y_prob'].append(probs[i])
                    if is_reg_valid:
                        current_data['y_target_diam'].append(target_diam_np[i])
                        current_data['y_pred_diam'].append(pred_diam_np[i])

                results['GLOBAL']['y_true'].append(labels_np[i])
                results['GLOBAL']['y_prob'].append(probs[i])
                if is_reg_valid:
                    results['GLOBAL']['y_target_diam'].append(target_diam_np[i])
                    results['GLOBAL']['y_pred_diam'].append(pred_diam_np[i])

    metrics = {}
    if results:
        for mat_name, data in results.items():
            if len(data['y_true']) < 5:
                # 对于数据量过少的材料，使用默认或占位符指标
                if mat_name != 'GLOBAL': continue
                if len(data['y_true']) == 0: continue

            y_true, y_prob = np.array(data['y_true']), np.array(data['y_prob'])

            # --- 决策指标 ---
            # --- 决策指标 ---

            if len(np.unique(y_true)) > 1:
                # 正常情况：有正负样本
                opt_thresh = find_optimal_threshold(y_true, y_prob, max_fpr=max_fpr_constraint)
                y_pred = (y_prob > opt_thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

                auc = roc_auc_score(y_true, y_prob)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            else:
                # 只有一类样本的情况
                unique_class = np.unique(y_true)[0]

                if unique_class == 0:
                    # 只有负样本的情况
                    # 这种情况下，我们应该关注FPR，因为任何正预测都是误报
                    if max_fpr_constraint is not None:
                        # 根据FPR约束选择阈值
                        thresholds = np.sort(np.unique(y_prob))
                        for thresh in thresholds:
                            y_pred_temp = (y_prob > thresh).astype(int)
                            # 计算这个阈值下的FPR
                            fp_count = np.sum(y_pred_temp)
                            fpr_temp = fp_count / len(y_true) if len(y_true) > 0 else 0.0

                            if fpr_temp <= max_fpr_constraint:
                                opt_thresh = thresh
                                y_pred = y_pred_temp
                                break
                        else:
                            # 如果没有阈值满足约束，使用最大阈值（不预测任何正样本）
                            opt_thresh = np.max(y_prob) + 0.001
                            y_pred = np.zeros_like(y_pred_temp)
                    else:
                        # 没有FPR约束时，保守预测：全部预测为负
                        opt_thresh = np.max(y_prob) + 0.001
                        y_pred = np.zeros_like(y_true)

                    # 重新计算指标
                    tn = len(y_true)  # 所有样本都是真负例
                    fp = 0
                    fn = 0  # 没有正样本，所以假负例为0
                    tp = 0  # 没有正样本，所以真正例为0

                    auc = 0.5  # 只有一类时，AUC无意义，设为0.5（随机水平）
                    fpr = 0.0  # 没有预测正样本，所以FPR=0
                    recall = 0.0  # 没有正样本，召回率定义为0

                else:
                    # 只有正样本的情况
                    if max_fpr_constraint is not None:
                        # 对于只有正样本的情况，我们可以设定一个合理的阈值
                        opt_thresh = np.percentile(y_prob, 50)  # 使用中位数作为阈值
                    else:
                        opt_thresh = 0.5

                    y_pred = (y_prob > opt_thresh).astype(int)

                    # 重新计算指标
                    tn = 0  # 没有负样本
                    fp = 0  # 没有负样本，所以假正例为0
                    fn = np.sum(y_pred == 0)  # 预测为负的正样本
                    tp = np.sum(y_pred == 1)  # 预测为正的正样本

                    auc = 0.5  # 只有一类时，AUC无意义，设为0.5
                    fpr = 0.0  # 没有负样本，FPR定义为0（或NaN，但这里设为0）
                    recall = tp / len(y_true) if len(y_true) > 0 else 0.0

            # --- 回归指标 ---
            y_target_diam = np.array(data['y_target_diam'])
            y_pred_diam = np.array(data['y_pred_diam'])

            diam_rmse = 0.0
            diam_r2 = 0.0

            if len(y_target_diam) > 1:
                try:
                    diam_rmse = np.sqrt(mean_squared_error(y_target_diam, y_pred_diam))
                    diam_r2 = r2_score(y_target_diam, y_pred_diam)
                except Exception as e:
                    logging.warning(f"Error calculating diameter metrics for {mat_name}: {e}")

            metrics[mat_name] = {
                "count": len(y_true),
                "auc": auc,
                "fpr": fpr,
                "recall": recall,
                "thresh": opt_thresh,
                "diam_rmse": diam_rmse,
                "diam_r2": diam_r2
            }

    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 加速固定输入尺寸的卷积网络
    if MLFLOW_ENABLED: mlflow.start_run()

    try:
        print("Loading Data...")
        combined_df = None
        if os.path.exists(CACHE_FILE):
            print(f"✅ Found cache file: '{CACHE_FILE}'. Loading data directly...")
            combined_df = pd.read_feather(CACHE_FILE)
            print(f"Cache loaded. Final dataset has {len(combined_df):,} rows.")
        else:
            print(f"❌ Cache file not found: '{CACHE_FILE}'. Starting to read and sample raw data...")
            file_paths = glob.glob(os.path.join(DATA_DIR, "*.feather"))

            sampled_dfs = []
            total_sampled_rows = 0

            if not file_paths:
                raise FileNotFoundError(f"No .feather files found in directory '{DATA_DIR}'.")

            print(f"Found {len(file_paths)} Feather files, starting sampling and merging...")

            for file_path in file_paths:

                try:
                    df = pd.read_feather(file_path)

                    if df.empty:
                        print(f"  - File '{os.path.basename(file_path)}' is empty, skipping.")
                        continue
                    else:
                        should_sample = True
                        real_area = ""

                        if 'area' in df.columns and not df['area'].empty:
                            real_area = df['area'].iloc[0]
                        # NOTE: 不再根据 area(AB/UB) 做筛选，全部样本都参与训练
                        print(f"  - Sampling without area filter: area={real_area!r}")
                        if should_sample:
                            if SAMPLE_FRACTION >= 1.0:
                                sampled_df = df
                            else:
                                sampled_df = df.sample(frac=SAMPLE_FRACTION)  # random_state=42
                            sampled_dfs.append(sampled_df)
                            total_sampled_rows += len(sampled_df)
                            print(f"  - File '{os.path.basename(file_path)}' sampled {len(sampled_df):,} rows.")

                except Exception as e:
                    logging.warning(f"Error processing file {file_path}: {e}")
                    continue

            if not sampled_dfs:
                print("\nNo valid data was sampled. Combined DataFrame is empty.")
                combined_df = pd.DataFrame()
            else:
                print(f"\nStarting to merge {len(sampled_dfs)} sampled DataFrames...")
                combined_df = pd.concat(sampled_dfs, ignore_index=True)
                print(f"Data merge complete. Final dataset has {len(combined_df):,} rows.")

                print(f"🚀 Writing combined data to cache file: '{CACHE_FILE}'...")
                combined_df.to_feather(CACHE_FILE)
                print("Cache write successful.")
        if not combined_df.empty:
            if LABEL_COLUMN in combined_df.columns:
                print(f"\nLabel column '{LABEL_COLUMN}' (OK/NOK) statistics:")
                decision_counts = combined_df[LABEL_COLUMN].value_counts(dropna=True)
                print(decision_counts.to_string())
                print(f"\nTotal {decision_counts.sum():,} records with valid labels.")
            else:
                print(f"\nWarning: Label column '{LABEL_COLUMN}' not found in the combined DataFrame.")
        else:
            print("\nFinal dataset is empty, skipping label statistics.")

        # Load external data (tessonic.csv, spot_detail.xlsx) as per original logic
        tes_df = pd.read_csv(os.path.join(AEQ_INPUT_ROOT, "tessonic.csv"))
        tes2_df = pd.read_csv(os.path.join(AEQ_INPUT_ROOT, "saic_data.csv"))
        tes_df = pd.concat([tes_df, tes2_df], ignore_index=True).drop_duplicates()
        # Ensure qrk2_decision is numerical (1 for NOK, 0 for OK/NaN reason)
        tes_df['qrk2_decision'] = tes_df['reason'].apply(lambda x: 0 if pd.isna(x) or str(x).lower() == 'ok' else 1)
        tes_df['diameter_1'] = tes_df['diameter_1']  # Ensure diameter column name consistency

        mat_df = pd.read_csv(os.path.join(AEQ_INPUT_ROOT, "material.csv"))  # pd.read_excel("spot_detail.xlsx")
        # mat_df['part_all'] = mat_df[['part1_material', 'part2_material', 'part3_material']].fillna(' ').agg(' '.join, axis=1)
        mat_df['part_all'] = mat_df['part_all'].astype(str)
        tes_df['is_tessonic'] = 1
        combined_df['is_tessonic'] = 0

        if 'spot_tag' not in mat_df.columns: raise ValueError("spot_detail.xlsx missing 'spot_tag'")
        full_df = pd.concat([tes_df, combined_df], ignore_index=True)
        full_df.dropna(subset=['imax', 'umax', 'spot_tag', 'qrk2_decision'], inplace=True)
        full_df = full_df.merge(mat_df[['spot_tag', 'part_all']], on='spot_tag', how='left')
        full_df['part_all'] = full_df['part_all'].fillna('unknown')

        full_df.dropna(subset=['imax', 'umax', 'spot_tag'], inplace=True)

        # --- 原始标签分布检查 (新增) ---
        print("\n========================================================")
        print("====== 标签 (qrk2_decision) 分布诊断 ======")

        # 1. 总体分布
        total_counts = full_df['qrk2_decision'].value_counts(dropna=False)
        print("\n--- 1. 全量数据标签分布 (包括NaN) ---")
        print(total_counts.to_string())
        print(f"总记录数: {len(full_df)}")

        # 2. 按数据源 (is_tessonic) 分布
        print("\n--- 2. 按数据源 ('is_tessonic') 分布 ---")
        tessonic_data = full_df[full_df['is_tessonic'] == 1]
        print(f"** TESSonic 数据源 ({len(tessonic_data)} 条记录): **")
        print(tessonic_data['qrk2_decision'].value_counts(dropna=False).to_string())

        combined_data = full_df[full_df['is_tessonic'] == 0]
        print(f"\n** Combined/Feather 数据源 ({len(combined_data)} 条记录): **")
        print(combined_data['qrk2_decision'].value_counts(dropna=False).to_string())

        # 3. 最终训练集中的正负样本比例 (去除NaN后)
        valid_labels = full_df['qrk2_decision'].dropna()
        num_valid = len(valid_labels)
        num_pos = valid_labels.sum()
        num_neg = num_valid - num_pos

        print("\n--- 3. 最终有效标签 (0/1) 比例 ---")
        if num_valid > 0:
            print(f"有效标签总数: {num_valid}")
            print(f"正样本 (1) 数量: {num_pos} ({num_pos / num_valid:.2%})")
            print(f"负样本 (0) 数量: {num_neg} ({num_neg / num_valid:.2%})")
            print(f"正样本权重 (neg/pos): {num_neg / (num_pos + 1e-5):.4f}")
        else:
            print("警告: 最终有效标签数量为零！")

        print("========================================================\n")
        # --- 原始标签分布检查 (结束) ---

        unique_mats = full_df['part_all'].unique().tolist()
        mat_map = {n: i for i, n in enumerate(unique_mats)}
        mat_map_rev = {i: n for n, i in mat_map.items()}
        print(f"Materials found: {len(unique_mats)}")
        if 'write_time' in full_df.columns:
            full_df['write_time'] = pd.to_datetime(full_df['write_time'], errors='coerce').fillna(
                pd.Timestamp('2024-01-01'))
        else:
            # 有些数据源(如 TESSonic)可能没有 write_time，给默认值避免 KeyError
            full_df['write_time'] = pd.Timestamp('2024-01-01')
        full_df.sort_values('write_time', inplace=True)
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱
        full_df = pre_clean_data(full_df)
        split_idx = int(len(full_df) * 0.8)

        train_df, val_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]
        # 从val_df丢弃“假NOk，即qrk2_decision>0且is_tessonic==0
        val_df = val_df[~((val_df['qrk2_decision'] > 0) & (val_df['is_tessonic'] == 0))]  # 去掉先检查训练逻辑的一致性

        train_loader = make_loader(train_df, mat_map, CONFIG["batch_size"], True, custom_collate_fn, cache_key="train")
        val_loader = make_loader(val_df, mat_map, CONFIG["batch_size"], False, custom_collate_fn, cache_key="val")
        print("\n=== Phase 1: Global Training ===")
        model = UIPRegreator().to(device)

        valid_train_labels = train_df['qrk2_decision'].dropna()
        num_pos_train = valid_train_labels.sum()
        # 动态计算 pos_weight 并打印出来供用户检查
        pos_weight = torch.tensor((len(valid_train_labels) - num_pos_train) / (num_pos_train + 1e-5)).to(device)
        print(f"✅ Decision Loss Positive Weight (pos_weight) calculated for Global Training: {pos_weight.item():.4f}")

        criteria = {
            'qrk2_decision': nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none'),
            'diameter_1': DiameterAwareLoss(reduction='none').to(device),  # Explicitly set reduction='none'
            'consistency': ConsistencyLoss(reduction='none').to(device),  # Explicitly set reduction='none'
            'regression': nn.MSELoss(reduction='none').to(device)  # for other regression
        }  # LOSS定义
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

        global_weights = train_model(model, train_loader, val_loader, CONFIG["global_epochs"], device, criteria,
                                     optimizer, mode="Global", resume=True)
        torch.save(global_weights, os.path.join(CONFIG["dirs"]["export"], "global_model.pth"))

        print("\n=== Phase 2: Evaluation & Fine-tuning ===")
        model.load_state_dict(global_weights)
        group_metrics = evaluate_per_material(model, val_loader, device, mat_map_rev, CONFIG["max_allowed_fpr"])

        # 打印 Global 指标
        print("\n--- Global Evaluation Summary ---")
        global_metrics = group_metrics.get('GLOBAL', {})
        print(
            f"Global Stats | Count: {global_metrics.get('count', 0):,} | AUC: {global_metrics.get('auc', 0.0):.4f} | FPR: {global_metrics.get('fpr', 0.0):.4f} | Recall: {global_metrics.get('recall', 0.0):.4f} | Diam RMSE: {global_metrics.get('diam_rmse', 0.0):.4f} | Diam R2: {global_metrics.get('diam_r2', 0.0):.4f}")

        # 打印详细指标
        print("\n--- Per-Material Evaluation Details ---")
        for mat_name, metric in group_metrics.items():
            if mat_name == 'GLOBAL': continue
            print(
                f"[{mat_name}] Count: {metric['count']:,} | AUC: {metric['auc']:.4f} | FPR: {metric['fpr']:.4f} | Recall: {metric['recall']:.4f} | RMSE: {metric['diam_rmse']:.4f} | R2: {metric['diam_r2']:.4f} | Thresh: {metric['thresh']:.4f}")

        # 处理 Fine-tuning
        inference_config = {
            "material_map": mat_map,
            "models": {"default": {"path": "global_model.pth", "threshold": global_metrics.get('thresh', 0.5)}}
        }

        for mat_name, metric in group_metrics.items():
            if mat_name == 'GLOBAL': continue

            needs_ft = (metric['fpr'] > CONFIG["max_allowed_fpr"] * 1.5) or (metric['recall'] < 0.85)

            if needs_ft and metric['count'] > 50:
                print(f"[{mat_name}] Triggering Fine-tuning (FPR:{metric['fpr']:.4f}, Recall:{metric['recall']:.4f})")

                ft_train_df = train_df[train_df['part_all'] == mat_name]
                ft_val_df = val_df[val_df['part_all'] == mat_name]
                valid_ft_labels = ft_train_df['qrk2_decision'].dropna()
                num_pos_ft = valid_ft_labels.sum()

                # Fine-tuning 阶段也使用动态计算的 pos_weight
                pos_weight_ft = torch.tensor((len(valid_ft_labels) - num_pos_ft) / (num_pos_ft + 10 + 1e-5)).to(
                    device)  # 避免极端微调
                print(f"[{mat_name}] FT Positive Weight: {pos_weight_ft.item():.4f}")

                ft_criteria = {
                    'qrk2_decision': nn.BCEWithLogitsLoss(pos_weight=pos_weight_ft, reduction='none'),
                    'diameter_1': DiameterAwareLoss(reduction='none').to(device),
                    'consistency': ConsistencyLoss(reduction='none').to(device),
                    'regression': nn.MSELoss(reduction='none').to(device),
                }

                ft_model = UIPRegreator().to(device)
                ft_model.load_state_dict(global_weights)
                ft_optim = torch.optim.Adam(ft_model.parameters(), lr=CONFIG["finetune_lr"])

                # 注意：这里我们使用 CustomDataset 包装了 ft_val_df
                ft_weights = train_model(ft_model,
                                         make_loader(ft_train_df, mat_map, 32, True, custom_collate_fn, cache_key=f"ft_train_{mat_name}"),
                                         make_loader(ft_val_df, mat_map, 32, False, custom_collate_fn, cache_key=f"ft_val_{mat_name}"),
                                         CONFIG["finetune_epochs"], device, ft_criteria, ft_optim,
                                         mode=f"FT_{mat_name}", resume=False)

                model_fname = f"model_{mat_name}.pth"
                torch.save(ft_weights, os.path.join(CONFIG["dirs"]["export"], model_fname))

                ft_model.load_state_dict(ft_weights)

                ft_metric_results = evaluate_per_material(ft_model,
                                                          make_loader(ft_val_df, mat_map, 32, False, custom_collate_fn, cache_key=f"ft_val_{mat_name}"), device, mat_map_rev,
                                                          CONFIG["max_allowed_fpr"])
                ft_metric = ft_metric_results[mat_name]

                inference_config["models"][mat_name] = {"path": model_fname, "threshold": ft_metric['thresh']}
                print(
                    f"[{mat_name}] FT Done. New AUC:{ft_metric['auc']:.4f} | New FPR:{ft_metric['fpr']:.4f}, Recall:{ft_metric['recall']:.4f}, RMSE:{ft_metric['diam_rmse']:.4f}, R2:{ft_metric['diam_r2']:.4f}")
            else:
                inference_config["models"][mat_name] = {"use_global": True, "threshold": metric['thresh']}

        with open(os.path.join(CONFIG["dirs"]["export"], "inference_config.json"), "w") as f:
            inference_config = json.loads(json.dumps(inference_config, default=float))
            json.dump(inference_config, f, indent=4)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        logging.error(f"Fatal error: {e}")
    finally:
        if MLFLOW_ENABLED: mlflow.end_run()