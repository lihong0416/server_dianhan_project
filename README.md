# server_dianhan_project

一个面向产线的 **批处理推理服务**：持续轮询 ClickHouse 中“当天未处理”的焊接记录，进行 **ML 模型推理 + 规则模型判定**，并将结果写回 ClickHouse，同时把 **NOK** 记录写入 MySQL 供业务系统查看；另外会落一份本地压缩 CSV 便于审计/排障。

---

## 1. 项目做什么

### 核心流程（server_qingdao.py）
- **读取数据**：从 ClickHouse 表 `tables.source_table` 拉取当天 `data_date=YYYYMMDD` 且 `alg_complete_time IS NULL` 的数据（每次最多 `batch_size` 行），并用 `ck_local_time` 做增量游标，避免扫全库。
- **数据校验**：过滤曲线/关键字段不合法的行（例如曲线为空、`umax`/`imax` 异常等）。
- **模型推理**：
  - **ML 模型（可开关）**：读取 `model_export_dir` 下导出的 `global_model.pth`（可选 `inference_config.json`），输入 4 通道曲线特征 `[U, I, R=U/I, P=U*I]`（长度 `target_len`，默认 1000），输出：
    - 二分类概率（OK/NOK）
    - 回归：`diameter / indentation / front_thickness / stack_thickness`
  - **规则模型（可开关）**：使用 `qdyq_peakrules_model.py`（依赖 `utils/`、`sputils/`），输出 `OK/NOK`（主要用于表面缺陷判断）。
- **融合策略（Soft-AND）**：
  - 默认：`ML=NOK` 且 `Rule=NOK` 才判为最终 `NOK`
  - 但若 ML 置信度**极高**，允许 **ML override**（即使规则未触发也可判 `NOK`）
- **写回结果**：
  - ClickHouse：将推理后的字段写回 `source_table`（要求表结构包含对应字段）
  - MySQL：仅写入 `NOK` 行到 `tables.result_table`
  - 本地：保存一份 `csv.gz` 到 `save_dir`

### 健康检查
服务启动后提供一个简单接口：
- `GET /`：返回服务运行信息、阈值与模型开关状态

---

## 2. 仓库结构（主要文件）

- `server_qingdao.py`：生产推理服务（FastAPI + 后台轮询）
- `config.yaml`：ClickHouse / MySQL / 模型路径 / 阈值 / 轮询参数等配置
- `Dockerfile`：GPU 镜像构建（CUDA 12.1 + conda python3.12 + torch cu121）
- `.gitlab-ci.yml`：GitLab CI/CD（构建推送 Harbor + SSH 到目标机拉起容器）
- `qdyq_peakrules_model.py`：规则模型（表面缺陷规则）
- `cols_to_drop_v2.py`：特征列定义/旧版数据处理（部分脚本会用到）
- `V3_test_lh_train.py`：训练脚本（导出 `model_export`：global + 可选分料号微调模型）
- `classification_and_regression.py`：**只做评估/验证**（不训练、不改模型）
- `sputils`：deploy
- `utils`：deploy

---

## 3. 快速开始（本地运行）

### 3.1 环境依赖
- Python 3.10+（Docker 里用的是 3.12）
- CUDA GPU（可选；无 GPU 也能跑但推理慢）
- 需要能访问 ClickHouse & MySQL 网络

安装依赖（示例）：
```bash
# 安装 Miniconda（不依赖你仓库里放安装包）
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -u -p $CONDA_DIR \
 && rm -f /tmp/miniconda.sh 
# 创建 conda 环境（env 名叫 train）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r  --channel https://repo.anaconda.com/pkgs/msys2
conda update --force conda
conda create -n aeq python=3.12 -y \
    && echo "conda activate train" >> ~/.bashrc
# 安装 PyTorch（CUDA 12.1 -> cu121）+ 依赖（清华 pip 源）
conda run -n aeq python -m pip install --upgrade pip \
 && conda run -n aeq pip install --no-cache-dir \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && conda run -n aeq pip install --no-cache-dir -r /alg/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
python server_qingdao.py
```

## 4. 效果大致展示

V1_lh_train.py  版本数据集为照顾小类的数据取样，为原来数据集的1/10，训练epoch25轮，下面为在测试集上的验证效果：
{
  "classification": {
    "n_eval": 335085,
    "threshold": 0.9999,
    "accuracy": 0.9893817986481042,
    "precision": 0.09614309670206819,
    "recall": 0.08140085186938002,
    "f1": 0.08815991799077397,
    "auc": 0.7579810250705518,
    "pos_rate_true": 0.006305862691555874,
    "pos_rate_pred": 0.005338943850067893
  },
  "regression": {
    "diameter_1": {
      "n_eval": 335085,
      "mae": 2.703149603181314,
      "mse": 12.712409172313922,
      "rmse": 3.5654465600137555,
      "r2": 0.003758778440864252
    },
    "indentation_1": {
      "n_eval": 335085,
      "mae": 0.019710651236416025,
      "mse": 0.0011195720913464177,
      "rmse": 0.033460007342294734,
      "r2": 0.8545963615868417
    },
    "front_thickness_1": {
      "n_eval": 335085,
      "mae": 0.04394573733804534,
      "mse": 0.005067901903897602,
      "rmse": 0.07118919794391283,
      "r2": 0.8120163334899242
    },
    "stack_thickness_1": {
      "n_eval": 335085,
      "mae": 0.17434677743080357,
      "mse": 0.08144845877625202,
      "rmse": 0.2853917636797741,
      "r2": 0.8487268339936465
    }
  }
}


V2_lh_train.py  版本数据集为原数据集的随机取样，为原来数据集的1/5，训练epoch20轮，为下面为在测试集上的验证效果：

{
  "classification": {
    "n_eval": 335085,
    "threshold": 0.9980697478991597,
    "accuracy": 0.9893221122998642,
    "precision": 0.10211841390548615,
    "recall": 0.0889730241362991,
    "f1": 0.09509357612544259,
    "auc": 0.7797558393336238,
    "pos_rate_true": 0.006305862691555874,
    "pos_rate_pred": 0.00549412835549189
  },
  "regression": {
    "diameter_1": {
      "n_eval": 335085,
      "mae": 2.401508298785366,
      "mse": 10.088233994943279,
      "rmse": 3.176198040888395,
      "r2": 0.20940913541509876
    },
    "indentation_1": {
      "n_eval": 335085,
      "mae": 0.017862179904118168,
      "mse": 0.0009543576825937692,
      "rmse": 0.03089268008110933,
      "r2": 0.8760534667939066
    },
    "front_thickness_1": {
      "n_eval": 335085,
      "mae": 0.040454741231357914,
      "mse": 0.004507804051400759,
      "rmse": 0.06714018209240097,
      "r2": 0.8327920410536007
    },
    "stack_thickness_1": {
      "n_eval": 335085,
      "mae": 0.15997509570089502,
      "mse": 0.06827588362221587,
      "rmse": 0.2612965434563111,
      "r2": 0.8731920869640155
    }
  }
}



V3_test_test_lh_train.py  版本数据为平均取样，之前版本因数据的正负样本分布及其不均匀，差不多为1000：6，现在对数据重新做抽取，抽取结果为：
正样本 (1) 数量: 113241 (19.11%)
负样本 (0) 数量: 479212 (80.89%)
正样本权重 (neg/pos): 4.2318
训练epoch25轮，为下面为在测试集上的验证效果：
{
  "classification": {
    "n_eval": 335085,
    "threshold": 0.9886414082762285,
    "accuracy": 0.9907695062446842,
    "precision": 0.25980392156862747,
    "recall": 0.2508282063416943,
    "f1": 0.2552371779436552,
    "auc": 0.940147429799705,
    "pos_rate_true": 0.006305862691555874,
    "pos_rate_pred": 0.006088007520479878
  },
  "regression": {
    "diameter_1": {
      "n_eval": 335085,
      "mae": 2.0265486273090585,
      "mse": 7.325212872459907,
      "rmse": 2.7065130467928484,
      "r2": 0.425940518329632
    },
    "indentation_1": {
      "n_eval": 335085,
      "mae": 0.01622594445956407,
      "mse": 0.0005459686266914559,
      "rmse": 0.0233659715546231,
      "r2": 0.9290927083713723
    },
    "front_thickness_1": {
      "n_eval": 335085,
      "mae": 0.03912594721322148,
      "mse": 0.002911919629521287,
      "rmse": 0.053962205565759515,
      "r2": 0.891988175990722
    },
    "stack_thickness_1": {
      "n_eval": 335085,
      "mae": 0.14039589826572513,
      "mse": 0.03699889904783171,
      "rmse": 0.19235097880653404,
      "r2": 0.9312824247160971
    }
  }
}

