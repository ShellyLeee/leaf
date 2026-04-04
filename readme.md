# Leaf Classification with Pure MLP (PyTorch)

## 1) 项目简介
这是一个基于 PyTorch 的叶片图像分类项目，模型为纯 MLP（无卷积层）。
项目支持训练、验证、测试，以及系统化超参数实验和训练曲线分析。

## 2) 项目结构
- `train.py`: 训练入口（train/val）
- `test.py`: 测试入口（仅 test）
- `cfgs/config.yaml`: 统一配置
- `datasets/`: 数据集与 dataloader 构建
- `models/mlp.py`: 纯 MLP 模型
- `trainers/trainer.py`: 训练循环、验证、checkpoint
- `common/`: loss/metrics/optimizer/scheduler/seed/logging/utils/config
- `scripts/`: 常用命令脚本
- `outputs/`: 训练产物目录

## 3) Environmental Setup
```bash
conda create -n leaf python=3.10 -y
conda activate leaf

# 安装 PyTorch（如需 CUDA，请按你的 CUDA 版本从官方命令替换）
pip install torch torchvision

# 项目依赖
pip install pandas numpy matplotlib scikit-learn pyyaml pillow tqdm
```

可选：检查 GPU 是否可用
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4) Dataset Preparation
配置文件默认使用：
- 图像目录：`/mnt/sharedata/ssd_large/users/liyx/leafdataset/images`
- 训练标签：`/mnt/sharedata/ssd_large/users/liyx/leafdataset/train.csv`
- 测试标签：`/mnt/sharedata/ssd_large/users/liyx/leafdataset/test.csv`

CSV 两列要求：
- 第 1 列：`image`
- 第 2 列：`label`

数据使用方式：
- `train.csv` 内部按 `train_ratio/val_ratio/test_ratio` 划分（默认 `0.7/0.1/0.2`）
- 其中 split 的 `test` 用于最终有标签评估
- 外部 `test.csv`（仅 `image` 列）用于最终提交预测，输出 `sample_submission.csv`

## 5) Training
```bash
python train.py --config cfgs/config.yaml
```
或
```bash
bash scripts/train.sh
```

## 6) Validation / Hyperparameter Tuning
验证集用于调参。可以改 `cfgs/config.yaml` 或使用 `--opts` 覆盖参数，例如：
- `model.num_hidden_layers`
- `model.hidden_dim` 或 `model.hidden_dims`
- `optimizer.lr`
- `optimizer.name` (SGD / Adam / AdamW)
- `model.dropout`
- `model.use_batchnorm`

示例：
```bash
python train.py --config cfgs/config.yaml --opts model.hidden_dim=1024 optimizer.lr=0.0005 output_dir=./outputs/exp_hd1024
```

## 7) Testing
```bash
python test.py --config cfgs/config.yaml --checkpoint outputs/mlp_baseline/checkpoints/best.pth
```
或
```bash
bash scripts/test.sh
```

可选：指定提交文件输出路径
```bash
python test.py --config cfgs/config.yaml --checkpoint outputs/mlp_baseline/checkpoints/best.pth --submission_path outputs/mlp_baseline/sample_submission.csv
```

## 8) Suggested Hyperparameter Experiments
可以优先尝试：
- hidden layers: `1 ~ 6`
- hidden dim: `128 / 256 / 512 / 1024`
- lr: `1e-1 ~ 1e-5`
- weight decay: `0 / 1e-5 / 1e-4 / 1e-3`
- optimizer: `SGD vs Adam`
- dropout / batchnorm / epochs

## 9) Outputs
默认输出到 `outputs/mlp_baseline/`，包括：
- `checkpoints/best.pth`, `checkpoints/latest.pth`
- `history.json`, `history.csv`
- `plots/loss_curve.png`, `plots/acc_curve.png`（和可选 f1 曲线）
- `config_used.yaml`
- `data_meta.json`
- `test_metrics.json`（测试后生成）

## 10) 纯 MLP 说明
模型结构是：
`Flatten -> Linear -> (BN) -> ReLU -> (Dropout) -> ... -> Linear(num_classes)`

项目中没有使用卷积层、池化层、Transformer 或预训练视觉模型。
