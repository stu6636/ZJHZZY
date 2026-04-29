# BloomGoal for ALFworld

本目录包含基于 ALFWorld 的 BloomGoal 相关实现与训练脚本。

## 安装指引

### 1. 安装 ALFWorld 及其依赖

```bash
git clone https://github.com/alfworld/alfworld.git
cd alfworld
pip install setuptools==63.2.0
pip install --no-build-isolation
pip install -e .[full]
```

### 2. 安装本项目依赖

```bash
pip install -r requirements.txt
```

说明：当前目录中依赖文件名为 `requirement.txt`。如果你未创建 `requirements.txt`，请使用：

```bash
pip install -r requirement.txt
```

## 项目结构

- `bloom_dist.py`：BloomGoal 算法主实现。
- `train.py`：训练模式脚本（基础训练流程）。
- `train_max.py`：长程训练脚本，持续训练并按区间记录统计信息。
- `base_config.yaml`：ALFWorld 环境与训练配置。
- `prompts/alfworld_3prompts.json`：提示词示例数据。
