# DIVL + TRQAM for LIBERO-Long

基于 **OpenPI / π0.5** 的 Distributional Implicit Value Learning（DIVL）与
Trust Region Q Adjoint Matching（TRQAM）研究代码，以及 LIBERO-Long 实验源码快照。

This is a research adaptation, **not the official implementation** of DIVL/LWD or TRQAM.
Upstream attribution and licenses are retained in [NOTICE](NOTICE).

## 核心代码

| 内容 | 位置 |
| --- | --- |
| TRQAM 路径采样、adjoint、路径 KL、dual、独立慢 BC 与 EMA | [`src/openpi/training/trqam.py`](src/openpi/training/trqam.py) |
| DIVL Critic 与训练更新 | [`divl_critic.py`](src/openpi/training/divl_critic.py)、[`divl_lwd_training.py`](src/openpi/training/divl_lwd_training.py) |
| DIVL 模型、分布式价值头与损失 | [`src/openpi/models/`](src/openpi/models/) 中的 `divl_*` |
| QAM 及缓存版计算 | [`qam.py`](src/openpi/training/qam.py)、[`qam_cached_training.py`](src/openpi/training/qam_cached_training.py) |
| 实际使用的训练、回放、原生闭环评测和审计脚本 | [`archive/libero_long/`](archive/libero_long/) |

`trqam.py` 的数学实现与实验中的 `pilot_actor_math_v1.py` 一致，仅增加来源说明。
G/H 使用独立 BC β 优化器，reference 以 0.005 的新权重跟随 β；主 Actor 使用 Critic 引导。
DIVL 和 Scalar Critic 在本轮 Actor 训练期间冻结。

## 安装与源码检查

推荐 Linux + NVIDIA CUDA。核心工程依赖见 `pyproject.toml` 和 `uv.lock`。

```bash
git clone https://github.com/du17183/divl-trqam.git
cd divl-trqam
uv sync --frozen
source .venv/bin/activate
python tools/verify_release.py
python -c "from openpi.training.trqam import tr_step, tr_dual; print('TRQAM import OK')"
```

`verify_release.py` 只执行源码完整性、语法与小型 CPU 公式检查，不加载权重或运行仿真。
CUDA、模型权重与 LIBERO 仿真依赖需自行准备；参见 [安装与复现边界](docs/REPRODUCTION.md)。

## 实验设置

- 主 Critic/QAM 池：50 条训练专家轨迹 + 240 条训练 Rollout。
- BC/慢 β 池：50 条专家 + 109 条成功 Rollout，来源 50:50；失败 Rollout 不作为 BC 标签。
- 真实决策起点为 0、5、10…；Critic 使用真实执行 H5 转移，BC 使用跨 chunk 的真实 H10 动作。
- 主 Actor：Global batch 32、FP32、AdamW、学习率 2e-5、Cosine500。
- 参数/EMA 在同一条轨迹的 100/250/500 步检查点分别评测；每套 10 任务 × 50 回合。
- 全部实验合计 19 条 Actor 轨迹、7100 次主更新、2200 次独立 β 更新、4000 次 Critic 更新、24000 回合评测。

## 结果摘要（成功率 %）

| 方法 | 100步 | 250步 | 500步 |
| --- | ---: | ---: | ---: |
| DIVL+TRQAM / Rollout | 34.2 | 51.6 | 62.6 |
| DIVL+TRQAM / Mixed | 39.2 | 50.0 | 65.0 |
| Scalar+TRQAM / Mixed | 33.0 | 49.8 | 64.6 |
| BC | 39.4 | 45.6 | 66.2 |
| BC EMA | 48.0 | 58.8 | 64.8 |

500步 Mixed DIVL+TRQAM 相对 BC 为 −1.2 pp，配对名义95%CI为 [−5.6, +3.2] pp。
**本轮没有证明其稳定超过 BC。** 单训练 seed；官方固定初态被反复用于开发，属于
standard/development benchmark，不是未见测试集。多重比较 CI 未校正。
Mixed 同时改变 Critic 与主 Actor 数据，不能分别做因果归因。

## 仓库范围

本仓库包含源码、配置示例、聚合结果和逐文件 SHA256 来源记录。
不包含模型权重、优化器大状态、原始轨迹、逐回合评测数据或凭据。
原实验服务器文件未被此发布过程修改。

历史编排脚本依赖当时的完整数据、检查点和审计链；保留它们是为了核查实际实现，
**并非配置完成即可一键重跑的公共训练器**。新环境需要重新建立数据清单与来源证明，不能复用旧进程身份。

## 来源

- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [Learning while Deploying / DIVL](https://arxiv.org/abs/2605.00416)
- [Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234)
- [Trust Region Q Adjoint Matching](https://arxiv.org/abs/2605.27079) / [official code](https://github.com/yonghdong/trqam)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

代码来源版本与发布变更详见 [provenance/source_manifest.json](provenance/source_manifest.json)。
