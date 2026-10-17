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

- **任务与样本量**：LIBERO-Long 的10个任务。每种方法、每个检查点/参数变体评测500回合，每任务50回合；配对比较使用对齐的官方初态与环境/策略种子。
- **数据配置**：R0为Rollout主池；R3（Mixed）主 Critic/QAM 池包含50条训练专家轨迹 + 240条训练Rollout。主池按任务均匀、任务内轨迹均匀、轨迹内chunk均匀，有放回采样，无专家来源配额。
- **BC监督**：BC/慢 β 池为50条专家 + 109条成功Rollout，来源50:50；失败Rollout不作为BC动作标签。
- **动作与转移**：真实决策起点为0、5、10…；Critic使用真实执行H5转移，BC使用跨chunk的真实H10动作；回合上限520步。
- **训练配置**：主 Actor Global batch 32、FP32、AdamW（β₁=0.9、β₂=0.999，weight decay 0.01，梯度范数裁剪1.0）、初始学习率2e-5、Cosine500；SFT初始化Seed0，训练Seed20261010。Actor训练期间Critic冻结。
- **检查点口径**：100/250/500步是同一条Cosine500训练轨迹的保存点，不是三个分别训练并挑选的最优结果；不能与早期Cosine100轨迹的100步直接等同。
- **预算区别**：BC参数与EMA来自同一条BC500轨迹，EMA不增加优化器更新；TRQAM另有独立的慢BC β优化器。因此主Actor更新次数相同不代表总计算预算相同。BC EMA也不是TRQAM内部β/reference的单独评测。
- **完整实验规模**：19条Actor轨迹、7100次主更新、2200次独立β更新、4000次Critic更新、48套完整500回合评测，共24000回合。下表摘录其中与DIVL+TRQAM和BC/EMA相关的结果。

## 100/250/500步总体对照（成功率 %）

| 方法 | 100步 | 250步 | 500步 |
| --- | ---: | ---: | ---: |
| DIVL+TRQAM / Rollout | 34.2 | 51.6 | 62.6 |
| DIVL+TRQAM / Mixed | 39.2 | 50.0 | 65.0 |
| Scalar+TRQAM / Mixed | 33.0 | 49.8 | 64.6 |
| BC | 39.4 | 45.6 | 66.2 |
| BC EMA | 48.0 | 58.8 | 64.8 |

总体表来自 [`cosine500_all_checkpoints.csv`](results/cosine500_all_checkpoints.csv) 和
[`bc_budget_matched.csv`](results/bc_budget_matched.csv)。以下逐任务主表使用 **Mixed DIVL+TRQAM（H-R3）**。

## 500步逐任务指标：Mixed DIVL+TRQAM

| Task | 任务简述 | 成功/回合 | 成功率 | 平均Return | 平均回合长度 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 字母汤罐、番茄酱罐放入篮子 | 29/50 | 58% | 0.580 | 400.56 |
| 1 | 奶油奶酪盒、黄油放入篮子 | 42/50 | 84% | 0.840 | 303.46 |
| 2 | 打开炉灶并放上摩卡壶 | 36/50 | 72% | 0.720 | 322.72 |
| 3 | 黑碗放入底部抽屉并关上 | 42/50 | 84% | 0.840 | 279.48 |
| 4 | 白杯放左盘、黄白杯放右盘 | 36/50 | 72% | 0.720 | 326.44 |
| 5 | 书放入收纳架后部隔间 | 38/50 | 76% | 0.760 | 273.98 |
| 6 | 白杯放盘上、布丁放盘右侧 | 22/50 | 44% | 0.440 | 403.64 |
| 7 | 字母汤罐、奶油奶酪盒放入篮子 | 41/50 | 82% | 0.820 | 317.28 |
| 8 | 两个摩卡壶放上炉灶 | 10/50 | 20% | 0.200 | 507.14 |
| 9 | 黄白杯放入微波炉并关门 | 29/50 | 58% | 0.580 | 427.54 |
| **总体** | **10任务等权** | **325/500** | **65.0%** | **0.650** | **356.22** |

表中Return按0/1终局成功奖励的回合累计值统计，故其均值与成功率对应；它不是Critic的折扣价值Q。
平均回合长度以仿真步数计，包含成功与失败回合，不能单凭长度判断策略优劣。
Task编号为本实验CSV中的0–9编号，完整官方任务名保存在逐任务CSV中。

## 逐任务检查点及BC/EMA对照

成功率单位为%；最后一列为500步 **Mixed DIVL+TRQAM−BC** 的百分点差。

| Task | TRQAM 100步 | TRQAM 250步 | TRQAM 500步 | BC 500步 | BC EMA 500步 | 相对BC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26 | 54 | 58 | 60 | 62 | -2 pp |
| 1 | 68 | 76 | 84 | 92 | 92 | -8 pp |
| 2 | 0 | 74 | 72 | 50 | 64 | +22 pp |
| 3 | 86 | 28 | 84 | 88 | 84 | -4 pp |
| 4 | 64 | 54 | 72 | 70 | 62 | +2 pp |
| 5 | 50 | 54 | 76 | 84 | 80 | -8 pp |
| 6 | 30 | 38 | 44 | 28 | 36 | +16 pp |
| 7 | 56 | 72 | 82 | 94 | 88 | -12 pp |
| 8 | 8 | 6 | 20 | 40 | 36 | -20 pp |
| 9 | 4 | 44 | 58 | 56 | 44 | +2 pp |
| **总体** | **39.2** | **50.0** | **65.0** | **66.2** | **64.8** | **−1.2 pp** |

完整的H-R3、BC参数、BC EMA在100/250/500步的90条逐任务聚合记录见
[`per_task_metrics.csv`](results/per_task_metrics.csv)，含成功数、回合数、成功率、平均Return和平均长度。
这些是任务级统计，不包含逐回合轨迹或原始评测数据。

## 结果解读与局限

- **总体未证明超过BC**：500步Mixed DIVL+TRQAM为65.0%，BC为66.2%，差值−1.2 pp，配对名义95%CI为 **[−5.6, +3.2] pp**。该区间覆盖0，不能据此宣称TRQAM优于BC，也不能将其解释为证明两者等效。
- **收益与退化并存**：500步Task2/6相对BC为+22/+16 pp，Task7/8为−12/−20 pp；Task8仅20%，平均长度507.14步。Task3随检查点为86%→28%→84%，呈现较大波动。这些逐任务差值是描述性结果，没有在本表提供逐任务配对显著性检验。
- **样本与泛化范围**：每任务50回合，1回合对应2个百分点；仅一个SFT初始化和一个训练seed，不能估计跨训练seed方差。官方固定初态被反复用于开发，属于 **standard/development benchmark，而非独立未见Final测试集**；“最终验收”表示交付与审计完成，不代表未见测试集结果。
- **统计口径**：总体按10任务等权，配对CI按任务分层、物理来源聚类Bootstrap10000次；名义多重比较CI未校正。这些CI不涵盖不同训练seed带来的不确定性。
- **归因限制**：Mixed同时改变Critic和主Actor训练数据，不能分别做因果归因；TRQAM内部β与EMA reference未单独闭环评测，不能把主Actor表现全部归于Q引导或慢BC。上述结果也不构成DIVL优于Scalar Critic的普遍证据。

所有表格取自已验收的最终聚合快照，源快照SHA256为
`ee738cb74c835185a8ea8ba60073bfafbd70e87fa236d3bae93d29da9f5445c6`。

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
