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

## DIVL+TRQAM：问题分析与下一轮方案

**证据范围**：以下“已有实验结论”来自本轮已完成的训练、诊断与闭环评测；“原因假设”只是候选解释，
尚未经过能区分原因的对照实验；“下一轮方案”尚未执行，不属于本轮结果。

### 已有实验结论

| 观察项 | 已有证据 | 可以得出的结论与边界 |
| --- | --- | --- |
| 相对强BC基线的增益尚未建立 | Mixed DIVL+TRQAM 500步65.0%，BC 66.2%；配对差−1.2 pp，名义95%CI [−5.6, +3.2] pp。Scalar+TRQAM为64.6%。 | 本轮没有证明超过BC，也没有建立DIVL相对Scalar的稳定优势；不能把区间覆盖0解读为等效或方法无效。 |
| Critic动作排序与局部引导证据有限 | Mixed DIVL同状态排序54.97%；Near-ID排序从R0的80.12%降到R3的49.52%，配对差−30.60 pp，名义95%CI [−44.29, −15.42] pp。DIVL正梯度相对等幅随机方向的Observed Return增益为+0.00819，CI [−0.00294, +0.02111]。 | Near-ID子集有退化证据；实际回报验证尚未证明正梯度稳定优于随机扰动。不能只凭价值损失、非零梯度或54.97%的点估计认定梯度有效或“等同随机”。整体排序R3−R0的CI覆盖0，不能外推为Critic全面退化。 |
| 记录中的KL约束未触及预算，优化尺度值得检查 | H-R3-Cosine500记录的raw路径KL最大0.405117，预算1.0，0次超预算；374/500条日志的**更新后有效λ**处于0.03下限；主Actor 500/500次更新触发梯度裁剪，500次均有效应用，β/reference也各更新500次。 | 现有日志不支持“KL太强把学习压住了”；也不能仅凭梯度裁剪认定梯度爆炸。记录中的批量平均KL不等于所有状态或每个任务的偏移保证，动态dual的收益需消融验证。 |
| 任务间收益与退化抵消 | 500步Task2/6相对BC为+22/+16 pp，Task7/8为−12/−20 pp；Task3为86%→28%→84%。 | 已观察到任务差异和检查点波动；尚不能证明其原因是任务间干扰、遗忘、采样噪声或某类失败机制。逐任务50回合的差值仍是描述性结果。 |
| 数据与模块贡献尚未拆开 | Mixed同时改变Critic和主Actor数据；TRQAM包含独立BC β和移动EMA reference，但β/reference未单独闭环评测。 | 无法把65%的表现全部归因于Q引导或慢BC，也无法把Mixed的变化分别归因于Critic或Actor。 |

H-R3总体成功率确实随检查点从39.2%→50.0%→65.0%提高，不能概括为“长训必然退化”；尚未建立的是相对强BC基线的稳定额外收益。

证据索引：[`critic_validation_summary.csv`](results/critic_validation_summary.csv)、
[`critic_paired_comparisons.csv`](results/critic_paired_comparisons.csv)、
[`trqam_optimizer_summary.csv`](results/trqam_optimizer_summary.csv)、
[`trqam_kl_dual_beta.csv`](results/trqam_kl_dual_beta.csv) 和上方逐任务表。
`lambda_parameter`的投影下限为0.01，乘以3后得到有效下限0.03，见
[`tr_dual` / `tr_step`](src/openpi/training/trqam.py)；日志中更新后的λ与当前步计算使用的前一λ应区分。

Critic验证的范围也有限：固定100个物理状态、60个物理来源用于动作排序；梯度检验使用原30个固定状态，
Mixed为Scalar/DIVL各追加30条正方向分支，随机/参考分支复用并核验原记录。Near-ID是原baseline与
independent-small的描述性子集。动作排序比较的CI使用原2000次Bootstrap，闭环成功率使用10000次。
heldout1000窗口中404个完整MC与596个截断Observed后缀分开统计；520步截断分支不能冒称完整MC。

### 待验证的原因假设

| 假设 | 为什么值得检查 | 如何验证或证伪 |
| --- | --- | --- |
| **H1：Q的局部动作梯度偏差限制了策略改进** | 排序与真实正梯度回报验证较弱。DIVL建模状态条件价值分布，不自动保证Q对动作的局部导数准确；TRQAM限制偏移也不自动纠正错误方向。 | 在新的、与训练来源分离的物理状态上，比较原动作、正/负梯度与等幅随机扰动，覆盖多个扰动幅度；核对Q预测变化与真实回报/成功、恢复结果是否一致。 |
| **H2：移动参考、KL尺度和优化尺度的组合不合适** | KL长期低于预算、有效λ到达下限且持续裁剪，提示应检查预算归一化与更新尺度；这些观察本身不能确定因果方向。 | 保持β数据、优化器与EMA一致，比较固定λ和动态dual；记录逐任务KL分布、实际动作偏移、adjoint/速度差项及交叉项、主/β梯度与实际参数更新。先在独立开发集校准预算，不直接假定调紧或放松就有效。 |
| **H3：固定离线数据和冻结Critic不能覆盖变化后的策略分布** | 主池只有290条轨迹并有放回抽样，重复更新不等于增加独立经验；本轮Actor更新时Critic冻结。 | 对照静态训练与“新策略采集→刷新Critic→更新Actor”，控制新增交互量；另做R0/R3 Critic数据 × R0/R3 Actor数据的2×2交叉对照，分离两种数据改动。 |
| **H4：稀疏长时程奖励的信用分配与成功率目标存在张力** | 原协议按执行步使用γ=0.99；仅从折扣量级看，350/520步后的终局奖励权重约0.030/0.0054。 | 按距成功/终止的距离检查TD与完整MC的一致性，核对timeout/bootstrap和实际执行步数；再单独做γ/目标的敏感性实验，并控制Q引导尺度变化。这是待验证的目标设计问题，不是已确认的实现错误。 |
| **H5：多任务共享更新或关键恢复状态覆盖不足** | Task2/6改善而Task7/8退化，Task3波动较大，但当前逐任务结果不能区分原因。 | 对齐失败阶段和物理初态，检查Task7/8的接触、恢复、纠错状态覆盖；做采样配额或单任务/共享模型对照，一次改变一个因素，并跟踪其他任务是否回退。 |

### 下一轮实验方案（尚未执行）

建议依次推进，先定位收益来源，再决定是否扩大训练。

| 优先级 | 实验 | 必须保留的对照与指标 | 决策用途 |
| --- | --- | --- | --- |
| **P0：拆分贡献** | 对已有100/250/500检查点分别评测β、EMA reference和TRQAM主Actor。 | 相同任务/初态、逐episode种子、推理步数与动作执行协议；报告逐任务成功/Return/长度、恢复与回退、主Actor−β/reference配对CI。 | 确认Q引导是否在其自身慢BC参考之上提供额外收益；不能用外部BC EMA代替这一对照。 |
| **P1：先验证并改进Critic** | 扩充独立来源的同状态正/负梯度与随机扰动检验；检查TD/MC、终止与截断mask、价值分布边界质量和局部Q校准。 | 成功与失败恢复状态都覆盖；完整MC与截断Observed分开；按任务/物理来源计算CI。数据配额、恢复数据、ensemble/梯度一致性门控或候选动作选择分别做消融，不同时全部修改。 | 预先定义梯度真实收益与校准的通过条件。若正梯度相对随机的CI仍覆盖0，则保持“未证实”，优先补验证或修Critic，再决定扩大Actor训练。 |
| **P2：隔离TRQAM机制** | 在同一β/reference下对照无Q引导、固定λ与动态dual；在独立开发集预先登记小规模KL预算/尺度网格。 | 匹配初始化、数据、主更新及β更新；同时报告总计算成本。检查真实动作漂移、逐任务KL分位数、裁剪比例和参数更新，避免仅比较总loss。 | 判断改善是否来自Q引导、移动参考或dual，以及预算是否实际起作用；不根据现有官方50初态反复挑最优参数。 |
| **P3：验证数据与信用分配假设** | 完成Critic×Actor数据2×2对照后，再独立比较静态与迭代数据刷新；γ/回报目标只作为另一个单因素实验。 | 交互量、数据可用性与BC基线公平对齐；新增数据与旧数据分开标记，保留训练/开发/测试来源边界。 | 区分数据覆盖、Critic刷新和折扣设计的作用，避免把多个改动的组合提升归因于某一模块。 |
| **P4：确认可重复增益** | 至少3个训练seed做初步复验，并以更多seed和独立未见初态确认；主要终点预先固定为500步，100/250步全部报告。 | 官方50初态继续标为开发基准；新初态/来源测试单独报告。预先规定最小有意义收益、允许的任务回退和统计比较，使用任务等权、物理来源配对CI并处理多重比较。 | 判断增益是否可重复、是否能泛化；单次最好检查点或平均成功率上升不足以证明稳定优越。 |

**当前建议的首要投入是P0贡献拆分与P1 Critic动作梯度验证。** 现有证据不足以支持直接增加Actor训练步数，
或仅通过调整KL就能解决问题。后续方案的阈值、seed、预算和统计分析应在新实验前确定；本次文档更新不表示这些实验已经执行。

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
