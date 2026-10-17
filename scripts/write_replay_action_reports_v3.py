# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Evidence-based reports; pending experiments are explicitly pending."""
from pathlib import Path
import json,datetime,hashlib,numpy as np,pandas as pd
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3';P=R/'reports';D=R/'data/action_sensitive_replay_v2'
def load(name):return json.loads((O/name).read_text())
def write(name,s): (P/name).write_text(s,encoding='utf-8')
def fmt(x):return '未完成/无可用估计' if x is None else f'{x:.6g}'
def ci(x):
    if x['mean'] is None:return 'NA（有效状态不足）'
    return f'{100*x["mean"]:.2f}% [95% CI {100*x["ci95"][0]:.2f}, {100*x["ci95"][1]:.2f}]，{x["states"]} 个有效状态'
def main():
    P.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    head='17561a508cfefe1b4170f16c52ac8adbb4d552f2'
    audit=f'''# 新版 LWD replay 设计审计

审计时间：{stamp}；项目 `/opt/divl-trqam`；复制项目与只读源仓库 HEAD 均为 `{head}`。

## 证据与历史

已读取六个指定模块，以及采集器、AIRBOT offline 入口、配置、Git log/blame/diff。完整带行号证据保存于 `/opt/divl-trqam/artifacts/replay_design_source_audit.txt` 和本轮 artifact 目录。

- `64a4ad7a`：LIBERO EpisodeReplay 和采集器。`git blame` 将 refresh/sample/composition 追溯至此提交。
- `d89813d9`：LIBERO n-step adapter（默认 td_steps=10）及 offline DIVL/QAM。
- `02bfc551` / `2659bce3`：AIRBOT offline / BC 统一入口，固定离线 replay。
- `17561a50`：dynamic beta，经 BC 更新后 EMA 到 reference；不会自动写新环境轨迹。
- `5a5621b3` 属于 `origin/test`，不是当前 HEAD 的祖先。其历史 LIBERO joint 入口和诊断文档仅作为设计背景，不能冒充当前运行代码。该旧入口是固定 replay、10000 critic warmup +20000 joint；诊断文档记录2999训练轨迹/196908chunks，包含不同来源策略和183条同状态 continuation，但不是当前项目结果。
- 当前未提交修复包含 FP32 master、完整 BC 序列/mask、remat、选择性读取；本轮没有改 QAM、DIVL loss 或 actor。

## 逐项回答

| 问题 | 代码事实 | 推断 / 未知 |
|---|---|---|
| 1 replay 来自什么 policy | `examples/libero/collect_divl.py:29-86,137-152` 从 WebSocket policy 服务取得 action，记录 policy_id；不限制唯一策略。AIRBOT模板 `examples/airbot/lwd_offline.md:9-10` 明示200遥操作+100rollout。 | 默认 LIBERO collector 可用于 frozen pi05；不等于必须仅用一种高成功率 policy。 |
| 2 是否不断刷新 | `divl_libero_adapter.py:25-38` refresh 只读取新增 NPZ。最新 `scripts/train_airbot_lwd.py:409-413,532-537` 固定 manifest 并重复采样，未自动采集。 | API 能刷新不等于训练调用了刷新；未发现 actor→collector→replay 自动反馈。 |
| 3 success/failure | `collect_divl_suites.py:1-5,46-56` 两者均计入 quota；EpisodeReplay composition 统计两者。AIRBOT入口 `train_airbot_lwd.py:427-434` 要至少一个成功 terminal，也允许 success-only 数据。 | 失败并非所有入口的硬要求；动作质量可辨监督仍是使用 QAM 的实际条件。 |
| 4 动作质量多样性 | 模块接受 recorded actions，未执行局部多样性检验。最新AIRBOT示例混合teleop+rollout。 | 学出动作价值依赖覆盖不同动作及未来；这属于学习条件推断，不是配置强制要求。 |
| 5 是否主要依靠 online exploration | 指定模块、当前collector及offline入口无 online exploration/actor环境更新循环。 | 不能声称原设计依靠在线探索。 |
| 6 perturbation/intervention/mixed replay | 当前collector不自动加扰动；AIRBOT显式mixed replay。`origin/test@5a5621b3` 的旧 prepare.py:71-73 支持branch group、旧诊断61-78行记载分叉数据。 | 当前HEAD没有保证每个state有多动作数据；旧分叉证明这种数据曾被用过，不能将其设为当前必备规模。 |
| 7 QAM前多少数据 | `qam_training.py:258-262` 明确offline联合更新不要求独立warmup；AIRBOT示例300episodes、联合从step1。另一个历史test入口warmup10000。 | 没有统一的“原版必须多少条/多少步”；本轮保持2000预算，禁止靠延长预算解决。 |
| 8 Q target动作 | `divl_lwd_training.py:86-87` 在当前state/recorded behavior action上取EMA双Q最小值拟合V；`96-113` next-state V的分位数形成TD标量，监督记录动作Q。 | QAM另在flow采样endpoint上求∂Q/∂a；这不为那些endpoint自动提供真实环境监督。 |
| 9 DIVL在哪些样本训练 | `divl_lwd_training.py:33-35,86-94` categorical V拟合相同replay minibatch的target min-Q标量（两邻近bin投影），不是success/failure分类，也不是直接输入全部候选真实G。 | 相似state下不同记录动作Q能产生有用distribution；仅有高成功单行为数据时未被保证。 |
| 10 dynamic beta是否意味着replay改变 | `beta_training.py:1-10,40-50,79-113` beta对记录行为序列做flow BC，再EMA到reference。 | “online beta”是每训练步更新模型，不等于online环境数据；固定replay也能启用dynamic beta。 |
| 11 当前200条90%成功是否忠实 | 形状、归一化、terminal和fixed replay本身兼容。只有一个frozen策略、20失败、10唯一train初态/task×2rep，未用专家；LIBERO pilot显式td_steps=1 (`scripts/train_libero_lwd.py:83-86`)，而LIBERO adapter默认10。 | 是缩小版可运行pilot，不能代表最新AIRBOT混合数据或旧LIBERO2999轨迹的覆盖。td1不是已证实bug：AIRBOT模板本来td1，LIBERO旧入口td10。必须记录此差异，不把恢复默认10宣称为已证实因果修复。 |

## QAM已有约束

`qam.py:119-157` 终端监督是−∂min(Q1,Q2)/∂A / temperature；`166-249` SDE路径+reference末端去噪；`252-320` reference drift的VJP逆向伴随；`354-365` velocity目标 `reference_velocity−.5σ²adjoint`。这些是reference相对约束，不是硬动作trust region。native action clip是controller合法区间，不是replay支持范围；epsilon/step-size检查是积分稳定性检查。当前LIBERO shim `sample_from_reference=True` 保留fixed reference路径；不能擅自用−Q actor loss替换。

## 当前结论

代码没有承诺在任意小规模frozen replay上自动获得可用action discrimination。应修复可验证的局部动作/未来覆盖，并保留严格held-out gate；不能用“本来需要online”作为没有代码证据的解释。本轮新replay只从固定train分区生成局部干预，policy/reward/controller/normalization不变。
'''
    write('new_lwd_replay_design_audit.md',audit)
    c=load('conditioning.json');text=f'# Critic action conditioning audit\n\n更新时间：{stamp}。只读step2000；没有更新任何参数。\n\n'
    text+='## 实际 action 注入\n\n`src/openpi/models/divl_heads.py:61-92`：7→256线性投影、H10位置编码、tanh、masked attention pooling；`95-135`：与640维state向量concat，再送入两套512/512 MLP。不是展平32维；25个actor padding维完全不进入Q；未执行step用attention −inf mask。Scalar/DIVL使用相同TwinQHead初始化和conditioner，训练后的参数可不同。quantile normalization为2(a−q01)/(q99−q01+1e−6)−1；分位数以外可超出[-1,1]，不额外截断normalized坐标。\n\n'
    text+='| Method | state norm | action pooled norm | state RMS | action RMS | ∂Q/∂a norm | ∂Q/∂z norm | norm ratio | first layer state/action contribution |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for m,v in c.items():
        mean=lambda key:float(np.mean(v[key]));text+=f'| {m} | {mean("state_norm"):.4f} | {mean("action_feature_norm"):.4f} | {mean("state_rms"):.4f} | {mean("action_rms"):.4f} | {mean("action_gradient_norm"):.5f} | {mean("state_feature_gradient_norm"):.5f} | {mean("gradient_ratio"):.4f} | {mean("first_layer_state_contribution_norm"):.4f} / {mean("first_layer_action_contribution_norm"):.4f} |\n'
    text+='\n## Action ablation（同一50states，固定随机置换/扰动）\n\n| Variant | Scalar mean absolute ΔQ | Scalar / Qstd | DIVL mean absolute ΔQ | DIVL / Qstd |\n|---|---:|---:|---:|---:|\n'
    for a in c['scalar']['ablations']:
        s=c['scalar']['ablations'][a];d=c['divl']['ablations'][a];text+=f'| {a} | {s["mean_abs_delta"]:.6f} | {s["relative_to_state_Q_std"]:.4f} | {d["mean_abs_delta"]:.6f} | {d["relative_to_state_Q_std"]:.4f} |\n'
    text+='\n未执行step action gradient最大值两组均0。动作分支被使用，但使用不等于质量判断正确。去除gripper信息的影响大于去除arm信息，可能反映动作阶段相关性；不是已证实gripper质量学习。DIVL state量级/首层贡献更大，提示encoder幅度可为后续共享conditioning消融方向；没有直接证明其导致排序失败。梯度比比较不同单位，不能视为严格重要性比例。BF16 encoder的批形状与原静态probe不同，Q的微小数值差异不用于更换主ranking结果；所有本轮ablation保持同一编码。\n'
    write('critic_action_conditioning_audit.md',text)
    design=f'''# Action-sensitive replay v2 design

更新时间：{stamp}。

## 决策

Case D（局部未来监督不足 + support/未见状态覆盖 + 待验证的target传播）。不是已证明padding bug，也没有足够Near-ID状态证明纯OOD问题。原有50-state小模型CV排序52.51% [47.00,58.21]，不能认定“换critic head就能学会”。

## 本轮最小数据修复

- 200原train轨迹，全部保留；每条固定10/30/50/70/90% chunk anchor，目标1000 states×5 candidates=5000 outcomes。
- 实际是10唯一train初态/task×2 policy repetitions=20轨迹/task，不能写成20唯一initial IDs/task。所有使用现有train分区；50 dev诊断states不进正式训练。
- 每state原recorded native behavior + 同一coherent Gaussian方向的±.25σ、±.5σ；σ来自local10neighbor/std并限于[.1taskstd,taskstd]；official quantile反归一化与native clipping保持不变。
- 同snapshot恢复、相同续跑noise seed，直到原剩余520步预算或first success；每state baseline完整续跑比对初始/下个anchor/RGB/policy plan/真实G，失败则全阶段blocked，不删除。
- 保存完整RGB/proprio/native reward/actions/next obs和实际prefix/mask；actor完整H10行为标签跨chunk拼接。初始完整snapshot另存states目录；无oracle actor输入。
- 同return/没有变化/失败结果一律保存。manifest和每episode SHA固定；三个GPU冻结推理服务的baseline门槛会检验跨设备一致性。
- 先用10trainstates/50outcomes测吞吐，再扩展；不开展更多dev rollout。

## Critic重训前条件

完整新replay后量化within-state action/G variance、不同G比例、success/failure mixture、支持范围。再做仅诊断小MLP的train-initial-state分组CV；时间点/两rep不得跨fold泄漏。如果新数据也不可学习，停止扩大和盲目重训，报告重新设计需求。

若存在足够可学习的监督，再执行两个2000update matched critics，B256、相同encoder/Q初始化、相同采样顺序/seed/optimizer/budget；保存250/500/1000/2000。可以恢复LIBERO原adapter的10chunk TD，必须两组一致、明确这是一项共享流程修正且尚未证明为根因。不能将同时改变replay和TD后的改善归因于其中一个变量。loss/QAM/DIVL结构不改。

采样需显式保存anchor与continuation混合规则，避免5000 anchors淹没在长continuation里；这是replay分布设计，不能改变reward/丢弃无差异branch。现有50×8真G仍为held-out诊断。

新critic若通过ranking/Spearman/Near-ID后，再在相同50states/epsilon运行新梯度方向真实续跑。旧600gradient outcomes仅验证旧方向，不能冒充新critic梯度收益。该限制保留原50states/400candidates/旧600记录，新增方向必须另存。

有critic通过广泛真实gradient门槛后立即QAM smoke→小actor update→fixed-reference pilot；dynamic beta后置。未通过不启动actor。
'''
    write('replay_v2_design.md',design)
    a=load('cpu_complete.json');future=load('future_diversity.json');ood=load('ood_analysis.json');control=load('positive_control.json')
    pairs=sum(v['similar_state_different_action_future_pairs'] for v in future['pair_summary'].values());total=sum(v['pairs'] for v in future['pair_summary'].values())
    progress=json.loads((D/'progress.json').read_text()) if (D/'progress.json').exists() else dict(states=10,outcomes=50,status='pilot_complete')
    stage=f'# 当前阶段完整进度\n\n更新时间：{stamp}；b300-1 `/opt/divl-trqam`。\n\n代码/replay/OOD/conditioning/正对照审计完成；train-only v2实际采集：{progress.get("states",0)} states / {progress.get("outcomes",0)} outcomes；状态 {progress.get("status")}。没有actor/QAM、dynamic beta、PPO训练。\n\n'
    answers=[
        ('原代码预期replay','最新入口为固定offline recorded actions；LIBERO支持新增文件refresh但没有自动online闭环；AIRBOT示例teleop+rollout；历史test分支有mixed policies/branches。'),
        ('当前200条是否符合','接口/fixed replay兼容，覆盖不足；不是原示例的混合来源，也不是旧LIBERO2999episode规模；原LIBERO默认td10而pilot显式td1，不是已证实bug。'),
        ('局部action variance',f'1000states，K10 mean={fmt(a["action_summary"]["variance_mean"]["mean"])}，p10/50/90={fmt(a["action_summary"]["variance_mean"]["p10"])}/{fmt(a["action_summary"]["variance_mean"]["p50"])}/{fmt(a["action_summary"]["variance_mean"]["p90"])}（normalized坐标）；有变化，不能声称action完全单一。'),
        ('局部future variance',f'mean={fmt(a["future_summary"]["future_variance"]["mean"])}，median={fmt(a["future_summary"]["future_variance"]["p50"])}；all-success76.6%，mixed23.4%。'),
        ('相似state/不同action/不同future pair',f'按预登记task内经验p10/p50阈值：{pairs}/{total}={100*pairs/total:.2f}%；visual近邻不是相同完整物理state，不能视为因果样本比例。'),
        ('Near-ID / Mid / OOD数',str(ood['counts'])+'；400个candidate不变，训练local-distance p50/p90校准。'),
        ('Near-ID Scalar ranking',ci(ood['ranking']['Near-ID']['scalar']['rank'])),
        ('Near-ID DIVL ranking',ci(ood['ranking']['Near-ID']['divl']['rank'])),
        ('完全不懂action还是OOD','action非零依赖；整体质量排序约随机。Near-ID只有7个informative states、CI宽，无法证明Near-ID可靠或纯OOD。Mid两组均有弱正向排序，但不足全局QAM准入。'),
        ('ablation Q变化','shuffle |ΔQ| Scalar0.01558 / DIVL0.01180；zero0.03448/0.01352；small.25σ0.00684/0.00275；medium.5σ0.01271/0.00478。'),
        ('padding/scale压制','无32维padding bug，真实7维，masked梯度0；action feature RMS .383/.395，state RMS .342/1.515。DIVL state第一层贡献33.48 vs action6.66，未证明因果压制。'),
        ('小模型held-out可学习性',ci(control['rank'])+f'；Spearman={fmt(control["spearman"]["mean"])}；train rank约87–89%，泛化未成立。'),
        ('问题分类','D：局部future单一、support/未见状态覆盖、小样本可学习性不足；C只剩scale可疑，尚不支持直接换模型。'),
        ('是否构造v2','已开始并保存实际train-only分叉；原数据不修改；目标1000states/5000outcomes，当前不是已完成5000。'),
        ('v2 outcomes数',str(progress.get('outcomes',0))+'，以progress/manifest实时值为准。'),
        ('v2是否增加局部差异','小批10states中7个state出现不同G，多为完成时间差；未出现新增混合成功/失败。完整1000states统计尚未完成，不能提前宣称有效。'),
        ('Scalar v2 ranking','未重训/未测。'),('DIVL v2 ranking','未重训/未测。'),
        ('真实gradient是否广泛贡献','旧DIVL正贡献99%来自2states，旧总体CI跨0；本轮没有新critic、新gradient证据。不能复用旧方向作为未来v2的梯度测试。'),
        ('是否允许QAM','否，STOP_ACTOR_QAM。达到固定gate后按用户授权立即smoke/pilot，目前未达到。')]
    for i,(q,v) in enumerate(answers,1):stage+=f'## {i}. {q}\n\n{v}\n\n'
    h=P/'history/recovery_20261007_current_stage_summary.md';h.parent.mkdir(exist_ok=True)
    if not h.exists() and (P/'current_stage_summary.md').exists():h.write_bytes((P/'current_stage_summary.md').read_bytes())
    write('current_stage_summary.md',stage)
    (O/'reports_written.json').write_text(json.dumps(dict(time=stamp,stage=progress,report_names=['new_lwd_replay_design_audit.md','current_replay_action_diversity.md','current_replay_future_diversity.md','counterfactual_ood_analysis.md','critic_action_conditioning_audit.md','action_learnability_positive_control.md','replay_v2_design.md','current_stage_summary.md']),indent=2))
    print('REPORTS_WRITTEN',stamp,progress)

if __name__=='__main__':main()
