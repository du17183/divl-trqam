# QAM loss 的简洁证明与 LWD 对照

**结论：当前加权速度回归与 QAM 的局部 loss 严格等价，但采样和离散化等细节不能由这个等式证明完全一致。** 本文以 AIRBOT 离线配置为准。

## 数学证明

记 $f_\theta$ 为 actor 速度，$f_\beta$ 为冻结 reference 速度，$g_w$ 为 lean adjoint，$\sigma_w^2=2(1-w)/w$。一次 actor 更新中，路径、reference 和 adjoint 均固定。

[QAM 原论文式 21](https://arxiv.org/html/2601.14234v4#S4) 的局部 loss 是

$$
\ell_w=\left\|\frac{2(f_\theta-f_\beta)}{\sigma_w}+\sigma_w g_w\right\|_2^2.
$$

定义回归目标

$$
y_w=f_\beta-\frac{\sigma_w^2}{2}g_w.
$$

对任意 $0<w<1$，直接提取 $2/\sigma_w$ 得

$$
\begin{aligned}
\ell_w
&=\left\|\frac2{\sigma_w}\left(f_\theta-f_\beta+\frac{\sigma_w^2}{2}g_w\right)\right\|_2^2\\
&=\boxed{\frac4{\sigma_w^2}\|f_\theta-y_w\|_2^2}.
\end{aligned}
$$

这正是 [当前代码](../src/openpi/training/qam_cached_training.py) 的写法：

```python
sigma2 = 2 * (1 - w) / w
target = stop_gradient(reference_velocity - 0.5 * sigma2 * adjoint)
loss = mean(4 / sigma2 * sum((velocity - target) ** 2))
```

两者逐点相等，没有省略常数项。对相同路径进行 batch 平均和时间加权求和后仍然相等。若 $J_\theta=\partial f_\theta/\partial\theta$，冻结监督后的参数梯度也相同：

$$
\boxed{\nabla_\theta\ell_w
=\frac8{\sigma_w^2}J_\theta^\top(f_\theta-f_\beta)+4J_\theta^\top g_w}.
$$

因此，`0.5` 和 `4 / sigma2` 都由论文公式直接推出，当前实现只是把平方残差改写成加权回归。

## 为什么这样实现

- **冻结监督。** `stop_gradient` 固定路径和 target，只对 actor 速度求参数梯度，形成局部监督更新。
- **Q 指导方向。** 终点使用 $g_1=-\nabla_a Q/\lambda$，其中 $\lambda=2$。负号来自终点代价 $-Q/\lambda$，使终点附近的速度修正包含 Q 上升方向。
- **反向传播 adjoint。** 使用 reference 漂移 $b_\beta=2f_\beta-a/w$，求解 $dg_w/dw=-J_{b_\beta}^\top g_w$；向后 Euler 更新为 $g_i=g_{i+1}+h_iJ_{b_\beta}^\top g_{i+1}$。它是连续 lean 方程的离散近似。
- **时间归约。** 动作坐标求和、batch 平均，再乘区间宽度近似时间积分；不在 $w=1$ 计算 loss，因为此处 $\sigma_w=0$。

## 当前实现与 LWD 的主要差异

依据 [LWD v4 式 9、IV-B、Algorithm 2 与 Appendix B2](https://arxiv.org/pdf/2605.00416v4)。论文未明确的实现细节不应直接判定为错误。

| 项目 | 当前 AIRBOT 实现 | LWD v4 |
| --- | --- | --- |
| 噪声系数平方 | $2(1-w)/w$，与 QAM 原论文一致 | 式 9 后写为 $2(1-w)w$，两者不等价 |
| Q 梯度的终点 | 新生成的无噪声动作 | 正文说生成终点，Algorithm 2 却写 replay 动作，存在歧义 |
| 时间与求解器 | 从 $\epsilon=0.1$ 放置标准高斯，10 个区间，最后一步 reference ODE | 公式写 $[0,1]$；未明确相同的截断与离散设置 |
| actor 的 Q 聚合 | $\min(Q_1,Q_2)$ | actor 公式写 $Q_\phi$，未明确双头聚合方式 |
| 动作 horizon | 32 | 实验设置为 30 |
| TD 步数 | 仅支持 1-step chunk TD | 离线短任务用 1-step，长任务用 10-step |

其中噪声系数的差异会实际改变 target，而不只是缩放 loss。例如取 $w=1/2$、$f_\beta=0$、$g_w=1$：

$$
y_{\rm current}=-1,\qquad y_{\rm LWD}=-\frac14.
$$

两个回归目标的最优值不同。当前采用除法有 QAM 原论文依据；不能仅凭这一差异确定 LWD 是否存在笔误。

固定 BC reference 采样、温度 2，以及 V → Q → EMA → actor 的更新顺序与 LWD 描述一致。固定 reference 采样不同于 QAM 原论文的当前 actor 采样，不应把它误列为与 LWD 的差异。

**上述证明保证相同监督下的局部 loss 和梯度等价，不保证有限步训练必然得到理论最优策略。**
