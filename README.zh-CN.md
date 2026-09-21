<div align="center">

# Jev-Evolve：你的 agent 是在做决策，还是在读选项顺序？

**同一个 schema、同一个模型、同一批 96 条样本，只因为选项列出的顺序不同，准确率是 0.188 或 0.542。测出你的类型化决策真正在依据什么，把它消掉，并且不再相信没跨过自身噪声下限的搜索。**

[![PyPI](https://img.shields.io/pypi/v/jev-evolve?logo=pypi&logoColor=white)](https://pypi.org/project/jev-evolve/)
[![Python](https://img.shields.io/pypi/pyversions/jev-evolve?logo=python&logoColor=white)](https://pypi.org/project/jev-evolve/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/novaleolin/jev-evolve/tests.yml?branch=main&label=tests&logo=github)](https://github.com/novaleolin/jev-evolve/actions)

**[快速开始](#快速开始) · [实测](#实测) · [为什么要 jev](#为什么要类型化决策又为什么要一个快的) · [工作方式](#工作方式) · [API](#api) · [常见问题](#常见问题) · [English](README.md)**

</div>

---

## Jev-Evolve 是什么

一个 agent 框架，每一个分支都是类型化问题：在这些选项里选一个、是或否、给一个分数。
不是先生成一段文本，再由下游去解析。

它之所以是现在这个样子，来自搭它时掉出来的一个发现。我们拿一个八选项的客服意图 schema，
规则是认真手写的，把它的文本进化了 14 代、43 个候选，**两次都没有任何改进**。
原因不在搜索：

```
  8 种选项顺序下答案发生变化        96/96  (100%)
    其中纯噪声                       0/96  (0%)
  各顺序准确率        0.188 到 0.542   spread 0.354
```

这个 schema 根本不是在依据它的 criteria 做判断，而是在依据每个选项碰巧排在第几位。
所有文本变异一直在和这个东西搏斗。

所以这个库按顺序做三件事：

| | |
|---|---|
| **1. 测** | `permutation_sensitivity` 跑 k 种顺序，外加 k 次固定顺序的对照，这样后端自身的随机性不会被算到顺序头上 |
| **2. 修** | `Marginalized` 把顺序平均掉。**held-out +15.6 点**，对比同一 schema 上全部文本算子合计 +0.0 |
| **3. 验** | `evolve` 报出同等规模的搜索在"什么都没改进"时能拿到多少，并且用 held-out 数据下判定 |

## 快速开始

```bash
pip install jev-evolve
python examples/order_check.py     # 20 秒，无需 API key、不下载模型
```

调优之前先体检。这是库里最便宜的一件事，而且它决定了你之后做的任何事情能不能有意义。
这个例子会在**两个** schema 上跑：一个不通过、一个通过——总在报警的检查不叫检查。

```python
from jev_evolve import permutation_sensitivity, Marginalized, run_policy

print(permutation_sensitivity(policy, backend, tasks, score, k=8).report())
```

```
  option orderings tried   8   (plus 8 controls with the order held fixed)
  answers that changed     96/96   (100.0%)
    of which noise alone   0/96   (0.0%)
    attributable to order  100.0%
  accuracy by ordering     0.188 to 0.542   spread 0.354
    control spread         0.000

  ORDER-DEPENDENT: this schema is partly measuring option
  position, so tuning its text will mostly fit noise.
  Wrap the backend in Marginalized(backend, k) and re-check.
```

然后修，就是套一层：

```python
run_policy(policy, Marginalized(backend, k=8), tasks, score)
```

## 实测

Qwen2.5-0.5B，banking77 中 8 个易混卡片意图共 96 条，每个选项一条手写规则。
`python experiments/order_dependence.py --k 8`。

同样八个选项的八种排列：

| 顺序 | 准确率 |
| :--- | ---: |
| 0（原样写的） | 0.458 |
| 1 | 0.302 |
| 2 | 0.438 |
| 3 | **0.542** |
| 4 | **0.188** |
| 5 | 0.417 |
| 6 | 0.385 |
| 7 | 0.438 |

你的 schema 最终带着哪个数上线，取决于你当初把选项敲成了什么顺序。

### 整个结论所依赖的那个对照

平均 k 次调用本身就可能带来收益——那只是集成，和选项顺序、和类型化决策、和这个库
统统没有关系。所以同样的 k 次调用再跑一遍，但**顺序不动**：

| 切分 | 原顺序 | 平均 k 次·顺序固定 | 平均 k 次·顺序打乱 |
| :--- | ---: | ---: | ---: |
| train | 0.458 | 0.458 | **0.635**（+0.177） |
| held-out | 0.448 | 0.448 | **0.604**（+0.156） |

固定顺序那一列与原顺序那一列**小数点后三位完全相等**。后端是确定性的，
所以集成的贡献恰好为零，全部收益来自打乱顺序。

### "搜一个好顺序"和"把顺序消掉"不是一回事

库的两半在这张表上合上。在训练集上挑最好的顺序**是一次搜索**，所以选择下限对它适用；
边缘化不选择任何东西，所以不适用。

| | train 增益 | floor | held-out 增益 |
| :--- | ---: | ---: | ---: |
| 从 8 个顺序里挑最好的 | +0.083 | **+0.073** | +0.115 |
| 把顺序边缘化掉 | 不适用 | 无 | **+0.156** |

那次搜索 +0.083 的表观训练增益，只比"8 个候选在 96 条样本上、彼此毫无真实差异时白拿的
+0.073"高出 0.010。绝大部分是选择本身。它确实有一部分迁移到了 held-out 的 +0.115，
但仍然不如把依赖消掉——后者 +0.156，不选择任何东西，而且**高于集合里运气最好的那个顺序**。

这就是"修掉一个偏置"优于"绕着它调参"的论证，落在一个 schema、一组真实数字上。

## 为什么要类型化决策，又为什么要一个快的

### 搜索空间的存在，来自"决策是类型化的"

靠生成文本做决策的 agent，交给调优循环的只有一段 prompt 和每条样本一个标量。
分支是类型化问题的 agent 交出的是结构，而结构就是可动手的东西的大部分。

| 循环能看到或改变什么 | 生成式 agent | 类型化决策 agent |
| :--- | :---: | :---: |
| 指令措辞 | 可以 | 可以 |
| 每个选项的描述文本 | 不行，选项不是声明出来的集合 | 可以 |
| **选项被呈现的顺序** | 无法与 prompt 分离 | 可以 |
| 哪个决策先跑 | 不行，它是一整团 | 可以 |
| 每个决策能看到什么 | 不行，只有一个上下文 | 可以 |
| 何时弃权而不是硬答 | 没有可校准的概率 | 可以 |
| 该磨哪个选项、对着哪个竞争者磨 | 没有概率分布 | 可以 |
| 是哪个决策点输掉了这条样本 | 只有一个标量结果 | 可以 |

第三行正是这个项目一开始站错了的那一边。对一个声明出来的选项集做排列，是一个定义明确的操作；
对自由文本 prompt 里的"那些选项"做排列，不是。

### 而这个修复的代价是每个答案 k 倍决策

对顺序做边缘化不是新想法。新的是**付得起**。同一条网络、同一批样本、同一组选项、
同一份描述文本下实测，8 个易混意图共 160 条：

| 模型 | 准确率 | p50 | p90 |
| :--- | ---: | ---: | ---: |
| typesafe/jev-1.13 | 0.856 | **450 ms** | **536 ms** |
| mistralai/mistral-nemo | **0.900** | 656 ms | 1206 ms |
| qwen/qwen3.7-flash | 0.881 | 764 ms | 917 ms |
| openai/gpt-5-nano | 0.881 | 703 ms | 907 ms |
| ibm-granite/granite-4.0-h-micro | 0.786 | 568 ms | 1064 ms |

请如实地读：开箱即用时，决策模型是最快的、尾部远比别人紧，同时比最好的便宜生成模型
**低 4.4 个点**。

这就是全部论证，而且它不是"快本身很好"。在 450ms、p90 536ms 下，
`Marginalized(backend, k=8)` 是一个几秒钟的决策；在 3 秒一次、p90 还宽一秒的模型上，
同样的修复是每个决策半分钟，没有人会把它上线。**一个已知正确、原本太贵的修正变得可行**，
而它在本地模型上值 held-out +15.6 点，是决策模型起步时那 4.4 点差距的三倍多。

## 工作方式

五个部件，每个都能单独用。

**Policy（策略）。** 把决策点写成数据：指令文本、每个选项的描述、选项列出的顺序、
每个决策点能看到哪些 state 字段、以及低于多少置信度就弃权。全都可搜索。

```python
from jev_evolve import Policy, choice, noul

policy = Policy({
    "in_scope": noul("这是银行账户相关的问题吗？", threshold=0.6),
    "intent":   choice("哪个意图？", {"lost_card": "...", "top_up": "..."},
                       threshold=0.3, state_fields=["text", "channel"]),
}, order=["in_scope", "intent"])
```

**Invariance（不变性）。** 检查与修复。先跑这个。

```python
from jev_evolve import permutation_sensitivity, Marginalized

s = permutation_sensitivity(policy, backend, tasks, score, k=8)
if not s.sound:
    backend = Marginalized(backend, k=8)
```

`Sensitivity` 带有 `.flip_rate`、来自对照的 `.noise_rate`，以及
`.excess_flip_rate`——归因于顺序的那部分。判定只看超出量，而且只看 flip rate、
不看 spread：max 减 min 这个统计量随 k 机械增长，在任何付得起的 k 上分辨率都太差，
所以它留在报告里当上下文，但没有投票权。

**Agent（循环）。** 按顺序跑决策点。`act(state, answer)` 应用每个回答，
你的工具调用写在这里。返回 `{jev_evolve.STOP: True}` 提前结束。

```python
def act(state, answer):
    if answer.name == "in_scope" and not answer:
        return {jev_evolve.STOP: True, "outcome": "handoff"}
    return {answer.name: answer.choice}

episode = Agent(policy, backend, act=act).run("t1", {"text": "卡丢了"})
```

决策点弃权或回答"否"时 `answer` 为假值。`answer.margin` 是与第二名的差距。

**Trace（轨迹）。** 每个决策、它的概率分布和耗时，存成 JSONL。

```python
jev_evolve.confusions(trace)       # (决策点, 选了什么, 本该选什么) -> 次数
jev_evolve.point_accuracy(trace)   # 哪个决策点吃掉了大部分损失
jev_evolve.overconfident(trace)    # 又错又自信，阈值救不了的那些
jev_evolve.cost(trace)             # 每条样本的决策数与秒数
```

**Evolve（进化）。** 一代代生成变异体、打分、保留更好的，并给出诚实的报告。

```python
from jev_evolve.mutate import default_operators

ops = default_operators(available_fields=["text", "channel"],
                        examples_by_label=INTENTS, trace=trace)
res = evolve(policy, backend, train, score, ops, heldout=held)
print(res.report())
res.best.save("policy.json")
```

一个"保留最优候选"的循环，即使什么都没改进也会报出提升。
`python examples/null_loop.py` 把这个循环跑在一个完全随机、不读策略的后端上，
真实提升恒为 0：

```
   5 generations  apparent gain +0.017   floor +0.064   NOT CONFIRMED
  20 generations  apparent gain +0.050   floor +0.086   NOT CONFIRMED
  60 generations  apparent gain +0.058   floor +0.101   NOT CONFIRMED
```

![空转循环报出的提升](docs/null.png)

在一个完全没有变好的 agent 上，报出的提升随代数增长。每一个按评测分数做选择的
自进化循环都有这个性质，绝大多数不报。这部分算术来自
[evalfloor](https://github.com/novaleolin/evalfloor)，是依赖而不是拷贝。

## 变异算子

六个盲改策略，两个读 agent 自己的轨迹。

| 算子 | 改什么 | 需要 |
| :--- | :--- | :--- |
| `mutate_option_order` | 选项的排列顺序 | 无 |
| `mutate_threshold_from_trace` | 弃权阈值，网格从后端自己的置信度里读出来 | 一份轨迹 |
| `mutate_order` | 哪个决策点先跑，也就改变了后面的点能看到什么 | 无 |
| `mutate_state_fields` | 某个决策点允许看到哪些字段 | 字段名 |
| `mutate_criteria_from_examples` | 用真实输入描述一个选项 | 带标注的数据 |
| `mutate_criteria_from_errors` | 用**漏掉的**输入描述一个选项 | 一份轨迹 |
| `mutate_from_confusions` | 重写最常被误选的那个选项 | 轨迹 + 改写器 |
| `mutate_instructions` | 重写决策点的指令文本 | 改写器 |

用 `mutate_threshold_from_trace`，不要用固定网格的 `mutate_threshold`。
在上面那个八选项任务上，固定网格里 ≤0.2 的值全是同一个 no-op，≥0.5 的值在五分之四的
决策上弃权：十一个网格点里真正起作用的只有两个，一次 15 代运行中有三分之一的候选槽位
喂给了不可能有帮助的值。

改写器是任意 `(text, slot, rng) -> text`。`rng` 是契约的一部分：
改写器若去拿全局 `random`，整个运行就不可复现，而一个专门告诉你"这次提升有多少是真的"
的工具，不能每次问都给不同答案。

## 后端

| 后端 | 是什么 | 成本 |
| :--- | :--- | :--- |
| `RuleBackend` | 一个 Python 函数。测试、基线、混合策略 | 免费 |
| `OverlapBackend` | `jev_evolve.demo` 里的词重叠打分器。示例与 CI | 免费 |
| `LocalBackend` | 任意 causal LM 的选项 logits，一次 prefill，不生成 | 本地算力 |
| `JevBackend` | 托管的类型化决策端点 | 按调用计费 |
| `Marginalized` | 包住上面任意一个，k 种顺序取平均 | 内层的 k 倍 |

`LocalBackend` 需要 `pip install "jev-evolve[local]"`。其余部分不需要任何额外依赖，
`import jev_evolve` 不会拉进任何模型库。

`JevBackend` 默认指向 OpenRouter 的 decisions API，传 `endpoint=` 可换成任何
同样收 `{model, state, questions}` 的服务。包里其它地方都不知道背后是哪一家。

托管厂商通常禁止用其输出去训练或构建竞品模型，而一次进化循环是几千个决策。
在你自己拥有的东西上进化，再验证胜者：

```python
from jev_evolve import JevBackend, validate_transfer

t = validate_transfer(policy, res.best, JevBackend(), heldout, score)
print(t.report())    # 恰好 2 * len(heldout) 次调用
```

`validate_transfer` 不报 floor，也确实不适用：两个固定策略在同一批样本上打分，
没有选择发生。若拿它跑多个候选再挑最好的，选择就又回来了，floor 也跟着回来。

## API

```python
from jev_evolve import Policy, Point, choice, noul       # 策略
from jev_evolve import permutation_sensitivity, Marginalized, Sensitivity
from jev_evolve import Agent, Answer, STOP               # 循环
from jev_evolve import Trace, Episode, Decision          # 记录
from jev_evolve import evolve, run_policy, Result        # 搜索
from jev_evolve import validate_transfer, Transfer       # 本地 -> 托管
from jev_evolve import confusions, point_accuracy, overconfident, cost

permutation_sensitivity(policy, backend, tasks, score, k=4)
# -> Sensitivity: .flip_rate .noise_rate .excess_flip_rate .sound .report()

evolve(policy, backend, tasks, score, operators,
       generations=20, candidates=4, heldout=None, act=None, seed=0)
# -> Result: .best .gain .floor .heldout .credible .verdict .report()

validate_transfer(base, evolved, backend, tasks, score)
# -> Transfer: .gain .paired .transferred .calls .p50_s .p90_s .report()
```

一条 task 是 `(id, state, label)` 或 `{"id":..., "state":..., "label":...}`。
`score(episode) -> float` 由你实现，0/1 能让下限的算术是精确的。

## 局限

`permutation_sensitivity` 只覆盖选项顺序。它在这里恰好是主导因素，
但它不是类型化决策唯一可能违反的不变性：state 字段顺序、选项命名、无关 state
都没有被这个检查覆盖，被它判为 STABLE 的 schema 仍可能在依据其中之一。

报出的 floor 假设候选之间独立、在 0/1 指标上各评测一次。进化循环里的候选因血缘而相关，
这让真实下限更高，所以打印的数是一个下界。

错误归因需要知道哪个决策点本该给出什么答案。只有一个 choice 点时 `default_gold`
能自动处理；多于一个就要自己传 `gold(episode) -> {point: answer}`。它拒绝猜，
因为猜错归因会让之后每一次变异都瞄准错误的决策点。

上面的测量是**单个模型、单个八选项任务**。机制本身是通用的、在别处也有充分记载，
但幅度不是常数——所以这个检查以"一个你在自己 schema 上跑的函数"的形式发布，
而不是一个从这份 README 里抄走的数字。

## 常见问题

**我的 agent 已经能用了，装这个能得到什么？**
在一批已有样本上跑 `permutation_sensitivity`。代价是 `2 * k * n` 个决策，
它告诉你这个 schema 是在依据 criteria 还是在依据排版。在做这里其它任何事之前，
这一条就值一个下午。

**选项顺序偏置不是早就知道的问题吗？**
是，这正是要点。问题已知、修法已知，而它被跳过的原因只有一个：贵 k 倍。
这个库加的是：带对照地测量它、用一层 wrapper 修它、
并且把决策放到为决策而生的模型上，让这个代价付得起。

**为什么不直接搜一个好顺序？**
上面测了：从八个里挑最好的，held-out 拿到 +0.115，而它训练集增益的绝大部分是选择噪声。
边缘化拿到 +0.156，且不选择任何东西。

**必须有 Jev 或 System One 模型吗？**
不必，而这个名字仍然成立。这个库要求的是决策**被类型化**，搜索空间就是由它构成的。
jev 这类模型是提供类型化决策最便宜、延迟最低的方式，正是它让 `Marginalized`
和几千决策的循环变得付得起。`LocalBackend` 和 `RuleBackend` 同样提供类型化决策。

**和 DSPy 之类的 prompt 优化器有什么不同？**
那些搜索的是 prompt 文本和少样本示例，那只是上面那张表里的一行。
其余各行只有在"决策是一个带具名选项和可校准概率的声明对象"时才存在。
而且这里一次运行结束给的是判定，不是一个最高分。

**为什么我的运行是 UNDERPOWERED？**
`d` 个分歧上的精确符号检验，p 值下界是 2^(1-d)，所以 5 个及以下永远到不了 0.05。
是你的 held-out 切分太小、判不了，这和"负结果"不是一回事。

MIT。
