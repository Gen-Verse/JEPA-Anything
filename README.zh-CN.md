# JEPA Anything

[English](README.md)

JEPA Anything 是一个 Codex Skill：它能自动把自然语言描述的世界模型任务转换为**可验证的 JEPA Anything 世界状态接口**。Skill 是主体，其余目录为验证器、代码骨架、确定性支撑库和维护示例：

```text
jepa-anything-skill/   Skill 指令、验证器、配置契约和代码骨架模板
jepa-anything-core/    OPF、损失、基线和审计的确定性支撑库
recipes/               用于检验 Skill 的任务设计示例
checkpoints/           可选的模型产物元数据契约
```

核心原则是：

> LLM 提出设计，确定性程序负责验证。

因此，Skill 不是自动训练器，也不会擅自把潜变量命名成“疾病因子”“速度因子”等语义概念。未经干预或其他验证实验确认，它们只能叫作中性的“预测坐标”。

## 验证范围

每份设计都必须经过机器检查：

- context 与 target 是否来自同一底层系统、同一设备/轨迹实例；
- observation、target descriptor、外生输入中是否存在目标泄漏；
- OPF 维度是否满足 `K * r == d`，并分别明确因子坐标拼接与从分析坐标回到完整状态的合成方式；
- projected target 与在线 encoder 是否都有逐坐标活性检验计划；
- 是否配置标准 JEPA 与无约束多头两类基线，并明确总可训练参数量和 predictor FLOPs 的容量匹配计划；
- 是否明确选择终端读出、重复转移或因子分析之一；
- 每项结论是否由相应实验和指标覆盖。

通过检查只表示**设计契约内部一致**，并不证明因子具有语义、模型性能良好、存在因果识别，或科学结论成立。

## 环境要求

- Python 3.10 或更高版本；
- 用自然语言设计任务时需要 Codex；
- JSON 配置的校验与代码生成只使用 Python 标准库；
- 读取 YAML 配置时才需要 `PyYAML`；
- 只有在生成项目中使用可选的 core 支撑库时才需要 PyTorch。

## 能转换哪些世界模型任务

只要任务的核心是“根据当前或历史信息预测同一系统的未来状态”，就属于本 Skill 的主要适用范围：

| 原始任务 | 典型输入 | 转换后的主要模式 |
| --- | --- | --- |
| 机器人或控制世界模型 | 观测历史、状态、已执行动作、未来计划动作 | `repeated_transition` |
| 视频未来预测 | 历史帧、时间位置、可选控制或相机运动 | 多步预测用 `repeated_transition`；单次表征消费用 `terminal_readout` |
| 工业或科学动力学 | 传感器轨迹、控制量、边界条件 | `repeated_transition` |
| 多智能体或图动力学 | 节点/边状态、交互、动作、时间 | 通常为 `repeated_transition` |
| 学习预测状态供下游任务使用 | 历史观测、一个未来目标、分类/回归/检索用途 | `terminal_readout` |
| 预测坐标分析 | 轨迹、探针、交换/消融/干预计划 | `factor_analysis` |
| 已有世界模型代码库改造 | 数据集、模型、rollout、评价和切分代码 | 保留原任务后生成 JEPA Anything 接口与骨架 |

普通静态分类或回归如果没有世界状态、未来预测或动力学意图，不应自动触发本 Skill。

## “自动调用”的准确含义

安装后，`agents/openai.yaml` 允许 Codex 根据请求语义隐式选择本 Skill。因此用户通常可以直接描述世界模型任务，不必写出 Skill 名称。

“自动”不表示后台守护进程，也不表示所有含糊请求都能保证命中。它表示：

- Codex 识别到世界模型、潜在动力学、动作条件预测、rollout、规划或控制意图时，可以自动加载本 Skill；
- Skill 加载后会主动完成 JEPA Anything 设计，不要求用户先填写 JEPA 字段；
- 请求同时可能匹配多个工作流或措辞过于模糊时，可用 `$jepa-anything-skill` 强制指定；
- 命令行验证器只接收已经生成的 JSON/YAML 配置，不负责把任意自然语言直接变成配置。

## 安装到 Codex

### 方式一：开发时使用符号链接

在仓库根目录执行：

```bash
SKILLS_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILLS_DIR"
ln -s "$(pwd)/jepa-anything-skill" \
  "$SKILLS_DIR/jepa-anything-skill"
```

这种方式会让仓库中的修改立即反映到 Skill。若目标路径已经存在，`ln` 会安全失败；请先检查现有安装，不要直接覆盖。

### 方式二：复制完整 Skill

将整个 `jepa-anything-skill/` 目录复制到：

```text
${CODEX_HOME:-$HOME/.codex}/skills/jepa-anything-skill/
```

不要只复制 `SKILL.md`。`agents/`、`scripts/`、`references/` 和 `assets/` 都是运行所需内容。安装完成后新建 Codex 任务或刷新 Skill 发现结果。

### 验证是否可用

先用显式调用做一次最小检查：

```text
使用 $jepa-anything-skill，把“根据过去 4 个机器人状态和动作预测未来 8 步状态”
转换为 JEPA Anything 设计。只说明转换方案，不训练。
```

如果 Codex 加载了 Skill，回复中应出现：原始任务到 context/target 的映射、一个使用模式、adapter/encoder 与 `d/K/r` 建议、验证计划，以及不训练的边界。

## 用户最少需要提供什么

用户无需提交完整配置，但任务中最好能确定下面四类事实：

| 必要事实 | 需要说明的内容 | 示例 |
| --- | --- | --- |
| 底层系统与实例 | 哪些 context 和 target 属于同一个具体对象或轨迹 | `episode_id`、`machine_id`、`scene_id` |
| 预测时可见信息 | 历史观测、当前状态，以及当时真正已知的动作或外部输入 | 过去 8 帧、当前关节状态、已选定控制量 |
| 未来目标与时距 | 预测什么、从何时开始、预测多少步 | 未来 16 步状态或下一段视频 |
| 状态如何使用 | 单次下游读出、多步转移，还是坐标分析 | MPC rollout、分类 probe、坐标消融 |

数据切分、指标、计算预算和输出位置也建议提供，但如果已有代码库中能找到，用户不必重复说明。

## Skill 会自动决定什么

以下内容属于 JEPA Anything 设计选择，不应反过来要求用户填写：

| 自动提出的内容 | 选择依据 |
| --- | --- |
| observation/token adapter | 数据模态、采样方式、缺失值、mask 和时间结构 |
| context–target 定义 | 原任务的预测边界和预测时可用信息 |
| target descriptor | 时距、目标位置、查询信息及其可用时间 |
| 外生输入 | 动作、控制计划、边界条件是否在预测时已知 |
| encoder 家族 | 向量、序列、图、图像/视频、音频或多模态结构 |
| `d/K/r` | 原模型潜变量宽度、任务规模和计算预算，同时满足 `K*r=d` |
| 下游模式 | 原任务是单次读出、多步状态转移还是坐标分析 |
| 四类损失 | 因子预测、投影几何、target 坐标活性、在线 encoder 活性 |
| 两类基线 | 标准 JEPA 与无约束多头，并规划参数量、FLOPs 和训练步数匹配 |
| 审计与评价 | 泄漏、几何、活性、容量以及结论—实验覆盖 |

这些自动选择必须写入配置并标记为可审查假设，而不是伪装成已证明的最优设置。

## 三种常用输入方式

### 1. 只有自然语言任务

```text
我有机器人轨迹数据，包含相机帧、关节状态、已执行动作和 episode_id。
使用过去 8 个观测以及每一步选定的动作，预测并滚动未来 16 步状态，
用于模型预测控制；不同 episode 不能跨数据切分。请转换成 JEPA Anything，
自动选择其余设计，生成配置和代码骨架，不要训练。
```

### 2. 转换已有世界模型代码库

```text
检查当前代码库的数据读取、模型输入输出、rollout、评价指标和数据切分。
保持原来的世界模型任务不变，把任务接口转换成 JEPA Anything。
在新的输出目录中生成设计配置、验证报告和代码骨架；不要覆盖现有实现，
不要启动训练。列出原实现到新接口的逐项对应关系和所有不确定假设。
```

Skill 应先从代码中恢复事实，再决定是否需要提问。它不应要求用户手工转述代码里已经明确的信息。

### 3. 强制显式调用

```text
使用 $jepa-anything-skill，把下面的世界模型任务转换为经过验证的
JEPA Anything 设计和不含训练循环的代码骨架：……
```

## 自动转换流程

```text
自然语言任务或已有代码库
          |
          v
恢复原始系统、实例、输入、动作、目标、时距、用途和指标
          |
          +---- 缺失因果关键事实 ----> 只询问必要问题
          |
          v
提出 adapter、encoder、d/K/r、损失、基线和审计
          |
          v
生成 schema 2.0 配置 ----> 确定性验证 ----> 修正失败项
                                            |
                                            v
                               生成代码骨架并运行契约测试
                                            |
                                            v
                         输出转换映射、文件路径和未解决假设
```

Skill 必须保留原始任务目标：不得静默更换数据集、预测目标、时距、动作可用性、切分规则或评价指标。JEPA Anything 改变的是世界状态接口和验证契约。

## 什么时候会提问

只有缺失信息可能改变任务或造成目标泄漏时，问题才是阻塞性的：

- 无法确定 context 和 target 是否属于同一设备、轨迹、场景或 episode；
- 不清楚某个未来动作、标签、统计量或外部输入在预测开始时是否已知；
- 不清楚究竟预测哪个未来目标或预测多远；
- 无法判断状态用于单次读出、多步 rollout 还是坐标分析。

adapter、encoder、OPF、`d/K/r`、损失权重初值、基线结构和审计项目不是需要用户回答的阻塞问题。Skill 应提出可修改的默认方案，并把假设记录在 `task.metadata.compiler_assumptions` 中。

## 输出文件与含义

成功转换后，典型目录为：

```text
generated/
  config/design.json         已验证的 schema 2.0 任务设计
  validation-report.json     机器检查结果；成功时 errors 为 0
  scaffold-manifest.json     生成文件及其 SHA-256 清单
  pyproject.toml             生成项目的最小 Python 包配置
  src/jepa_task/
    adapter.py               observation/token adapter 接口
    model.py                 encoder、predictor、OPF 和状态合成接口
    activity.py              projected target 与在线 encoder 活性接口
    capacity.py              参数量、predictor FLOPs 与步数审计接口
    usage.py                 终端读出、重复转移或因子分析接口
    evaluation.py            指标、实验、基线和结论覆盖接口
    contracts.py             生成配置的运行时不变量
  tests/test_contract.py      可立即运行的生成契约测试
```

除文件外，Skill 的回复还应包含：

1. 原始世界模型任务到 JEPA Anything 的逐项转换表；
2. 从代码或用户描述中确认的事实；
3. 自动提出的 adapter、encoder、`d/K/r` 和使用模式；
4. 验证器结果及生成测试结果；
5. 仍需数据负责人确认的假设；
6. 明确说明尚未训练、没有实验结果、没有给坐标赋予语义。

## 如何判断转换已经完成

同时满足以下条件才算完成设计阶段：

- `validation-report.json` 中 `valid` 为 `true` 且 `summary.errors` 为 `0`；
- context 和 target 共享底层系统与实例标识，且不存在时间或字段泄漏；
- `K*r=d`，分析坐标拼接和完整状态合成均有明确接口；
- 标准 JEPA 与无约束多头基线均已配置容量匹配计划；
- 每项计划结论都有对应实验、指标、基线和审计；
- 生成目录中的契约测试全部通过；
- 未解决假设被明确列出，没有被伪装成事实。

这只表示 JEPA Anything **任务设计和接口转换完成**，不表示模型已经训练或性能已经得到验证。

## 不经过 Codex，直接使用命令行

仓库自带的 recipe 可以用于熟悉完整流程。

### 校验已有设计

```bash
mkdir -p work/quickstart

python3 jepa-anything-skill/scripts/validate_design.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --pretty \
  --output work/quickstart/validation-report.json
```

退出码为 `0` 表示全部检查通过，`1` 表示设计被确定性规则拒绝，`2` 表示输入或命令使用错误。报告中的 `valid: true` 只说明配置契约一致，不代表模型已经得到训练或结论已获实验证实。

### 从已验证设计生成代码骨架

```bash
python3 jepa-anything-skill/scripts/generate_scaffold.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --output-dir work/quickstart/generated-task \
  --config-format json
```

目标目录必须不存在或为空，防止意外覆盖已有工作。生成后运行：

```bash
python3 -m unittest discover \
  -s work/quickstart/generated-task/tests \
  -p 'test_*.py' \
  -v
```

这些测试验证生成文件与设计契约是否一致，不会训练模型。

## 选择唯一的使用模式

每个设计必须只选一个主模式：

| 模式 | 适用场景 | 设计重点 |
| --- | --- | --- |
| `terminal_readout` | 编码一次状态后完成分类、回归或检索 | 明确冻结/微调协议、readout 输入和终端指标 |
| `repeated_transition` | 反复应用转移模型做多步预测或控制 | 明确外生输入、rollout 计划和逐时距误差 |
| `factor_analysis` | 分析预测坐标的活性、正交性或可分性 | 使用中性坐标名称，并配置干预、交换或消融验证 |

如果任务同时包含多个用途，应先选择主要声明对应的模式；其他用途作为后续独立评估，不要在一个模糊模式中混合。

## 配置契约与必须满足的不变量

- 原始世界模型到 JEPA Anything 的转换规则：`jepa-anything-skill/references/world-model-conversion.md`；
- 人可读字段说明：`jepa-anything-skill/references/config-contract.md`；
- 机器规则实现：`jepa-anything-skill/scripts/validate_design.py`；
- 可运行示例：`recipes/synthetic-linear-dynamics/design.expected.json`；
- 完整工作协议：`jepa-anything-skill/SKILL.md`。

最容易出错的约束包括：

- context 与 target 必须能追溯到同一底层系统实例；
- context、descriptor 和预测时输入不得直接或间接包含 target；
- OPF 配置必须满足 `K * r == d`；
- `coordinate_layout` 只描述预测坐标拼接；`synthesis.kind` 必须另外定义如何恢复完整状态；
- projected target 和在线 encoder 都必须配置逐坐标活性检查；
- 标准 JEPA 与无约束多头基线都必须存在，并给出参数量和 predictor FLOPs 的容量匹配计划；
- 声明的每项结论必须映射到实际实验与指标，不能把“计划检查”写成“已有证据”。

## 可选：使用 core 支撑库

只校验设计和生成骨架不需要安装 core。需要复用 OPF、损失、基线或几何审计张量实现时，在仓库根目录运行：

```bash
python3 -m pip install -e './jepa-anything-core[dev]'
```

最小接口示例：

```python
import torch
from jepa_anything_core import OrthogonalFactorProjection

opf = OrthogonalFactorProjection(
    state_dim=12,
    num_factors=3,
    factor_dim=4,
    learnable=True,
)
state = torch.randn(128, 12)
coordinates = opf(state)                 # [128, 3, 4]
restored = opf.compose(coordinates)       # [128, 12]
```

完整张量接口和前置条件见 `jepa-anything-core/README.md`。

## 维护者检查

修改 Skill、验证器、模板或 recipe 后，在仓库根目录运行：

```bash
make check
```

该命令会运行 core 测试、Skill 测试、示例设计校验、代码骨架契约测试和 checkpoint manifest 检查。`make help` 可查看单独的检查命令。

## 常见问题

- **提示找不到 Skill**：确认 `jepa-anything-skill/SKILL.md` 位于 Codex Skill 目录中的一级子目录下，然后新建任务或刷新发现结果。
- **描述了世界模型但没有自动调用**：隐式选择取决于请求语义。补充未来预测、动作条件、rollout 或世界状态用途，或者直接写 `$jepa-anything-skill`。
- **Skill 一开始问了很多 JEPA 参数**：这不符合预期。用户只需要补充因果边界所必需的事实；adapter、encoder、`d/K/r`、损失和基线应由 Skill 自动提出。
- **转换已有代码库时重复询问代码里已有的信息**：明确要求先检查数据读取、模型输入输出、rollout、评价和切分代码，并只询问无法恢复的关键事实。
- **`No module named yaml`**：JSON 无需额外依赖；如需读取 YAML，请安装 `PyYAML`。
- **设计校验失败**：读取报告中 `status: "fail"` 检查的 `code`、`path` 和 `message`，修改设计后重新校验；不要绕过失败项直接生成。
- **输出目录错误**：换用一个不存在或为空的目录；生成器有意拒绝覆盖非空目录。
- **生成后测试通过但没有模型结果**：这是预期行为。骨架测试只验证任务接口，训练和实验执行需要用户另行明确启动。

## 提案与证据的边界

自然语言先由 LLM 转成可审查的 JSON/YAML 提案；确定性验证器拒绝违反不变量的设计；只有通过的设计才能生成代码骨架。训练、下载 checkpoint 与发表结论都属于后续显式动作，不在 Skill 的自动行为范围内。

`jepa-anything-skill/` 是主要入口。core 提供生成实现可复用的确定性机制，recipe 是维护中的端到端任务示例，checkpoint manifest 只定义未来模型产物的元数据格式。配置中的预期检查仍是计划，只有后续执行流程产生的测量记录才能作为证据。

## 许可证

Apache-2.0，详见 `LICENSE`。
