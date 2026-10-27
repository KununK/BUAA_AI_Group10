# 视频情感识别与不确定性后训练

**BUAA-2026AI-课程大作业**。基于 Qwen2.5-Omni-7B，完成视频情感识别、模态冲突判断和澄清问题生成，探索推理优化、监督微调（SFT）、强化学习（GRPO）、蒸馏及多模态专项训练。

最终的单模型方案的 Kaggle Public 得分为 **0.93876**，额外的多模型方案为 **0.94958**。

- [课程比赛](https://www.kaggle.com/competitions/video-emotion-uncertainty-2026/overview)
- [基座模型：Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [SFT 框架：LLaMAFactory](https://github.com/hiyouga/LLaMAFactory)
- GRPO 使用项目实现的训练循环，复用 Hugging Face TRL 的配置与工具。

## 任务说明

模型根据视频画面、台词或音频判断人物情感，在模态冲突或依据不足时表达不确定，并生成中立的澄清问题。

| 输出字段 | 含义 | 取值 |
| --- | --- | --- |
| `pred_emotion` | 整体情感 | `positive`、`negative`、`neutral` |
| `is_uncertain` | 是否需要进一步确认 | `yes`、`no` |
| `clarifying_question` | 澄清问题 | 问题文本；无需提问时为 `none` |

数据包含两种模态协议：**VT** 比较画面与文本，**VA** 比较画面与音频。标注中的 `M1` 为视觉情绪，`M2` 为另一模态情绪，`M12` 为整体情绪。

训练集、验证集、测试集分别包含 **3354、666、734** 条样本。根据训练数据的标签分布，后期实验的整体情感采用正负二分类，辅助模态判断保留中性类别。

比赛加权总分为：

```text
总分 = 0.5 × 情感识别 Macro-F1
     + 0.3 × 冲突识别准确率
     + 0.2 × 一致样本校准率
```

冲突识别准确率统计冲突样本中输出 `yes` 的比例；一致样本校准率统计一致样本中输出 `no` 的比例。澄清问题用于提交和交互展示。

## 实验流程

| 阶段 | 内容 |
| --- | --- |
| 原始模型基线 | 使用原始 Omni 模型直接进行视频推理 |
| 基线推理优化 | 比较帧率、模态输入、提示词和冲突判断规则 |
| 第一轮 SFT | 学习整体情绪和联合模态情绪判断 |
| 第一轮 GRPO | 利用情感标签与模态冲突奖励进行后训练 |
| 第二轮 SFT | 分别训练整体、视觉、文本和音频判断任务 |
| 第二轮 GRPO | 在独立模态任务基础上进行奖励优化 |
| 蒸馏数据准备 | 对训练集中的 2000 条样本生成并筛选标注、简短证据和问句 |
| 对照 SFT | 使用官方分类标签、直接冲突监督、问句及匹配样本回放 |
| 蒸馏 SFT | 加入简短证据与联合判断监督 |
| 视觉增强训练 | 在蒸馏模型基础上训练视觉 LoRA，提升视觉情绪感知 |
| 音频冲突专项训练 | 在视觉增强模型基础上强化画面与音频的冲突判断 |

对照 SFT 与蒸馏 SFT 均从第二轮 SFT 模型出发。两组使用相同的问句与样本安排，匹配样本分别采用官方直接判断目标和带简短证据的联合目标。

**单模型方案**使用视觉增强模型完成各项判断，通过多次调用和外部决策规则生成结果。**额外多模型方案**由视觉增强模型负责整体情感与画面—文本判断，音频冲突专项模型负责画面—音频判断。

## 实验结果

| 方案 | 验证集加权分数 | Kaggle Public |
| --- | ---: | ---: |
| 原始模型基线 | — | 0.56789 |
| 基线推理优化 | — | 0.77091 |
| 第一轮 SFT | — | 0.87520 |
| 第二轮 SFT | 0.89602 | — |
| 第二轮 GRPO | 0.89636 | 0.90261 |
| 对照 SFT + 决策优化 | 0.93037 | 0.93772 |
| 蒸馏 SFT + 决策优化 | 0.93313 | 0.93786 |
| 单模型方案 | 0.93635 | 0.93876 |
| 额外多模型方案 | 0.93837 | **0.94958** |

验证集用于模型选择与决策门限调整。决策优化采用按视频分组的五折交叉验证，测试阶段使用验证集确定的门限。Kaggle Public 分数来自实际提交反馈。

## 目录结构

```text
.
├── README.md
├── requirements.txt       # 主要依赖
├── configs/sft/           # 训练配置
├── scripts/
│   ├── common/            # 模型加载、提示词、媒体处理与公共工具
│   ├── data/              # 数据准备、音轨提取与预处理
│   ├── train/             # SFT、GRPO 与专项训练
│   └── eval/              # 推理、指标计算与提交文件导出
├── Dataset/               # 本地数据
├── outputs/               # 训练权重、日志和推理结果
└── LlamaFactory/          # 本地框架源码
```

## 环境准备

实验使用 Python 3.12、PyTorch 2.8.0、Transformers 4.57.1、PEFT 0.18.1、TRL 0.24.0，训练显卡为 A100 80GB。以下命令均在项目根目录执行。

```bash
uv venv --python 3.12
source .venv/bin/activate

git clone https://github.com/hiyouga/LLaMAFactory.git LlamaFactory
git -C LlamaFactory checkout c683e59edd8f893abc5e1403184a1d0644817d2a
uv pip install -r requirements.txt -e ./LlamaFactory
```

下载基座模型并准备课程数据，将模型路径与本机数据路径填入相应配置。公共推理模块的模型位置在 `scripts/common/baseline.py` 中配置。

```text
Dataset/
├── train.jsonl
├── val.jsonl
├── test_release.jsonl
├── sample_submission.csv
└── videos/
```

## 基本使用

### 基线推理

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.common.baseline \
  --split val --input-mode protocol --label-set ternary \
  --fps 2 --run-name baseline
```

结果保存到 `outputs/baseline/val/` 下的时间戳目录，包括模型输出、预测 CSV 和指标。使用 `--split test` 可生成测试集预测。

### SFT 与蒸馏

准备蒸馏材料后，构建训练数据、预处理缓存，再启动 LLaMAFactory 训练：

```bash
python -m scripts.data.prepare_distillation --stage prepare

python -m scripts.data.prepare_distillation --stage tokenize \
  --config configs/sft/distill_v3_teacher.yaml

CUDA_VISIBLE_DEVICES=1 python -m scripts.train.distill_sft \
  --config configs/sft/distill_v3_teacher.yaml --max-hours 7
```

对照训练使用 `configs/sft/distill_v3_control.yaml`，并通过 `--stop-after 800` 设置匹配的训练步数。训练输出包括 LoRA 权重、检查点和训练日志。蒸馏所需的本地材料路径见 `scripts/data/prepare_distillation.py`。

### GRPO 与评估

GRPO 入口为 `scripts/train/grpo.py`，奖励函数定义在 `scripts/train/rewards.py`。独立模态评估、直接冲突评估与决策组合分别由以下模块提供：

```bash
python -m scripts.train.grpo --help
python -m scripts.eval.evaluate --help
python -m scripts.eval.evaluate_distillation --help
python -m scripts.eval.fuse --help
```

推理结果保存原始生成内容、预测标签、澄清问题和评估指标。测试提交文件采用以下列顺序，覆盖全部 734 条测试样本：

```csv
sample_id,pred_emotion,is_uncertain,clarifying_question
```

## 数据说明

课程数据仅限课程使用。仓库提供代码、配置与依赖说明；原始数据、蒸馏材料、模型权重、缓存和运行结果保存在本地，由 `.gitignore` 排除。
