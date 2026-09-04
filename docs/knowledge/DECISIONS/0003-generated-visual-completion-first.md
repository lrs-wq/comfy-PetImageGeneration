# 0003：生成视觉门耗尽后完成优先

- 状态：accepted（canonical 身份不安全终态回退由 0009 部分取代；动作完成优先仍有效）
- 日期：2026-09-01
- 负责人：项目负责人
- 取代：0001 中“身份视觉门耗尽后任务失败”的部分
- 影响节点：`comfy.module.pipeline`、`comfy.module.quality`、`comfy.api.local_video`
- 部分取代：`0009-canonical-identity-safe-anchor.md` 规定 canonical 不得在全部生成候选身份不安全时继续采用生成图

## 背景

项目验收目标是不让任务因画面质量结论失败，允许模型更自由地生成。任务 `20260901T055619Z-d6d84380a1` 已经得到三轮可读取的 canonical 图片，但严格绿幕、canonical 内部洞和身份色盘门依次把它们及上传身份抠图全部拒绝，最终在动作生成前失败。这类失败反映的是视觉偏差，不是模型或媒体无法运行。

历史稳定性图集已经采用“只阻断运行失败、不阻断画面差异”的合同。生产流水线若仍让身份、姿态或透明视觉门终止任务，会与该合同和负责人要求冲突。

## 决策

- canonical 和动作候选继续执行原有严格审核、定向重试与固定预算；严格合格候选始终优先。
- canonical 严格候选耗尽后，从已有 Qwen 生成图中选择最低风险者。严格语义抠图失败时，对同一生成图执行不阻断视觉质量的 RemBG 语义抠图，只要求得到可用主体 alpha 和可读取的归一化媒体。
- 每个动作仍完整生成五轮。没有严格身份/动作/毛面/透明合格候选时，从具备可用媒体文件的五轮生成候选中选择最低风险者，标记 `best_effort`，继续 SCAIL 与产物发布。
- 身份色盘、花纹、动作接触、姿态、构图、毛面、背景残留和 canonical 内部洞等结论保留为重试、排序和审计信号；固定预算耗尽后不再把任务置为 `failed`。
- 原始上传图只用于身份锚定、审计和回绘佐证，不能作为动作首帧、SCAIL 动作参考或最终降级输出。
- 运行完整性仍失败关闭：模型或进程无输出、源/候选媒体不可读取、无法取得可用主体 alpha、归一化失败、摘要变化、最终主产物缺失或不可读取都必须失败。不得伪造 `completed`。
- 所有完成优先选择都必须记录策略、轮次、SHA-256、严格失败原因、视觉问题和 `best_effort` 状态，供历史图集与任务详情审计。

## 备选方案

- 继续逐个修补视觉门直到每种输入都严格通过：未采用，因为开放式生成仍会产生新的视觉边界，无法保证完成。
- 失败后发布原始上传图：未采用，因为它不是动作生成结果，也会掩盖模型链路未完成。
- 完全删除质量门：未采用，因为这些信号仍能指导五轮重试、严格优先选择和诊断。
- 对运行完整性错误也强制写成完成：未采用，因为会发布损坏或不存在的媒体。

## 后果

- 任务画面可能存在身份、姿态、背景或透明边缘问题，但只要生成媒体完整，就会以可审计的 `best_effort` 结果完成。
- 固定五轮 GPU 成本不变；canonical 最多增加一次对同一生成图的宽松语义抠图，不增加 Qwen 候选轮次。
- 依赖严格质量字段的调用方必须读取 `best_effort` 与视觉审计，而不能把 `completed` 等同于画面严格合格。
- 0001 的固定五轮、生成结果选优和原图禁用条款仍有效；只有“无身份合格候选即失败”被取代。

## 证据

- `src/comfy_pet_pipeline/api/local_video.py`
- `src/comfy_pet_pipeline/api/local_video_parts/identity_quality.py`
- `src/comfy_pet_pipeline/api/local_video_parts/job_runner.py`
- `tests/integration/test_prompt_translation_profile.py`
- `tests/integration/test_local_video_qwen_prompt_workflows.py`
- `docs/api/LOCAL_VIDEO_API.md`
