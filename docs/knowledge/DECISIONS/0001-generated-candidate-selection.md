# 0001：五轮生成候选与最优结果选择

- 状态：accepted
- 日期：2026-08-31
- 负责人：项目负责人
- 影响节点：`comfy.module.pipeline`、`comfy.module.identity`、`comfy.module.quality`

## 背景

首版知识图谱曾将失败策略记录为“终端降级回原始上传图”。这不符合当前正式策略：系统应在有限生成预算内产生多个候选，并在生成结果中取最优。

## 决策

- 每个动作固定完成 `5` 轮 Qwen 候选生成，不因前序候选合格而提前停止。
- 在通过身份底线的生成候选中，按身份保持、动作正确性和视觉质量选择综合最优结果。
- 原始上传图可用于建立身份、身份审计和必要的身份回绘佐证，但不能作为五轮生成失败后的最终降级输出。
- 未通过颜色、条纹等身份底线的候选，不因生成预算耗尽而重新放行。
- 语义抠图是五轮内候选的后处理，不计作第六个生成候选；下游恢复只能复用已经选定的生成候选。

## 备选方案

- 失败后直接返回原始上传图：未采用，因为这不是动作生成结果。
- 预算耗尽后选择身份审计失败的候选：未采用，因为会破坏宠物身份一致性。

## 后果

- 最坏情况需要承担五轮生成的 GPU 时间和等待时间。
- 候选记录必须保留轮次、身份审计、动作门禁、质量分数和最终选择原因。
- 异常恢复必须校验逐动作五轮选择记录与所选候选摘要，旧任务中的原图或 canonical 派生文件不能凭文件名重新获得输出资格。
- 文档和 Agent 不得再将“降级回原图”描述为当前策略。
- `v2.26.40` 已在 `v2.26.39` 生产基线上按本决策修改实现、恢复边界、健康接口、测试和现行文档；历史版本说明继续保留当时行为，仅用于追溯。

## 证据

- `src/comfy_pet_pipeline/api/local_video.py`
- `src/comfy_pet_pipeline/api/local_video_parts/prompting_workflows.py`
- `src/comfy_pet_pipeline/api/local_video_parts/job_runner.py`
- `src/comfy_pet_pipeline/api/local_video_parts/subject_replacement.py`
- `tests/integration/test_recoverable_output_guarantee.py`
- `docs/api/LOCAL_VIDEO_API.md`
