# 0004：舔毛身份与几何职责分离并保持完成优先

- 状态：accepted
- 日期：2026-09-01
- 负责人：项目负责人
- 补充：0003 的生成视觉门完成优先合同
- 影响节点：`comfy.module.pipeline`、`comfy.module.quality`、`comfy.api.local_video`

## 背景

任务 `20260901T083322Z-72165d7e84` 已按完成优先策略发布，但舔毛首帧仍保留横卧 canonical 的整体姿态，随后 SCAIL 输出又接近灰色 driver 外观。旧 advisory 工作流只把 canonical 作为 Qwen Image 1，以文字描述要求舔毛；这不足以跨越横卧到直立坐姿的大几何差异。旧姿态门先分别裁剪候选与导图再拉伸至同一大小，局部抬爪指标无法识别全身宽高比错误。SCAIL 后也没有把输出外观同时与参考身份和 driver 比较。

## 决策

- 猫舔毛 Qwen 首帧使用职责分离的两个输入：Image 1 仅承载 canonical 身份、毛色、花纹、比例与渲染风格；Image 2 仅承载去外观单色动作几何。
- Qwen 对全画布自由重绘，不要求保留 Image 1 的当前姿态、方向或裁切；不得复制 Image 2 的颜色、明暗、材质、纹理、脸或身份。
- 首帧姿态门先保持候选与导图各自宽高比进行全身轮廓注册，显式审核全身宽高比与轮廓 IoU，再运行既有抬爪、嘴部接触、肢体连续性和外观门。
- 全身与局部视觉问题继续指导五轮 Qwen 重试和最低风险候选排序；视觉预算耗尽后仍选可读取生成候选继续，不把视觉偏差升级为任务失败。
- 猫舔毛 SCAIL 输出抽取首、中、末三帧，在正式身份遮罩内对比 Qwen 参考身份、输出和灰模 driver 的色度、暗色花纹与纹理。疑似 driver 外观接管时最多生成三轮，并依次降低姿态强度。
- 三轮 SCAIL 视觉审核均拒绝或审核本身不可用时，选择身份罚分最低的真实生成视频、标记 `best_effort` 并继续发布。模型无输出、媒体不可读、摘要变化或最终产物缺失仍按运行完整性失败关闭。

## 备选方案

- 继续只用 canonical 与文字描述动作：未采用，因为历史异常已经证明大姿态转换不足。
- 把完整灰模 RGB 当作外观参考：未采用，因为灰度、材质与脸部会污染身份；只允许单色几何角色。
- SCAIL 身份门首次拒绝即终止：未采用，因为这与完成优先验收合同冲突。
- 完全不审核 SCAIL 身份：未采用，因为审核仍能驱动三轮选优并暴露灰模接管。

## 后果

- Qwen 获得更明确的大姿态几何，同时保留单一身份权威；模型可以更自由地完成横卧到直立舔毛的全身变换。
- 单个猫舔毛动作的 SCAIL 最多从一轮增加到三轮；通过后可提前结束，预算耗尽仍必须发布最低风险的完整生成结果。
- `completed` 继续只代表运行与媒体完整，不等同于所有身份和姿态视觉门严格通过；调用方应读取重试、问题和 `best_effort` 字段。

## 证据

- `src/comfy_pet_pipeline/api/local_video_parts/job_runner.py`
- `src/comfy_pet_pipeline/api/local_video_parts/prompting_workflows.py`
- `src/comfy_pet_pipeline/api/local_video_parts/subject_replacement.py`
- `src/comfy_pet_pipeline/quality/grooming_pose.py`
- `tests/integration/test_local_video_api_subject_replacement.py`
- `tests/integration/test_local_video_qwen_prompt_workflows.py`
- `tests/unit/test_grooming_pose_gate_terminal.py`
