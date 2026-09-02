# 0005：舔毛前肢拓扑优先且永不阻断完成

- 状态：accepted
- 日期：2026-09-02
- 负责人：项目负责人
- 补充：0003 的完成优先合同与 0004 的舔毛身份/几何职责分离
- 影响节点：`comfy.module.pipeline`、`comfy.module.quality`、`comfy.api.local_video`、`comfy.verification.historical_stability_gallery`

## 背景

生产任务 `20260901T102809Z-58d2285192` 已把横卧 canonical 改成直立舔爪，但选中首帧的抬起前肢与胸部融合，关节链路和腿数难以辨认。旧门只在固定前肢 ROI 统计前景与边缘；导图及五张候选的 ROI 均为 `13904/13904` 前景像素，而毛发和条纹又能满足边缘密度，所以“有前景/有边缘”不等于“前肢与躯干拓扑分离”。

## 决策

- 复用已绑定 SHA-256 的公开单色 Qwen 导图，从其绿色 C 形内部开通通道抽取“抬起前肢与躯干分离”语义。不向模型新增私有图像、不读取上传外观来构造该证据。
- 将导图通道边界与候选的低频、有方向边缘比较，记录边界覆盖、平均距离和平均方向一致性；用连续罚分排序，不依赖单一硬阈值决定任务成败。
- 五轮候选保持真实生成 `5/5`。第 4、5 轮切换为解剖优先自由重绘：几何导图是粗略动作意图，模型可调整头、肩、关节、比例、取景和次要花纹，以恰好四条与身体连接的腿和清晰肩—肘—腕—爪为优先。
- 选择时先排除大姿态风险，再比较前肢拓扑和解剖优先轮次，然后比较身份与其他视觉债务。身份相似不得压过明显的融肢风险。
- 前肢拓扑只作为 advisory：不改变最低门的 `acceptable`，不阻断终态。五轮都有视觉偏差时仍选择媒体完整的最低风险真实生成候选，继续 SCAIL 与发布。
- 不读取历史首帧缓存、不用缓存替代任何一轮、不追加第 6 轮。模型/进程无输出、媒体不可读、主体 alpha 不可用、摘要变化或主产物缺失仍是运行完整性失败。
- API 状态和 live 图集显示拓扑策略、选中证据、是否偏离和“永不阻断完成”标记，但图集 PASS 仍只根据任务完成和媒体可读。

## 备选方案

- 继续仅使用 ROI 前景与边缘密度：未采用，因为毛发/条纹会在融肢候选上产生假证据。
- 把拓扑阈值变成新失败门：未采用，因为视觉证据受毛发、条纹和模型随机性影响，且违反完成优先合同。
- 把 canonical 身份像素完全锁死：未采用，因为跨越坐姿和抬臂的大几何变化需要重建关节与遮挡。

## 后果

- 舔毛首帧在保持身份参考的同时，允许后两轮为正确解剖更自由地偏离像素级导图和 canonical 原始比例。
- 同一五轮候选集会先将腿部拓扑风险更低的结果送入 SCAIL，身份审核继续参与后续排序。
- `completed` 仍只代表运行与媒体完整；调用方和测试台可通过新诊断区分拓扑通过、偏离与 best-effort。

## 证据

- `src/comfy_pet_pipeline/quality/grooming_pose.py`
- `src/comfy_pet_pipeline/api/local_video_parts/job_runner.py`
- `src/comfy_pet_pipeline/api/local_video_parts/identity_quality.py`
- `src/comfy_pet_pipeline/api/local_video_parts/prompting_workflows.py`
- `src/comfy_pet_pipeline/regression/live_stability_gallery.py`
- `tests/unit/test_grooming_pose_gate_terminal.py`
- `tests/integration/test_local_video_qwen_prompt_workflows.py`
- `tests/integration/test_local_video_api_pose_guard.py`
