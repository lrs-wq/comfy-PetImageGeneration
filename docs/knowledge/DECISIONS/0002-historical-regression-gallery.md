# 0002：历史图集只阻断运行失败，不阻断画面差异

- 状态：accepted
- 日期：2026-09-01
- 负责人：项目负责人
- 影响节点：`comfy.project`、`comfy.module.pipeline`、`comfy.runtime.data`、`comfy.api.local-video`

## 背景

项目积累了大量真实任务，但没有固定的历史输入集合，也没有统一的修改后检查入口。已有测试主要验证局部函数；提示词、模型调用、队列、生成和产物发布仍可能在真实输入上失败。

本项目当前的回归目标是稳定完成，不要求图像达到固定审美或质量阈值。因此用历史画面结论作为硬门禁会把正常的生成差异误报为失败，也不符合负责人确认的验收标准。

完整 Qwen + SCAIL 生成受单卡 GPU 和在线队列约束，不适合塞入 pre-commit；但只重放历史文件又不能证明修改后的线上服务能完成新任务。

## 决策

- 从已完成历史任务中选择猫狗、单图/三视图、重复/真实多视图和不同动作的真实输入，按 SHA-256 固化为服务器侧版本化语料。
- Git 只保存来源任务 ID、相对资源路径、摘要与动作范围；媒体留在 `/data/comfy-pet-pipeline/archive/regression-gallery/`，不进入源码或知识图谱。
- 每次仓库修改后运行 `bash ops/scripts/verify-change.sh`，统一执行 pytest、导入检查、语料完整性和历史输入合同检查。
- 16 个既有质量边界案例保留为 `MATCH/DRIFT` 诊断。阈值结论变化或诊断异常写入报告，但不改变命令退出码。
- 影响运行链路的服务修改在安全重启后运行 live smoke。默认串行提交已由现行代码实测完成的一组重复三视图猫 `grooming` 和一组户外金毛犬 `standing`；其他历史输入保留在总览中，只有现行代码完成性稳定后才提升为必跑 live canary。
- live 图集只有三个硬断言：任务终态为 `completed`、每个请求动作有主动画、主动画可下载并按媒体格式读取。不读取身份、姿态、毛发、接触或其他视觉质量结论。
- live 运行在发现已有活动任务时默认拒绝提交，避免测试抢占单卡在线队列。
- 已存在且摘要不同的语料不得自动覆盖；更换输入时提升语料版本。

## 备选方案

- 把历史媒体提交到 Git：未采用，因为会扩大仓库并破坏运行数据和隐私边界。
- 让历史质量阈值阻断：未采用，因为负责人明确不要求画面质量，只要求任务不失败。
- 每次提交自动跑全部 GPU canary：未采用，因为单卡串行耗时长且会干扰在线任务。
- 只验证历史任务的旧状态：未采用，因为旧任务完成不能证明修改后的代码仍能完成；必须有服务重启后的新任务。
- 只跑单元测试：未采用，因为 mock 无法覆盖 API、队列、模型进程和输出发布的完整链路。

## 后果

- 所有修改都有快速、可由 pre-commit 调用的统一检查入口。
- 服务行为修改增加两例真实 GPU smoke 的运行成本；它们串行执行并在在线队列忙时停止。
- 图像变差不会让稳定性门禁失败，但仍可通过诊断图和 live 第一帧总览人工发现。
- 其他输入或动作的完成性需要显式选择对应 canary，不能由快速历史重放替代。

## 证据

- `tests/regression/historical_gallery.json`
- `src/comfy_pet_pipeline/regression/historical_gallery.py`
- `src/comfy_pet_pipeline/regression/live_stability_gallery.py`
- `ops/scripts/run-live-stability-gallery.py`
- `ops/scripts/verify-change.sh`
- `.githooks/pre-commit`
- `docs/operations/REGRESSION_GALLERY.md`
