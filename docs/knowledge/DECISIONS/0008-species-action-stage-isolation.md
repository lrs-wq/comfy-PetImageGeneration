# 0008：猫狗与双动作按阶段隔离执行

- 状态：accepted
- 日期：2026-09-04
- 负责人：项目负责人
- 影响节点：`comfy.module.pipeline`、`comfy.execution.species_action_stage_isolation`、`comfy.action.cat_default`、`comfy.action.dog_default`

## 背景

默认任务虽然分别声明猫 `eating + grooming`、狗 `eating + standing`，但 Qwen 首帧与 SCAIL 动画仍在共享大循环中顺序执行。前序动作抛出普通异常时会直接退出循环，后序动作根本没有机会运行；犬链路还曾通过把 `standing` 排到 `eating` 前来规避这一问题，形成明显的顺序耦合。终态恢复也曾复用循环最后一次读取的计时对象，使多个已存在动作可能继承同一份时间参数。

## 决策

- 每个默认动作建立不可变的 `ActionExecutionScope`，隔离键固定为 `species:action`，并分别保存请求索引与执行索引。猫狗使用互不相交的键；动作执行不得依赖交换列表顺序。
- Qwen 动作首帧、上传动作导图配置和 SCAIL 动画阶段必须从共享任务配置深拷贝当前动作的有效配置，只向该阶段暴露当前动作切片。
- Qwen 与 SCAIL 的单动作普通异常必须绑定到 `species:action:stage` 记录；完成必要的 ComfyUI 清理后继续尝试兄弟动作。所有动作都经过屏障后再聚合抛出，交给既有逐动作恢复保留成功产物并只恢复缺失动作。
- 用户取消属于全局终止信号，必须立即传播，不得被动作隔离捕获。模型运行完整性、媒体摘要和禁止原图输出等既有失败关闭边界不变。
- 每个既有动作的输出、质量指标与时间参数必须作为同一动作的元组保存和恢复，禁止使用循环外残留变量。

## 备选方案

- 继续把高风险动作排到列表末尾：未采用，因为只能隐藏当前已知失败，新增动作或另一物种出现异常时仍会跳过兄弟动作。
- 捕获异常后把整个任务直接标记为成功：未采用，因为会掩盖运行完整性失败；隔离只保证兄弟动作得到执行机会，最终状态仍由聚合失败与逐动作恢复决定。
- 为四个组合复制四套执行函数：未采用，因为会造成同类修复在猫狗或不同动作间漂移；共享执行框架保留，但状态和异常边界按作用域隔离。

## 后果

- 调整猫或狗的一个姿态时，配置快照、失败、计时和恢复结果不会污染同任务的另一个姿态，也不会与另一物种共用隔离键。
- 单动作失败不再让兄弟动作静默缺席；代价是失败任务可能继续占用 GPU 完成剩余动作，然后才给出聚合结果。
- 新增动作阶段时必须接入同一作用域与阶段屏障，并补充“前一动作失败、后一动作仍执行”的回归测试。

## 证据

- `src/comfy_pet_pipeline/api/local_video_parts/action_execution.py`
- `src/comfy_pet_pipeline/api/local_video_parts/job_runner.py`
- `src/comfy_pet_pipeline/api/local_video_parts/subject_replacement.py`
- `src/comfy_pet_pipeline/api/local_video_parts/uploaded_references.py`
- `tests/unit/test_action_execution_isolation.py`
- `tests/integration/test_local_video_api_subject_replacement.py`
- `tests/integration/test_recoverable_output_guarantee.py`
