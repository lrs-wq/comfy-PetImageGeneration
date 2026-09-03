# 0006：每次修改后执行猫狗双行为线上自测

- 状态：accepted
- 日期：2026-09-03
- 负责人：项目负责人
- 修订：覆盖 0002 中“仅运行链路修改后 smoke”以及猫狗各单行为的默认矩阵
- 影响节点：`comfy.project`、`comfy.verification.historical_stability_gallery`、`comfy.api.local_video`

## 背景

旧规则只在会影响运行链路的服务修改后执行 live smoke，默认猫仅运行 `grooming`、狗仅运行 `standing`。这无法保证每次修改都真实经过猫狗两条完整生成链，也遗漏了两个物种共有且风险较高的 `eating` 路径。项目负责人要求把修改后自测提升为固定猫狗双行为矩阵。

## 决策

- 每次仓库修改都必须先通过 `bash ops/scripts/verify-change.sh`，再在生产队列为空时运行 `.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke`。
- 默认 smoke 恰好串行提交两个新任务：猫运行 `eating + grooming`，狗运行 `eating + standing`；四个行为每个恰好执行一次。
- `select_cases` 对默认 suite 硬校验物种和动作矩阵。缺少猫或狗、缺少任一动作、动作重复、顺序漂移或新增额外默认案例都失败关闭。
- 默认矩阵第一组失败时仍继续尝试第二个物种，保证猫狗均留下真实执行结果；任一案例失败时总报告和退出码仍为失败。
- 显式 `--case` 只用于追加专项覆盖，不能替代默认四动作矩阵。队列繁忙时等待清空后重跑，不把 `--allow-busy` 当作日常规避手段。
- pre-commit 继续只执行快速门禁，因为真实 Qwen + SCAIL 受单卡队列和生成时长约束；任务完成前由执行者负责补跑 live smoke，不得在缺失或失败时宣称修改完成。
- live PASS 语义保持不变：新任务终态为 `completed`，每个请求动作发布可下载、格式可读的主动画。身份、姿态、毛发、接触等画质结论仍为非阻断诊断。

## 备选方案

- 只把四动作写入文档：未采用，因为清单可能退回单动作而不被发现。
- 把完整 GPU smoke 放入 pre-commit：未采用，因为单卡任务耗时长且在线队列可能繁忙；保留显式的修改后必跑门禁。
- 继续只在运行链路修改后执行：未采用，因为负责人要求每次修改后都留下猫狗双行为真实自测。

## 后果

- 每次修改增加两个串行生产 canary、共四个动作的 GPU 成本和等待时间。
- 猫狗 canonical、共同吃饭链路以及各自第二默认行为都会在每次修改后被真实执行。
- 默认矩阵配置漂移会被单元测试和 live 脚本自身提前拒绝；单个物种失败不会掩盖另一个物种的执行结果，完成性与画质评价的边界不变。

## 证据

- `tests/regression/historical_gallery.json`
- `src/comfy_pet_pipeline/regression/live_stability_gallery.py`
- `tests/unit/test_regression_galleries.py`
- `AGENTS.md`
- `docs/operations/REGRESSION_GALLERY.md`
