# 历史稳定性图集

## 验收目标

图集只回答一个问题：修改后的服务能否把历史真实输入完整处理完。画面好坏、毛色、姿态和接触阈值不属于阻断条件。

成功合同固定为：

1. 新任务到达 `completed`；
2. 每个请求动作都发布一个主动画产物；
3. 主产物可通过 API 下载，并能按 PNG/APNG、GIF 或 MP4 格式读取。

恢复、降级或质量诊断字段不影响结果。只要满足以上三条，稳定性 canary 就通过。

## 两级运行

### 每次仓库修改：快速图集

```bash
cd /root/comfy-pet-pipeline
bash ops/scripts/verify-change.sh
```

该命令依次运行默认 pytest、流水线导入检查和历史图集。历史图集会：

- 校验 32 个历史媒体文件的 SHA-256；
- 校验 6 组 canary 的视图数量与唯一视图数量；
- 生成输入总览图；
- 重放 16 个既有质量样例作为非阻断诊断。

质量样例显示 `MATCH` 或 `DRIFT`，都不会让命令失败。只有代码测试、导入、语料完整性或输入合同失败才会阻断。服务器仓库的 pre-commit hook 调用同一命令，未提交修改也必须显式运行一次。

### 服务重启后：真实完成性 canary

确认没有在线任务后运行：

```bash
.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke
```

默认 smoke 串行提交两组历史输入，避免与单卡 GPU 并发：

| canary | 输入覆盖 | 动作 |
| --- | --- | --- |
| `orange-tabby-duplicate-three-view` | 猫，三视图字段但三个文件相同 | `grooming` |
| `golden-retriever-outdoor` | 犬，户外复杂背景、两个不同视角 | `standing` |

脚本在提交前检查 API、worker、ComfyUI、模型和当前活动任务。发现已有 `queued/running` 任务会拒绝运行；仅在明确接受排队影响时使用 `--allow-busy`。

修改提示词、工作流、模型调用、队列、任务状态、输出发布或 GPU 生成链路时，服务安全重启后必须跑 smoke。修改动作专属路径时可显式运行对应历史案例；显式案例会使用清单中的完整动作集合：

```bash
.venv/bin/python ops/scripts/run-live-stability-gallery.py \
  --case white-cat-real-three-view
```

历史输入只有在现行代码下重复确认完成后才增加 `smoke_actions`，避免把会被现有视觉身份门随机拒绝的输入放进完成性必跑集合。

若 SSH 或终端在生成期间断开，任务仍由服务继续执行。使用原 job ID 恢复轮询和产物校验，不要重复提交：

```bash
.venv/bin/python ops/scripts/run-live-stability-gallery.py \
  --case golden-retriever-outdoor \
  --resume-job <existing-job-id>
```

## 历史输入集合

| canary | 物种与输入特征 | 完整动作集合 |
| --- | --- | --- |
| `orange-tabby-duplicate-three-view` | 橘虎斑，重复三视图 | `eating / grooming` |
| `tabby-kitten-colored-background` | 幼猫、彩色背景、重复三视图 | `eating / grooming` |
| `white-cat-real-three-view` | 浅色猫，三个不同视角 | `eating / grooming` |
| `labrador-six-actions` | 拉布拉多，三个不同视角 | 六动作 |
| `pomeranian-single-view` | 白色长毛犬，单图 | `eating` |
| `golden-retriever-outdoor` | 户外复杂背景，两组不同视角 | `eating / standing` |

集合来自已经完成过的历史任务。`source_job_id` 只用于来源追踪；每次 live 运行都会产生新的任务 ID，不复用历史输出作为通过结论。

## 数据与报告

Git 只保存 [`tests/regression/historical_gallery.json`](../../tests/regression/historical_gallery.json) 中的来源任务 ID、相对路径、SHA-256、案例标签和动作范围。原始上传图、生成图和动画只留在服务器数据根目录，不进入 Git 或知识图谱。

服务器首次使用时执行：

```bash
.venv/bin/python ops/scripts/run-historical-regression-gallery.py --materialize-only
```

脚本只把缺失文件复制到 `/data/comfy-pet-pipeline/archive/regression-gallery/v1/`。已存在但 SHA 不同的文件会失败关闭，绝不覆盖基线。

报告位置：

```text
/data/comfy-pet-pipeline/outputs/regression_gallery/<UTC>-<git-sha>/
  gate-gallery.png          # 非阻断质量诊断
  canary-inputs.png         # 历史输入总览
  index.html
  report.json

/data/comfy-pet-pipeline/outputs/regression_gallery/live/<UTC>-<git-sha>/
  live-gallery.png          # 新任务各动作第一帧，仅供查看
  thumbnails/
  index.html
  report.json               # 新任务 ID、终态和可读产物摘要
```

报告不保存 API key，也不以质量字段决定通过与否。

## 更新图集

1. 只选已完成且输入文件仍可确认来源的历史任务。
2. 在清单新增来源相对路径与 SHA-256；不记录用户身份、提示词或敏感字段。
3. 为新 canary 写清物种、视图、完整动作；只有适合每次 smoke 的小集合才增加 `smoke_actions`。
4. 需要替换媒体时提升 `corpus_root` 版本，不覆盖旧版本文件。
5. 运行 `--materialize-only`、`bash ops/scripts/verify-change.sh` 和 live dry-run；服务修改还要完成 live smoke。

非阻断质量样例可以保留，方便定位行为变化，但不需要为画面差异更新基线来“修绿”稳定性测试。
