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

### 结构化库存

当前 v1 语料共 `6` 个 canary、`16` 张输入视图和 `15` 张历史姿态关键帧。输入文件按 SHA-256 固化；姿态关键帧只用于查看每个历史动作对应的生成姿态，不是质量基线，也不参与完成性硬门禁。

| canary | 来源任务 | 输入视图 | 历史姿态关键帧 |
| --- | --- | ---: | --- |
| `orange-tabby-duplicate-three-view` | `20260831T094915Z-5c93eda6e9` | 3 | `pose_01=eating`、`pose_02=grooming` |
| `tabby-kitten-colored-background` | `20260831T015134Z-376a202e84` | 3 | `pose_01=eating`、`pose_02=grooming` |
| `white-cat-real-three-view` | `20260828T112414Z-38b93945ef` | 3 | `pose_01=eating`、`pose_02=grooming` |
| `labrador-six-actions` | `20260827T100639Z-1d8b9ff966` | 3 | `pose_01=eating`、`pose_02=drinking`、`pose_03=running`、`pose_04=sleeping`、`pose_05=toileting`、`pose_06=standing` |
| `pomeranian-single-view` | `20260821T075838Z-0e6161264e` | 1 | `pose_01=eating` |
| `golden-retriever-outdoor` | `20260831T015110Z-d8182c974d` | 3 | `pose_01=eating`、`pose_02=standing` |

输入语料位于 `/data/comfy-pet-pipeline/archive/regression-gallery/v1/canaries/`；姿态关键帧仍保留在对应来源任务的 `/data/comfy-pet-pipeline/outputs/local_video_api/<source_job_id>/` 中。Git 和知识图谱只记录上述非敏感资源身份、位置、数量和关系，不复制原图、关键帧、动画或运行日志。

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

live 报告同时显示每个动作的非阻断诊断：首帧选中轮次、最低门状态、舔毛前肢拓扑状态/策略/选中证据、拓扑是否阻断完成，以及 SCAIL 舔毛身份审核状态和选中轮次。这些字段仅用于定位选择原因；图集的 PASS 仍只取决于 `completed` 和主媒体可读。

## 已核验完成性快照

2026-09-01 在提交 `a0cd7ce` 上运行默认 smoke，两组 canary 均完成（`2/2`）：重复三视图橘虎斑猫的 `grooming` 与户外金毛犬的 `standing` 都发布了可读取的 `512×512`、`100` 帧主 APNG。该快照只证明当次完成性合同通过，不建立任何画质结论；新的服务修改仍必须重新运行 smoke。

同日提交 `89ddcb9` 部署后的最新默认 smoke 再次 `2/2`：猫任务 `20260901T081100Z-b0f4ee4801`、狗任务 `20260901T081913Z-33cde52c21` 均完成，报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T081055Z-89ddcb9392/`。针对完成优先策略的原失败输入复测 `20260901T083322Z-72165d7e84` 也为 `1/1`：复用源任务 `20260901T055619Z-d6d84380a1` 的三张输入和两个动作，canonical 使用完成优先回退，两个动作各生成固定五轮并以 best-effort 继续，最终两个 `512×512`、`100` 帧主 APNG 可读取；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T083311Z-89ddcb9392/`。

提交 `7e6bbfc` 部署后的 v2.26.52 默认 smoke 为 `2/2`：猫舔毛任务 `20260901T095724Z-22d0da39ca` 与犬站立任务 `20260901T100834Z-1b066e1d40` 都 `completed`，各自主 APNG 可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T095713Z-7e6bbfcb71/`。随后使用旧异常任务 `20260901T083322Z-72165d7e84` 的三张 SHA 锁定输入只复跑舔毛，新任务 `20260901T102809Z-58d2285192` 为 `1/1`：五个首帧候选都从横卧 canonical 重绘为与导图宽高比一致的直立抬爪舔舐，SCAIL 首轮身份审核通过，最终主 APNG 可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T102758Z-7e6bbfcb71/`。两份报告仍只以完成性和媒体可读性决定通过，姿态指标用于诊断与本次修复证据。

## 更新图集

1. 只选已完成且输入文件仍可确认来源的历史任务。
2. 在清单新增来源相对路径与 SHA-256；不记录用户身份、提示词或敏感字段。
3. 为新 canary 写清物种、视图、完整动作；只有适合每次 smoke 的小集合才增加 `smoke_actions`。
4. 需要替换媒体时提升 `corpus_root` 版本，不覆盖旧版本文件。
5. 运行 `--materialize-only`、`bash ops/scripts/verify-change.sh` 和 live dry-run；服务修改还要完成 live smoke。

非阻断质量样例可以保留，方便定位行为变化，但不需要为画面差异更新基线来“修绿”稳定性测试。
