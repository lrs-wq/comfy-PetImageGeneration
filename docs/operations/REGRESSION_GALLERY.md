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

### 每次仓库修改后：真实完成性 canary

确认没有在线任务后运行：

```bash
.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke
```

默认 smoke 串行提交两组历史输入，避免与单卡 GPU 并发：

| canary | 输入覆盖 | 动作 |
| --- | --- | --- |
| `orange-tabby-duplicate-three-view` | 猫，三视图字段但三个文件相同 | `eating / grooming` |
| `golden-retriever-outdoor` | 犬，户外复杂背景、两个不同视角 | `eating / standing` |

该默认矩阵在每次仓库修改后都必须执行：恰好一组猫、两种行为各一次；恰好一组狗、两种行为各一次。脚本会在提交前硬校验 `cat = eating + grooming`、`dog = eating + standing`，缺物种、缺行为、重复行为或额外默认案例都会直接失败。第一组失败时默认仍继续提交第二个物种，保证猫狗都被实际尝试，最终报告仍按任一失败返回非零。pre-commit 只运行快速门禁，不能替代这次真实 GPU 自测。

脚本在提交前检查 API、worker、ComfyUI、模型和当前活动任务。发现已有 `queued/running` 任务会拒绝运行；应等待队列清空后重跑，不以显式 `--case` 规避默认矩阵。修改动作专属路径时仍可追加对应历史案例；显式案例会使用清单中的完整动作集合：

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

live 报告同时显示每个动作的非阻断诊断：首帧选中轮次、最低门状态、舔毛前肢拓扑状态/策略/选中证据、拓扑是否阻断完成，以及 SCAIL 舔毛身份审核状态和选中轮次。吃饭链路还记录全帧运动、局部口鼻跨度、口鼻追踪来源，以及静态食盆可靠帧、时序恢复和 alpha 核心保留率。这些字段仅用于定位选择原因；图集的 PASS 仍只取决于 `completed` 和主媒体可读。

### v2.26.60 历史异常离线回放

犬任务 `20260903T055308Z-7ca587d08e` 的吃饭参考红色食盆框为 `[66,303,205,424]`，宽高比 `1.149 < 1.40`，确认食盆分割已吞进下口鼻。新版准备路径以 `reference_bowl_mask_occludes_muzzle` 转入完整白膜恢复；诊断运行 `dog-mouth-fix-replay-20260903` 生成 `81` 帧完整低—高—低几何、静态食盆框 `[58,364,182,411]`，身份中心纵向跨度 `15 px`、高度跨度 `30 px`。旧主视频经过新版输出门回放时，全帧邻帧 MAE 为 `0.503358`，首帧 excursion 为 `5.43438`，但口鼻纵向跨度只有 `0.599152 px`，因此明确命中 `eating_muzzle_motion_too_low`；轻微画面变化不再掩盖固定嘴部。

猫任务 `20260903T053156Z-51a505bf26` 的旧吃饭 SCAIL 视频与引导回放显示，食盆在源帧 `25–36` 的语义 alpha 可降为零。新版从 `81` 帧中找到 `69` 个有界画布支持率至少 `55%` 的可靠帧，以第 `14` 帧为最高支持代表，构建 `10791` 像素时序支持区和 `9858` 像素保护核心。修复后 alpha `220` 下的食盆核心最低保留率为 `0.998478`，而旧输入最低为 `0`；`unsupported_guidance_forced_opaque=false`，人工对比未再出现旧帧的碗体消失，也没有初版直接填充宽泛引导造成的碗下白块。该离线回放只验证确定性准备、输出门和透明恢复，不能替代部署后的默认猫狗双行为 live smoke。

## 已核验完成性快照

提交 `ebbc987` 部署为 `v2.26.59` 后，使用户外金毛三视图直接运行 `eating + standing`；生产报告 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260903T040438Z-ebbc987b86/report.json` 为 `1/1`，预检 API 为 `2.26.59`，worker、ComfyUI、模型健康且提交前队列为空。任务 `20260903T040446Z-6021b80b4f` 为 `completed`，两个动作都真实完成固定 `5/5`，分别按完成优先选择第 `5/4` 轮生成候选；状态确认只加载 canonical Image 1、`image2_input_name=null`，没有原图/canonical 输出替代，`degraded_output_actions=[]`。

eating 与 standing 主 APNG 都可读取、`512×512 / 100` 帧，SHA-256 分别为 `096514bc02ceb9081bed2344e041776c07024d6efd84d40cc6464d19b5906f4a` 与 `5d6957cecd7bafa9f5d53be5b8f8566368a8de3f6253a437b81e8d13ee2fbee2`。人工抽检 canonical、两个动作首帧以及动画首/中/末帧，未再观察到 v2.26.57 的低多边形切面或 v2.26.58 的平滑无纹理犬体；毛束、鼻眼材质与身体自阴影均明显恢复。该人工结论单独记录，图集 PASS 仍只由完成终态和媒体可读性决定。线上 LoRA 字段为 `requested_user_loras=[]`、`effective_canonical_user_loras=[]`，动作只保留 Qwen Lightning 推理加速策略，没有可继续移除的风格化 LoRA。

提交 `6f20788` 部署为 `v2.26.57` 后，默认生产 smoke 报告 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T112749Z-6f207881c2/report.json` 为 `2/2`。预检版本为 `2.26.57`，worker、ComfyUI 和模型均健康；猫任务 `20260902T112757Z-05b61e8e16` 与户外金毛任务 `20260902T113839Z-dec1cf3d6c` 均 `completed`。

随后把源任务 `20260902T061115Z-93095dbfff` 的三张 SHA 锁定输入与 `eating + standing` 精确复跑为 `20260902T115351Z-20c5865423`；报告 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T115345Z-6f207881c2/report.json` 为 `1/1`。两个动作均真实完成 `5/5`，分别选中第 `3/4` 轮且都为 `validated`；两份主 APNG 可读、`512×512 / 100` 帧，SHA-256 分别为 `5493f32ef8e85bf49db96b750a1bfbd10d4c8413b3aad151e56fca8ccc8d71fa` 与 `abdca496b714ac18d58fa6297413a93be63698e64a12569d720ae5def37a419e`。canonical 暖色恢复把上半身白色从 `0.113897` 降至 `0.000031`、暖色从 `0.794310` 升至 `0.971048`，修改 `94864` 个候选像素，alpha 完全一致且没有复制源空间像素；两个动作候选自身已经通过暖色门，故动作后修色未触发。吃饭 SCAIL 使用 `81` 个真实源帧，邻帧灰度 MAE `0.550668`（门限 `0.55`）、相对首帧最大 MAE `8.539383`（门限 `3.0`）、动作 excursion ratio `2.846461`，不是静态首帧循环。站立使用 `reference_guided_hybrid_semantic_plus_bounded_canvas_matte_v2`；独立对 100 帧 alpha 以阈值 `8` 扫描，最大封闭透明组件仅 `21 px`，没有大于 `64 px` 的显著面部或身体孔洞。

提交 `a538013` 部署为 `v2.26.56` 后，默认生产 smoke 报告 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T095548Z-a53801311e/report.json` 为 `2/2`。猫舔毛任务 `20260902T095553Z-261629d3e2` 与户外金毛犬站立任务 `20260902T100536Z-f08b885750` 均 `completed`，两个主 APNG 都可读取且为 `512×512 / 100` 帧；后者使用与 `v2.26.55` 失败任务相同的犬输入，canonical 暖色恢复被接受，归一化内部洞不再终止任务。

随后把源任务 `20260902T061115Z-93095dbfff` 的三张 SHA 锁定输入、犬类 Profile 与 `eating + standing` 精确复跑为 `20260902T102135Z-c2efcdd105`；报告 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T102129Z-a53801311e/report.json` 为 `1/1`。两个动作都真实生成 `5/5`，任务终态 `completed`，两份主 APNG 均可读、`512×512 / 100` 帧。吃饭 SCAIL 输出含 `81` 个真实源帧，邻帧灰度运动均值 `0.554275`（门限 `0.55`）、相对首帧最大灰度变化 `8.866577`（门限 `3.0`）、轮廓最大变化 `0.103970`，不再是静态首帧循环；站立使用 `reference_guided_hybrid_semantic_plus_bounded_canvas_matte_v2`，引导式内部洞修补共处理 `80` 个组件、`10554` 像素且保留外部自然间隙。

该精确复跑同时暴露了动作选中后的最后一处视觉窗口：eating 第 3 轮虽是五轮最低风险媒体，上半身中性白色仍由 canonical 的 `1.8088%` 扩张到 `8.2580%`。`v2.26.57` 在这份实际选中 cutout 上离线运行宠物专属暖色分布恢复，正式食盆分割锁定 `6892` 个道具像素，恢复 `18921` 个宠物像素；白色降为 `0`、暖色升至 `0.983364`、alpha 完全一致。该步骤不新增模型轮次，任何异常均只跳过，不改变完成终态。

`v2.26.55` 部署后的首轮默认 smoke 中，猫任务完成，但户外金毛犬站立任务 `20260902T092351Z-8709d1e5f0` 在 canonical 最终化阶段失败。诊断确认模型和抠图均已产出可读媒体，失败仅来自尺度相关的内部洞比例：原始候选 `954 / 286084 = 0.003335`，512 归一化结果 `405 / 82514 = 0.004908`。`v2.26.56` 对这份失败媒体离线运行候选内有界修补后，同一门的最大残余洞降至 `12 / 82927 = 0.000145` 并通过；画布 ROI 外未修改。该失败 smoke 保留为发现证据，不能记作通过；新提交仍需重新运行默认 smoke。

`v2.26.55` 对源任务 `20260902T061115Z-93095dbfff` 完成了部署前的同媒体离线回放。吃饭准备路径再次复现 `no_face_muzzle_anchor`，但新策略生成完整 `81` 帧白膜动作几何并通过整体、下半身与下四分之一运动审计，不再生成静态姿态。站立原视频经自适应混合 matte 重建后，旧 APNG 中分别出现在 `14/100` 和 `16/100` 帧的头部/身体封闭洞均降为 `0/100`；输出仍为可读 `512×512 / 100` 帧 APNG。该记录证明历史媒体的确定性后处理和准备合同，不替代服务部署后的默认 smoke 与原输入真实模型复跑。

2026-09-01 在提交 `a0cd7ce` 上运行默认 smoke，两组 canary 均完成（`2/2`）：重复三视图橘虎斑猫的 `grooming` 与户外金毛犬的 `standing` 都发布了可读取的 `512×512`、`100` 帧主 APNG。该快照只证明当次完成性合同通过，不建立任何画质结论；新的服务修改仍必须重新运行 smoke。

同日提交 `89ddcb9` 部署后的最新默认 smoke 再次 `2/2`：猫任务 `20260901T081100Z-b0f4ee4801`、狗任务 `20260901T081913Z-33cde52c21` 均完成，报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T081055Z-89ddcb9392/`。针对完成优先策略的原失败输入复测 `20260901T083322Z-72165d7e84` 也为 `1/1`：复用源任务 `20260901T055619Z-d6d84380a1` 的三张输入和两个动作，canonical 使用完成优先回退，两个动作各生成固定五轮并以 best-effort 继续，最终两个 `512×512`、`100` 帧主 APNG 可读取；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T083311Z-89ddcb9392/`。

提交 `7e6bbfc` 部署后的 v2.26.52 默认 smoke 为 `2/2`：猫舔毛任务 `20260901T095724Z-22d0da39ca` 与犬站立任务 `20260901T100834Z-1b066e1d40` 都 `completed`，各自主 APNG 可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T095713Z-7e6bbfcb71/`。随后使用旧异常任务 `20260901T083322Z-72165d7e84` 的三张 SHA 锁定输入只复跑舔毛，新任务 `20260901T102809Z-58d2285192` 为 `1/1`：五个首帧候选都从横卧 canonical 重绘为与导图宽高比一致的直立抬爪舔舐，SCAIL 首轮身份审核通过，最终主 APNG 可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260901T102758Z-7e6bbfcb71/`。两份报告仍只以完成性和媒体可读性决定通过，姿态指标用于诊断与本次修复证据。

提交 `45082ff` 部署后的 v2.26.54 默认 smoke 为 `2/2`：猫舔毛任务 `20260902T035216Z-c146ae8758` 与犬站立任务 `20260902T040146Z-a1586dbfa7` 均完成，主 APNG 都可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T035208Z-45082ffb95/`。再以源任务 `20260901T083322Z-72165d7e84` 的三张 SHA 锁定输入精确复跑舔毛，任务 `20260902T041526Z-67216d465d` 为 `1/1`：严格生成五轮且未读缓存、未增加第六轮，选中第 5 轮解剖优先候选，前肢拓扑覆盖率 `0.904762`、平均匹配距离 `2.804203 px`、方向一致性 `0.888213`，SCAIL 首轮通过，最终主 APNG 可读取、`512×512 / 100` 帧；报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T041515Z-45082ffb95/`。最低身份门的 `degraded_passed` 仍是非阻断诊断，图集 PASS 只取完成性和媒体可读性。

同版随后使用原失败任务 `20260902T032017Z-cf5f87991a` 的三张 SHA 锁定输入、犬类 Profile 与 `eating + standing` 精确复跑，新任务 `20260902T054129Z-7711e8fff7` 为 `1/1`。该任务再次命中与旧任务相同的 `no_face_muzzle_anchor`、`static_pose_fallback_forbidden` 和 `white_driver_motion_signal_missing`；两个动作都真实生成五轮且选中候选的身份视觉字段均为 `false`。新版没有在 67.5% 失败，而是从 eating 第 4 轮的 SHA 锁定 Qwen cutout 制作降级循环并完成；eating 主 GIF 与 standing 主 APNG 均可读取、`512×512 / 100` 帧，未使用原图或 canonical 代替输出。报告位于 `/data/comfy-pet-pipeline/outputs/regression_gallery/live/20260902T054122Z-172a78f4d0/`。

## 更新图集

1. 只选已完成且输入文件仍可确认来源的历史任务。
2. 在清单新增来源相对路径与 SHA-256；不记录用户身份、提示词或敏感字段。
3. 为新 canary 写清物种、视图、完整动作；只有适合每次 smoke 的小集合才增加 `smoke_actions`。
4. 需要替换媒体时提升 `corpus_root` 版本，不覆盖旧版本文件。
5. 运行 `--materialize-only`、`bash ops/scripts/verify-change.sh` 和 live dry-run；每次仓库修改都必须完成猫狗双行为 live smoke。

非阻断质量样例可以保留，方便定位行为变化，但不需要为画面差异更新基线来“修绿”稳定性测试。
