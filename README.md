# Comfy Pet Pipeline

猫狗图片与双动作 GIF 生成服务。调用方上传一张必选正面图，以及最多两张可选侧面图；系统先强制二分类为猫或狗，再生成并归一化 canonical 全身参考。猫生成 `eating`、`grooming`，狗生成 `eating`、`standing`，其中狗的 `standing` 暂时同时映射到兼容字段 `grooming`。

当前 API 版本：`2.26.65`。猫舔毛导出保持原始时间方向，优先完整正向视频，接缝不合适时选择通过头爪运动与边界检查的自然周期正向重复；无法找到安全周期时保留正向动作并记录接缝告警。GIF、APNG 和白底 MP4 共用周期方案，原始生成视频单独保留。上传图先建立共享身份 canonical；低饱和灰黑/暖棕混合被毛会阻断高饱和橘化、条纹严重放大以及源图不存在的红粉眼鼻特征。前两轮生成仍不合格时，最后一轮改用原宠物安全抠图作为 Qwen 输入，以 `denoise=0.70` 且禁用用户 LoRA 轻度重绘；所有生成候选仍不安全时，只把上传身份安全抠图作为 canonical 锚点，绝不把身份漂移的生成图传给动作链路。该例外不把上传图当成动作首帧或最终动画输出。其后的每个物种/动作组合都有稳定隔离键、独立配置快照、Qwen 首帧异常边界、SCAIL 动画异常边界、几何引导、seed、实验目录和输出路径。单个动作发生普通执行异常时，系统先清理该动作运行态并继续兄弟动作，全部可运行动作尝试完成后再统一失败关闭或进入既有生成候选降级恢复；用户取消仍立即终止整任务。每个动作固定完成 `5` 轮 Qwen 候选生成；历史首帧缓存只写审计、不参与读取或候选计数。动作五轮结束后先选择严格合格候选；如果所有候选都只有视觉门问题，则改选媒体完整且风险最低的生成候选，以 `best_effort` 继续 SCAIL。模型无输出、媒体不可读取、无法取得可用主体 alpha、摘要不一致或进程失败仍属于运行完整性错误并阻断。猫支持 `eating / grooming`，默认犬和中型犬支持 `eating / standing`，测试平台显式选择小型犬后支持 `eating / drinking / running / sleeping / toileting / standing`；API 路由和请求字段保持不变。

## 主要入口

- 简化 GIF API：`http://39.106.120.253:8051/api`
- 本地视频 API：`http://127.0.0.1:8770/api/local-video`
- 测试界面：`https://ai.hzzootech.com/pet-generation/`
- ComfyUI：`http://127.0.0.1:8188`

简化 API 使用同一个 `/api` 端点，以 `type=upload` 提交任务并返回查询 key，以 `type=query` 和 key 查询进度及最终 GIF。请求字段、猫狗字段差异和狗 `grooming → standing` 兼容映射见 [简化 API 文档](docs/api/SIMPLE_PET_GIF_API.md)。底层接口见 [本地视频 API 文档](docs/api/LOCAL_VIDEO_API.md)。

## 目录

```text
src/comfy_pet_pipeline/   生产 Python 包
web/platform/             测试台页面
profiles/                 受版本控制的主体配置与清单
extensions/comfyui/       本项目维护的 ComfyUI 自定义节点
apps/dressup/             独立换装演示应用
tests/                    unit、integration、e2e 测试
tools/                    资产、诊断、实验和迁移工具
ops/                      systemd、Nginx、环境模板和运维脚本
requirements/             分场景依赖文件
docs/                     API、架构、运维及历史资料
```

根目录中的少量同名 Python 文件是兼容入口，供旧脚本和外部调用继续导入；新代码应直接从 `src.comfy_pet_pipeline` 导入。完整存放规则见 [项目目录与数据边界](docs/operations/PROJECT_LAYOUT.md)。

## 开发与检查

```bash
cd /root/comfy-pet-pipeline
.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python ops/scripts/run-historical-regression-gallery.py --materialize-only
bash ops/scripts/verify-change.sh
```

历史稳定性图集从既有任务中按 SHA-256 固化 6 组生成 canary 输入；媒体只存放在 `/data/comfy-pet-pipeline/archive/regression-gallery/`，不进入 Git。`verify-change.sh` 是每次修改后的快速门禁，并由服务器仓库的 pre-commit hook 调用。16 个旧质量样例仍生成诊断图，但画面差异不会阻断。

每次仓库修改完成快速门禁后，都必须在队列为空时运行 `.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke`。默认矩阵固定为猫 `eating + grooming` 与狗 `eating + standing`，每个行为各真实运行一次并产生可读主产物；脚本会拒绝缺物种、缺行为或重复行为的默认矩阵。它只要求任务 `completed`、每个动作有主产物且产物可读取，不评价画面质量。动作专属 `--case <id>` 只能追加，不能替代这四项必测。覆盖范围、运行时机和报告说明见 [历史稳定性图集](docs/operations/REGRESSION_GALLERY.md)。其他专项端到端脚本仍需从 `tests/e2e/` 显式运行。

## 部署

敏感配置只存放于 `/etc/comfy-pet-pipeline/`，仓库仅提交 `ops/env/*.example`。运行模型、任务产物、Profile 资产和验证归档位于 `/data/comfy-pet-pipeline/`。ComfyUI 运行代码位于 `/opt/comfyui/current`，仓库根目录的 `ComfyUI` 仅作为兼容软链接。

服务模板位于 `ops/systemd/`，部署说明见 [服务器部署文档](docs/operations/SERVER_DEPLOYMENT.md)。历史版本说明已归档到 `docs/history/README_RELEASE_HISTORY.md`。
