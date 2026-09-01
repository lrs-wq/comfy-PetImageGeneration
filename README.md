# Comfy Pet Pipeline

猫狗图片与双动作 GIF 生成服务。调用方上传一张必选正面图，以及最多两张可选侧面图；系统先强制二分类为猫或狗，再生成并归一化 canonical 全身参考。猫生成 `eating`、`grooming`，狗生成 `eating`、`standing`，其中狗的 `standing` 暂时同时映射到兼容字段 `grooming`。

当前 API 版本：`2.26.51`。上传图先建立共享身份 canonical，各动作再使用独立提示词、单色几何引导、seed 偏移、SCAIL 实验目录和输出路径。每个动作固定完成 `5` 轮 Qwen 候选生成；历史首帧缓存只写审计、不参与读取或候选计数，因此不会替代、增加或跳过本次任务的五轮预算。每轮仍读取上一轮的身份、动作和透明质量问题，既有条纹、吃饭接触和浅色消色差暗斑修复继续优先争取严格合格结果。五轮结束后先选择严格合格候选；如果所有候选都只有身份配色、花纹、动作、毛面、构图或透明视觉门问题，则改选可读取、可归一化且风险最低的生成候选，标记 `best_effort` 后继续 SCAIL 和产物发布。canonical 同样先执行严格透明与内部洞检查；严格抠图不通过时，复用最佳 Qwen 生成图做非阻断 RemBG 语义抠图，不再因身份色盘或内部洞视觉结论切换到上传图或终止任务。原始上传图只用于身份锚定、审计和回绘佐证，绝不作为动作首帧或最终降级输出。模型无输出、媒体不可读取、无法取得可用主体 alpha、摘要不一致或进程失败仍属于运行完整性错误并阻断。猫支持 `eating / grooming`，默认犬和中型犬支持 `eating / standing`，测试平台显式选择小型犬后支持 `eating / drinking / running / sleeping / toileting / standing`；API 路由和请求字段保持不变。

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

服务安全重启后，运行 `.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke` 提交猫、狗各一组已由现行代码实测完成的真实历史输入。它只要求任务 `completed`、每个动作有主产物且产物可读取，不评价画面质量。动作专属修改可用 `--case <id>` 显式扩展；历史输入只有在现行代码下完成性稳定后才进入必跑 smoke。覆盖范围、运行时机和报告说明见 [历史稳定性图集](docs/operations/REGRESSION_GALLERY.md)。其他专项端到端脚本仍需从 `tests/e2e/` 显式运行。

## 部署

敏感配置只存放于 `/etc/comfy-pet-pipeline/`，仓库仅提交 `ops/env/*.example`。运行模型、任务产物、Profile 资产和验证归档位于 `/data/comfy-pet-pipeline/`。ComfyUI 运行代码位于 `/opt/comfyui/current`，仓库根目录的 `ComfyUI` 仅作为兼容软链接。

服务模板位于 `ops/systemd/`，部署说明见 [服务器部署文档](docs/operations/SERVER_DEPLOYMENT.md)。历史版本说明已归档到 `docs/history/README_RELEASE_HISTORY.md`。
