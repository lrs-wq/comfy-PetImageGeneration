# Comfy Pet Pipeline

宠物图片与双动作 GIF 生成服务。调用方上传一张必选正面图，以及最多两张可选侧面图；系统先生成并归一化宠物 canonical 全身参考，再分别生成“吃饭”和“舔毛”动作首帧，最后驱动白模动作视频并输出两个 GIF。

当前 API 版本：`2.26.1`。生产策略默认只对运行完整性失败关闭，生成物的细粒度视觉审核以 advisory 记录为主，避免审核门导致正常链路中断。

## 主要入口

- 简化 GIF API：`http://39.106.120.253:8051/api`
- 本地视频 API：`http://127.0.0.1:8770/api/local-video`
- 测试界面：`https://ai.hzzootech.com/pet-generation/`
- ComfyUI：`http://127.0.0.1:8188`

简化 API 使用同一个 `/api` 端点，以 `type=upload` 提交任务并返回查询 key，以 `type=query` 和 key 查询进度及最终 `eating.gif`、`grooming.gif`。请求字段和响应示例见 [简化 API 文档](docs/api/SIMPLE_PET_GIF_API.md)。底层接口见 [本地视频 API 文档](docs/api/LOCAL_VIDEO_API.md)。

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
.venv/bin/python -m pytest -q
```

端到端脚本不会进入默认 pytest 集合，必须显式运行 `tests/e2e/` 中的目标脚本。它们可能调用 GPU 和生成链路。

## 部署

敏感配置只存放于 `/etc/comfy-pet-pipeline/`，仓库仅提交 `ops/env/*.example`。运行模型、任务产物、Profile 资产和验证归档位于 `/data/comfy-pet-pipeline/`。ComfyUI 运行代码位于 `/opt/comfyui/current`，仓库根目录的 `ComfyUI` 仅作为兼容软链接。

服务模板位于 `ops/systemd/`，部署说明见 [服务器部署文档](docs/operations/SERVER_DEPLOYMENT.md)。历史版本说明已归档到 `docs/history/README_RELEASE_HISTORY.md`。
