# Comfy Pet Pipeline

猫狗图片与双动作 GIF 生成服务。调用方上传一张必选正面图，以及最多两张可选侧面图；系统先强制二分类为猫或狗，再生成并归一化 canonical 全身参考。猫生成 `eating`、`grooming`，狗生成 `eating`、`standing`，其中狗的 `standing` 暂时同时映射到兼容字段 `grooming`。

当前 API 版本：`2.26.9`。生产策略默认只对运行完整性失败关闭，生成物的细粒度视觉审核以 advisory 记录为主；上传动作候选仍执行保证身份与动作底线的有限重试。犬类吃饭首帧固定从已通过三视图审计的 canonical 单图执行四步动作编辑，Qwen 不再读取中型犬/小型犬白膜轮廓；SCAIL 只读取白膜的抬头/低头时间曲线，在已批准首帧坐标内生成动作，耳形、口吻、体型与尾形不再由白膜覆盖。偏置到主体一侧的食盆允许外侧没有前爪，只要内侧前爪、食盆面积和边界合同可靠；参考动作提取失败时允许身份安全的静态吃饭回退继续提交，不再被旧白膜策略断言误杀。动作候选一旦未通过原始上传图锚定的颜色/条纹身份审计，重试耗尽后也不会重新放行；犬类站立仍使用 canonical 身份图并把动作几何交给 SCAIL。终端降级强制优先锚定原始上传图，绝不使用动作白膜冒充上传宠物。测试平台显式选择小型犬后可用 `eating / drinking / running / sleeping / toileting / standing` 六条白膜，中型犬仍为 `eating / standing`；默认 API 犬类路由、动作集合和请求字段保持不变。

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
.venv/bin/python -m pytest -q
```

端到端脚本不会进入默认 pytest 集合，必须显式运行 `tests/e2e/` 中的目标脚本。它们可能调用 GPU 和生成链路。

## 部署

敏感配置只存放于 `/etc/comfy-pet-pipeline/`，仓库仅提交 `ops/env/*.example`。运行模型、任务产物、Profile 资产和验证归档位于 `/data/comfy-pet-pipeline/`。ComfyUI 运行代码位于 `/opt/comfyui/current`，仓库根目录的 `ComfyUI` 仅作为兼容软链接。

服务模板位于 `ops/systemd/`，部署说明见 [服务器部署文档](docs/operations/SERVER_DEPLOYMENT.md)。历史版本说明已归档到 `docs/history/README_RELEASE_HISTORY.md`。
