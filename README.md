# Comfy Pet Pipeline

猫狗图片与双动作 GIF 生成服务。调用方上传一张必选正面图，以及最多两张可选侧面图；系统先强制二分类为猫或狗，再生成并归一化 canonical 全身参考。猫生成 `eating`、`grooming`，狗生成 `eating`、`standing`，其中狗的 `standing` 暂时同时映射到兼容字段 `grooming`。

当前 API 版本：`2.26.42`。上传图先建立共享身份 canonical，各动作再使用独立提示词、单色几何引导、seed 偏移、SCAIL 实验目录和输出路径。每个动作固定完成 `5` 轮 Qwen 候选生成，不命中缓存跳过、也不在首个合格候选处提前停止；每一轮会把上一轮身份门的具体问题传给下一轮，而不是无条件声称发生了重着色。若原图确有浅条纹但候选把条纹数量、面积和对比度放大，前三轮中姿态与食盆安全的候选可在第 4、5 轮仅作为动作几何底稿，由 canonical 和原始正面图重绘真实身份；原图没有条纹证据时仍失败关闭，身份阈值不放宽。五轮结束后只从通过颜色、条纹等身份底线的生成候选中选择综合最优者。原始上传图只用于建立身份、身份审计和必要的身份回绘佐证，不能作为最终降级输出；五轮后没有身份合格候选时任务准确记录 `5/5` 身份预算耗尽且未发布，不再先标记“接受最佳降级”，下游 SCAIL 或透明提取失败时也只能复用已选中的生成候选制作降级循环。canonical 透明完整性门只把明显穿过主体中线的缺口视为内部洞，正常肩部、腰部和臀部侧轮廓凹陷不再误杀；语义降级抠图在被接受前执行同一门禁，生成 canonical 的抠图确实无效时会改用上传身份安全抠图继续建立身份 canonical，但不会把原始上传图发布为动作首帧或最终结果。犬类吃饭、喝水、如厕提示把白/灰主体像素限定为零外观权威的动作坐标遮罩，并在专用提示后重新绑定上传色盘。犬类吃饭的 SCAIL 动作不足时仍最多进行 `5` 次逐步增强、强度上限受 `1.35` 约束的尝试。猫支持 `eating / grooming`，默认犬和中型犬支持 `eating / standing`，测试平台显式选择小型犬后支持 `eating / drinking / running / sleeping / toileting / standing`；API 路由和请求字段保持不变。

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
