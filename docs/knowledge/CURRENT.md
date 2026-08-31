# Comfy Pet Pipeline 当前知识

## 目标

上传一张必选宠物正面图和最多两张侧面图，先确定猫/狗物种并生成 canonical 身份参考，再生成相应动作 GIF/视频，同时尽量保持毛色、花纹、耳形、口吻、体型和尾形身份一致。

## 当前对外能力

- 简化 GIF API：端口 `8051`，同一 `/api` 端点使用 `type=upload` 提交、`type=query` 查询。
- 本地视频 API：端口 `8770`，路径 `/api/local-video`。
- ComfyUI：端口 `8188`，承担底层工作流执行。
- 测试平台：用于显式测试犬型及扩展动作，不代表默认公开 API 动作集合已经变化。

## 当前动作规则

- 猫默认生成 `eating`、`grooming`。
- 狗默认生成 `eating`、`standing`；`standing` 同时映射到兼容字段 `grooming`。
- 测试平台显式选择小型犬时可测试 `eating / drinking / running / sleeping / toileting / standing`。
- 中型犬测试路径仍以 `eating / standing` 为主。

## 关键不变量

- 每个动作固定完成 `5` 轮 Qwen 候选生成，不因前序候选合格而提前停止；随后从通过身份底线的生成候选中选择综合罚分最低者。
- 原始上传图只用于建立身份、身份审计和必要的身份回绘佐证，不作为生成失败时的最终降级输出。
- 上传动作候选如果未通过颜色/条纹身份审计，重试耗尽后也不能重新放行。
- 犬类吃饭动作中，白膜只提供受控动作几何或时间曲线，不能覆盖耳形、口吻、体型和尾形。
- 根目录 Python 文件主要是兼容入口；新实现优先放在 `src/comfy_pet_pipeline/`。

## 质量与失败策略

- 运行完整性失败会阻断任务。
- 细粒度视觉质量审核以 advisory 记录为主，除明确身份底线外不应被旧策略误杀。
- e2e 脚本可能调用 GPU 和完整生成链路，不进入默认 pytest 集合。

## 异常恢复边界

- `v2.26.40` 已在 `v2.26.39` 生产基线上将动作生成预算统一为固定五轮，禁用会跳过本轮生成的首帧缓存捷径，并删除预算耗尽后追加 canonical 第六轮的恢复分支。
- 语义抠图只算对五轮内既有生成候选的后处理，不新增候选轮次。
- SCAIL、透明提取或服务恢复等下游异常，只能从已经选定的生成动作候选制作降级循环；恢复前必须校验逐动作五轮选择记录及候选 SHA-256。
- 异常发生在候选选定之前，或五轮都没有身份合格候选时，任务失败；服务重启场景可重新入队，但不能输出原始上传图。

## 数据与部署边界

- 敏感配置位于 `/etc/comfy-pet-pipeline/`，仓库只保留示例。
- 模型、任务产物、Profile 运行资产和验证归档位于 `/data/comfy-pet-pipeline/`。
- ComfyUI 生产运行代码位于 `/opt/comfyui/current`；仓库中的 `ComfyUI` 是兼容入口。
- systemd、Nginx 和环境模板位于 `ops/`。

## 阅读顺序

1. API 任务读取 `docs/api/`。
2. 生成行为读取 `docs/architecture/PET_TWO_ACTION_PIPELINE.md`。
3. 文件或数据位置读取 `docs/operations/PROJECT_LAYOUT.md`。
4. 部署任务读取 `docs/operations/SERVER_DEPLOYMENT.md` 和 `ops/`。

实时分支、HEAD、未提交文件和知识漂移状态由 `kg context` 动态生成，不在本文件硬编码。
