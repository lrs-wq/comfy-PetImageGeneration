# 0016：Comfy 接入独立审核服务，结果异步交付与后续拦截

- 状态：accepted
- 日期：2026-09-30

## 决策

Content Review 是独立服务，通过现有统一审核接口接入 Comfy；Comfy 的历史页承担人工复核，不假定 Content Review 已有复核后台。按用户明确要求，结果审核不阻塞首次下载或成功回调。结果后来被机器拒绝或人工驳回才拦截后续交付，已交付内容无法追回。输入审核仍是生成前门禁。

将生成完成与审核完成分别记录。输出审核有独立线程、任务锁和持久化恢复，避免上一个任务的结果审核拖住其他任务完成。内部生成读取保留中间输出；对外查询隐藏未完成生成和已拒绝结果，文件接口校验交付摘要。视觉质量降级、重启恢复和过期的成功回调均不得覆盖已记录的拒绝。

审核决定记录文件摘要、本地版本、上游实际版本与检测状态。平台服务器绑定既有登录账号，独立审核密钥不发浏览器。已有无记录任务明确标“未审核”；测试自动复核与真实人工操作区分，不能伪装人工审核证明。

## 边界

实现不证明交付前已经审核，也不保证能追回已下载或客户端缓存的内容。2026-09-30 上游实际运行的图片 NSFW/NudeNet/vision safety、视频 ASR 关闭；视频采样不是逐帧。上游实际数值阈值未由现有 API 完整提供。控制逻辑测试、少量真实接口样例和真实生成 smoke 不能替代全面违法内容准确率评测或实际人工审核记录。

## 证据

- `src/comfy_pet_pipeline/api/content_moderation.py`
- `src/comfy_pet_pipeline/api/content_moderation_routes.py`
- `src/comfy_pet_pipeline/api/content_moderation_platform.py`
- `src/comfy_pet_pipeline/api/local_video.py`
- `src/comfy_pet_pipeline/api/job_callback.py`
- `web/platform/history.html`
- `tests/integration/test_content_moderation.py`
- `ops/scripts/verify-content-moderation.py`
- `docs/api/CONTENT_MODERATION.md`
