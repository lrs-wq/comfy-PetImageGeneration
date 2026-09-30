# Comfy 内容审核与后续交付拦截

版本：local-video `2.27.0`、历史平台 `1.14.0`；本地策略 `comfy-content-moderation-v1`。

Comfy 调用独立 Content Review 网关的现有 `/v1/moderation?brief=false` 接口。Content Review 的分类器、词库及阈值仍由其自身管理；Comfy 管理任务状态、异步结果审核、人工复核、下载拦截和证据记录。

## 交付时点

用户要求下载和回调无需等待审核完成，因此**结果审核在生成完成后异步执行**。生成任务先进入 `completed`，立即提供下载并发送原有成功回调。`review`、人工复核未完成、结果审核超时或异常都不阻塞首次交付。

结果后来被机器判为 `block` 或被人工驳回时，任务变为 `blocked`，后续业务下载返回 `403`；查询隐藏输出，简化 GIF API 映射为 `failed`。尚未发送的成功回调会再次检查当前拒绝状态和文件摘要，必要时改为无媒体链接的失败通知。**已经下载的文件、已发送的成功回调及客户端自行缓存的内容无法追回。该实现不证明所有内容在交付前已经审核。**

输入审核仍在主体生成前执行。输入 `block` 停止生成；`review` 等待后台复核；超时、无效响应或审核不可用进入 `moderation_error`，允许后台显式重审。审核中断不会触发视觉质量降级恢复来绕过拒绝。已有任务没有审核记录的标为“未审核”，不会补写通过证明，也不自动批量补审旧媒体。

## 生效规则

以下是 Comfy 的执行规则，不是对违法内容识别率的承诺。部署后通过 `GET /api/local-video/moderation/rules` 或已登录历史页的 `/pet-generation/api/moderation/rules` 导出当前开关、规则版本及上游实时健康状态。

| 规则 | 行为 |
| --- | --- |
| CM01 | 上传的全部身份视图、原始提示及动作文本在生成前审核；不只审核抠图。动画上传保留完整动画供视频采样审核。 |
| CM02 | 全部声明的交付媒体在生成完成后异步审核，包括降级生成结果；诊断 JSON 只绑定摘要。 |
| CM03 | 输入机器拒绝停止生成；结果机器拒绝拦截后续交付。 |
| CM04 | 输入待复核暂停生成；结果待复核进入后台，首次交付继续。 |
| CM05 | 输入审核异常暂停生成；结果审核异常记录为异常而不是通过，提供重审，首次交付继续。文件摘要不匹配时拒绝对应下载或尚未发送的成功回调。 |
| CM06 | 人工决定绑定本地规则版本、当前文件摘要及审核记录 revision；另存上游实际策略版本、检测状态、风险标签、分数和 trace ID。 |
| CM07 | 已判违规或人工驳回的结果拦截后续业务下载和尚未发出的成功回调；后台仍可在鉴权后查看证据媒体。 |
| CM08 | 可要求所有结果继续接受人工复核，但这一要求不阻塞首次下载或成功回调。 |

GIF/APNG/动画 WebP 经白底 MP4 转换后调用视频审核，避免只检查第一帧。实际覆盖受 Content Review 的视频抽帧策略限制，**不是逐帧识别**。返回中出现部分审核不可用标签、缺少策略版本、内容 ID 不匹配或风险与动作矛盾时记录审核异常。

2026-09-30 实际上游健康检查显示：图片服务启用了肤色启发式和 PaddleOCR，NSFW 模型、NudeNet、多模态 vision safety 关闭；视频以 `1 FPS` 采样、最多 `60` 帧，ASR 关闭。不能把配置文件中的开关当作模型实际启用证明。规则导出返回实际运行健康信息；当前 API 未提供全部实际数值阈值，不能将 YAML 默认值伪装为已核对的运行阈值。肤色启发式也可能把金毛等宠物照片判为待复核。

## 人工复核后台

入口：`https://ai.hzzootech.com/pet-generation/history`，沿用已有平台登录。详情新增“内容审核”，可查看机器记录、输入原图和生成结果、下载审核 JSON、填写复核依据并通过/驳回，或重新机器审核。

生成页遇输入拒绝、待复核或审核异常时停止轮询并恢复提交按钮，提示前往历史页复核/重审。后台放行后的后续结果从历史页查看。

操作账号由后台设置为 `platform:<已配置登录账号>`，不能由浏览器指定；账号代表现有平台账号，不代表新增了实名或多用户权限系统。写操作要求自定义请求头防止跨站提交。私有审核密钥仅留在服务器，普通生成 API Key 无法访问复核媒体或人工决定接口。

输入通过后重新排入生成队列；结果通过仅保存复核决定。结果驳回后可重审，已撤销交付的结果在重审期间仍保持拦截，审核明确通过后再恢复。机器 `block` 或审核异常不能直接人工“通过”，须先重审；人工说明不能为空，过期 revision 或文件变更须刷新重审。取消状态不可通过复核。

## 配置与接口

| 环境变量 | 默认值/作用 |
| --- | --- |
| `COMFY_CONTENT_MODERATION_ENABLED` | `0`；部署启用后新任务持久化 `required=true`，之后关闭全局开关也不解除这些任务的已记录拒绝。 |
| `COMFY_CONTENT_REVIEW_URL` | `http://127.0.0.1:8040` |
| `COMFY_CONTENT_REVIEW_TIMEOUT` | `300` 秒 |
| `COMFY_CONTENT_MODERATION_REQUIRE_OUTPUT_REVIEW` | `1`；机器通过的结果也进入人工复核，不影响首次交付。 |
| `COMFY_CONTENT_REVIEW_ADMIN_KEY` | 无默认值；local-video 与历史平台共享独立复核凭证。 |
| `LOCAL_VIDEO_OUTPUT_ROOT` | 默认 `/data/comfy-pet-pipeline/outputs/local_video_api`；隔离测试可覆盖，模型仍使用原数据根目录。 |
| `COMFY_CONTENT_REVIEW_ALLOW_TEST_DECISIONS` | 默认关闭；只用于隔离生成回归，将 `review_origin=automated_test` 保存为 `verification_review`，不作为人工审核证明；生产保持关闭。 |

部署配置使用私有环境文件，禁止将密钥值提交到 Git。新增接口：

| 接口 | 鉴权/用途 |
| --- | --- |
| `GET /api/local-video/moderation/rules` | 原生成 API Key；实时规则导出 |
| `GET /api/local-video/jobs/{id}/moderation` | `X-Review-Key`；完整审核记录 |
| `POST /api/local-video/jobs/{id}/moderation/review` | `X-Review-Key`；`decision`、`reviewer`、`note`、`revision`；人审来源默认 `human` |
| `POST /api/local-video/jobs/{id}/moderation/retry` | `X-Review-Key`；显式重审，结果重审不重新生成 |
| `GET /api/local-video/jobs/{id}/review-files/{filename}` | `X-Review-Key`；私有复核预览 |

历史平台提供同名 `/pet-generation/api/jobs/video/{id}/...` 鉴权代理，不将复核凭证发给浏览器。普通文件接口保留原鉴权并检查拒绝标志和交付文件 SHA-256。

每个新任务的 `status.json` 保存审核状态，任务目录的 `content_moderation.json` 保存完整事件。事件保留每次机器审核、上游响应标识与模型状态、文件 SHA-256、人工账号和说明；不保存 OCR 原文。服务重启从持久化状态恢复未完成的结果审核，输入/结果重审不会自动覆盖人工驳回。

## 验证记录的真实范围

- `tests/integration/test_content_moderation.py` 使用模拟上游判定验证立即交付、拒绝后的下载/回调拦截、鉴权、过期/文件变更、取消/重启、动画上传和降级恢复边界；这是控制逻辑测试，不是识别准确率测试。
- `.venv/bin/python ops/scripts/verify-content-moderation.py --report <私有报告路径>` 调用真实 Content Review，记录中性文本、一个配置风险词、普通图形和动画样例的结果及运行健康信息。只证明所列样例及接口传输；未测试违法图片/视频准确率，未生成真实人工审核证明。
- `bash ops/scripts/verify-change.sh` 和默认猫狗四动作 live smoke 仍是必需检查。隔离 smoke 中固定历史金毛输入可能被肤色启发式转为待复核；明确标注的自动测试复核只用于完成生成回归，不能填入“实际人工复核记录”。
- 数据、报告和测试媒体留在私有数据目录，不进入知识图谱或 Git。完整的违法内容分类覆盖、准确率/误报率及真实人工操作记录，**没有做的仍写“没有”**。
