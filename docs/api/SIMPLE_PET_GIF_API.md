# 宠物双姿态 GIF API

服务地址：`http://39.106.120.253:8051`

这个 API 固定生成两个 512×512 透明背景 GIF：

- `eating`：吃饭
- `grooming`：舔毛

调用方不能修改动作、提示词、尺寸或输出格式。所有业务请求都使用 `POST /api`，通过表单字段 `type` 区分上传和查询。

## 鉴权

所有上传、查询和 GIF 下载请求都必须携带请求头：

```text
X-API-Key: <API_KEY>
```

缺少或填写错误时返回 HTTP `401`。API Key 由服务部署方单独提供，不写入本文档。

## 1. 上传并提交任务

请求：`POST /api`，类型：`multipart/form-data`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `type` | 是 | 固定为 `upload` |
| `front` | 是 | 宠物正面图片 |
| `left` | 否 | 宠物左侧图片；必须与 `right` 同时上传 |
| `right` | 否 | 宠物右侧图片；必须与 `left` 同时上传 |

仅上传正面图：

```bash
curl -X POST 'http://39.106.120.253:8051/api' \
  -H 'X-API-Key: <API_KEY>' \
  -F 'type=upload' \
  -F 'front=@/path/to/front.png'
```

上传三视图：

```bash
curl -X POST 'http://39.106.120.253:8051/api' \
  -H 'X-API-Key: <API_KEY>' \
  -F 'type=upload' \
  -F 'front=@/path/to/front.png' \
  -F 'left=@/path/to/left.png' \
  -F 'right=@/path/to/right.png'
```

成功提交返回 HTTP `202`：

```json
{
  "key": "20260807T120000Z-abcd1234",
  "status": "queued"
}
```

`key` 是后续查询任务的唯一凭据。侧面图为可选；如使用侧面图，`left` 和 `right` 必须成对提交。

### 三视图字段对应关系

服务严格按照 multipart 字段名保存并传递图片，不会根据画面内容自动识别或交换 `front`、`left`、`right`。调用方必须在提交前完成视角对应：

| 字段 | 应上传的画面 | 识别要点 |
| --- | --- | --- |
| `front` | 宠物正面图 | 正脸朝向镜头，双眼、鼻部和两侧耳朵基本同时可见 |
| `left` | 宠物左侧视图 | 主要展示宠物左侧身体；通常宠物头部朝画面右侧 |
| `right` | 宠物右侧视图 | 主要展示宠物右侧身体；通常宠物头部朝画面左侧 |

如果只有一张可靠的正面图，只提交 `front`。不要为了凑齐三张图，把侧面图填入 `front`，也不要把正面图填入 `left` 或 `right`。

#### 错位案例：`20260813T091107Z-a272f25801`

该任务虽然被记录为 `front_left_right` 三视图模式，但实际提交内容如下：

| 接口字段 | 服务端文件 | 实际画面 | 尺寸 |
| --- | --- | --- | --- |
| `front` | `input.png` | 编号 3、头部朝画面右侧的侧面图 | 563×416 |
| `left` | `input_left.png` | 编号 2、头部朝画面左侧的侧面图 | 560×407 |
| `right` | `input_right.png` | 真正的正面图 | 514×584 |

任务目录中的三张输入图与送入 ComfyUI/Qwen 的副本 SHA-256 完全一致，因此错位发生在接口提交字段映射阶段，不是后续生成链路重新排序导致的。该任务的视图审核状态为 `skipped_advisory`，只记录了三个字段角色，没有对“正面图是否确实为正面”进行阻塞校验。

针对该案例，正确字段映射应为：

| 正确字段 | 应使用的原任务文件 |
| --- | --- |
| `front` | `input_right.png`，真正的正面图 |
| `left` | `input.png`，编号 3 的侧面图 |
| `right` | `input_left.png`，编号 2 的侧面图 |

#### 三视图错误返回

8051 会同步校验字段是否齐全，并在任务进入后台处理后校验 `front` 槽位是否包含可靠的宠物正面图。由于正面识别是异步执行的，上传接口先返回任务 key，调用方需要通过查询接口取得最终校验结果。

未上传 `front` 时返回 HTTP `422`：

```json
{
  "detail": "front is required for type=upload."
}
```

只上传 `left` 或只上传 `right` 时返回 HTTP `422`：

```json
{
  "detail": "left and right must be uploaded together."
}
```

如果三个字段都存在，上传请求先返回 HTTP `202`：

```json
{
  "key": "20260813T091107Z-a272f25801",
  "status": "queued"
}
```

后台检测到 `front` 不是可靠正面图时，任务不会进入 Qwen 或 SCAIL，查询接口稳定返回：

```json
{
  "key": "20260813T091107Z-a272f25801",
  "status": "failed",
  "progress": 0.0,
  "stage": "front_view_not_reliably_frontal",
  "action": null,
  "error": "front_view_not_reliably_frontal: front must contain a reliable frontal pet image.",
  "gifs": {
    "eating": null,
    "grooming": null
  }
}
```

调用方应以 `status=failed` 结束轮询，并以 `stage=front_view_not_reliably_frontal` 作为稳定、可编程判断的错误码。任务 `20260813T091107Z-a272f25801` 中的编号 3 侧面图被放入 `front`；在本规则生效后，按原字段重新提交会得到上述失败结果。

当前错误码只保证识别 `front` 槽位不可靠，不保证判断 `left` 与 `right` 是否互换。调用方仍需在提交前正确映射两张侧面图。

## 2. 查询进度和结果

请求：`POST /api`，类型：`multipart/form-data`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `type` | 是 | 固定为 `query` |
| `key` | 是 | 上传接口返回的任务 key |

```bash
curl -X POST 'http://39.106.120.253:8051/api' \
  -H 'X-API-Key: <API_KEY>' \
  -F 'type=query' \
  -F 'key=20260807T120000Z-abcd1234'
```

返回示例：

```json
{
  "key": "20260807T120000Z-abcd1234",
  "status": "completed",
  "progress": 100.0,
  "stage": "completed",
  "action": null,
  "error": null,
  "gifs": {
    "eating": "http://39.106.120.253:8051/file/20260807T120000Z-abcd1234/eating_subject_replacement_transparent_512.gif",
    "grooming": "http://39.106.120.253:8051/file/20260807T120000Z-abcd1234/grooming_subject_replacement_transparent_512.gif"
  }
}
```

任务状态：

| `status` | 含义 |
| --- | --- |
| `queued` | 已进入队列 |
| `running` | 生成中 |
| `completed` | 两个 GIF 均已生成 |
| `failed` | 生成失败，原因见 `error` |
| `cancelled` | 任务已取消 |

### `stage` 字段枚举

`stage` 表示任务当前执行到的具体步骤，主要用于展示详细进度；业务逻辑应以 `status` 判断任务是否完成或失败。8051 服务通常会原样返回底层生成服务的阶段名称。

当前 8051 双 GIF 链路可能返回以下值：

| `stage` | 含义 |
| --- | --- |
| `queued` | 任务已进入队列，尚未开始处理 |
| `front_view_not_reliably_frontal` | `front` 槽位未检测到可靠宠物正面图；任务终止，`status` 为 `failed` |
| `waiting_for_gpu` | 正在等待 GPU 或 ComfyUI 空闲；Qwen 和 SCAIL 开始前都可能出现 |
| `generating_cartoon_reference` | Qwen 正在根据上传视图生成宠物全身基准图 |
| `retrying_cartoon_reference` | 全身基准图未满足要求，正在重新生成；仅特定审核配置下出现 |
| `generating_cartoon_keyframes` | Qwen 正在生成吃饭或舔毛动作首帧 |
| `retrying_cartoon_keyframe` | 动作首帧未满足最低要求，正在重新生成 |
| `restoring_grooming_identity` | 正在恢复舔毛首帧的面部及身份一致性 |
| `finalizing_keyframes` | 正在对 Qwen 首帧进行抠图、归一化和结果固化 |
| `subject_references_ready` | 两个动作的参考首帧已准备完成，可以进入视频生成 |
| `preparing_subject_replacement` | 正在准备 SCAIL 输入、动作视频和参考图 |
| `generating_subject_replacement` | SCAIL 正在生成当前动作视频 |
| `auditing_final_video_fur_surface` | 正在检查最终视频的毛发表面；当前线上为非阻塞审核，通常不会出现 |
| `extracting_transparent_animation` | 正在从动作视频提取透明背景 GIF |
| `completed` | 吃饭和舔毛两个 GIF 均已生成 |
| `failed` | 任务失败，具体原因见 `error` |
| `cancel_requested` | 底层已收到取消请求，正在停止任务 |
| `cancelled` | 任务已取消 |
| `output_contract_failed` | 8051 专用阶段：底层报告完成，但缺少 `eating` 或 `grooming` GIF，8051 将任务改判为失败 |

正常情况下，阶段顺序大致如下：

```text
queued
→ waiting_for_gpu
→ generating_cartoon_reference
→ generating_cartoon_keyframes
→ finalizing_keyframes
→ subject_references_ready
→ waiting_for_gpu
→ eating: preparing_subject_replacement
→ eating: generating_subject_replacement
→ eating: extracting_transparent_animation
→ grooming: preparing_subject_replacement
→ grooming: generating_subject_replacement
→ grooming: extracting_transparent_animation
→ completed
```

重试、舔毛身份恢复和审核均为条件阶段。某些阶段持续时间很短，因此客户端轮询时可能观察不到。`action` 在 SCAIL 阶段通常为 `eating` 或 `grooming`，在 Qwen 首帧阶段可能为 `null`。

底层 8770 服务还保留以下兼容阶段。当前 8051 的“上传宠物图并生成两个 GIF”链路正常情况下不会进入这些阶段，但调用方应允许服务返回未知的 `stage` 字符串，以便兼容后续升级。

| `stage` | 含义 |
| --- | --- |
| `preparing_uploaded_identity_references` | 已退役的上传图直接构造 SCAIL 参考图线路 |
| `verifying_profile_standard_direct` | 校验预置宠物档案的直接输出资源 |
| `copying_profile_standard_direct` | 复制预置档案动作结果 |
| `matting_profile_standard_direct_outputs` | 对预置结果进行抠图 |
| `publishing_profile_standard_transparency` | 发布预置透明动画 |
| `generating_keyframes` | 旧版普通关键帧生成 |
| `completed_keyframes_only` | 只生成关键帧，不生成视频 |
| `switching_to_video_model` | 从 Qwen 图像模型切换到视频模型 |
| `generating_videos` | 旧版 Wan 视频生成 |
| `generating_controlled_running_video` | 旧版受控跑步视频生成 |
| `stabilizing_transparent_animation` | 旧版透明动画稳定处理 |

客户端接入建议：

- 以 `status` 判断任务生命周期，使用 `stage` 展示细分进度。
- 遇到未知 `stage` 时直接展示原值并继续轮询，不要将其自动判定为失败。
- 当 `status=failed` 时读取 `error`；当 `status=completed` 时确认 `gifs.eating` 和 `gifs.grooming` 均不为 `null`。

生成期间尚未产出的 GIF URL 为 `null`。建议每 10～20 秒查询一次。

## 3. 下载 GIF

查询结果中的 URL 仍需携带同一个 API Key：

```bash
curl \
  -H 'X-API-Key: <API_KEY>' \
  'http://39.106.120.253:8051/file/<key>/<filename>' \
  -o result.gif
```

8051 服务只开放两个目标透明 GIF，不暴露中间 PNG、MP4 或内部服务地址。

## 错误格式

错误响应使用 FastAPI 标准格式：

```json
{
  "detail": "错误原因"
}
```

常见状态码：`401` 鉴权失败，`404` key 或结果不存在，`422` 字段不合法，`502/503` 生成服务暂不可用。

## 在线说明

- Swagger UI：`http://39.106.120.253:8051/docs`
- OpenAPI JSON：`http://39.106.120.253:8051/openapi.json`
- 健康检查：`http://39.106.120.253:8051/health`
