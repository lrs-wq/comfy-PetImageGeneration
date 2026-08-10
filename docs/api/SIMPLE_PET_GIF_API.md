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
