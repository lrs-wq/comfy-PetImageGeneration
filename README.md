# 萌宠图片与视频生成平台（Comfy Pet Pipeline）

本项目是一套面向宠物图片的本地 AI 生成平台：用户上传一张宠物参考图和提示词后，系统先生成保留原宠物身份特征的 3D 卡通基准图，再生成奔跑、吃饭、喝水、趴卧等动作首帧，最后可继续生成约 5 秒的视频或 512×512 透明背景 GIF。

当前生产代码位于 `/root/comfy-pet-pipeline`，模型、输入、输出、任务状态和实验资产位于 `/data/comfy-pet-pipeline`。Git 只管理源码、文档和部署模板，不管理模型、LoRA、用户图片、生成视频、虚拟环境或第三方源码副本。

## 1. 当前状态

| 项目 | 当前值 |
|---|---|
| 主分支 | `main` |
| Codeup 远程 | `git@codeup.aliyun.com:6a02dee0405bafb07e12ff8c/comfy-pet-pipeline.git` |
| 服务器代码目录 | `/root/comfy-pet-pipeline` |
| 服务器数据目录 | `/data/comfy-pet-pipeline` |
| 本地视频 API 版本 | `2.18.4` |
| 测试平台版本 | `1.9.0` |
| 图片基础模型 | Qwen Image Edit 2511 FP8 Mixed |
| 当前视频引擎 | Wan2.2 I2V A14B FP8，高/低噪声双模型 |
| 当前视频规格 | 121 帧、24 FPS，约 5 秒 |
| 透明 GIF 规格 | 512×512、100 帧、20 FPS、5 秒 |
| 当前 GPU | NVIDIA A10，23028 MiB 显存 |
| 当前内存与 Swap | 58 GiB RAM、32 GiB Swap |

> 上表记录的是 2026-07-23 的服务器运行基线。磁盘占用、服务 PID、任务数量和模型文件会随运行变化。

## 2. 能力范围

生产链路提供以下能力：

- 上传 PNG、JPEG 或 WebP 宠物参考图。
- 输入中文或英文提示词；中文会先由服务器本地翻译模型转换为英文，再送入 Qwen。
- 使用参考图作为宠物身份唯一标准，保留毛色、花纹位置、脸型、眼睛、耳形、鼻色、毛长、体型比例和尾巴特征。
- 生成统一风格的萌系 3D 卡通基准图。
- 生成奔跑、吃饭、喝水、趴卧四种默认动作首帧，也允许自定义姿态。
- 上传并融合多个 Qwen Image/Edit LoRA；前端支持逐个添加、单独删除和文件名去重。
- 分别控制 Qwen 步数、CFG、降噪强度、Wan 步数、CFG、Shift 和随机种子。
- 只生成图片，或跳过视频阶段；也可以从动作首帧继续生成视频。
- 输出普通 MP4，或输出 512×512 透明背景 GIF。
- 对宠物身份、毛色、完整身体、四肢、爪子落地、姿态重复、绿幕质量和透明抠图执行自动质量检查与重试。
- 查询任务状态、下载结果、终止视频任务，并在历史页面查看提示词、LoRA 和生成参数。
- 保留图片抠图失败时的原始生成图；视频链路如果不能形成有效透明主体则按错误返回，避免输出不可用动画。

项目还保留了换装、三视图、传统 DashScope 链路和 3D 重建实验，但它们不等同于当前四姿态生产主链路。

## 3. 线上访问入口

| 用途 | 外部路径 | 内部服务 | 鉴权 |
|---|---|---|---|
| 图片/视频测试平台 | `https://ai.hzzootech.com/pet-generation/` | `127.0.0.1:8050` | 应用层 HTTP Basic Auth |
| 生成历史 | `https://ai.hzzootech.com/pet-generation/history` | `127.0.0.1:8050/history` | 应用层 HTTP Basic Auth |
| 本地视频 API 文档 | `https://ai.hzzootech.com/api/local-video/docs` | `127.0.0.1:8770/docs` | API 请求使用 `X-API-Key` |
| 宠物换装 MVP | `https://ai.hzzootech.com/pet-dressup/` | `127.0.0.1:8766` | Nginx Basic Auth |
| ComfyUI | 不直接暴露公网 | `127.0.0.1:8188` | 仅供内部服务调用 |

Nginx 对公网监听 80/443。域名 HTTP 请求会重定向到 HTTPS；裸 IP 的普通根路径返回 404。生成平台、API 和换装服务均通过独立路径反向代理，ComfyUI 不直接暴露。

## 4. 总体架构

```mermaid
flowchart LR
    U[浏览器或 API 调用方] -->|HTTPS| N[Nginx 80/443]

    N -->|/pet-generation/| P[测试平台 FastAPI<br/>127.0.0.1:8050]
    N -->|/api/local-video/| A[本地视频 API<br/>127.0.0.1:8770]
    N -->|/pet-dressup/| D[换装 MVP<br/>127.0.0.1:8766]

    P -->|内部 HTTP + X-API-Key| A
    A --> Q[单工作线程任务队列]
    Q --> T[中文本地翻译<br/>Opus-MT zh-en]
    Q --> C[ComfyUI<br/>127.0.0.1:8188]

    C --> GPU[NVIDIA A10 GPU]
    C --> M[Qwen / Wan / LoRA / VAE / Text Encoder]
    Q --> QC[身份、毛色、姿态、绿幕与透明主体 QC]
    QC --> POST[FFmpeg + OpenCV + rembg 后处理]

    P --> H[历史记录与文件代理]
    A --> S[/data/comfy-pet-pipeline/outputs/local_video_api]
    H --> S
    M --> MD[/data/comfy-pet-pipeline/models]
```

### 4.1 服务职责

| 服务 | systemd 单元 | 端口 | 主要职责 |
|---|---|---:|---|
| Nginx | `nginx.service` | 80/443 | TLS、反向代理、上传大小、超时和访问隔离 |
| 测试平台 | `pet-generation-platform.service` | 8050 | Web UI、Basic Auth、图片任务入口、LoRA 上传、历史记录和文件代理 |
| 本地视频 API | `local-video-api.service` | 8770 | 任务队列、Qwen/Wan 工作流、QC、恢复、取消、状态持久化和下载 |
| ComfyUI | `comfyui.service` | 8188 | GPU 模型加载、Qwen 图片推理和 Wan 视频推理 |
| 换装服务 | `pet-pipeline.service` | 8766 | 叠加式换装和可选 Qwen 精修 |
| NVIDIA 常驻服务 | `nvidia-persistenced.service` | — | 保持驱动上下文，减少 GPU 服务反复初始化 |

`local-video-api.service` 强依赖 `comfyui.service`；测试平台依赖 ComfyUI 和本地视频 API。服务均以 `root` 运行，只监听回环地址，由 Nginx 对外提供入口。

## 5. 代码与数据分离

### 5.1 代码目录

```text
/root/comfy-pet-pipeline/
├── local_video_api.py                  # 生产生成引擎与 API
├── pet_generation_platform.py         # 测试平台后端
├── pet_generation_platform.html       # 测试平台前端
├── pet_generation_history.html        # 历史记录前端
├── transparent_animation.py           # 透明 GIF/APNG 后处理
├── ComfyUI-DashScope-*/                # 自定义 ComfyUI 节点
├── pet_dressup_mvp/                    # 换装 MVP
├── deploy/                             # 部署模板与检查脚本
├── docs/                               # 运维文档
├── ComfyUI/                            # 运行时第三方源码，不进入主仓库
├── vendor/                             # 运行时第三方依赖，不进入主仓库
├── .venv/                              # Web/API Python 环境，不进入主仓库
└── *.env                               # 服务器配置与凭据，不进入主仓库
```

项目目录当前约 1.3GB，主要来自 `.venv`、ComfyUI 和第三方依赖；Git 首次提交实际只有约 836KB 源码与文档。

### 5.2 数据目录

```text
/data/comfy-pet-pipeline/
├── models/
│   ├── comfyui/                        # Qwen、Wan、VAE、文本编码器、LoRA
│   ├── translation/opus-mt-zh-en/      # 中文到英文本地翻译模型
│   ├── videoxfun/                      # Wan2.2-Fun Control 模型
│   └── comfyui-3d-pack/                # 3D 实验模型
├── comfyui/
│   ├── input/                          # ComfyUI 输入
│   ├── output/                         # ComfyUI 原始输出
│   └── temp/                           # ComfyUI 临时文件
├── outputs/
│   ├── local_video_api/                # 正式 API 任务、status.json 和输出文件
│   ├── test_platform/                  # 测试平台图片任务
│   ├── dressup/                        # 换装输出
│   └── *_test*/                        # 各类试验输出
├── inputs/                             # 固定测试输入
├── assets/                             # 换装等运行资产
├── experiments/                        # VideoX-Fun 和各种实验环境
├── archive/                            # 历史产物归档
└── downloads/                          # 下载过程目录
```

2026-07-23 数据盘快照：

| 路径 | 约占用 |
|---|---:|
| `/data/comfy-pet-pipeline/models` | 161GB |
| `/data/comfy-pet-pipeline/outputs` | 2.1GB |
| `/data/comfy-pet-pipeline/comfyui` | 824MB |
| `/data/comfy-pet-pipeline/archive` | 972MB |
| `/data/comfy-pet-pipeline/experiments` | 253MB |
| `/data/comfy-pet-pipeline` 总计 | 165GB |

`/data` 数据盘约 1007GB，当前可用约 689GB。系统盘约 197GB，当前可用约 91GB。

## 6. 完整生成链路

### 6.1 请求接收与鉴权

测试平台使用 HTTP Basic Auth。它接收参考图片、提示词、模式、动作、LoRA 和推理参数，并把视频任务转发到内部 `local-video-api`。直接调用本地视频 API 时，必须在请求头中提供：

```http
X-API-Key: <服务器配置的 API Key>
```

API Key、平台用户名和密码只保存在服务器环境文件中，不写入代码，也不进入 Git。

### 6.2 输入规范化

`normalize_uploaded_image()` 会校验上传类型和大小、处理方向信息、统一图像模式并生成 ComfyUI 可读取的输入文件。默认图片上传上限为 15MiB。

### 6.3 中文提示词本地翻译

系统通过 `contains_han()` 判断文本中是否含中文。中文提示词会先规范化，再使用：

```text
/data/comfy-pet-pipeline/models/translation/opus-mt-zh-en
```

翻译为英文。默认四姿态基准提示词使用代码中已验证的固定中英对照缓存，避免通用翻译器破坏动作定义。翻译模型缺失时，服务会给出明确的缺失路径信息。

### 6.4 卡通身份基准

当前默认图片配置为 `pet-multipose-20260717-v2`，来源任务为 `20260717T094459Z-f738613526`。它要求：

- 上传图片中的宠物是唯一身份标准。
- 保留毛色与花纹位置、脸型、眼睛颜色和间距、耳形、鼻色、毛长、体型比例和尾巴。
- 转换为高品质萌系 3D 卡通风格。
- 不穿衣服、不带配饰。
- 不继承参考图片中的姿态和肢体坐标；动作完全由当前姿态提示词控制。

当前基准 LoRA：

| LoRA 文件 | 权重 |
|---|---:|
| `宠物小猫 _ 写实摄影_v3.0.safetensors` | 0.50 |
| `Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors` | 0.60 |

基准图片默认参数：Qwen 6 步、CFG 1.0、Denoise 1.0、Seed `20260715`、LoRA 模式 `replace`。

### 6.5 四个默认动作首帧

| 动作 | 首帧核心约束 | 视频核心约束 | Seed 偏移 |
|---|---|---|---:|
| 奔跑 | 三分之四侧面、完整四肢、四爪落在同一基线、静态起跑准备姿态 | 原地猫科步态循环，前后腿按规律交替，身体小幅起伏，不长时间腾空 | 101 |
| 吃饭 | 低位四足进食姿态，浅米色猫粮碗与爪子留绿幕间隙，猫粮留在碗内 | 碗固定，头颈与下颌微动，前爪保持着地，不用爪抓食物 | 211 |
| 喝水 | 低位俯身，小型水碗，短小粉色舌尖与水面接触 | 小幅舔水、细微水纹，不产生长舌、口水或水柱 | 307 |
| 趴卧 | Sphinx 式趴卧，前爪平行前伸，后腿收拢，尾巴完整 | 呼吸、耳头微动和轻微尾尖动作，整身自然响应，不固定上半身 | 401 |

动作首帧由 `build_qwen_prompt()` 组合身份基准、用户提示词、动作规则、完整身体规则、绿幕规则与负面提示词。视频提示词由 `build_wan_prompt()` 组合动作生物力学、身份锁定、镜头稳定和负面约束。

### 6.6 姿态与身份质量检查

系统不会直接接受第一张生成图，而会执行多层检查：

- 主体必须完整、不能裁切头部、躯干、四肢、爪子或尾巴。
- 奔跑首帧必须识别出足够数量的分离着地爪，避免 `fewer_than_three_distinct_grounded_paws`。
- 主体占画面比例不能过大，画布角落应保持透明或干净绿幕。
- 身份调色板与参考图比较，防止灰白猫被错误重绘为暖色毛发等问题。
- 吃饭/喝水动作会排除碗区域，避免把碗颜色误判成宠物毛色。
- 通过轮廓 IoU、dHash 和姿态签名检测不同动作是否实际生成成同一张图。
- 绿幕检查会评估背景是否足够接近纯绿、角落是否干净以及前景比例是否合理。
- Qwen 姿态默认最多尝试 3 次；奔跑与碗类动作可有额外恢复尝试。

重试时可能降低用户 LoRA 总强度，减少 LoRA 对四肢结构和动作的干扰。默认用户 LoRA 上限为 4 个，单文件最大 2GiB，总融合文件最大 4GiB。

### 6.7 背景恢复与抠图

首选路径使用纯 `#00FF00` 绿幕进行色度抠图。背景不合格时，系统会：

1. 评估绿幕候选是否仍可可靠抠图。
2. 尝试语义背景恢复和主体候选选择。
3. 必要时使用 `rembg` 的 `u2net` 模型作为回退。
4. 再次检查主体面积、画布角落、爪子和透明度。

异常策略：

- 图片任务抠图不可靠时，返回原始生成图片并记录背景移除失败，不因为抠图失败丢弃整张图片。
- 视频/透明动画任务如果无法形成完整有效主体，则正常报错，不输出质量不可控的透明动画。

### 6.8 视频生成

当前服务器通过 `wan14.env` 启用：

| 参数 | 当前值 |
|---|---|
| 引擎 | `wan2.2-i2v-a14b-fp8` |
| 高噪声模型 | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` |
| 低噪声模型 | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |
| VAE | `wan_2.1_vae.safetensors` |
| 帧数 | 121 |
| FPS | 24 |
| 采样步数 | 20 |
| 高低噪声切换步 | 10 |
| CFG | 3.5 |
| Shift | 5.0 |

121 帧按 24 FPS 播放约为 5.04 秒。A14B FP8 依靠模型卸载和系统内存/Swap 在 A10 24GB 上运行，速度明显慢于 5B，但通常具有更好的细节和运动一致性。

代码仍保留 `wan2.2-ti2v-5b` 作为默认回退配置，并保留 Wan2.2-Fun-5B-Control 实验链路。当前 `FUN_CONTROL_ENABLED=false`，所以生产任务不会进入 Fun Control；即使未来启用，失败时也允许回退到普通 Wan。

### 6.9 透明动画后处理

`transparent_animation.py` 使用 OpenCV、FFmpeg、Pillow 和 rembg 完成：

- 从原始视频抽取目标帧。
- 估计主体掩码和锚点。
- 跟踪主体位移并进行平滑的全局稳定修正。
- 进行时间中值、Alpha 边缘和透明度处理。
- 输出 RGBA 帧、透明 GIF/APNG，并计算动画质量指标。

稳定处理只修正整体漂移，不应冻结上半身或把身体局部强制固定。趴卧动作仍允许胸腹呼吸、头耳联动和尾尖运动。

透明 GIF 固定输出 512×512、100 帧、20 FPS，时长 5 秒。Wan 原始视频使用 121 帧、24 FPS，后处理阶段重采样为 GIF 规格。

### 6.10 任务队列、恢复与取消

本地视频 API 使用单工作线程队列，默认队列上限为 8。每个任务写入独立目录：

```text
/data/comfy-pet-pipeline/outputs/local_video_api/<job_id>/
├── status.json
├── workflow-*.json
├── 输入与中间首帧
├── MP4/GIF/APNG 输出
└── QC、恢复和失败信息
```

服务启动时会读取历史 `status.json` 恢复任务视图。运行中的任务可以通过取消接口设置取消标志，并向 ComfyUI 发送中断请求。任务终态为 `completed`、`failed` 或 `cancelled`。

### 6.11 历史页面

测试平台从正式视频输出和独立图片任务中统一读取历史记录，支持：

- 按关键词搜索任务号、提示词或文件名。
- 按图片/视频类型筛选。
- 按任务状态筛选。
- 按 LoRA 名称筛选。
- 查看图片、提示词、内置/用户 LoRA、权重、Qwen/Wan 参数、Seed 和输出文件。
- 使用游标分页，避免一次加载全部历史任务。

## 7. 模型与运行环境

### 7.1 Qwen 图片模型

相对于 `/data/comfy-pet-pipeline/models/comfyui`：

| 类型 | 文件/目录 |
|---|---|
| 扩散模型 | `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` |
| 文本/视觉编码器 | `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | `vae/qwen_image_vae.safetensors` |
| Lightning LoRA | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| 用户 LoRA | `loras/user_uploads/` |

### 7.2 Wan 视频模型

| 类型 | 文件/目录 |
|---|---|
| A14B 高噪声模型 | `models/comfyui/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` |
| A14B 低噪声模型 | `models/comfyui/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |
| 5B 回退模型 | `models/comfyui/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` |
| UMT5 编码器 | `models/comfyui/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| 当前 A14B VAE | `models/comfyui/vae/wan_2.1_vae.safetensors` |
| Fun Control 模型 | `models/videoxfun/Wan2.2-Fun-5B-Control/` |

### 7.3 Python 环境

系统使用两个隔离环境：

| 环境 | 位置 | Python | 用途 |
|---|---|---:|---|
| ComfyUI Conda | `/opt/miniconda3/envs/comfyui` | 3.11 | PyTorch/CUDA、Qwen、Wan 和 ComfyUI GPU 推理 |
| Web/API venv | `/root/comfy-pet-pipeline/.venv` | 3.11（当前部署） | FastAPI、平台、后处理、DashScope 辅助工具 |

仓库中的 `environment-comfyui.yml` 声明 PyTorch 2.11.0、TorchVision 0.26.0、TorchAudio 2.11.0 和 CUDA 12.8 wheel。`environment.yml` 是较早的通用 `comfy-pet-video` 环境定义，目标 Python 3.10；生产服务器实际服务命令以 systemd 文件为准。

### 7.4 主机资源

| 资源 | 当前值 |
|---|---|
| GPU | NVIDIA A10，约 24GB 显存，驱动 580.173.02 |
| RAM | 58GiB |
| Swap | `/data/swapfile`，32GB，当前有实际使用 |
| 系统盘 | 197GB，总体约 52% 使用 |
| 数据盘 | 1007GB，总体约 28% 使用 |

Swap 是 A14B 模型 CPU offload 和高峰内存压力下的重要稳定保障，但不能替代显存；若 Swap 持续大量增长，推理会明显变慢，应检查并发、模型卸载和遗留进程。

## 8. API 说明

完整参数以 [LOCAL_VIDEO_API.md](LOCAL_VIDEO_API.md) 和 FastAPI `/docs` 为准。

### 8.1 路由

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/local-video/health` | 检查 API、模型和运行配置 |
| `POST` | `/api/local-video/jobs` | 创建图片/视频任务 |
| `GET` | `/api/local-video/jobs` | 列出最近任务 |
| `GET` | `/api/local-video/jobs/{job_id}` | 查询任务状态和输出 |
| `POST` | `/api/local-video/jobs/{job_id}/cancel` | 取消排队或运行任务 |
| `GET` | `/api/local-video/jobs/{job_id}/files/{filename}` | 下载任务文件 |

### 8.2 创建任务主要参数

| 参数 | 说明 |
|---|---|
| `image` | PNG/JPEG/WebP 参考图 |
| `prompt` | 用户提示词，1–2000 字符 |
| `poses` | JSON 数组、换行或 `|` 分隔的动作列表 |
| `pose_count` | 输出动作数量，默认 4，服务上限默认 6 |
| `resolution` | `480p`、`720p` 或 `512x512` |
| `output_format` | `mp4` 或 `transparent_gif` |
| `skip_video` | `true` 时只生成所选动作首帧 |
| `request_kind` | `image` 或 `video`，记录调用界面模式 |
| `image_profile` | 图片基准配置，默认 `pet-multipose-20260717-v2` |
| `user_loras` | 已上传 LoRA 的 JSON 数组，包含模型名、原文件名和强度 |
| `user_lora_mode` | `replace` 或 `stack` |
| `qwen_steps` | 1–30 |
| `qwen_cfg` | 0.1–5.0 |
| `qwen_denoise` | 0.1–1.0 |
| `wan_steps` | 4–40 |
| `wan_cfg` | 1.0–8.0 |
| `wan_shift` | 1.0–12.0 |
| `seed` | 0 到 64 位有符号范围内的非负整数 |

### 8.3 调用示例

```bash
curl -X POST 'https://ai.hzzootech.com/api/local-video/jobs' \
  -H 'X-API-Key: <YOUR_API_KEY>' \
  -F 'image=@pet.jpg' \
  -F 'prompt=保留猫咪原始毛色、脸型和眼睛特征，生成高品质3D卡通形象' \
  -F 'pose_count=4' \
  -F 'output_format=transparent_gif' \
  -F 'qwen_steps=6' \
  -F 'wan_steps=20' \
  -F 'seed=20260715'
```

只生成动作首帧：

```bash
curl -X POST 'https://ai.hzzootech.com/api/local-video/jobs' \
  -H 'X-API-Key: <YOUR_API_KEY>' \
  -F 'image=@pet.jpg' \
  -F 'prompt=保留宠物身份，生成完整全身3D卡通形象' \
  -F 'pose_count=4' \
  -F 'skip_video=true' \
  -F 'request_kind=image'
```

查询和取消：

```bash
curl -H 'X-API-Key: <YOUR_API_KEY>' \
  'https://ai.hzzootech.com/api/local-video/jobs/<JOB_ID>'

curl -X POST -H 'X-API-Key: <YOUR_API_KEY>' \
  'https://ai.hzzootech.com/api/local-video/jobs/<JOB_ID>/cancel'
```

## 9. 配置文件与安全

### 9.1 不进入 Git 的服务器配置

| 文件 | 用途 |
|---|---|
| `.env` | DashScope/ComfyUI/换装等基础配置 |
| `.web-password` | 换装页面生成的本地密码 |
| `local_video_api.env` | 本地视频 API Key、数据路径和运行参数 |
| `pet_generation_platform.env` | 测试平台用户名、密码、基础路径和队列设置 |
| `wan14.env` | A14B 引擎、模型名、121帧/24FPS和采样参数 |
| `fun_control.env` | Fun Control 开关、脚本、模型和动作控制视频路径 |

所有 `*.env`、密钥、证书和密码文件都被 `.gitignore` 排除。`.env.example` 只提供空值或公开默认项。

### 9.2 Git 忽略范围

当前规则排除：

- `.venv`、Python 缓存、日志、PID、数据库和临时文件。
- 输入、上传、输出、生成结果和媒体文件。
- `*.safetensors`、`*.ckpt`、`*.pt`、`*.onnx`、`*.gguf`、NumPy 张量和压缩包。
- `models/`、`checkpoints/`、`loras/` 和 `datasets/`。
- `backups/`、`*.bak*`、`*.backup*` 和模型下载地址文件。
- `ComfyUI/`、`vendor/`、`comfyui_custom_nodes/` 和 `experiments/` 等下载的第三方项目。
- 图片、视频、音频和 3D 产物。

如确实需要提交少量文档图片，应先确认无隐私内容，再使用明确的例外规则；不要直接取消全局媒体忽略规则。

## 10. 部署与启动

### 10.1 基础软件

- Linux + NVIDIA 驱动。
- Miniconda/Conda。
- Python venv。
- Nginx、systemd、FFmpeg。
- 可访问的数据盘 `/data`。

### 10.2 创建环境

ComfyUI：

```bash
cd /root/comfy-pet-pipeline
conda env create -f environment-comfyui.yml
```

Web/API 环境：

```bash
cd /root/comfy-pet-pipeline
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-server.txt
```

3D 实验依赖很重，且 NumPy 版本要求与生产 API 不完全一致，应使用独立环境安装 `requirements-3d-server.txt`，不要直接混装进生产 `.venv`。

### 10.3 准备目录与软链接

`deploy/setup-layout.sh` 会把 ComfyUI 模型、输入、输出和临时目录链接到 `/data`，并链接自定义节点。该脚本会安全检查目标路径，但会重建 `/root/comfy-pet-pipeline/ComfyUI`，执行前必须确认 `/root/comfyui-core.tar.gz` 是正确版本并做好备份。

预期软链接：

```text
/root/comfy-pet-pipeline/ComfyUI/models  -> /data/comfy-pet-pipeline/models/comfyui
/root/comfy-pet-pipeline/ComfyUI/input   -> /data/comfy-pet-pipeline/comfyui/input
/root/comfy-pet-pipeline/ComfyUI/output  -> /data/comfy-pet-pipeline/comfyui/output
/root/comfy-pet-pipeline/ComfyUI/temp    -> /data/comfy-pet-pipeline/comfyui/temp
```

### 10.4 systemd 操作

```bash
systemctl daemon-reload
systemctl enable --now nvidia-persistenced.service
systemctl enable --now comfyui.service
systemctl enable --now local-video-api.service
systemctl enable --now pet-generation-platform.service
systemctl enable --now pet-pipeline.service
```

查看状态：

```bash
systemctl status comfyui.service --no-pager
systemctl status local-video-api.service --no-pager
systemctl status pet-generation-platform.service --no-pager
systemctl status pet-pipeline.service --no-pager
```

重启生产链路：

```bash
systemctl restart comfyui.service
systemctl restart local-video-api.service
systemctl restart pet-generation-platform.service
```

### 10.5 Nginx

```bash
nginx -t
systemctl reload nginx
```

仓库中的 `deploy/nginx/label-studio-proxy.conf` 是基础部署模板。服务器当前 `/etc/nginx/conf.d/label-studio-proxy.conf` 已包含 `/pet-generation/`、`/api/local-video/`、TLS 和其他同机业务路由，不能未经比较就用仓库基础模板覆盖当前生产配置。

同样，仓库当前只保存了 ComfyUI、换装和 NVIDIA 服务模板；线上 `local-video-api.service`、`pet-generation-platform.service` 及 drop-in 配置位于 `/etc/systemd/system/`。部署新服务器时应依据本 README 的端口、环境文件和依赖关系补齐这些单元。

## 11. 运维检查

### 11.1 健康检查

```bash
curl -H 'X-API-Key: <YOUR_API_KEY>' \
  http://127.0.0.1:8770/api/local-video/health

curl -u '<PLATFORM_USER>:<PLATFORM_PASSWORD>' \
  http://127.0.0.1:8050/api/health
```

### 11.2 日志

```bash
journalctl -u comfyui.service -n 200 --no-pager
journalctl -u local-video-api.service -n 200 --no-pager
journalctl -u pet-generation-platform.service -n 200 --no-pager
journalctl -u nginx.service -n 100 --no-pager
```

实时跟踪：

```bash
journalctl -fu local-video-api.service
```

### 11.3 GPU、内存与 Swap

```bash
nvidia-smi
free -h
swapon --show
df -h / /data
```

### 11.4 常见问题

#### 模型缺失

错误包含 `Required local model files are missing` 时，对照第 7 节检查 diffusion model、text encoder、VAE 和 LoRA。模型不在 Git 中，重新克隆代码不会自动得到模型。

#### `Animation source has only 81 frames; at least 100 are required`

说明送入透明动画后处理的视频不足 100 帧。当前 Wan 配置应为 121 帧、24 FPS；检查 `wan14.env`、工作流 JSON 和实际输出帧数。

#### `fewer_than_three_distinct_grounded_paws`

奔跑首帧没有检测到足够多的分离落地爪。常见原因是四肢互相遮挡、身体过大、爪子被裁切或 LoRA 过强。系统会自动重试；持续失败时降低用户 LoRA、减少夸张动作并检查参考图是否只包含特写头部。

#### `achromatic_source_recolored_with_warm_fur`

参考宠物是灰/白等低彩度毛色，但生成结果被改为暖棕/橙色。检查两项基准 LoRA 是否存在、权重是否为 0.50/0.60，并确认身份调色板保护没有被关闭。

#### `foreground_fraction_too_large`

主体过度占满画面，容易造成爪、尾巴或耳朵缺角。提示词应保留完整全身和画面四周留白；系统恢复提示词会主动缩小主体。

#### `nontransparent_canvas_corner`

透明结果的画布角落仍有前景或背景残留。图片模式会保留原图；视频透明模式会报错，避免输出带硬边背景的 GIF。

#### 四个姿态生成相同图片

检查姿态是否正确提交、动作 Seed 偏移是否生效，以及历史任务是否使用 `pet-multipose-20260717-v2`。系统使用轮廓 IoU 和 dHash 拦截重复姿态。

#### LoRA 看起来没有生效

确认文件已上传到 `models/comfyui/loras/user_uploads/`，文件名校验通过，任务历史记录中出现该 LoRA，并检查 `replace/stack` 模式和实际权重。动作重试可能自动降低用户 LoRA 权重以保护四肢结构。

#### 视频四肢运动不自然

先检查首帧是否具备清晰、分离、可见的四肢。视频模型会放大首帧结构错误。奔跑提示词要求猫科接触相位、推蹬顺序、前后腿交替和短暂身体起伏，禁止长时间四脚腾空。

#### 画面抖动或脸部扭曲

减少动作幅度、镜头运动和表情变化；保持头、眼、鼻、嘴作为一个连贯结构。检查 `transparent_animation.py` 的稳定化是否仅修正全局漂移，避免重新加入局部或上半身强制固定。

#### A14B 推理缓慢或内存压力大

检查是否存在并发任务、残留 Python/ComfyUI 进程、Swap 持续增长或显存没有释放。平台和本地 API 默认均为小队列，但 ComfyUI 仍应保持单 GPU 串行推理。

## 12. Git 工作流

### 12.1 拉取与查看状态

```bash
cd /root/comfy-pet-pipeline
git status --short --branch
git pull --ff-only origin main
```

### 12.2 提交代码

```bash
git add README.md local_video_api.py pet_generation_platform.py
git diff --cached --stat
git diff --cached --check
git commit -m update-pet-generation-pipeline
git push origin main
```

提交前必须确认：

- 没有 `.env`、密码、证书、API Key 或带签名下载 URL。
- 没有模型、LoRA、视频、图片、数据库、任务状态和用户上传文件。
- 没有 `.venv`、ComfyUI 全量源码、`vendor` 或实验环境。
- 单个文件不应异常增大；模型文件不要使用普通 Git 上传。

### 12.3 第三方仓库

运行目录中的以下项目可能保留自己的 `.git`，但主仓库通过 `.gitignore` 整体排除，不会形成不可用的嵌套 Git 引用：

- `/root/comfy-pet-pipeline/vendor/kiuikit`
- `/root/comfy-pet-pipeline/vendor/utils3d`
- `/data/comfy-pet-pipeline/experiments/VideoX-Fun`

## 13. 所有受 Git 管理文件说明

以下清单对应初始仓库中的 82 个受控文件。运行时新增文件如任务输出、模型和环境配置不会列入此表。

### 13.1 仓库基础文件

| 文件 | 说明 |
|---|---|
| `.env.example` | 基础环境变量示例，只包含空 API Key 和公开默认值。 |
| `.gitattributes` | 统一 Python、Shell、HTML、JSON、YAML、Markdown 等文本为 LF，并标记常见模型/媒体为二进制。 |
| `.gitignore` | 排除凭据、环境、缓存、模型、数据、媒体、备份和第三方源码。 |
| `README.md` | 本文档，项目入口、架构、部署、API、运维与文件索引。 |

### 13.2 架构、提示词与报告

| 文件 | 说明 |
|---|---|
| `CAT_MODULAR_HQ_IMPLEMENTATION_PLAN_20260702.md` | 高还原模块化猫咪 3D 建模实施计划，覆盖主体、器官、毛发、装配和验收。 |
| `CAT_MODULAR_MODELING_PLAN.md` | 模块化 3D 建模的总体拆分、锚点规范、技术路线和风险。 |
| `COMFY_PET_VIDEO_ARCHITECTURE.md` | ComfyUI 宠物动作视频最初架构设计，包括节点、五类动作、队列、重试和 API 草案。 |
| `LOCAL_VIDEO_API.md` | 本地多姿态 5 秒动画 API 的调用说明、参数和版本变更。 |
| `LOCAL_VIDEO_PROMPTS_V210.md` | 四动作图片/视频提示词英文版本、组合方式和负面提示词。 |
| `LOCAL_VIDEO_PROMPTS_V210_ZH.md` | 四动作提示词完整中文版本、统一基准、调色板保护和系统纠姿提示词。 |
| `LORA_STYLE_TEST_REPORT_20260715.md` | Qwen 萌宠 LoRA 的兼容性、固定 Seed A/B、四姿态和视频测试记录。 |
| `OPEN_SOURCE_PET_VIDEO_PIPELINE.md` | 纯开源图片与视频链路选择、目录、工作流、参数和质量筛选方案。 |
| `PET_TWO_ACTION_PIPELINE.md` | 单图六姿态与吃饭/奔跑双动作旧链路文档。 |
| `PET_VIDEO_DRIFT_RECORD.md` | 历史跑偏产物、原因及多候选/QC/结构控制改进记录。 |
| `PET_VIDEO_IMPLEMENTATION_REPORT.md` | 多姿态与动作视频阶段汇报，包含效果、成本、风险和本地开源配置。 |
| `docs/SERVER_DEPLOYMENT.md` | 服务器代码/数据布局、基础运维和能力边界说明。 |

### 13.3 当前生产服务与前端

| 文件 | 说明 |
|---|---|
| `local_video_api.py` | 核心生产服务：任务队列、中文翻译、图片基准、多 LoRA、Qwen/Wan 工作流、QC、重试、抠图、恢复、取消、状态和下载。 |
| `pet_generation_platform.py` | 测试平台 FastAPI 后端：Basic Auth、图片队列、LoRA 上传、视频 API 转发、历史聚合和文件代理。 |
| `pet_generation_platform.html` | 测试平台主页面：上传参考图、提示词、模式切换、姿态选择、LoRA 列表、参数、终止按钮和结果显示。 |
| `pet_generation_history.html` | 历史记录页面：分页、筛选、提示词、LoRA、参数和图片/动画预览。 |
| `transparent_animation.py` | 透明主体提取、位移跟踪、平滑稳定、Alpha 处理、GIF/APNG 导出和动画 QC。 |
| `prompt_translator.py` | 本地中文到英文翻译命令行封装。 |
| `download_prompt_translation_model.py` | 下载并保存本地提示词翻译模型。 |
| `local_model_trial.py` | 直接构造 Qwen/Wan ComfyUI 工作流的本地模型试运行工具。 |

### 13.4 传统图片/视频流水线与 CLI

| 文件 | 说明 |
|---|---|
| `pipeline_cli.py` | 统一命令行入口，组织姿态、三视图、吃饭视频和双动作任务。 |
| `pet_pose_generator.py` | DashScope/Qwen 旧版姿态图生成、下载、去背景和透明姿态表拼接。 |
| `pet_three_view_generator.py` | 使用多参考图生成正面、侧面、背面三视图。 |
| `pet_video_i2v.py` | 旧版吃饭/奔跑首帧、视频提交、轮询下载和后处理。 |
| `pet_two_action_pipeline.py` | 串联姿态、吃饭视频和奔跑视频的旧版双动作总流程。 |
| `run_open_api_probe.py` | 最小化开放 API 连通性和响应探测。 |
| `run_wan22_resume_from_poses.py` | 从已有姿态图恢复/继续 Wan2.2 视频阶段。 |
| `submit_four_pose_control_test.py` | 提交四姿态 Control 测试任务。 |
| `running_cycle_test.py` | 构造奔跑循环工作流并测试两周期视频。 |
| `wan14_smoke_test.py` | Wan A14B 模型加载和最小生成冒烟测试。 |
| `wan14_sha256.txt` | Wan A14B 模型文件校验摘要，不包含模型本体。 |

### 13.5 LoRA 测试工具

| 文件 | 说明 |
|---|---|
| `lora_style_ab_test.py` | 固定 Seed 比较不同 LoRA 强度的风格差异。 |
| `lora_pose_keyframe_test.py` | 使用 LoRA 生成四动作首帧并记录结构效果。 |

### 13.6 自定义 ComfyUI 节点

| 文件 | 说明 |
|---|---|
| `ComfyUI-DashScope-PetVideo/__init__.py` | 注册基础宠物姿态表节点包。 |
| `ComfyUI-DashScope-PetVideo/nodes/__init__.py` | 基础节点模块初始化文件。 |
| `ComfyUI-DashScope-PetVideo/nodes/qwen_cat_pose_sheet.py` | `DashScopeCatPoseSheet`，生成猫咪姿态表。 |
| `ComfyUI-DashScope-PetVideo-I2V/__init__.py` | 注册吃饭图生视频节点包。 |
| `ComfyUI-DashScope-PetVideo-I2V/nodes/__init__.py` | I2V 节点模块初始化文件。 |
| `ComfyUI-DashScope-PetVideo-I2V/nodes/eating_video_pose2.py` | `DashScopePose2EatingVideo`，由指定姿态生成吃饭视频。 |
| `ComfyUI-DashScope-PetVideo-TwoActions/__init__.py` | 注册双动作节点包。 |
| `ComfyUI-DashScope-PetVideo-TwoActions/nodes/__init__.py` | 双动作节点模块初始化文件。 |
| `ComfyUI-DashScope-PetVideo-TwoActions/nodes/two_action_pipeline.py` | `DashScopePetTwoActionVideos`，统一生成吃饭和奔跑视频。 |

### 13.7 换装 MVP

| 文件 | 说明 |
|---|---|
| `pet_dressup_mvp/README.md` | 换装 MVP 的运行、目录和能力说明。 |
| `pet_dressup_mvp/index.html` | 换装页面结构。 |
| `pet_dressup_mvp/styles.css` | 换装页面样式。 |
| `pet_dressup_mvp/app.js` | 前端宠物/服装选择、图层叠加、导出和精修调用。 |
| `pet_dressup_mvp/server.py` | 换装 FastAPI 服务、Qwen 精修、平面背景移除和静态页面。 |
| `pet_dressup_mvp/scripts/generate_demo_assets.py` | 生成演示服装、眼镜、帽子、项圈等透明覆盖素材。 |
| `pet_dressup_mvp/data/avatar_manifest.json` | 可选宠物头像/角色清单。 |
| `pet_dressup_mvp/data/wardrobe_manifest.json` | 可选服装和配饰清单。 |

### 13.8 3D 与图像资产实验

| 文件 | 说明 |
|---|---|
| `run_comfy3d_hunyuan_cat.py` | Hunyuan3D 单视图猫咪重建、网格后处理和预览。 |
| `run_comfy3d_hunyuan_mv_cat.py` | Hunyuan3D 多视图猫咪重建和视图裁切。 |
| `run_comfy3d_stablefast3d_cat.py` | StableFast3D 模型加载、几何生成、GLB/预览输出。 |
| `run_comfy3d_triposr_cat.py` | TripoSR 单图 3D 重建和 GLB 导出。 |
| `run_hunyuan_texgen_cat_pose.py` | Hunyuan TexGen 对已有猫咪姿态网格生成纹理。 |
| `refine_hunyuan_texgen_cat_texture.py` | 清理纹理噪点、增强毛发感、顶点色和预览。 |
| `improve_hunyuan2mini_vertex_texture_cat.py` | 通过投影和微位移增强 Hunyuan2 Mini 猫咪顶点纹理。 |
| `add_blender_fur_layer_cat.py` | 调用 Blender 清理网格、创建毛发、嘴垫、胡须、材质、灯光和渲染。 |
| `clean_cat_whisker_components.py` | 删除错误生成的独立胡须/碎片网格组件。 |
| `prepare_clean_alpha_cat_pose.py` | 清理透明姿态图遮罩、最大连通域、居中和白底调试图。 |
| `prepare_no_whisker_cat_pose.py` | 生成去除错误胡须的姿态输入。 |

### 13.9 部署文件

| 文件 | 说明 |
|---|---|
| `deploy/setup-layout.sh` | 建立 `/root` 与 `/data` 目录布局、重建 ComfyUI 并创建模型/输入/输出软链接。 |
| `deploy/configure-services.sh` | 安装基础 systemd 模板、配置 FFmpeg、换装密码和 Nginx，并启动换装服务。 |
| `deploy/smoke-test.sh` | 检查换装鉴权、换装健康接口和 Label Studio 根路径。 |
| `deploy/submit-lora-style-test.sh` | 提交 LoRA 风格测试任务。 |
| `deploy/verify_gpu.py` | 验证 PyTorch CUDA 可用性、设备名和显存。 |
| `deploy/verify_pipeline.py` | 验证核心 Python 模块和流水线依赖。 |
| `deploy/nginx/pet-pipeline.conf` | 可嵌入现有 server 块的 `/pet-dressup/` 反向代理片段。 |
| `deploy/nginx/label-studio-proxy.conf` | Label Studio + 换装服务的基础 Nginx 配置模板。 |
| `deploy/systemd/comfyui.service` | ComfyUI 8188 服务模板。 |
| `deploy/systemd/pet-pipeline.service` | 换装 MVP 8766 服务模板。 |
| `deploy/systemd/nvidia-persistenced.service` | NVIDIA Persistence Daemon 服务模板。 |

### 13.10 依赖定义

| 文件 | 说明 |
|---|---|
| `environment.yml` | 通用 `comfy-pet-video` Conda 环境定义。 |
| `environment-comfyui.yml` | ComfyUI GPU Conda 环境与 CUDA 12.8 PyTorch 版本。 |
| `requirements.txt` | 云 API、FastAPI、队列、图像视频预处理和开发工具的完整依赖集合。 |
| `requirements-server.txt` | 当前 Web/API 服务的精简依赖。 |
| `requirements-pipeline.txt` | 在 server 依赖上增加 rembg/onnxruntime 的生产流水线依赖。 |
| `requirements-3d-server.txt` | Hunyuan、StableFast3D、TripoSR 等 3D 实验依赖。 |

## 14. 推荐阅读顺序

1. 本 `README.md`：先理解当前生产架构。
2. [LOCAL_VIDEO_API.md](LOCAL_VIDEO_API.md)：了解 API 参数和调用。
3. [LOCAL_VIDEO_PROMPTS_V210_ZH.md](LOCAL_VIDEO_PROMPTS_V210_ZH.md)：了解四姿态基准与提示词约束。
4. [LORA_STYLE_TEST_REPORT_20260715.md](LORA_STYLE_TEST_REPORT_20260715.md)：了解当前 LoRA 选择依据。
5. [docs/SERVER_DEPLOYMENT.md](docs/SERVER_DEPLOYMENT.md)：了解服务器目录与基础运维。
6. `local_video_api.py`：阅读当前生产实现。
7. `pet_generation_platform.py`、`pet_generation_platform.html`：阅读测试平台实现。
8. `transparent_animation.py`：阅读透明动画与稳定后处理。

## 15. 已知边界与后续建议

- 模型和数据不在 Git 中，新服务器不能只靠 `git clone` 完成恢复，必须单独准备 `/data/comfy-pet-pipeline/models`。
- 当前生产视频使用 A14B FP8，但 A10 24GB 需要较多 CPU offload，单任务耗时长，不适合高并发。
- Fun Control 5B 已保留实验代码和模型，但当前关闭；不能把实验结果当作生产默认。
- 3D 实验依赖与生产 API 的 NumPy/PyTorch 组合可能冲突，必须隔离环境。
- 线上 Nginx 和两个核心 systemd 单元比仓库部署模板更新，后续应把脱敏后的 `local-video-api.service`、`pet-generation-platform.service` 和完整 Nginx 路由模板纳入版本控制。
- 历史任务会持续增长，建议定期归档 `/data/comfy-pet-pipeline/outputs/local_video_api`，但不要删除仍在历史页面引用的文件。
- 如需团队共享模型，应使用对象存储、制品库或专用模型仓库，并保存 SHA256；不要把大模型直接推送到普通 Git。
