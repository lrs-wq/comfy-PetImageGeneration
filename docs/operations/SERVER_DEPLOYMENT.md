# 服务器部署

## 运行目录

```text
/root/comfy-pet-pipeline/       Git 代码、文档和 Python 虚拟环境
/etc/comfy-pet-pipeline/        机器级环境变量与密码（0600）
/opt/comfyui/current/           ComfyUI 运行时代码
/opt/comfyui/custom-nodes/      第三方 ComfyUI 节点
/data/comfy-pet-pipeline/       模型、输入、输出、任务、Profile 资产和归档
```

仓库根目录的 `ComfyUI` 是指向 `/opt/comfyui/current` 的兼容软链接。模型、输入、输出和临时目录继续由 ComfyUI 软链接到 `/data/comfy-pet-pipeline/`。

## 首次配置

```bash
cd /root/comfy-pet-pipeline
install -d -m 0700 /etc/comfy-pet-pipeline
install -m 0600 ops/env/common.env.example /etc/comfy-pet-pipeline/common.env
install -m 0600 ops/env/api.env.example /etc/comfy-pet-pipeline/api.env
install -m 0600 ops/env/fun-control.env.example /etc/comfy-pet-pipeline/fun-control.env
install -m 0600 ops/env/wan.env.example /etc/comfy-pet-pipeline/wan.env
install -m 0600 ops/env/platform.env.example /etc/comfy-pet-pipeline/platform.env
```

编辑这些文件填入真实 API key、密码和模型路径；不要把值写入 Git。Nginx 中 JoyAI 的授权 map 单独存放在 `/etc/nginx/snippets/joyai-hls-upstream-authorization.conf`。

## 安装依赖与服务

```bash
cd /root/comfy-pet-pipeline
.venv/bin/python -m pip install -r requirements/server.txt
bash ops/scripts/configure-services.sh
```

systemd 运行以下服务：

| 服务 | 监听地址 | 用途 |
| --- | --- | --- |
| `comfyui.service` | `127.0.0.1:8188` | Qwen/ComfyUI 工作流 |
| `local-video-api.service` | `127.0.0.1:8770` | 底层图片与视频编排 |
| `pet-gif-api.service` | `0.0.0.0:8051` | 简化双 GIF API |
| `pet-generation-platform.service` | `127.0.0.1:8050` | 测试界面 |
| `pet-pipeline.service` | `127.0.0.1:8766` | 换装演示 |

## 运维检查

```bash
systemctl status comfyui local-video-api pet-gif-api pet-generation-platform pet-pipeline
journalctl -u local-video-api -f
curl -fsS http://127.0.0.1:8188/system_stats
curl -fsS http://127.0.0.1:8770/api/local-video/health
curl -fsS http://127.0.0.1:8051/health
curl -fsS http://127.0.0.1:8766/health
```

测试台由 Nginx 暴露在 `https://ai.hzzootech.com/pet-generation/`。ComfyUI 不公开；需要时使用 SSH 隧道：

```bash
ssh -L 8188:127.0.0.1:8188 root@39.106.120.253
```

## 工具入口

旧版 CLI 保留兼容入口：

```bash
cd /root/comfy-pet-pipeline
.venv/bin/python -m tools.legacy.pipeline_cli pose
.venv/bin/python -m tools.legacy.pipeline_cli three-view
.venv/bin/python -m tools.legacy.pipeline_cli two-actions
```

`tests/e2e/` 和部分 `tools/experiments/` 会提交真实 GPU 任务，不属于日常部署健康检查，必须显式执行。
