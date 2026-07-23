# Server deployment

## Layout

```text
/root/comfy-pet-pipeline/       code, configuration templates, documentation, virtualenv
/data/comfy-pet-pipeline/
  assets/dressup/               wardrobe and pet PNG assets
  inputs/                       source uploads
  outputs/                      generated images, videos, and metadata
  archive/                      historical project media
```

The web service runs on `127.0.0.1:8766` under systemd. Nginx publishes it below `/pet-dressup/` and applies HTTP Basic authentication. ComfyUI runs on `127.0.0.1:8188` from the Conda environment named `comfyui`.

The dress-up URL is `http://39.106.120.253/pet-dressup/`; the user is `petadmin`. Retrieve the generated password over SSH with `cat /root/comfy-pet-pipeline/.web-password`.

ComfyUI is intentionally not public. Open an SSH tunnel from an administrator workstation, then browse to `http://127.0.0.1:8188`:

```bash
ssh -L 8188:127.0.0.1:8188 aliyun-gogelila
```

## Operations

```bash
systemctl status pet-pipeline
journalctl -u comfyui -f
systemctl status comfyui
journalctl -u pet-pipeline -f
systemctl restart pet-pipeline
curl http://127.0.0.1:8766/health
curl http://127.0.0.1:8188/system_stats
```

Run the cloud workflow:

```bash
cd /root/comfy-pet-pipeline
source .venv/bin/activate
python pipeline_cli.py pose
python pipeline_cli.py three-view
python pipeline_cli.py two-actions
```

Use the ComfyUI Conda environment directly:

```bash
conda activate comfyui
cd /root/comfy-pet-pipeline/ComfyUI
python main.py --listen 127.0.0.1 --port 8188
```

The default input is `/data/comfy-pet-pipeline/inputs/1.jpg`. Set `DASHSCOPE_API_KEY` in `/root/comfy-pet-pipeline/.env`; never put the key in source control.

## Capability boundary

The Alibaba Cloud host has one NVIDIA A10 with 23028 MiB VRAM. NVIDIA driver 580.173.02 and persistence mode are enabled. DashScope workflows, ComfyUI, ComfyUI-3D, Hunyuan3D, StableFast3D, and TripoSR can run on the server; some 3D workflows may require CPU offload because of the 24 GB VRAM limit. Blender and Unity desktop authoring remain workstation-oriented.
