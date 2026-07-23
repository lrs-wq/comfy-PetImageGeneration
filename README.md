# Pet Media Pipeline

This workspace contains one production-capable cloud pipeline and several local GPU/3D experiments.

## Production flow

1. `pipeline_cli.py pose` turns a pet photo into transparent cartoon poses with Qwen Image Edit.
2. `pipeline_cli.py three-view` creates front, side, and back references.
3. `pipeline_cli.py eat-video` creates an eating keyframe and Wan image-to-video result.
4. `pipeline_cli.py two-actions` runs the pose, eating-video, and running-video stages together.
5. `pet_dressup_mvp/` provides the browser dress-up editor and optional Qwen refinement.

On the server, code and documentation live in `/root/comfy-pet-pipeline`; images, videos, inputs, and outputs live in `/data/comfy-pet-pipeline`.

The server ComfyUI runtime uses the Conda environment `comfyui`. The separate `.venv` at the project root is only for the web application and DashScope command-line workflows.

## Non-production experiments

- `ComfyUI/` and the three `ComfyUI-DashScope-*` packages are workflow experiments.
- `run_comfy3d_*`, Hunyuan, Blender, and Unity files are 3D experiments requiring a GPU or desktop application.
- Historical generated media remains under `outputs/` and is deployed only to the data disk.

See `docs/SERVER_DEPLOYMENT.md` for the server layout and operations.
