# Pet Dressup MVP

This is a 2D pet dress-up application with an optional DashScope image-refinement API.

## Run

```powershell
cd D:\gogelila\comfy\apps\dressup
python -m uvicorn server:app --host 127.0.0.1 --port 8766
```

Open:

`http://127.0.0.1:8766/`

## Structure

- `data/avatar_manifest.json`: canonical pet image, anchors, and occlusion overlays.
- `data/wardrobe_manifest.json`: dress-up items, slots, anchors, and scale rules.
- `assets/pets/`: pet avatar image.
- `assets/overlays/`: head and front-paw occlusion layers.
- `assets/items/`: transparent PNG wardrobe items.
- `scripts/generate_demo_assets.py`: regenerates demo wardrobe and overlay PNGs.
- `server.py`: serves the app and calls Qwen Image Edit for final outfit fusion.

Runtime media can be kept outside the code directory. The server recognizes:

- `PET_PIPELINE_DATA_ROOT`: data root used for outputs.
- `PET_DRESSUP_ASSET_ROOT`: PNG asset directory mounted at `/assets`.
- `PET_DRESSUP_OUTPUT_ROOT`: refinement output directory.
- `DASHSCOPE_API_KEY`: preferred API-key source.
- `DASHSCOPE_API_KEY_FILE`: optional fallback key file.

## Current MVP

- Real-time Canvas compositing.
- Slots: back accessory, top, collar, eyewear, hat.
- Upload replacement pet image for preview.
- Per-item fine tuning for scale, x/y offset, and rotation.
- Transparent or colored export to PNG.
- AI fusion render: sends the current canvas to `qwen-image-edit-plus` and asks the model to redraw clothing, fur occlusion, lighting, and edges as a unified final image.

For production, the upload step should call your ComfyUI/Qwen cartoonization workflow first, then generate an avatar manifest and occlusion overlays for that generated pet.
