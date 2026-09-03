# Comfy Pet Pipeline Agent Instructions

## Knowledge bootstrap

- Start with `python3 /root/project-knowledge/bin/kg context .`.
- Read `docs/knowledge/CURRENT.md` before changing behavior.
- Treat source code, configuration, tests, and generated evidence as more authoritative than summaries.
- Follow graph `evidence` paths instead of loading every document into context.

## Task routing

- Simplified GIF API: `docs/api/SIMPLE_PET_GIF_API.md` and `src/comfy_pet_pipeline/api/gif_api.py`.
- Local video API: `docs/api/LOCAL_VIDEO_API.md` and `src/comfy_pet_pipeline/api/local_video.py`.
- Pet generation flow: `docs/architecture/PET_TWO_ACTION_PIPELINE.md` and `src/comfy_pet_pipeline/pipeline/`.
- Runtime layout and data boundaries: `docs/operations/PROJECT_LAYOUT.md`.
- Deployment: `docs/operations/SERVER_DEPLOYMENT.md` and `ops/`.

## Knowledge update contract

- Update `docs/knowledge/GRAPH.json` when a service, API, model, config, deployment, event, or relationship changes.
- Update `docs/knowledge/CURRENT.md` when supported behavior, active constraints, compatibility rules, or known risks change.
- Append one JSON object to `docs/knowledge/CHANGES.jsonl` for a semantic change; omit formatting-only changes.
- Add an ADR under `docs/knowledge/DECISIONS/` for a durable architectural choice or reversal.
- Keep knowledge changes in the same commit as the code they describe.
- Run `python3 /root/project-knowledge/bin/kg refresh .` and `python3 /root/project-knowledge/bin/kg check .` before finishing.

## Safety and verification

- After every repository modification, run `bash ops/scripts/verify-change.sh`; it runs the default pytest suite, pipeline import verification, and the SHA-pinned historical input gallery. Quality differences in that gallery are diagnostic only. Do not report a change complete while the command is failing or has not been run.
- Materialize the server-only historical corpus once with `.venv/bin/python ops/scripts/run-historical-regression-gallery.py --materialize-only`; never commit the copied media.
- After every repository modification, once the fast check passes and the live queue is clear, run `.venv/bin/python ops/scripts/run-live-stability-gallery.py --suite smoke`. The mandatory matrix is exactly one cat job with `eating + grooming` and one dog job with `eating + standing`; every one of those four actions must run once and publish a readable primary animation. The script rejects an incomplete or duplicate default matrix and still attempts the second species if the first fails, while preserving an overall failing result. It asserts completion and media readability, not visual quality. Explicit `--case` runs are supplementary and do not replace the mandatory matrix. Do not report a change complete while this live smoke is missing or failing.
- Never add `.env` values, credentials, raw uploads, generated media, model weights, or `/data` contents to knowledge files.
- Preserve compatibility entrypoints at the repository root unless the task explicitly removes them.
