# 2026-08-10 目录迁移记录

分支：`refactor/project-layout`

## 已迁移

- 生产 Python 代码：根目录模块迁到 `src/comfy_pet_pipeline/`，根目录保留轻量兼容入口。
- 页面：迁到 `web/platform/`。
- Profile JSON：迁到 `profiles/`。
- 自研 ComfyUI 节点：迁到 `extensions/comfyui/`。
- 测试、工具、依赖、文档与部署文件：分别迁到 `tests/`、`tools/`、`requirements/`、`docs/`、`ops/`。
- ComfyUI 运行时：迁到 `/opt/comfyui/current`，根目录保留兼容软链接。
- 第三方 `ComfyUI-3D-Pack`：迁到 `/opt/comfyui/custom-nodes/`。
- 机器环境配置：迁到 `/etc/comfy-pet-pipeline/`，文件权限为 `0600`；旧根目录副本保存在 `/etc/comfy-pet-pipeline/archive/root-config-pre-layout-20260810/` 供回滚。

## 已归档

- `.codex-backups` → `/data/comfy-pet-pipeline/archive/backups/code-history-20260810`
- `.codex-validation` 和旧 `deploy/staging` → `/data/comfy-pet-pipeline/archive/validation/`
- 旧 `backups`、`deploy/backups`、根目录 84 个 `.bak/.backup` 文件 → `/data/comfy-pet-pipeline/archive/backups/`
- `cat/`、`dog/` → `/data/comfy-pet-pipeline/archive/legacy-inputs/`
- `experiments/UnityCatCharacter` → `/data/comfy-pet-pipeline/archive/experiments/`
- 未被当前代码引用的 `vendor/` → `/data/comfy-pet-pipeline/archive/upstream/vendor-20260810`

## 已删除

- `.codex-staging`：约 17 GB、224149 个临时文件，可由历史变更重新生成。
- `.codex-deploy`：约 6.7 MB 的部署中间副本。
- 旧 `diagnostics/__pycache__`、根目录 `__pycache__` 和 `.pytest_cache`。

服务与 Nginx 的迁移前副本位于 `/data/comfy-pet-pipeline/archive/backups/20260810-pre-project-layout/`。
