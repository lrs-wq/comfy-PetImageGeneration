# 犬类基础白膜资产目录

当前目录版本为 `dog_white_model_catalog_v1 / 20260824-v1`，配置文件位于 `profiles/morphology/dog_white_model_catalog_v1.json`。

## 分类

- `neutral_dog`（中性犬）：保留当前生产正在使用的 `eating` 和 `standing` 两条白膜。原文件路径和 SHA-256 不变，不迁移、不重编码。
- `small_dog`（小型犬）：2026-08-24 新上传六条源视频，包含 `sleeping`、`standing_head_turn`、`drinking`、`running`、`toileting`、`eating`。
- `medium_dog`（中型犬）：预留，尚未上传资产。

服务器资产根目录：

```text
/data/comfy-pet-pipeline/profile_assets/dog_white_model_catalog/20260824-v1/
```

小型犬源视频保持上传字节不变，以英文动作名落盘；配置中同时记录原中文文件名、SHA-256、分辨率、帧率、帧数、时长和音轨信息。

## 当前启用状态

本次只完成资产上传、完整性校验和分类登记，不改变生产路由。`neutral_dog` 继续作为现有回退类别；`small_dog` 标记为 `source_uploaded_pending_runtime_normalization`，在进入生产前还必须逐动作完成：

1. 选择有效动作周期并归一化为 `512×512 / 81 帧 / 16 FPS`；
2. 移除音轨、非动作水印和非几何背景信息；
3. 验证主体完整性、首尾循环、食盆/饮水接触及动作语义；
4. 生成冻结摘要、几何参考和动作专用审计；
5. 上线显式形态路由后，才允许将小型犬上传任务切换到该资产组。

在完成上述步骤前，禁止把六条原始视频直接作为 SCAIL 生产 driver，也禁止根据文件夹名称自动改变现有任务路由。
