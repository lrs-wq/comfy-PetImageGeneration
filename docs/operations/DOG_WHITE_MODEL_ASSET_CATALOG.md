# 犬类基础白膜资产目录

当前目录版本为 `dog_white_model_catalog_v1 / 20260824-v1`，配置文件位于 `profiles/morphology/dog_white_model_catalog_v1.json`。

## 分类

- `neutral_dog`（中性犬）：保留当前生产正在使用的 `eating` 和 `standing` 两条白膜。原文件路径和 SHA-256 不变，不迁移、不重编码。
- `small_dog`（小型犬）：2026-08-24 新上传六条源视频；其中 `eating` 与 `standing_head_turn` 已归一化为测试平台的“吃饭 / 站立”动作，其余四条继续保留为待处理源素材。
- `medium_dog`（中型犬）：测试平台暂时显式复用当前 `neutral_dog` 的两条已验证资产，不移动、不重编码原文件。

服务器资产根目录：

```text
/data/comfy-pet-pipeline/profile_assets/dog_white_model_catalog/20260824-v1/
```

小型犬源视频保持上传字节不变，以英文动作名落盘；配置中同时记录原中文文件名、SHA-256、分辨率、帧率、帧数、时长和音轨信息。

## 测试平台启用状态

生产/API 自动路由仍保持关闭，`neutral_dog` 继续作为现有犬类回退类别。测试平台通过已有的 `subject_profile` 字段显式选择以下配置，不新增或改变生产 API 参数：

- 中型犬：`dog_medium_white_model_test_v1`，动作 `eating / standing`，资产别名为 `neutral_dog`；
- 小型犬：`dog_small_white_model_test_v1`，动作 `eating / standing`，使用 `small_dog/runtime-v1`。

小型犬两条测试动作已经完成：中心正方形裁剪、`512×512 / 81 帧 / 16 FPS`、移除音轨、首尾闭环检查、透明几何参考与 SHA-256 冻结。上传宠物图仍是身份、犬种、体型、脸部、毛色、花纹和毛长的唯一外观标准；白膜只提供动作几何和时序。

其余小型犬动作进入生产前仍必须逐动作完成：

1. 选择有效动作周期并归一化为 `512×512 / 81 帧 / 16 FPS`；
2. 移除音轨、非动作水印和非几何背景信息；
3. 验证主体完整性、首尾循环、食盆/饮水接触及动作语义；
4. 生成冻结摘要、几何参考和动作专用审计；
5. 上线显式形态路由后，才允许将小型犬上传任务切换到该资产组。

在完成上述步骤前，禁止把其余原始视频直接作为 SCAIL driver，也禁止根据文件夹名称或自动分类改变现有生产任务路由。测试平台选择中型犬或小型犬时，以该显式选择为准；分类结果只保留作审计，不得静默改回默认类别。
