# 萌宠猫咪 Qwen Image LoRA 测试报告

测试日期：2026-07-15  
生产接口：`http://39.106.120.253/api/local-video`  
接口版本：`2.2.0`

## 1. LoRA 文件

- 原始文件：`D:/download/萌宠猫咪_Qwen Image_萌宠猫星.safetensors`
- 文件大小：236,116,856 bytes
- SHA-256：`bfc6b0e4a87c6b4a68a9d24f5bf35da1142b431871327f1340ba35af176c7176`
- 服务器文件：`/data/comfy-pet-pipeline/models/comfyui/loras/pet3d_cat_qwen_image_r16.safetensors`
- 结构：1440 个 LoRA 张量，rank 16，覆盖 Qwen Image transformer blocks
- 元数据：空；无法从文件内确认训练基座版本、触发词、推荐强度和许可证

服务器与本地 SHA-256 一致，safetensors payload 长度和文件长度一致。

## 2. 兼容性结论

LoRA 可以由当前 ComfyUI `LoraLoaderModelOnly` 正常识别和加载。

- baseline Qwen：720 个唯一 patch 目标
- 叠加风格 LoRA：840 个唯一 patch 目标
- ComfyUI 日志中没有 `key not loaded`、异常或 OOM
- 当前 Lightning 加速 LoRA继续保留，风格 LoRA串联在其后

增加的唯一 patch 数不是 LoRA 的总模块数；大部分模块与 Lightning LoRA指向同一基础权重并叠加，其他模块增加新的 patch 目标。

## 3. 固定 seed A/B

测试输入：服务器 `/data/comfy-pet-pipeline/inputs/1.jpg`  
seed：`20260715`  
提示词：线上标准卡通身份图提示词  
唯一变量：风格 LoRA strength

| 版本 | Comfy执行时间 | 观察 |
|---|---:|---|
| baseline | 40.97秒 | 卡通化成立，但头脸偏窄、眼睛和毛发质感不够统一 |
| strength 0.65 | 21.96秒（模型已缓存） | 圆脸、大眼、柔软长毛和萌系3D风格最均衡 |
| strength 0.85 | 22.20秒（模型已缓存） | 风格更强，但身份泛化风险和模板化倾向更高 |

选择 `0.65` 作为当前生产强度。0.65 和 0.85 都通过线上严格绿幕检测和透明抠图。

## 4. 四动作首帧测试

使用 strength 0.65，对同一张标准卡通身份图生成：

1. 奔跑
2. 吃饭
3. 喝水
4. 趴卧

结果：

- 四张均成功生成完整主体并完成透明抠图。
- 圆脸、大眼、长毛、粉鼻、额头和眼周花纹保持较一致。
- 未生成服装、眼镜、项圈或多余角色。
- 喝水图生成了单个粉色舌头和单个水碗。
- 奔跑、喝水、趴卧构图可直接用于视频首帧。

发现的问题：

- 原始照片没有展示完整身体，模型在未知区域补出了较明显的虎斑身体纹路；LoRA 有强化这种卡通虎斑模板的倾向。
- 吃饭首帧出现少量碗外/嘴边猫粮，后续可通过动作专项负面提示词修正。
- 毛发下缘存在少量绿幕反光，当前抠图可处理，但仍有进一步 despill 空间。

## 5. 完整5秒奔跑链路

任务号：`20260715T020936Z-f147ad9342`

完整执行：

```text
真实照片
-> LoRA标准卡通身份图
-> LoRA奔跑首帧
-> Wan2.2 TI2V 5B，121帧
-> 透明抠图和主体平移校正
-> 512x512 GIF/APNG，100帧，20fps，5秒
```

任务约 3分31秒完成，无 API、CUDA 或文件错误。

格式验证：

| 项目 | GIF | APNG |
|---|---:|---:|
| 尺寸 | 512×512 | 512×512 |
| 帧数 | 100 | 100 |
| 时长 | 5000ms | 5000ms |
| 帧时长 | 50ms | 50ms |
| 无限循环 | 是 | 是 |
| 四角透明 | 是 | 是 |
| 柔和半透明 Alpha | GIF格式不支持 | 是 |
| 文件大小 | 6,523,282 bytes | 16,974,209 bytes |

视觉检查：

- 10个均匀采样时间点中，头型、双眼、鼻子和口鼻区域没有融化或独立漂移。
- 视频保持同一萌系3D风格，没有切换成写实或2D画风。
- 上半身和头部保留自然奔跑动作；没有重新启用像素冻结或首帧覆盖。
- `head_lock.mode=disabled`，`locked_pixels=0`。
- 主体平移校正仍只使用每帧最大1.25像素的有限速度补偿；奔跑动作本身保留约20–34像素的自然姿态变化范围。

## 6. 生产部署状态

当前生效配置：

```text
QWEN_STYLE_LORA=pet3d_cat_qwen_image_r16.safetensors
QWEN_STYLE_LORA_STRENGTH=0.65
QWEN_STYLE_PROFILE=pet3d_cat_lora_v1
```

`GET /api/local-video/health` 返回：

```json
{
  "status": "ok",
  "version": "2.2.0",
  "qwen_style": {
    "enabled": true,
    "profile": "pet3d_cat_lora_v1",
    "lora": "pet3d_cat_qwen_image_r16.safetensors",
    "strength": 0.65
  }
}
```

生产代码备份：`/root/comfy-pet-pipeline/local_video_api.py.bak-v210-20260715`

回滚只需清空 `QWEN_STYLE_LORA` 或恢复 profile 为 baseline，再重启 `local-video-api`；不需要删除模型文件或修改 ComfyUI。

## 7. 结论

此 LoRA 可以用于当前 Qwen-Image-Edit-2511 + Lightning 链路，推荐强度为 `0.65`。它能明显增强目标萌系3D卡通风格的一致性，Wan 2.2 能稳定继承该风格，完整透明GIF链路可正常完成。

不建议把强度直接提高到 0.85 以上。当前下一项优化应是收紧“未知身体区域不得新增虎斑”和“吃饭首帧不得出现碗外食物”的提示词，而不是再提高 LoRA 强度。
