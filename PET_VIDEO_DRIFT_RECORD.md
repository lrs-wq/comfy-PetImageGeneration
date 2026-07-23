# 猫咪视频链路跑偏产物记录

## 1. 记录目的

本文档单独记录一次跑偏实验产物，避免把不稳定结果混入正式链路文档。

实验目录：

```text
D:/gogelila/comfy/outputs/cat_two_action_pipeline_wan22_side_eat
```

本次实验使用：

| 模块 | 模型 / 能力 |
| --- | --- |
| 姿态图生成 | DashScope API `qwen-image-edit` |
| 视频生成 | DashScope API `wan2.2-i2v-flash` |
| 动作公共姿态 | `pose_02_transparent.png` |
| 输出视频 | 吃饭、奔跑 |

说明：这里记录的是 API 实验产物，不是“完全本地开源权重”产物。

## 2. 跑偏产物清单

### 2.1 跑偏姿态图

以下 3 张姿态图存在明显身份或姿态跑偏，暂不进入正式示例：

| 文件 | 问题概述 |
| --- | --- |
| [`pose_01_transparent.png`](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_01_transparent.png) | 脸部灰色面罩过重，和参考猫身份特征不一致 |
| [`pose_04_transparent.png`](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_04_transparent.png) | 动作方向和身体比例偏离预期，姿态不像“伸展” |
| [`pose_05_transparent.png`](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_05_transparent.png) | 抬爪姿态可读，但脸部、眼睛和花纹稳定性不足 |

六姿态总览：

![六姿态总览](outputs/cat_two_action_pipeline_wan22_side_eat/poses/cat_6_poses_transparent_sheet.png)

跑偏姿态图：

![pose_01](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_01_transparent.png)

![pose_04](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_04_transparent.png)

![pose_05](outputs/cat_two_action_pipeline_wan22_side_eat/poses/transparent/pose_05_transparent.png)

### 2.2 单独记录的视频

本次视频基于 `pose_02_transparent.png` 继续生成。虽然吃饭视角已经调整为偏侧，但整组视频仍归为实验记录，暂不作为正式链路样例。

| 视频 | 路径 | 记录原因 |
| --- | --- | --- |
| 吃饭视频 | [`eat_5s_30fps_16x9.mp4`](outputs/cat_two_action_pipeline_wan22_side_eat/videos/eat/final/eat_5s_30fps_16x9.mp4) | 侧面吃饭动作可用，但身份稳定性仍受上游姿态图影响 |
| 奔跑视频 | [`run_5s_30fps_16x9.mp4`](outputs/cat_two_action_pipeline_wan22_side_eat/videos/run/final/run_5s_30fps_16x9.mp4) | 动作方向可读，但身体比例、腿部运动和身份一致性仍需优化 |

吃饭视频预览：

![吃饭视频预览](outputs/cat_two_action_pipeline_wan22_side_eat/preview/eat_contact_sheet.jpg)

<video controls width="720" src="outputs/cat_two_action_pipeline_wan22_side_eat/videos/eat/final/eat_5s_30fps_16x9.mp4"></video>

奔跑视频预览：

![奔跑视频预览](outputs/cat_two_action_pipeline_wan22_side_eat/preview/run_contact_sheet.jpg)

<video controls width="720" src="outputs/cat_two_action_pipeline_wan22_side_eat/videos/run/final/run_5s_30fps_16x9.mp4"></video>

## 3. 跑偏原因简述

### 3.1 六姿态图是独立生成的

当前六张姿态图分别调用图像编辑模型生成。模型每次都会重新解释“同一只猫”的身份特征，因此容易出现：

- 脸部花纹漂移
- 眼睛颜色变化
- 灰色面罩被强化
- 身体比例变胖或变幼态
- 姿态和描述不完全匹配

### 3.2 缺少强身份约束

当前只靠 prompt 描述身份，没有使用更强的身份保持机制，例如：

- 猫咪专属 LoRA
- IP-Adapter / reference adapter
- 多参考图 identity embedding
- CLIP / DINO 相似度筛选
- 局部脸部特征约束

所以模型会把“可爱 3D 毛绒猫”风格放大，弱化原图中的具体身份。

### 3.3 缺少结构控制

姿态图没有接入 Canny、Depth、Pose、Mask 等结构控制，导致动作姿态容易被模型自由发挥。

典型表现：

- `pose_04` 没有稳定形成预期伸展动作
- `pose_05` 虽然有抬爪，但身体和脸部稳定性不足
- 姿态图之间不像同一个角色的连续资产

### 3.4 视频模型会放大首帧问题

Wan 图生视频主要依赖首帧。如果首帧身份或比例已经有偏差，视频会继续放大这些问题。

吃饭视频虽然偏侧视角更符合需求，但仍存在：

- 身体比例偏圆
- 花纹细节有漂移
- 动作过程中脸部状态有变化

奔跑视频则更明显，因为奔跑需要四肢运动，模型更容易在腿部和身体比例上发生漂移。

## 4. 后续解决方法

### 4.1 生成多候选并筛选

每个姿态不要只生成 1 张，建议每个姿态生成 4-8 张候选：

```text
pose_01_candidate_01.png
pose_01_candidate_02.png
...
pose_01_candidate_08.png
```

然后人工或自动筛选，只把通过 QC 的图进入 `accepted/`。

### 4.2 增加姿态图 QC

建立固定 QC 规则：

- 是否仍是同一只猫
- 脸部花纹是否接近参考图
- 眼睛颜色是否稳定
- 身体比例是否接近参考图
- 是否只有一只猫
- 是否全身完整
- 是否没有多余道具、文字、水印

不通过的结果进入：

```text
outputs/.../poses/rejected/
```

### 4.3 使用更强身份保持方案

优先级从低到高：

1. 强化 prompt 中的身份描述
2. 多参考图输入
3. IP-Adapter / reference adapter
4. 基于该猫训练轻量 LoRA
5. 本地 Qwen-Image-Edit / FLUX Kontext 加结构控制

最稳的方案是训练猫咪专属 LoRA 或使用多参考身份保持模型。

### 4.4 增加结构控制

对六个姿态图使用固定结构条件：

- Canny 控制轮廓
- Depth 控制体积关系
- Mask 限制主体区域
- 手工绘制姿态草图作为参考

这样能减少 `pose_04`、`pose_05` 这类动作跑偏。

### 4.5 视频阶段使用通过 QC 的首帧

视频不要直接使用自动生成的第一张首帧。建议流程改为：

```text
姿态候选 -> QC 通过 -> 动作首帧候选 -> QC 通过 -> 视频生成
```

对于吃饭视频，首帧必须检查：

- 侧面或三分之四侧面
- 猫粮碗位置正确
- 猫嘴接近猫粮
- 全身完整
- 只有一只猫

对于奔跑视频，首帧必须检查：

- 侧面
- 四肢清晰
- 尾巴完整
- 身体不过胖
- 方向明确

### 4.6 视频生成多次取优

同一首帧建议至少生成 2-4 个视频候选，选择动作最稳定的一段。

重点比较：

- 身份是否漂移
- 动作是否明确
- 腿部是否合理
- 是否出画
- 是否有闪烁或变形

## 5. 处理结论

本次产物保留为实验记录，不进入正式展示链路。

可复用部分：

- `pose_02_transparent.png` 可以继续作为侧面动作输入候选
- 吃饭视频的侧面构图方向是对的
- Wan2.2 API 调用链路已经跑通

需要重做部分：

- 六姿态图需要加入多候选和 QC
- `pose_01`、`pose_04`、`pose_05` 需要重新生成
- 奔跑视频需要更好的侧面奔跑首帧和更多候选
- 正式文档只能记录 QC 通过后的结果

