# 猫咪高还原模块化 3D 建模实施计划

## 1. 项目目标

把 `D:\gogelila\comfy\outputs\cat_poses_comfy\transparent` 下的猫咪图像制作成一套高还原、可继续编辑的模块化 3D 资产。核心要求不是“整只猫一次性图生 3D”，而是将猫拆成主体、五官、耳朵、脚掌、尾巴、胡须、毛发和材质层，分别生成或制作后在 Blender 中统一装配。

最终目标资产应满足：

- 主体比例、轮廓、坐姿、尾巴形态接近参考图。
- 花纹、眼睛、鼻口、脚掌、胡须和胸口长毛可单独编辑。
- 毛发质感接近原图的柔软长毛猫风格，而不是塑料玩具或低密度刺毛。
- 保留 Blender 展示版和实时 GLB/FBX 版本两条输出线。
- 所有可由建模大模型生成的模型部件优先使用大模型生成，再由 Blender 修整和装配。

## 2. 当前本地资产盘点

### 输入图像

目录：

`D:\gogelila\comfy\outputs\cat_poses_comfy\transparent`

已确认有 6 张透明 PNG：

| 文件 | 尺寸 | 用途建议 |
| --- | --- | --- |
| `pose_01_transparent.png` | 548 x 854 | 体态补充、轮廓参考 |
| `pose_02_transparent.png` | 781 x 746 | 主参考图，正面三分之二视角，五官和毛色最清楚 |
| `pose_03_transparent.png` | 685 x 723 | 侧向/姿态补充 |
| `pose_04_transparent.png` | 651 x 842 | 长毛和身体比例补充 |
| `pose_05_transparent.png` | 737 x 843 | 尾巴和站姿/坐姿体块补充 |
| `pose_06_transparent.png` | 784 x 795 | 背侧、尾巴和花纹补充 |

### 已有模型资产

当前工程已经有可复用的 3D 结果：

- `D:\gogelila\comfy\outputs\cat_3d_hunyuan2mini_turbo*`  
  已有 6 个 pose 的 Hunyuan2mini 整猫结果，可作为主体候选和比例对照。

- `D:\gogelila\comfy\outputs\cat_3d_triposr`  
  已有 TripoSR 快速重建结果，可作为粗轮廓和失败对照，不建议作为最终高质量主体。

- `D:\gogelila\comfy\outputs\cat_3d_blender_fur`  
  已有 Blender 毛发实验文件，可复用其中粒子毛发/曲线毛发参数。

- `D:\gogelila\comfy\outputs\cat_modular_model_mvp`  
  已有第一版模块化资产：主体、眼睛、鼻子、嘴垫、耳朵、脚掌、胡须和装配 blend。

- `D:\gogelila\comfy\outputs\cat_modular_model_art_v2`  
  当前最有价值的基线资产，包含：
  - 主体：`base_body/body_art_clean.glb`
  - 装配：`assembly/cat_modular_art_v2_assembled.blend`
  - 分部件：`parts/eyes`、`nose`、`muzzle`、`ears`、`paws`、`whiskers`
  - 锚点：`assembly/art_v2_anchors.json`
  - 预览：front / side / back 三视图

### Blender 环境

Blender 未加入 PATH，但本机可用：

`C:\Program Files\Blender Foundation\Blender 4.5\blender.exe`

后续脚本应显式使用这个路径调用 Blender。

## 3. 大模型优先级

优先级以“质量、可控性、能否接入当前本地流程”为准。

1. **Hunyuan3D 2.1 / 可访问时的 Hunyuan3D 2.5**
   - 主体 body shell、耳朵、尾巴、嘴鼻块、脚掌等主要 mesh 优先使用这一类大模型生成。
   - Hunyuan3D 2.1 适合作为本地可落地首选，因为官方提供 shape + PBR texture 管线。
   - Hunyuan3D 2.5 若有可调用服务或模型权重，应作为最高质量候选，特别适合主体和 PBR 材质。

2. **Hunyuan3D-Omni**
   - 如果能部署，优先用于受控部件生成，尤其是 bbox / point cloud / skeleton 约束下的身体、尾巴、耳朵、脚掌。
   - 价值在于减少单图生成时比例漂移和隐藏面胡乱补全。

3. **SPAR3D / Stable Fast 3D**
   - 用于快速生成候选，尤其补全背面或检查 Hunyuan 输出是否出现大形体错误。
   - 不作为最终质感首选，但适合做 A/B 对照。

4. **TripoSR**
   - 只作为快速粗模或轮廓参考。
   - 当前已有结果可保留，但不建议作为最终高还原模型源。

5. **Blender 程序化建模**
   - 对于胡须、眼球高光、毛发、眼睛玻璃层、毛发 cards 等细长/透明/重复结构，Blender 程序化结果会比大模型更稳定。
   - 这类对象虽然不完全由大模型生成，但它们是为了保证质量而必须独立制作的部件。

## 4. 工作目录

新一轮高质量版本建议独立输出，避免覆盖现有 MVP 和 art_v2：

```text
D:\gogelila\comfy\outputs\cat_modular_model_hq\
  references\
    original\
    cleaned\
    masks\
    crops\
    multiview_board\
  candidates\
    body\
    ears\
    muzzle_nose\
    paws\
    tail\
  base_body\
    body_raw.glb
    body_selected.glb
    body_clean.blend
    body_clean.glb
  parts\
    eyes\
    nose\
    muzzle\
    ears\
    paws\
    tail\
    whiskers\
  textures\
    pbr\
    fur_color\
    fur_masks\
    bake\
  fur\
    blender_hair.blend
    hair_cards.glb
  assembly\
    anchors.json
    cat_hq_assembled.blend
    cat_hq_render.blend
    cat_hq_realtime.glb
    previews\
  reports\
    model_candidates.md
    qa_report.md
    build_manifest.json
```

## 5. 实施阶段

### 阶段 A：参考图整理和语义拆分

目标：把 6 张透明猫图整理成可供大模型和 Blender 使用的参考集。

工作：

- 选 `pose_02_transparent.png` 为主参考图。
- 其他 pose 用于补充身体侧面、尾巴形态和花纹连续性。
- 生成以下输入图：
  - `clean_alpha`：清理边缘透明像素和黑边。
  - `no_whisker`：去除胡须线，避免大模型把胡须生成为碎片。
  - `no_face_detail`：弱化眼睛、鼻口和嘴线，只保留主体体积。
  - `body_mask`、`head_mask`、`ear_mask`、`muzzle_mask`、`paw_mask`、`tail_mask`、`fur_mask`。
  - 部件裁剪图：眼睛、耳朵、鼻子/嘴垫、前脚、尾巴。

验收：

- alpha 边缘干净。
- 胡须和嘴边细线不会进入主体生成图。
- 毛色和花纹参考没有被过度模糊。

### 阶段 B：主体 Body Shell 大模型生成

目标：生成高质量、无五官干扰的主体模型。

工作：

- 使用 `no_face_detail` / `no_whisker` 图输入 Hunyuan3D 2.1 或更高质量模型。
- 至少生成 6-12 个 body 候选，记录 seed、输入图、模型版本和参数。
- 只评估身体大形体：头身比例、胸口长毛体积、四肢位置、尾巴根部、坐姿重心。
- 不接受眼睛、鼻子、胡须、嘴线等被主体模型错误固化。
- 对最佳 body 做 Blender 清理：
  - 删除 floating fragments。
  - 修正嘴鼻区域不需要的凹凸。
  - 平滑异常褶皱。
  - 保持坐标规范：面朝 `-X`，高度 `+Z`，宽度 `Y`。

验收：

- 轮廓接近 pose_02，侧面能参考其他 pose 合理补全。
- 主体不是球形玩具感，胸口和脸颊有长毛体积。
- 脚、尾巴和耳朵根部位置稳定。
- mesh 适合后续贴附部件和毛发。

### 阶段 C：独立部件生成与制作

目标：所有高风险细节都脱离主体，变成可替换部件。

#### 1. 眼睛

- 优先使用 Blender 生成干净眼球、虹膜、瞳孔、高光和透明角膜。
- 眼睛纹理可从参考图采样或由图像模型生成 2D 虹膜贴图。
- 不建议直接依赖图生 3D 大模型生成眼睛，因为容易出现黑洞、歪斜、贴图糊。

验收：左右对称，眼神接近原图，瞳孔和高光清楚。

#### 2. 鼻子与嘴垫

- 使用鼻口裁剪图输入 Hunyuan3D / Omni 生成候选小部件。
- 同时保留 Blender 程序化版本作为高稳定 fallback。
- 鼻子采用柔和粉色 PBR 材质，嘴垫为浅白/奶油色，嘴线用细曲线或贴图表达。

验收：鼻子不黑、不碎，嘴垫不穿脸，正面表情接近参考图。

#### 3. 耳朵

- 若主体耳朵质量好，只添加耳内片和耳缘毛。
- 若主体耳朵变形，则用 Hunyuan/Omni 单独生成左右耳或耳朵对，再在 Blender 中替换。

验收：耳尖锐但不破碎，耳内粉色层自然，左右角度可微调。

#### 4. 脚掌与脚垫

- 脚掌体积来自主体或大模型候选。
- 脚趾分隔、粉色脚垫、可选小爪子作为独立 mesh。
- 对前脚优先还原，因为正面最容易看到。

验收：脚垫位置贴合，不漂浮，不穿模，近景可看。

#### 5. 尾巴

- 优先测试大模型尾巴候选；若尾巴环纹或弯曲错误，用 Blender 曲线扫掠生成尾巴体块，再贴图/烘焙环纹。
- 尾巴毛发单独 mask，长度略高于身体短毛。

验收：尾巴粗细、弯曲、环状花纹接近参考图。

#### 6. 胡须

- 不使用大模型生成最终胡须。
- 在 Blender 中用 Bezier 曲线生成，每侧 3-5 根，独立材质和 bevel depth。
- 胡须根部绑定到嘴垫 anchor。

验收：胡须是细线，不是片状碎 mesh；转视角不穿脸。

### 阶段 D：锚点系统与 Blender 装配

目标：所有部件用锚点和法线装配，而不是手动猜位置。

工作：

- 复用并升级 `cat_modular_model_art_v2\assembly\art_v2_anchors.json` 的结构。
- 每个 anchor 保存：
  - `position`
  - `normal`
  - `local_axes`
  - `scale_hint`
  - `part_name`
  - `manual_offset`
- 对主体 mesh 做 raycast 定位，自动放置眼睛、鼻子、嘴垫、耳内片、脚垫、胡须根部。
- Blender 脚本装配时保留每个部件为独立 Collection：
  - `CAT_BODY`
  - `CAT_EYES`
  - `CAT_MUZZLE`
  - `CAT_EARS`
  - `CAT_PAWS`
  - `CAT_TAIL`
  - `CAT_WHISKERS`
  - `CAT_FUR`

验收：

- 重新运行脚本可稳定生成同一装配结果。
- 每个部件可单独隐藏、替换和导出。
- 部件不穿模、不漂浮。

### 阶段 E：材质、PBR 和花纹还原

目标：恢复参考图的奶油白长毛、棕色面罩和尾巴环纹。

工作：

- 主体使用 Hunyuan3D Paint / TexGen / PBR 管线生成 base color。
- 对脸部、耳朵、尾巴、背部花纹进行投影修正。
- PBR 至少保留：
  - base color
  - roughness
  - normal 或 bump
  - optional AO
- 毛发颜色从主体贴图采样，避免统一白毛覆盖花纹。
- 对眼鼻嘴脚垫使用独立材质，不混入主体贴图。

验收：

- 正面棕色面罩、额头条纹、尾巴环纹清晰。
- 白毛区域不是纯白塑料，有轻微色彩层次。
- 眼睛、鼻子和嘴垫有合适高光，但不过分卡通。

### 阶段 F：毛发系统

目标：在 Blender 展示版中实现可信的猫毛质感。

工作：

- 使用两层毛发：
  - 身体短密底毛：短、密、轻微 roughness。
  - 胸口/脸颊/耳缘/尾巴局部长毛：更长、更蓬松、方向受 mask 控制。
- 生成毛发 mask：
  - `fur_body_short`
  - `fur_chest_long`
  - `fur_cheek_long`
  - `fur_ear_edge`
  - `fur_tail`
  - `fur_exclude_eyes_nose_mouth_pads`
- 毛发颜色跟随主体贴图或 vertex color。
- 实时版另做 hair cards / mesh fur，避免 GLB 导出后毛发丢失。

验收：

- 毛发不遮挡眼睛、鼻子和嘴垫。
- 胸口有长毛体积，脸颊和耳缘有柔软边缘。
- 花纹仍然可见，不被白色毛发全部盖住。

### 阶段 G：质量评估与迭代

每轮输出必须生成：

- 正面 render。
- 侧面 render。
- 背面 render。
- 与 pose_02 的正面叠图对比。
- 部件穿模检查报告。
- 模型候选评分表。

评分维度：

| 维度 | 权重 | 判定 |
| --- | --- | --- |
| 主体轮廓还原 | 25% | 头身比例、胸口、脚、尾巴 |
| 五官还原 | 20% | 眼距、眼神、鼻口比例 |
| 花纹还原 | 20% | 面罩、额头纹、尾巴环纹 |
| 毛发质感 | 20% | 蓬松度、密度、方向、颜色跟随 |
| 模块化可编辑性 | 10% | 是否可单独替换部件 |
| 导出稳定性 | 5% | blend/GLB 是否可打开 |

必须解决的问题：

- 当前 `art_v2` 前脸偏玩具化、眼鼻嘴比例过于卡通。
- 主体花纹和毛发层次不足。
- 胸口长毛体积还不够像参考图。
- 需要更高质量的主体大模型候选，而不是继续在低保真主体上堆部件。

## 6. 交付物

第一阶段交付：

- `references` 清理图、mask、部件裁剪图。
- 主体候选比较报告。
- 最佳主体 `body_clean.glb` / `body_clean.blend`。

第二阶段交付：

- 独立部件模型：
  - eyes
  - nose
  - muzzle
  - ears
  - paws
  - tail
  - whiskers
- `anchors.json`
- 初版装配 `cat_hq_assembled.blend`

第三阶段交付：

- Blender 毛发展示版 `cat_hq_render.blend`
- 实时预览版 `cat_hq_realtime.glb`
- front / side / back / beauty render
- `qa_report.md`
- `build_manifest.json`

## 7. 风险与处理

| 风险 | 处理 |
| --- | --- |
| 单图大模型无法准确补背面 | 使用 6 张 pose 作为多视角约束；必要时用 SPAR3D 或 Hunyuan3D-Omni 补点云/约束 |
| 胡须被生成成碎片 | 主体输入去胡须；胡须只用 Blender 曲线 |
| 眼睛变成贴图糊点 | 眼睛独立建模，不依赖主体贴图 |
| 毛发导出 GLB 后消失 | 保留 Blender 展示版；实时版另做 hair cards/mesh fur |
| 大模型输出过度卡通 | 用 pose_02 做正面叠图评分，低分主体直接淘汰 |
| Hunyuan3D 2.5 无本地权重/接口 | 本地优先使用 Hunyuan3D 2.1 或现有 Hunyuan2mini 管线，2.5 作为可用时的高质量外部候选 |

## 8. 推荐执行顺序

1. 建立 `outputs\cat_modular_model_hq` 工作目录。
2. 生成 6 张输入图的清理版、去胡须版、无五官主体版和 mask。
3. 用 Hunyuan3D 2.1/2.5 生成主体候选，并和现有 Hunyuan2mini、art_v2 主体对比。
4. 选出主体后清理 mesh，固定坐标和比例。
5. 生成或制作鼻口、耳朵、脚掌、尾巴候选部件。
6. 用 Blender 程序化制作眼睛、胡须和基础毛发。
7. 生成 `anchors.json`，脚本化装配所有部件。
8. 做 PBR/贴图/毛发 mask 修正。
9. 输出正面/侧面/背面预览和叠图评分。
10. 根据评分迭代主体或部件，直到达到高还原度。

## 9. 下一步建议

下一步不直接重建整只猫，而是先执行“阶段 A + 阶段 B”：

- 创建 `cat_modular_model_hq` 目录。
- 从 6 张透明 PNG 生成干净参考集。
- 调用可用的大模型管线生成多个主体候选。
- 与当前 `cat_modular_model_art_v2` 作正面和侧面对比。

通过主体候选评审后，再进入部件生成和 Blender 毛发装配。这样能避免在不够好的主体上反复修局部，成本最低，质量也最稳。

