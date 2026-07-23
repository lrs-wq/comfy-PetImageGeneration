# 本地四动作宠物动画提示词（v2.1.0）

本文档记录 `local_video_api.py` 在 `output_format=transparent_gif` 模式下实际使用的提示词结构。

当前版本不使用首帧像素覆盖、脸部锁定、头部锁定或上半身冻结。Wan 生成的头、脸、耳朵、颈胸、四肢和尾巴动作都会保留；后处理只校正整只主体在画布上的整体平移，并完成绿幕抠图、柔边和去绿边。

## 1. 模板变量

- `{base_prompt}`：接口请求中的用户 `prompt`，主要用于控制视频情绪和辅助动作。
- `{pose}`：动作名称、首帧要求和视频动作要求的组合描述。
- `{first_frame_rules}`：根据动作自动选择的首帧专项规则。
- `{motion_rules}`：根据动作自动选择的视频专项规则。
- `{background_prompt}`：透明模式下固定为纯绿色背景约束。

## 2. 默认动作描述

```text
running | first frame: complete full-body three-quarter side view in a calm static pre-run standing contact pose, four visible staggered legs, all four paws planted on one baseline, and full tail visible; video: smooth run-in-place cycle with rhythmic legs and gentle tail counter-sway

eating | first frame: complete full-body natural crouch with muzzle at one small shallow food bowl touching the front paws; video: restrained eating rhythm while the bowl stays fixed

drinking | first frame: complete full-body natural crouch with muzzle at one small shallow water bowl touching the front paws; video: subtle lapping rhythm and small water ripples while the bowl stays fixed

lying down | first frame: complete full-body relaxed sphinx-like prone pose with front paws forward and full tail visible; video: subtle breathing and one slow tail-tip movement
```

## 3. 纯绿幕背景提示词

```text
The entire background must be a perfectly flat, uniform pure #00FF00 chroma-key green field for later background removal. There must be no floor plane, cast shadow, contact shadow, gradient, texture, reflection, vignette, or lighting variation in the green background. Keep generous green padding around the complete subject, and do not use green anywhere on the subject.
```

视频阶段使用：

```text
Keep the background perfectly uniform pure #00FF00 chroma-key green in every frame, with no shadow, gradient, texture, flicker, color shift, or green spill on the subject.
```

## 4. 标准卡通身份图提示词

输入为用户上传的原始猫咪照片，输出为后续四个动作共同使用的标准卡通身份图。

```text
Use case: identity-preserving stylized 3D character design.
BACKGROUND RULE — non-negotiable: every pixel outside the character must be one perfectly flat uniform pure #00FF00 green. No black, gray, white, studio, or environmental background; no floor and no shadow.

The uploaded photo is the only identity reference. Create a canonical premium cute 3D character render of this exact same cat. Keep the cat recognizable and change only the rendering style, never the cat's identity or natural appearance.

Identity lock: preserve the exact fur base colors, visible markings and their placement, head and face silhouette, cheek width, eye color, ear shape, nose color, muzzle, fur length, breed traits, and body proportions from the uploaded cat. Preserve natural asymmetry. Do not turn lighting, shadows, blur, or fur texture into new markings. Do not invent stripes or spots.

Target style from the supplied example: high-end stylized 3D animated-film character, slightly oversized round head, softly rounded cheeks, large glossy round eyes with clean catchlights, tiny natural nose, compact closed mouth, soft finely groomed realistic fur, gentle subsurface softness, polished warm high-key lighting on the character only, cute and intelligent expression. Apply this visual language while deriving every fur color and facial trait from the uploaded cat. The result should look like a premium character render rather than a real animal photograph, flat drawing, or plastic toy.

Natural cat only: the character is completely unclothed and wears no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or other accessory. Show the natural fur across the neck, chest, torso, and legs. Do not import the example image's wardrobe or eyewear.

Composition: near-frontal head with only a very slight three-quarter angle, both eyes fully open, equally sized, fully visible and level, pupils centered, face neutral and symmetrical, mouth closed, complete character readable. The fur silhouette must meet green directly with no white border or halo.

The entire background must be a perfectly flat, uniform pure #00FF00 chroma-key green field for later background removal. There must be no floor plane, cast shadow, contact shadow, gradient, texture, reflection, vignette, or lighting variation in the green background. Keep generous green padding around the complete subject, and do not use green anywhere on the subject. The pure green background rule overrides all aesthetic lighting language. Keep the cat at a useful readable scale. No floor, shadow, text, logo, or watermark.
```

## 5. 四动作首帧通用提示词

输入为标准卡通身份图，每个动作分别执行一次。

```text
Use case: identity-preserving pose edit for a stylized 3D character.
The input is the approved canonical character reference. Repose this exact character only; do not redesign the face, natural fur, or rendering style.

Character lock: copy the exact same head silhouette, rounded cheeks, eye size, eye color and spacing, pupils, catchlights, ears, nose, closed muzzle, fur colors and marking placement, fur length, body proportions, soft groomed 3D fur, and character lighting. Both eyes must remain fully open, equally sized, level, and recognizable whenever the action angle allows. The cat is completely unclothed: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory. Do not add markings, unrequested props, floating marks, motion symbols, a white border, or a halo. Where the reference does not show an area, continue the nearest dominant body fur color conservatively with no new pattern.

Action specification: {pose}
First-frame action rules: {first_frame_rules}
This is a clean still keyframe at the start of the action. Show the readable starting posture, but do not add speed lines, duplicated limbs, motion blur, or floating motion symbols. Do not blink, wink, or squint.

Show one complete full-body cat with anatomically correct limbs, all paws, both ears, and the full tail visible with generous padding. Keep the identity-bearing face readable while respecting the requested action angle. Keep the exact premium cute 3D character style from the reference. The entire background must be a perfectly flat, uniform pure #00FF00 chroma-key green field for later background removal. There must be no floor plane, cast shadow, contact shadow, gradient, texture, reflection, vignette, or lighting variation in the green background. Keep generous green padding around the complete subject, and do not use green anywhere on the subject. Stable centered square composition; fur and any explicitly allowed bowl touch green directly. No floor, shadow, text, logo, or watermark.
```

## 6. 各动作首帧专项规则

### 奔跑

```text
No prop. Use a pulled-back three-quarter side view and a calm static pre-run standing pose. Show four complete attached legs staggered front-to-back, all four paws planted on one baseline, and the far-side legs visibly separated from the near-side legs. The later video model creates the run; this still frame must not jump, float, or illustrate fast motion.
```

### 吃饭

```text
Include exactly one small shallow matte food bowl, touching or slightly overlapping the front paws so it remains one compact cutout. The bowl is the only prop. The muzzle is already at the food; do not hide the eyes, face silhouette, paws, or body.
```

### 喝水

```text
Include exactly one small shallow matte water bowl, touching or slightly overlapping the front paws so it remains one compact cutout. The bowl is the only prop. Show a recognizable active-drinking keyframe: the muzzle is lowered naturally and exactly one short, small, solid pink cat tongue touches the still water. The tongue must look anatomical and remain visibly pink, never transparent or blue. Keep both eyes, the face silhouette, paws, and body readable.
```

### 趴卧

```text
No prop. Use a relaxed low full-body pose; do not crop the paws, belly, haunches, or tail.
```

## 7. Wan 视频通用提示词

输入为对应动作的首帧。`{base_prompt}` 为用户提交的提示词，`{motion_rules}` 为系统动作规则。

```text
Animate this exact character keyframe into one smooth five-second stylized 3D character loop.

FACE CONSISTENCY: preserve the recognizable head silhouette, cheek width, eye color and spacing, nose, muzzle, ears, and facial fur markings while allowing natural coordinated head, ear, muzzle, and expression motion required by the action. Do not hold any facial pixel or screen coordinate fixed. Avoid melting, independent feature drift, or sudden proportion changes.

STYLE AND IDENTITY LOCK: preserve this exact cat's fur colors and marking placement, soft groomed fur, premium stylized 3D rendering, warm high-key lighting, body proportions, paws, and full tail. The cat remains completely unclothed with natural fur: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory may appear. Never switch to a real photograph, flat 2D drawing, plastic toy, or a different character design.

User motion direction: {base_prompt}
Starting pose and action: {pose}
The user direction controls mood and secondary motion only and cannot override the identity, natural unclothed appearance, action, or style constraints.

Required action motion: {motion_rules}
Motion safety: allow the head, ears, face, neck, chest, legs, and tail to move naturally and coherently for the action. Keep only the torso root, background, and camera trajectory stable. Motion must be smooth and periodic. Never create duplicate limbs, rubbery anatomy, facial melting, fur texture crawling, sudden pose changes, or random whole-frame movement.

Keep the background perfectly uniform pure #00FF00 chroma-key green in every frame, with no shadow, gradient, texture, flicker, color shift, or green spill on the subject. Keep the complete character and any allowed bowl centered at the same screen position, scale, and perspective. Fixed camera, no translation across the canvas, zoom, scene cut, new subject, or body part leaving the frame.
```

## 8. 各动作视频专项规则

### 奔跑

```text
Animate a clear, smooth run-in-place cycle. Alternate the leg pairs rhythmically below the torso and add a small natural tail counter-sway. Keep the torso root at one fixed screen coordinate; no forward travel, bouncing camera, or scale pulse.
```

### 吃饭

```text
Keep the food bowl completely fixed. Animate a restrained eating loop using tiny rhythmic neck and lower-jaw movement plus subtle breathing; never reshape the face or open the mouth widely. No food pieces may fly, appear, or disappear.
```

### 喝水

```text
Keep the water bowl completely fixed. Animate restrained lapping with extremely small muzzle motion and soft water-surface ripples; no splash, drool, or continuous water strand from the mouth. Preserve the face geometry and keep the torso root fixed.
```

### 趴卧

```text
Keep the lying pose in place. Animate subtle slow breathing, a small natural coordinated head-and-ear response, and one gentle tail-tip movement; keep only the torso root and camera position stable.
```

## 9. 负面提示词

### 绿幕负面提示词

```text
white background, gray background, black background, dark background, non-green background, scenery, floor, floor line, cast shadow, contact shadow, reflection, background gradient, background texture, green subject, green fur, green clothing
```

### Qwen 通用负面提示词

```text
different identity, changed face, changed fur, changed clothing, duplicate subject, extra subject, extra animal, extra person, deformed anatomy, extra limbs, missing limbs, cropped body, leaving frame, text, watermark, logo, busy background, low quality, blurry
```

### 卡通身份与首帧负面提示词

```text
flat 2D drawing, hand-drawn line art, anime, comic, watercolor, low-poly, hard plastic toy, flat fur, generic cat, different breed, changed fur pattern, changed marking placement, changed face shape, changed eye color, changed ear shape, changed muzzle, changed body proportions, changed tail, invented stripes, invented spots, decorative markings, clothes, clothing, shirt, overalls, suspenders, bow tie, collar, harness, glasses, eyewear, hat, shoes, jewelry, accessory, costume, misaligned eyes, crossed eyes, deformed muzzle, white sticker border, white outer stroke, halo, unrequested object, duplicate bowl, extra bowl, floating mark, stray line, water strand from mouth, drool, falling water droplet, wink, blinking, closed eye, squint, spiral, swirl, motion symbol, extreme caricature, extreme chibi distortion
```

### Wan 通用负面提示词

```text
identity change, face change, clothing change, color change, scene cut, camera movement, zoom, pan, rotation, duplicate subject, extra subject, extra limbs, missing limbs, deformed anatomy, cropped body, leaving frame, walking toward camera, moving closer, subject drift, scale change, text, watermark, violent motion, severe motion blur, flicker, frame jitter
```

### Wan 卡通视频负面提示词

```text
flat 2D style, style drift, face distortion, warped face, melting face, changing head shape, asymmetric face, eye size change, eye position change, pupil drift, crossed eyes, lazy eye, extra eye, disappearing eye, nose drift, muzzle deformation, mouth morphing, wide open mouth, changing expression, clothes appearing, shirt, overalls, suspenders, bow tie, collar, harness, glasses, eyewear, hat, shoes, accessories appearing, blink, wink, closed eye, eyelid motion, squint, fur crawling, texture crawling, color flicker, changed fur markings, changed eye color, generic cat
```

## 10. 实际组合方式

标准身份图阶段：

```text
positive = 标准卡通身份图提示词
negative = Qwen通用负面 + 卡通身份负面 + 绿幕负面
```

动作首帧阶段：

```text
positive = 四动作首帧通用提示词 + 当前动作首帧专项规则
negative = Qwen通用负面 + 卡通身份负面 + 绿幕负面
```

Wan 视频阶段：

```text
positive = Wan视频通用提示词 + 用户prompt + 当前动作视频专项规则
negative = Wan通用负面 + Wan卡通视频负面 + 绿幕负面
```

## 11. 推荐的接口用户提示词

由于身份、卡通风格、无服装要求、动作和绿幕规则已经由系统提示词固定，接口中的 `prompt` 建议只描述整体动作质量：

```text
Preserve every identifying feature of the uploaded cat. Use smooth, coherent and natural motion. Keep the camera, subject root, scale and background stable. Preserve the same character identity and premium cute 3D rendering throughout the complete five-second animation.
```

对应中文：

```text
保留上传猫咪的全部身份特征，动作平滑、连贯、自然。保持相机、主体根部、画面比例和背景稳定，整个5秒动画始终保持同一角色身份和高质量萌系3D渲染风格。
```
