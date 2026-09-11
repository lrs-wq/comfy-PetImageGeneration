# 0010：舔毛输出保持原始时间方向

- 状态：accepted
- 日期：2026-09-11
- 影响节点：`comfy.media.grooming_forward_timeline`、`comfy.api.local_video`、`comfy.module.pipeline`

## 背景

任务 `20260905T083916Z-0124acd4b4` 的原始 SCAIL 视频为 81 帧、16 FPS。v2.26.2 引入的导出规则丢弃前半段，生成 `40→80→40` 回文序列，GIF/APNG/白底 MP4 因而在后半程倒放。原首尾 MAE 门只验证接缝，无法判断时间方向，甚至优先放行回文动画。

## 决策

- 以原始生成视频的源帧顺序作为时间权威。抠图前不改顺序，在全部真实帧提取完成后选择导出时间方案。
- 首先保留完整正向视频。只有全身及头部/抬爪区域的首尾接缝不合格时，搜索连续的 0.5–3 秒自然周期；头爪运动幅度、边界差分方向余弦至少 0.2、全部内部与最后回到开头的接缝共同检查。接缝不超过相邻差异中位数 1.5 倍或 0.5 的绝对误差地板。
- 保持既有动画 100 帧/20 FPS 与视频 81 帧/16 FPS，各自时长不变。两种采样共用源窗口和整数周期重复次数，动画速度限制为 0.85–1.15 倍；视频因原时长多 1/16 秒有相应微小速率差。逐帧像素来自真实源 matte，只允许正向采样及明确周期边界回到周期开头，不能反转、融合 RGB 或用静态帧替代动作。
- 导出与 API 接受均重建期望的两套源帧映射验证顺序，不能只相信 forward 标签或 closed_loop 布尔值。原回文映射、单一格式倒序、越界和未声明周期重置均拒绝。
- 找不到安全自然周期时保留完整正向源动作，并记录接缝告警；视觉接缝不触发生成首帧降级。运行完整性与正向源顺序仍失败关闭，与完成优先视觉规则一致。
- 选中原始 SCAIL 视频另存 `grooming_subject_replacement_source.mp4` 并记录 SHA-256，白底导出不能覆盖这一原件。

## 后果与验证

原任务离线回放选出源帧 67–78、重复 6 次、动画速度 0.9 倍；头爪边界方向余弦 0.285436、最大循环接缝比 1.109283，最终 GIF 首尾比 1.360783。APNG 的 100 帧逐字节匹配选定源 matte，MP4 的 81 帧全部按最近源图匹配预期映射。短周期可能减少动作变化；无自然周期时仍可能有循环跳变，但不能以再次倒放来隐藏。原始生成视频保留可供复查。此确定性媒体回放之外，每次修改仍必须通过猫狗四动作 live smoke。

## 证据

- `src/comfy_pet_pipeline/media/grooming_temporal.py`
- `src/comfy_pet_pipeline/pipeline/subject_replacement.py`
- `src/comfy_pet_pipeline/api/local_video_parts/subject_replacement.py`
- `src/comfy_pet_pipeline/api/local_video_parts/health.py`
- `tests/unit/test_grooming_forward_timeline.py`
- `tests/integration/test_scail2_subject_replacement_masking.py`
- `docs/api/LOCAL_VIDEO_API.md`
