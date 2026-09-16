# 0014：犬动作保持驱动起点，身份参考不锁定旧姿态

- 状态：accepted
- 日期：2026-09-16
- 补充：0013

## 问题与证据

犬任务 `20260915T091236Z-0dfe6ada1d` 的进食原始 SCAIL 输出已跟随完整白模低—高—低周期，发布阶段却把首帧几何导图选中的第 67 帧当作动画起点。81 帧环移后，末尾 14 帧被提前，形成约 1.75 秒的原始节奏延迟。处理后的白底 MP4 与 APNG 都受影响；不能把白底发布 MP4 当作未加工 SCAIL 视频分析。

站立的 Qwen 提示没有包含实际首帧详细姿态，仅引用未定义的“指定姿态”，同时要求保留 canonical 整体轮廓。上传图的肩后翘尾因此保留，SCAIL 又按白模生成髋部低位尾巴，形成双尾。独立对照补足文字姿态、重定位单尾后消除了该样本的双尾。

## 决策

- 犬进食发布从 SCAIL 原始第 0 帧开始，几何静帧选择不得改变完整驱动的时间原点。猫的既有接触导图起点合同不变。
- 犬 Qwen 提示必须传入 `qwen_first_frame_spec`。进食明确侧身低头、头左臀右；站立明确白模实际为坐姿、胸高相机和髋部近爪高度的单尾。身份保持被毛、脸部和解剖比例，不要求保留 canonical 旧轮廓、肢体位置或尾巴位置。
- 站立 SCAIL 继承相同低位单尾约束。仍保持 canonical 单图、白模仅进 SCAIL、Qwen 固定五轮与完成优先规则。
- GIF、APNG、MP4 使用各动作实际播放时长，APNG 不得把 10 秒进食误报为全局默认 5 秒。

## 验证与边界

原任务真实视频以起点 0 离线重导出，头部轮廓高度的高位区间与准备白模同为第 16–63 帧，最佳相关偏移由 14 帧变为 0。该测量是局部图像信号，不是通用关节识别。站立独立生成对照由双尾变为髋部低位单尾；提示不能保证所有犬型的解剖质量，既有运动门也不等价于完整白模保真度审核。

本次保留站立现有后半段 5 秒驱动窗口，完整原片为约 10 秒；片段范围与本次错误帧序、双尾修复独立。原任务复跑及猫狗四动作 live smoke 必须在交付前完成。

## 证据

- `src/comfy_pet_pipeline/api/local_video.py`
- `src/comfy_pet_pipeline/api/local_video_parts/prompting_workflows.py`
- `src/comfy_pet_pipeline/api/local_video_parts/subject_replacement.py`
- `src/comfy_pet_pipeline/pipeline/subject_replacement.py`
- `tests/integration/test_local_video_qwen_service_inputs.py`
- `tests/integration/test_local_video_qwen_prompt_workflows.py`
