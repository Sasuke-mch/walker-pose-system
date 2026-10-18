# Project status

## 已验证链路

### YOLO26x-pose

已完成：

- 图片序列输入
- 视频输入
- 模拟实时丢帧
- Docker GPU 推理
- 骨架视频与 JSONL 输出

一次 MOT17-05 30 帧测试结果：

- processed_frames: 30
- dropped_frames: 0
- effective_fps: 6.169
- mean_model_ms: 129.451
- mean_roundtrip_ms: 149.287

### YOLO26x + PMPose

已完成：

- YOLO26x 人体检测服务
- PMPose-b 姿态服务
- 矩形 mask
- 多帧完整链路

一次 MOT17-05 5 帧小规模测试结果：

- processed_frames: 5
- dropped_frames: 0
- effective_fps: 2.612
- mean_model_ms: 316.743
- mean_roundtrip_ms: 366.208

这组 5 帧结果仅用于验证链路，不能单独用于稳定性能结论。

## 精度实验摘要

OCHuman 2500 张验证图像上的现有结果：

| 方法 | AP | AP50 | AP75 | AR |
|---|---:|---:|---:|---:|
| YOLO26x-pose | 0.4712 | 0.7986 | 0.4749 | 0.6027 |
| YOLO26x + PMPose | 0.6472 | 0.8010 | 0.7353 | 0.7794 |
| YOLO26x + ProbPose | 0.5726 | 0.7820 | 0.6529 | 0.7041 |
| YOLO26x + BBoxMaskPose | 0.5839 | 0.7482 | 0.6745 | 0.7634 |
| YOLO26x + Sapiens2 | 0.5971 | 0.7862 | 0.6780 | 0.7412 |

具体实验口径以 `benchmarks/` 中的原始汇总文件和实验记录为准。
