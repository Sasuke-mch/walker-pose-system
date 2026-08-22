# 输出格式

## raw_predictions.json

模型原始输出。不同模型字段不同，保留模型特有信息。

## common_predictions.json

统一结构至少包含：

```text
frame_index
file_name
person_id
bbox_xyxy
bbox_score
keypoints[name, x, y, score]
```

对 17 点模型使用 COCO17 名称。其他模型暂时使用 `keypoint_0` 等模型原生编号。
