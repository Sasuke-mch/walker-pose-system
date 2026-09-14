# 局部平面短时传播：数据契约与验收边界

每帧平面只在该帧左相机坐标系中表达：`n^T X + d = 0`。输出没有永久世界坐标系。

`direct_planes.jsonl` 的 `status="direct"` 记录必须来自独立的地面身份与双目几何验收，包含：

```json
{"local_ground_state_version":"local_ground_state_v1","frame_index":12,"frame_id":"pair_0012.png","observation_state":"direct","source":"independent_direct_plane_input","reason":null,"quality":{},"plane_in_left_camera":{"normal_toward_camera_unit":[0,1,0],"offset_mm":1000}}
```

非直接观测帧记录 `observation_state="unavailable"`，不得附带伪平面。每个输出都有 `source`、`reason` 和 `quality`。

`relative_poses.jsonl` 只接受相邻帧相对位姿，并强制下列字段：

```json
{"from_frame_index":12,"to_frame_index":13,"status":"accepted","R_to_from":[[1,0,0],[0,1,0],[0,0,1]],"q_to_from":[0,0,0],"feature_domain":"static_background_excluding_person_walker_ground","static_3d_correspondence_count":42,"ransac_inlier_count":35,"ground_region_used_for_motion":false}
```

该契约要求位姿估计的 3-D 对应仅来自静态背景，且已排除人体、助步器和被拟合地面区域。若字段缺失、不是 `accepted`，或使用了地面区域，传播程序将拒绝该边。

若 `X_t = R X_{t-1}+q`，则：

\[
n_t=R n_{t-1},\qquad d_t=d_{t-1}-n_t^Tq.
\]

状态只有 `direct`、`propagated` 和 `unavailable`。传播年龄到上限、位姿缺失或不合格、帧序不相邻都必须变为 `unavailable`；不允许静默外推。

评估命令以源端 `direct` 平面和仅由中间静态背景位姿组成的预测，与目标端独立 `direct` 平面比较法向夹角和相机-平面距离差。目标端不参与预测。这是内部一致性，不能称为真实地面精度、真实接触或步态准确度。
