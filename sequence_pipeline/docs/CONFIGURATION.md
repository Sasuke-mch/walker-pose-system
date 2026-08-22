# 配置字段

每个模型主要字段：

- `docker_image`：镜像名。
- `repo.host_path`：相对项目根目录的模型源码目录。
- `repo.container_path`：容器内挂载位置。
- `weight.host_path`：主机权重路径。
- `weight.container_dir`：权重父目录在容器中的挂载位置。
- `workdir`：Docker 工作目录。
- `command`：容器内命令列表。
- `required_paths`：预检必须存在的文件。
- `expected_outputs`：判断阶段是否成功的文件。

不要把 `D:\...` 写进模型脚本。需要本地绝对路径时写到 `configs/local.json`。
