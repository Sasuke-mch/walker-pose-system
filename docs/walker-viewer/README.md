# 交互式助步器运动查看器

这是一个纯静态 Three.js 页面。它读取 `data/motion.json`，不需要 Python 后端，适合部署到 GitHub Pages。

## 本地预览

在仓库根目录运行：

```powershell
python -m http.server 8000 --directory .
```

然后打开：

```text
http://127.0.0.1:8000/docs/walker-viewer/
```

不能直接双击 `index.html`，因为浏览器通常禁止本地页面通过 `fetch` 读取 JSON。

## GitHub Pages

在 GitHub 仓库设置中选择 `Settings → Pages → Deploy from a branch`，分支选择 `master`，目录选择 `/docs`。部署后地址为：

```text
https://Sasuke-mch.github.io/walker-pose-system/walker-viewer/
```

## 重新导出数据

使用 `realtime_app/tools/export_interactive_walker_viewer.py` 从动态地面、严格/显示双目、交互审计和粗助步器模型生成公开数据。导出保持当前帧直出，不做插值或显示平滑。

页面使用 jsDelivr 上固定版本的 Three.js，因此观看者需要能够访问互联网。

## 不通过GitHub直接发送

运行 `realtime_app/tools/build_standalone_walker_viewer.py` 可生成单个HTML文件。该文件已内嵌样式、程序和448帧运动数据，接收者双击即可打开；Three.js仍从jsDelivr加载，因此首次打开需要联网，但不需要GitHub、Python或本项目目录。

当前生成文件位于：

```text
research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/standalone_web/walker_motion_viewer.html
```
