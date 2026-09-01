# YOLO 检测权重

钻石检测（SAHI）模块需要本目录下的 `best.pt`。

- 开发/打包：将训练好的 YOLO 权重命名为 `best.pt` 放在此处，或于 `app_config.json` 的 `yolo_path` 指向可访问路径。
- 机台包：打包脚本会复制为 `dist/缺陷分类系统/detect_weights/best.pt`；也可在机台上直接覆盖该文件热更新。
- 打包时若找不到权重，组装将失败（避免打出缺少检测模型的空包）。

`best.pt` 体积较大，默认不纳入 git；请从训练产出或备份目录自行放置。
