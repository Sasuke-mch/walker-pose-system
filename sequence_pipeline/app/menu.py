from __future__ import annotations

from pathlib import Path

from .config import LoadedConfig


def interactive_request(config: LoadedConfig) -> dict:
    print("\n图像序列姿态估计程序\n")
    input_dir = Path(input("输入图片目录：").strip().strip('"'))
    output_raw = input("输出根目录（直接回车使用项目下 sequence_results）：").strip().strip('"')
    output_dir = Path(output_raw) if output_raw else config.project_root / "sequence_results"

    print("\n请选择模型，可输入多个编号，例如 1,2,5：")
    for i, key in enumerate(config.model_order, 1):
        print(f"{i}. {config.model(key).get('display_name', key)} [{key}]")
    print(f"{len(config.model_order) + 1}. 全部模型")
    raw = input("编号：").strip()
    if raw == str(len(config.model_order) + 1):
        models = "all"
    else:
        indexes = [int(x.strip()) for x in raw.split(",") if x.strip()]
        models = ",".join(config.model_order[i - 1] for i in indexes)

    save_vis = input("保存可视化图片？[y/N]：").strip().lower() in {"y", "yes", "1"}
    return {"input_dir": input_dir, "output_dir": output_dir, "models": models, "save_vis": save_vis}
