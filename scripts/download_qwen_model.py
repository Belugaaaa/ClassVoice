from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_TARGET_DIR = "models/qwen3-0.6b"


def download_from_modelscope(model_id: str, target_dir: str, revision: str | None) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "请先安装 ModelScope SDK：pip install -r requirements-modelscope.txt"
        ) from exc

    kwargs = {
        "model_id": model_id,
        "local_dir": target_dir,
    }
    if revision:
        kwargs["revision"] = revision

    return snapshot_download(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="从魔搭 ModelScope 下载 Qwen 模型权重。")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"魔搭模型 ID，默认：{DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help=f"本地保存目录，默认：{DEFAULT_TARGET_DIR}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选：指定分支或版本。",
    )
    args = parser.parse_args()

    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"ModelScope model: {args.model_id}")
    print(f"Target dir: {target_path}")
    model_dir = download_from_modelscope(args.model_id, str(target_path), args.revision)
    print(f"Done: {model_dir}")
    print("在 ClassVoice 页面侧边栏将模型名/本地路径填写为：")
    print(target_path.as_posix())


if __name__ == "__main__":
    main()
