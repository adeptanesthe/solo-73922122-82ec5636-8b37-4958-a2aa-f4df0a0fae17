"""调用本地 /repair 接口的示例请求脚本。

用法:
    python3 examples/call_repair.py [base_url]

成功后把返回的补线掩膜写到 examples/output/repaired_mask.png，
并在终端打印每对端点的判定（apply / review）与总灰度代价。
"""

from __future__ import annotations

import base64
import json
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))


def main(base_url: str = "http://127.0.0.1:8000") -> None:
    with open(os.path.join(HERE, "film_gray.png"), "rb") as f:
        image_bytes = f.read()
    with open(os.path.join(HERE, "film_mask.png"), "rb") as f:
        mask_bytes = f.read()
    with open(os.path.join(HERE, "pairs.json"), "r", encoding="utf-8") as f:
        pairs = f.read()

    resp = httpx.post(
        f"{base_url.rstrip('/')}/repair",
        files={
            "image": ("film_gray.png", image_bytes, "image/png"),
            "mask": ("film_mask.png", mask_bytes, "image/png"),
        },
        data={"pairs": pairs},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    print(json.dumps(
        {k: v for k, v in body.items() if k != "mask_png_base64"},
        ensure_ascii=False, indent=2,
    ))

    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "repaired_mask.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(body["mask_png_base64"]))
    print("saved repaired mask ->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
