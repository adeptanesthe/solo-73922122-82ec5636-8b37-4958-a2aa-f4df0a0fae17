"""流式 /repair 演示请求：先预览指定端点对，再收整图结果。

用法:
    python3 examples/call_repair_stream.py [base_url]

请求体与普通 /repair 相同，额外加两个表单字段：
    stream=true         返回 NDJSON（每行一个事件）
    preview_pairs=2,1   要先看的端点对编号（1 基，顺序即预览顺序）

示例数据含一个与预览对没有直接/间接候选冲突的独立端点组（第 3 对），
它会在预览事件发出之后才参与最优求解；最终 result 事件与普通请求等价。
"""

from __future__ import annotations

import base64
import json
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))


def main(base_url: str = "http://127.0.0.1:8000") -> None:
    with open(os.path.join(HERE, "stream_gray.png"), "rb") as f:
        image_bytes = f.read()
    with open(os.path.join(HERE, "stream_mask.png"), "rb") as f:
        mask_bytes = f.read()
    with open(os.path.join(HERE, "stream_pairs.json"), "r", encoding="utf-8") as f:
        pairs = f.read()

    url = f"{base_url.rstrip('/')}/repair"
    files = {
        "image": ("stream_gray.png", image_bytes, "image/png"),
        "mask": ("stream_mask.png", mask_bytes, "image/png"),
    }
    data = {"pairs": pairs, "stream": "true", "preview_pairs": "2,1"}

    result = None
    with httpx.stream("POST", url, files=files, data=data, timeout=30) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event"] == "preview":
                print(
                    f"[preview] pair {event['pair_id']}: {event['decision']}"
                    f"（{event['path_count']} 条最低代价路径）"
                )
                for path in event["paths"]:
                    print(f"    turns={path['turns']} cost={path['cost']}"
                          f" cells={len(path['new_cells'])}")
            else:
                result = event
                print(
                    f"[result] status={event['status']}"
                    f" min_total_cost={event.get('min_total_cost')}"
                    f" apply={event.get('applied_pair_count')}"
                    f" review={event.get('review_pair_count')}"
                )

    if result is None or "mask_png_base64" not in result:
        print("无整图掩膜可写（可能为无解或输入错误）")
        return
    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stream_repaired_mask.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(result["mask_png_base64"]))
    print("saved repaired mask ->", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
