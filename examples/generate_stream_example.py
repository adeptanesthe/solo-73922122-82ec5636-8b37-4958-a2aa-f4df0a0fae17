"""生成流式演示用的灰度图、掩膜与端点对。

场景（64x64，8 位灰度），与 film_* 系列相互独立：
* 第 1 对：水平断口 (10,12)-(22,12)，断口内有明显更暗的直线通道
  -> 唯一最低代价路径，预览即可判为可补（apply）；
* 第 2 对：L 形断口 (38,10)-(48,22)，区域内灰度均匀
  -> 多条同代价路径，预览判为待复核（review）；
* 第 3 对：独立端点组 (6,48)-(16,48)，远离前两对、补线区内互不冲突
  -> 与预览对没有直接/间接候选冲突，流式求解会把它推迟到预览之后。

输出:
    examples/stream_gray.png   8 位灰度原图
    examples/stream_mask.png   二值掩膜（只含端点，不含补线）
    examples/stream_pairs.json 请求体 pairs 字段内容
"""

from __future__ import annotations

import json
import os

from PIL import Image

SIZE = 64
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    gray = bytearray([210] * (SIZE * SIZE))
    mask = bytearray(SIZE * SIZE)

    def set_mask(x: int, y: int) -> None:
        mask[y * SIZE + x] = 255

    def set_gray(x: int, y: int, value: int) -> None:
        gray[y * SIZE + x] = value

    # ---- 第 1 对：水平断口，暗通道铺出唯一最优直线 ----
    for x in range(4, 11):
        set_mask(x, 12)
    for x in range(22, 29):
        set_mask(x, 12)
    for x in range(11, 22):
        set_gray(x, 12, 40)
    pair1 = {"a": [10, 12], "b": [22, 12], "region": [8, 8, 24, 16]}

    # ---- 第 2 对：L 形断口，区域内严格均匀灰度 ----
    for y in range(4, 11):
        set_mask(38, y)
    for x in range(48, 55):
        set_mask(x, 22)
    for y in range(8, 25):
        for x in range(36, 51):
            set_gray(x, y, 210)
    pair2 = {"a": [38, 10], "b": [48, 22], "region": [36, 8, 50, 24]}

    # ---- 第 3 对：独立端点组，远离前两对 ----
    for x in range(2, 7):
        set_mask(x, 48)
    for x in range(16, 21):
        set_mask(x, 48)
    for x in range(6, 16):
        set_gray(x, 48, 60)
    pair3 = {"a": [6, 48], "b": [16, 48], "region": [4, 44, 18, 52]}

    Image.frombytes("L", (SIZE, SIZE), bytes(gray)).save(
        os.path.join(OUT_DIR, "stream_gray.png"), format="PNG", optimize=True
    )
    Image.frombytes("L", (SIZE, SIZE), bytes(mask)).save(
        os.path.join(OUT_DIR, "stream_mask.png"), format="PNG", optimize=True
    )
    with open(os.path.join(OUT_DIR, "stream_pairs.json"), "w", encoding="utf-8") as f:
        json.dump([pair1, pair2, pair3], f, ensure_ascii=False, indent=2)

    print("written: stream_gray.png, stream_mask.png, stream_pairs.json")


if __name__ == "__main__":
    main()
