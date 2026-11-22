"""生成一组可用于演示的底片灰度图与二值掩膜。

场景（64x64，8 位灰度）:
* 背景灰度约 210，模拟底片底色；
* 掩膜上有两段清晰走线，各自在中间褪色断开，形成 4 个断点（2 对）；
* 第 1 对断口两侧灰度分布有唯一最低代价直线路径 -> 直接补；
* 第 2 对断口放在不同行不同列、灰度均匀，存在多条同代价路径 -> 待复核。

输出:
    examples/film_gray.png   8 位灰度原图
    examples/film_mask.png   二值掩膜（只含端点，不含补线）
    examples/pairs.json      请求体 pairs 字段内容
"""

from __future__ import annotations

import json
import os
import random

from PIL import Image

SIZE = 64
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    rng = random.Random(42)
    gray = bytearray(210 for _ in range(SIZE * SIZE))
    # 轻微噪点，让灰度总和更真实（确定性随机）
    for i in range(len(gray)):
        gray[i] = max(0, min(255, gray[i] + rng.randint(-6, 6)))

    mask = bytearray(SIZE * SIZE)

    def set_mask(x: int, y: int) -> None:
        mask[y * SIZE + x] = 255

    # ---- 第 1 对：水平断口，y=18，断点 (12,18) 与 (24,18) ----
    # 两端各保留一截清晰走线
    for x in range(6, 13):
        set_mask(x, 18)
    for x in range(24, 31):
        set_mask(x, 18)
    # 断口内做一条明显更暗（灰度更低）的直线通道，制造唯一最优
    for x in range(13, 24):
        gray[18 * SIZE + x] = 40 + rng.randint(0, 8)
    pair1 = {"a": [12, 18], "b": [24, 18], "region": [10, 14, 26, 22]}

    # ---- 第 2 对：L 形断口，断点 (44,14) 与 (54,26)，区域内严格均匀灰度 ----
    # 均匀灰度下所有合法路径同代价、且路径不一致 -> 待复核
    for y in range(8, 15):
        set_mask(44, y)
    for x in range(54, 60):
        set_mask(x, 26)
    for y in range(12, 29):
        for x in range(42, 57):
            gray[y * SIZE + x] = 210
    pair2 = {"a": [44, 14], "b": [54, 26], "region": [42, 12, 56, 28]}

    Image.frombytes("L", (SIZE, SIZE), bytes(gray)).save(
        os.path.join(OUT_DIR, "film_gray.png"), format="PNG", optimize=True
    )
    Image.frombytes("L", (SIZE, SIZE), bytes(mask)).save(
        os.path.join(OUT_DIR, "film_mask.png"), format="PNG", optimize=True
    )
    with open(os.path.join(OUT_DIR, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump([pair1, pair2], f, ensure_ascii=False, indent=2)

    print("written:", os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
