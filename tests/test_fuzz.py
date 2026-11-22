"""把独立暴力交叉校验包装为 pytest 测试（小规模随机案例）。"""

import random

from tests.fuzz_crosscheck import fuzz_once


def test_fuzz_matches_brute_force_200_cases():
    rng = random.Random(20261122)
    for _ in range(200):
        fuzz_once(rng)
