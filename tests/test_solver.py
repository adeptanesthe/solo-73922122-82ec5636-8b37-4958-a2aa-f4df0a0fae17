"""求解器单元测试：覆盖枚举、约束、代价优化与一致性判定。"""

import pytest

from app.solver import (
    MAX_PAIRS,
    MAX_REGION_SIDE,
    RepairError,
    enumerate_paths,
    solve,
    validate_pair_inputs,
)


def make_grid(width, height, fill=200):
    gray = [bytearray([fill] * width) for _ in range(height)]
    mask = [bytearray(width) for _ in range(height)]
    return gray, mask


def set_mask(mask, cells):
    for x, y in cells:
        mask[y][x] = 1


def set_gray(gray, values):
    for (x, y), v in values.items():
        gray[y][x] = v


def specs_from(payload, width, height):
    return validate_pair_inputs(payload, width, height)


def assert_legal(result, specs, mask0):
    """对返回的已补路径做独立的合法性复核。"""
    assert result.feasible
    applied = set()
    for detail in result.pair_details:
        if detail["decision"] != "apply":
            continue
        cells = [tuple(c) for c in detail["new_cells"]]
        # 不覆盖原走线
        for c in cells:
            assert mask0[c[1]][c[0]] == 0
        # 不与其他补线重叠
        assert not (applied & set(cells))
        # 不同补线之间不四邻接
        for c in cells:
            x, y = c
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + dx, y + dy)
                assert n not in applied
        applied.update(cells)


# ---------- 路径枚举 ----------

def test_enumerate_straight_line():
    paths = enumerate_paths((2, 5), (6, 5), 0, 0, 10, 10)
    assert [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)] in paths
    # 同行直线只有一条
    assert len(paths) == 1


def test_enumerate_includes_l_and_zigzags():
    paths = enumerate_paths((2, 2), (5, 6), 0, 0, 9, 9)
    # 两条 1 转弯 L 形
    assert any(p[0] == (2, 2) and p[-1] == (5, 6) for p in paths)
    assert any({(2, 2), (5, 2), (5, 6)} <= set(p) for p in paths)
    assert any({(2, 2), (2, 6), (5, 6)} <= set(p) for p in paths)
    # 2 转弯 H-V-H 与 V-H-V 各列/行都出现
    hvh = [p for p in paths if p[0] == (2, 2) and (4, 2) in p and (4, 6) in p]
    assert hvh
    vhv = [p for p in paths if (2, 4) in p and (5, 4) in p]
    assert vhv
    # 所有路径转弯 <= 2
    for p in paths:
        turns = 0
        prev = None
        for (x1, y1), (x2, y2) in zip(p, p[1:]):
            d = "v" if x1 == x2 else "h"
            if prev is not None and d != prev:
                turns += 1
            prev = d
        assert turns <= 2
    # 无重复
    keys = [tuple(p) for p in paths]
    assert len(keys) == len(set(keys))


def test_enumerate_region_boundary_limits_columns():
    # 补线区只有 3 列时，H-V-H 的中间列只能取区内列
    paths = enumerate_paths((1, 1), (3, 5), 1, 1, 3, 5)
    for p in paths:
        for x, y in p:
            assert 1 <= x <= 3 and 1 <= y <= 5


# ---------- 基础补线 ----------

def test_straight_horizontal_repair_minimal_cost():
    gray, mask = make_grid(12, 12, fill=200)
    set_mask(mask, [(2, 5), (6, 5)])  # 两个端点
    specs = specs_from([{"a": [2, 5], "b": [6, 5]}], 12, 12)
    result = solve(specs, mask, gray, 12, 12)
    assert result.feasible
    assert result.decisions == ["apply"]
    detail = result.pair_details[0]
    assert detail["turns"] == 0
    # 新增 3 个像素，每个 200
    assert detail["cost"] == 600
    assert detail["new_cells"] == [[3, 5], [4, 5], [5, 5]]
    assert_legal(result, specs, mask)


def test_cost_drives_choice_of_path():
    # L 形两条路径：经过便宜像素的那条应胜出且唯一
    gray, mask = make_grid(20, 20, fill=200)
    a, b = (3, 3), (8, 8)
    set_mask(mask, [a, b])
    # 拐角 (8,3) 路径上设置廉价像素；另一条拐角 (3,8) 保持 200
    set_gray(gray, {(4, 3): 10, (5, 3): 10, (6, 3): 10, (7, 3): 10, (8, 4): 10,
                    (8, 5): 10, (8, 6): 10, (8, 7): 10})
    specs = specs_from([{"a": list(a), "b": list(b)}], 20, 20)
    result = solve(specs, mask, gray, 20, 20)
    assert result.feasible
    assert result.decisions == ["apply"]
    cells = {tuple(c) for c in result.pair_details[0]["new_cells"]}
    assert (4, 3) in cells and (8, 4) in cells
    assert (3, 4) not in cells
    assert result.pair_details[0]["turns"] == 1
    # 9 个新增像素：8 个廉价像素(10) + 拐角 (8,3) 一个普通像素(200) = 280
    assert result.pair_details[0]["cost"] == 280
    assert_legal(result, specs, mask)


def test_does_not_overwrite_existing_trace():
    # 一条 1 转弯 L 形路径的拐角被独立旁线占据：候选既不能覆盖旁线，
    # 也不能与其四邻接，只能改走 2 转弯折线保持一格间距绕行
    gray, mask = make_grid(24, 24, fill=100)
    a, b = (3, 3), (9, 7)
    set_mask(mask, [a, b])
    # 独立旁线（与本对走线不相连）挡住拐角 (9,3) 及其邻接位
    set_mask(mask, [(8, 3), (9, 3)])
    specs = specs_from([{"a": list(a), "b": list(b),
                         "region": [2, 1, 10, 9]}], 24, 24)
    result = solve(specs, mask, gray, 24, 24)
    assert result.feasible
    detail = result.pair_details[0]
    barrier = {(8, 3), (9, 3)}
    # 所有备选路径都必须满足约束（无覆盖、无邻接旁线）
    alts = detail["alternatives"]
    assert alts
    for alt in alts:
        cells = {tuple(c) for c in alt["new_cells"]}
        assert not any(mask[y][x] for x, y in cells)
        for x, y in cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                assert (x + dx, y + dy) not in barrier
    assert_legal(result, specs, mask)


def test_new_pixels_touch_trace_only_at_own_endpoints():
    # 旁路有一段原走线；任何与其四邻接的候选都应被剔除
    gray, mask = make_grid(20, 20, fill=100)
    a, b = (3, 3), (3, 9)
    set_mask(mask, [a, b])
    # 在右侧 x=5 列放一段与潜在绕行路径四邻接的原走线
    set_mask(mask, [(5, y) for y in range(4, 9)])
    specs = specs_from([{"a": list(a), "b": list(b)}], 20, 20)
    result = solve(specs, mask, gray, 20, 20)
    assert result.feasible
    cells = {tuple(c) for c in result.pair_details[0]["new_cells"]}
    # 所有新增像素都不得与那段旁线四邻接
    for x, y in cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n not in (a, b):
                assert mask[n[1]][n[0]] == 0
    assert_legal(result, specs, mask)


# ---------- 一致性判定 ----------

def test_tie_between_two_paths_marks_review():
    gray, mask = make_grid(20, 20, fill=100)
    a, b = (3, 3), (8, 8)
    set_mask(mask, [a, b])
    specs = specs_from([{"a": list(a), "b": list(b)}], 20, 20)
    result = solve(specs, mask, gray, 20, 20)
    assert result.feasible
    # 均匀灰度下两条 L 形同代价，且大量 2 转弯路径同代价 -> 待复核
    assert result.decisions == ["review"]
    assert result.applied_cells == []
    detail = result.pair_details[0]
    assert detail["alternative_count"] >= 2
    # 所有路径的新增像素数都等于曼哈顿距离 (8-3)+(8-3)=9，均匀灰度下同代价
    costs = {alt["cost"] for alt in detail["alternatives"]}
    assert costs == {100 * 9}


def test_unique_optimum_marks_apply_even_with_many_candidates():
    gray, mask = make_grid(20, 20, fill=100)
    a, b = (3, 3), (8, 8)
    set_mask(mask, [a, b])
    # 让一条 L 形明显更便宜，形成唯一最优
    cheap = [(x, 3) for x in range(4, 9)] + [(8, y) for y in range(4, 8)]
    set_gray(gray, {c: 5 for c in cheap})
    specs = specs_from([{"a": list(a), "b": list(b)}], 20, 20)
    result = solve(specs, mask, gray, 20, 20)
    assert result.feasible
    assert result.decisions == ["apply"]
    assert result.pair_details[0]["turns"] == 1
    assert_legal(result, specs, mask)


# ---------- 多对冲突 ----------

def test_two_pairs_mutually_blocking_returns_no_solution():
    gray, mask = make_grid(10, 10, fill=100)
    # 两条水平端点上下紧邻（y=4 与 y=5），补线区只给各自一行：
    # 直线是唯一合法路径，而两条直线互相四邻接 -> 无全局可行方案
    set_mask(mask, [(1, 4), (5, 4), (1, 5), (5, 5)])
    payload = [
        {"a": [1, 4], "b": [5, 4], "region": [1, 4, 5, 4]},
        {"a": [1, 5], "b": [5, 5], "region": [1, 5, 5, 5]},
    ]
    specs = specs_from(payload, 10, 10)
    result = solve(specs, mask, gray, 10, 10)
    assert not result.feasible
    assert "冲突" in result.reason


def test_two_pairs_with_room_both_apply():
    gray, mask = make_grid(24, 24, fill=100)
    set_mask(mask, [(2, 3), (8, 3), (2, 15), (8, 15)])
    # 区域 y 范围互不相邻：对 1 到 y=6，对 2 从 y=13 起
    payload = [
        {"a": [2, 3], "b": [8, 3], "region": [1, 1, 10, 6]},
        {"a": [2, 15], "b": [8, 15], "region": [1, 13, 10, 18]},
    ]
    specs = specs_from(payload, 24, 24)
    result = solve(specs, mask, gray, 24, 24)
    assert result.feasible
    assert result.decisions == ["apply", "apply"]
    cells = [tuple(c) for c in result.applied_cells]
    ys_above = {y for _, y in cells if y < 10}
    ys_below = {y for _, y in cells if y > 10}
    assert ys_above and ys_below
    assert max(ys_above) <= 6 and min(ys_below) >= 13
    assert_legal(result, specs, mask)


def test_conflict_forces_detour_in_joint_optimum():
    # 第 1 对是水平直线（灰度压到 10，固定为直线）；
    # 第 2 对为 L 形端点，其 1 转弯直连路径与第 1 对四邻接，
    # 联合最优被迫选择 2 转弯绕行，代价可精确核对。
    gray, mask = make_grid(20, 20, fill=100)
    a1, b1 = (2, 3), (7, 3)
    a2, b2 = (2, 4), (7, 8)
    set_mask(mask, [a1, b1, a2, b2])
    set_gray(gray, {(x, 3): 10 for x in range(3, 7)})
    payload = [
        {"a": list(a1), "b": list(b1), "region": [1, 1, 10, 6]},
        {"a": list(a2), "b": list(b2), "region": [1, 1, 10, 10]},
    ]
    specs = specs_from(payload, 20, 20)
    result = solve(specs, mask, gray, 20, 20)
    assert result.feasible
    d1 = next(d for d in result.pair_details if d["pair_id"] == 1)
    d2 = next(d for d in result.pair_details if d["pair_id"] == 2)
    # 第 1 对直线：4 个廉价像素，唯一最优直接补
    assert d1["decision"] == "apply"
    assert d1["turns"] == 0 and d1["cost"] == 40
    # 第 2 对的 1 转弯直连路径都与第 1 对冲突；剩余多条同代价 2 转弯绕行
    # （以及一条 1 转弯的远侧 L），路径不一致 -> 待复核，不写入掩膜
    assert d2["decision"] == "review"
    assert d2["alternative_count"] >= 2
    for alt in d2["alternatives"]:
        cells = {tuple(c) for c in alt["new_cells"]}
        # 任何备选都不得走到与第 1 对直线相邻的 y=4 行
        assert all(y != 4 for _, y in cells)
        assert not any(mask[y][x] for x, y in cells)
    # 联合最优 = 40（第 1 对）+ 800（第 2 对 8 个新增像素 × 100）
    assert result.min_total_cost == 840
    # 待复核对不产生新增像素
    assert {tuple(c) for c in result.applied_cells} == {
        (3, 3), (4, 3), (5, 3), (6, 3)
    }
    assert_legal(result, specs, mask)


def test_eight_pairs_accepted_ninth_rejected():
    gray, mask = make_grid(60, 60, fill=100)
    payload = []
    for i in range(8):
        y = 2 + i * 6
        set_mask(mask, [(2, y), (6, y)])
        payload.append({"a": [2, y], "b": [6, y], "region": [1, y - 1, 8, y + 1]})
    specs = specs_from(payload, 60, 60)
    result = solve(specs, mask, gray, 60, 60)
    assert result.feasible
    assert len(result.decisions) == 8

    payload.append({"a": [2, 55], "b": [6, 55]})
    with pytest.raises(RepairError, match=str(MAX_PAIRS)):
        specs_from(payload, 60, 60)


# ---------- 输入校验 ----------

def test_endpoints_must_be_on_mask():
    gray, mask = make_grid(10, 10)
    set_mask(mask, [(2, 2)])  # 只有 a 在掩膜上
    specs = specs_from([{"a": [2, 2], "b": [7, 7]}], 10, 10)
    with pytest.raises(RepairError, match="端点"):
        solve(specs, mask, gray, 10, 10)


def test_region_side_limit():
    _, mask = make_grid(40, 40)
    set_mask(mask, [(2, 2), (30, 2)])
    with pytest.raises(RepairError, match=str(MAX_REGION_SIDE)):
        specs_from([{"a": [2, 2], "b": [30, 2]}], 40, 40)


def test_duplicate_endpoint_rejected():
    _, mask = make_grid(20, 20)
    set_mask(mask, [(2, 2), (5, 5), (8, 8)])
    payload = [
        {"a": [2, 2], "b": [5, 5]},
        {"a": [5, 5], "b": [8, 8]},
    ]
    with pytest.raises(RepairError, match="共用"):
        specs_from(payload, 20, 20)


def test_out_of_range_endpoint_rejected():
    with pytest.raises(RepairError, match="超出"):
        specs_from([{"a": [0, 0], "b": [99, 99]}], 10, 10)


def test_default_region_is_square_and_contains_endpoints():
    specs = specs_from([{"a": [3, 3], "b": [9, 7]}], 20, 20)
    s = specs[0]
    width = s.x1 - s.x0 + 1
    height = s.y1 - s.y0 + 1
    assert width == height
    assert width <= MAX_REGION_SIDE
    assert s.x0 <= 3 <= s.x1 and s.x0 <= 9 <= s.x1
    assert s.y0 <= 3 <= s.y1 and s.y0 <= 7 <= s.y1
