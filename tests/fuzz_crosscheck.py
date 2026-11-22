"""独立暴力校验器：与 solver 完全独立的穷举实现，用于 fuzz 交叉核对。

不导入 app.solver 的任何求解逻辑，只复用输入校验得到的区域坐标，
路径枚举、合法性检查、组合穷举都在这里重新实现一遍。
"""

from __future__ import annotations

import itertools
import random
import sys

from app.solver import validate_pair_inputs

N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def seg(x0, y0, x1, y1):
    if x0 == x1:
        s = 1 if y1 >= y0 else -1
        return [(x0, y) for y in range(y0, y1 + s, s)]
    s = 1 if x1 >= x0 else -1
    return [(x, y0) for x in range(x0, x1 + s, s)]


def brute_paths(a, b, x0, y0, x1, y1):
    ax, ay = a
    bx, by = b
    out = []
    if ax == bx or ay == by:
        out.append(seg(ax, ay, bx, by))
    if ax != bx and ay != by:
        out.append(seg(ax, ay, bx, ay) + seg(bx, ay, bx, by)[1:])
        out.append(seg(ax, ay, ax, by) + seg(ax, by, bx, by)[1:])
        for cx in range(x0, x1 + 1):
            if cx in (ax, bx):
                continue
            out.append(seg(ax, ay, cx, ay) + seg(cx, ay, cx, by)[1:]
                       + seg(cx, by, bx, by)[1:])
        for cy in range(y0, y1 + 1):
            if cy in (ay, by):
                continue
            out.append(seg(ax, ay, ax, cy) + seg(ax, cy, bx, cy)[1:]
                       + seg(bx, cy, bx, by)[1:])
    # 去重
    seen, uniq = set(), []
    for p in out:
        k = tuple(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def legal_candidate(path, a, b, mask, w, h):
    eps = {a, b}
    new = [p for p in path if p not in eps]
    for x, y in new:
        if mask[y][x]:
            return None
        for dx, dy in N4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and (nx, ny) not in eps:
                return None
    return tuple(new)


def fuzz_once(rng: random.Random) -> None:
    w = h = rng.randint(8, 16)
    gray = [bytearray(rng.randint(0, 255) for _ in range(w)) for _ in range(h)]
    mask = [bytearray(w) for _ in range(h)]
    n_pairs = rng.randint(1, 5)
    payload = []
    used = set()
    attempts = 0
    while len(payload) < n_pairs and attempts < 200:
        attempts += 1
        ax, ay = rng.randrange(w), rng.randrange(h)
        bx, by = rng.randrange(w), rng.randrange(h)
        if (ax, ay) == (bx, by) or (ax, ay) in used or (bx, by) in used:
            continue
        side = rng.randint(3, 10)
        x0 = max(0, min(ax, bx) - rng.randint(0, 2))
        y0 = max(0, min(ay, by) - rng.randint(0, 2))
        x1 = min(w - 1, x0 + side - 1)
        y1 = min(h - 1, y0 + side - 1)
        if not (x0 <= ax <= x1 and x0 <= bx <= x1 and y0 <= ay <= y1 and y0 <= by <= y1):
            continue
        mask[ay][ax] = 1
        mask[by][bx] = 1
        used.update(((ax, ay), (bx, by)))
        payload.append({"a": [ax, ay], "b": [bx, by],
                        "region": [x0, y0, x1, y1]})
    if not payload:
        return

    # 再随机撒一些原走线像素作为干扰（避开端点与已用点）
    for _ in range(rng.randint(0, w)):
        x, y = rng.randrange(w), rng.randrange(h)
        if (x, y) not in used:
            mask[y][x] = 1

    specs = validate_pair_inputs(payload, w, h)

    # 暴力：逐对候选
    pair_cands = []
    for s in specs:
        cands = []
        for p in brute_paths(s.a, s.b, s.x0, s.y0, s.x1, s.y1):
            new = legal_candidate(p, s.a, s.b, mask, w, h)
            if new is None:
                continue
            cost = sum(gray[y][x] for x, y in new)
            cands.append((new, cost))
        # 同路径去重
        dedup = {}
        for new, cost in cands:
            dedup.setdefault(new, cost)
        pair_cands.append(list(dedup.items()))

    # 每对的端点列表，供兼容性检查把端点邻域纳入禁占
    endpoint_list = [(s.a, s.b) for s in specs]

    def compatible(cells_list, endpoint_list):
        all_sets = [set(c) for c in cells_list]
        forbid = []
        for cells, eps in zip(all_sets, endpoint_list):
            f = set(cells)
            for x, y in list(cells) + list(eps):
                f.add((x, y))
                for dx, dy in N4:
                    f.add((x + dx, y + dy))
            forbid.append(f)
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                if all_sets[i] & all_sets[j]:
                    return False
                # 双向四邻接（含对方端点邻域）
                if forbid[i] & all_sets[j] or forbid[j] & all_sets[i]:
                    return False
        return True

    best = None
    opt_choices = [set() for _ in payload]
    for combo in itertools.product(*[range(len(c)) for c in pair_cands]):
        cells = [pair_cands[i][combo[i]][0] for i in range(len(payload))]
        if not compatible(cells, endpoint_list):
            continue
        total = sum(pair_cands[i][combo[i]][1] for i in range(len(payload)))
        if best is None or total < best:
            best = total
            for i in range(len(payload)):
                opt_choices[i] = {pair_cands[i][combo[i]][0]}
        elif total == best:
            for i in range(len(payload)):
                opt_choices[i].add(pair_cands[i][combo[i]][0])

    # 求解器结果
    from app.solver import solve, stream_repair
    result = solve(specs, mask, gray, w, h)

    # 流式结果必须与普通请求等价；预览对按随机顺序取
    preview_ids = [rng.randrange(len(specs))]
    if len(specs) > 1 and rng.random() < 0.5:
        extra = rng.randrange(len(specs))
        if extra not in preview_ids:
            preview_ids.append(extra)
    events = list(stream_repair(specs, mask, gray, w, h, preview_ids))
    stream_result = events[-1][1]
    assert events[-1][0] == "result"
    assert stream_result.feasible == result.feasible
    if result.feasible:
        assert stream_result.min_total_cost == result.min_total_cost, payload
        assert stream_result.decisions == result.decisions, payload
        assert sorted(map(tuple, stream_result.applied_cells)) == sorted(
            map(tuple, result.applied_cells)
        ), payload
        assert stream_result.pair_details == result.pair_details, payload
    # 有解时预览事件必须与普通结果中该对的判定一致；无解时不得有预览
    preview_events = [p for k, p in events if k == "preview"]
    if best is None:
        assert preview_events == [], payload
    else:
        assert [p["pair_id"] - 1 for p in preview_events] == preview_ids
        for ev in preview_events:
            detail = next(
                d for d in result.pair_details if d["pair_id"] == ev["pair_id"]
            )
            assert ev["decision"] == detail["decision"], payload
            n_paths = (len(detail["alternatives"])
                       if detail["decision"] == "review" else 1)
            assert ev["path_count"] == n_paths, payload
            assert ev["decision"] == ("apply" if n_paths == 1 else "review")

    if best is None:
        assert not result.feasible, f"暴力无解但求解器报告有解: {payload}"
        return
    assert result.feasible, f"暴力有解({best})但求解器无解: {payload} {result.reason}"
    assert result.min_total_cost == best, (
        f"代价不一致: 暴力={best} 求解器={result.min_total_cost} payload={payload}"
    )
    # 逐对一致性判定核对
    for i, s in enumerate(specs):
        detail = next(d for d in result.pair_details if d["pair_id"] == i + 1)
        brute_paths_opt = opt_choices[i]
        if len(brute_paths_opt) == 1:
            assert detail["decision"] == "apply", f"第{i+1}对应唯一却未补: {payload}"
            only = next(iter(brute_paths_opt))
            assert [list(c) for c in only] == detail["new_cells"]
        else:
            assert detail["decision"] == "review", f"第{i+1}对分歧却被补: {payload}"
            solver_alts = {tuple(tuple(c) for c in a["new_cells"])
                           for a in detail["alternatives"]}
            assert solver_alts == brute_paths_opt, (
                f"第{i+1}对备选集合不一致\n{solver_alts}\n{brute_paths_opt}"
            )


def main():
    rng = random.Random(20261122)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for i in range(n):
        fuzz_once(rng)
    print(f"fuzz {n} cases: all matched brute force")


if __name__ == "__main__":
    main()
