"""补线核心求解器。

业务规则（与需求逐条对应）:

* 每张图每边至多 512 像素，掩膜与原图同尺寸、同方向。
* 最多 8 对端点；每对指定一块包含两端的补线区，每边至多 24 像素
  （省略区域时自动生成包住两端的方形区）。
* 候选路径在补线区内沿横/竖方向前进，最多拐两次（0/1/2 转弯）。
* 新增像素（端点除外，端点本就属于原走线）不得覆盖原走线掩膜；
  新增像素只能在该对自己的两个端点处与原走线四邻接。
* 不同补线之间不得重叠，也不得四邻接。
* 目标：在能同时接通全部端点对的方案中，使新增像素的原图灰度总和最小。
* 只补在所有最低代价方案中路径一致的端点对，其余列为待复核。
* 若无法同时接通，返回无解（由上层返回原掩膜与无解提示）。

求解流程拆成三步，普通请求与流式请求共用：

1. ``prepare_repair``：校验端点、枚举候选、构建跨对冲突与连通分量；
2. ``solve``：直接算完全部分量，返回与历史版本一致的 ``SolveResult``；
3. ``stream_repair``：先确认所有分量可同时接通，再只算预览对所在分量，
   发出预览事件后才计算与预览对无（直接/间接）候选冲突的分量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# 硬约束上限
MAX_SIDE = 512
MAX_PAIRS = 8
MAX_REGION_SIDE = 24

# 四邻域偏移
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class RepairError(ValueError):
    """输入数据或补线规则相关的错误。"""


@dataclass(frozen=True)
class Candidate:
    """一条候选补线路径。

    cells:   新增像素坐标（不含两个端点），按从 a 到 b 的顺序排列
    cellset: 同上，frozenset，用于快速冲突判断
    turns:   转弯次数（0/1/2）
    cost:    新增像素的原图灰度总和（越大表示底色越深、代价越高）
    forbid:  新增像素与本对端点、以及它们各自的四邻域，
             供跨对“不得重叠 / 四邻接”双向判断
    """

    cells: tuple[tuple[int, int], ...]
    cellset: frozenset[tuple[int, int]]
    turns: int
    cost: int
    forbid: frozenset[tuple[int, int]]

    def as_payload(self) -> list[list[int]]:
        return [[x, y] for x, y in self.cells]


@dataclass
class PairSpec:
    """一对端点及其补线区。"""

    pair_id: int
    a: tuple[int, int]
    b: tuple[int, int]
    x0: int
    y0: int
    x1: int  # 含
    y1: int  # 含
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class SolveResult:
    """求解结果。"""

    feasible: bool
    reason: str = ""
    min_total_cost: int = 0
    # 每对的判定："apply"（所有最低代价方案路径一致，已补）或
    # "review"（存在分歧，待人工复核）
    decisions: list[str] = field(default_factory=list)
    applied_cells: list[list[int]] = field(default_factory=list)
    pair_details: list[dict] = field(default_factory=list)
    # 每对按路径去重后的最低代价候选（待复核对用于人工参考）
    alternatives: list[list[Candidate]] = field(default_factory=list)


def _segment(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """生成轴对齐线段上的像素（含两端），仅用于横/竖方向。"""
    if x0 == x1:
        step = 1 if y1 >= y0 else -1
        return [(x0, y) for y in range(y0, y1 + step, step)]
    step = 1 if x1 >= x0 else -1
    return [(x, y0) for x in range(x0, x1 + step, step)]


def _turns_of(points: list[tuple[int, int]]) -> int:
    """根据路径顶点统计 90 度转弯次数。"""
    turns = 0
    prev_dir = None
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 == x2:
            direction = "v"
        else:
            direction = "h"
        if prev_dir is not None and direction != prev_dir:
            turns += 1
        prev_dir = direction
    return turns


def enumerate_paths(
    a: tuple[int, int],
    b: tuple[int, int],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[list[tuple[int, int]]]:
    """枚举补线区内 a->b 之间转弯不超过两次的全部折线。

    包含：直线（0 转弯）、L 形（1 转弯，两种拐角）、
    三段折线（2 转弯，H-V-H 取区内每一列、V-H-V 取区内每一行）。
    返回的每条路径都含端点 a、b。重复路径去重。
    """
    ax, ay = a
    bx, by = b
    paths: list[list[tuple[int, int]]] = []

    if ax == bx and ay == by:
        return [[a]]

    # 0 转弯：同行或同列的直线
    if ax == bx or ay == by:
        paths.append(_segment(ax, ay, bx, by))

    if ax != bx and ay != by:
        # 1 转弯：两个 L 形拐角
        paths.append(_segment(ax, ay, bx, ay) + _segment(bx, ay, bx, by)[1:])
        paths.append(_segment(ax, ay, ax, by) + _segment(ax, by, bx, by)[1:])

        # 2 转弯 H-V-H：两端不同列时，取区内每一列作为竖段所在列
        for cx in range(x0, x1 + 1):
            if cx == ax or cx == bx:
                continue  # 退化为 1 转弯/直线，已覆盖
            p = (
                _segment(ax, ay, cx, ay)
                + _segment(cx, ay, cx, by)[1:]
                + _segment(cx, by, bx, by)[1:]
            )
            paths.append(p)

        # 2 转弯 V-H-V：两端不同行时，取区内每一行作为横段所在行
        for cy in range(y0, y1 + 1):
            if cy == ay or cy == by:
                continue
            p = (
                _segment(ax, ay, ax, cy)
                + _segment(ax, cy, bx, cy)[1:]
                + _segment(bx, cy, bx, by)[1:]
            )
            paths.append(p)

    # 去重（顺序折叠为元组后比较）
    seen: set[tuple[tuple[int, int], ...]] = set()
    unique: list[list[tuple[int, int]]] = []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _build_candidate(
    path: list[tuple[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
    region: tuple[int, int, int, int],
    mask: list[bytearray],
    gray: list[bytes] | list[bytearray],
    width: int,
    height: int,
) -> Candidate | None:
    """校验一条路径并构造 Candidate；违反规则则返回 None。"""
    x0, y0, x1, y1 = region
    endpoints = {a, b}
    new_cells = [p for p in path if p not in endpoints]

    # 新增像素必须全部落在补线区内（端点按输入校验在区内）
    for x, y in new_cells:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        # 不得覆盖原走线
        if mask[y][x]:
            return None

    # 新增像素与原走线四邻接时，只允许邻接到本对端点（端点是该走线的残段）。
    # 若邻接到的是其它原走线像素，该候选路径不合法。
    for x, y in new_cells:
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if mask[ny][nx] and (nx, ny) not in endpoints:
                return None

    cost = sum(gray[y][x] for x, y in new_cells)
    cellset = frozenset(new_cells)
    # 禁占集合：本对新增像素、它们的四邻域，以及本对端点及其四邻域。
    # 其它补线若与本对端点（原走线）四邻接/重叠，同样违反
    # “新增像素只能在自己端点处与原走线四邻接”的规则。
    forbid = set(cellset)
    guard_cells = list(new_cells) + list(endpoints)
    for x, y in guard_cells:
        forbid.add((x, y))
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                forbid.add((nx, ny))
    return Candidate(
        cells=tuple(new_cells),
        cellset=cellset,
        turns=_turns_of(path),
        cost=cost,
        forbid=frozenset(forbid),
    )


def validate_pair_inputs(
    pairs: list[dict], width: int, height: int
) -> list[PairSpec]:
    """校验 API 层传入的端点对参数，并解析补线区。"""
    if not pairs:
        raise RepairError("至少需要一对端点")
    if len(pairs) > MAX_PAIRS:
        raise RepairError(f"端点对数量超过上限 {MAX_PAIRS}")

    specs: list[PairSpec] = []
    seen_endpoints: set[tuple[int, int]] = set()
    for idx, pair in enumerate(pairs):
        try:
            ax = int(pair["a"][0])
            ay = int(pair["a"][1])
            bx = int(pair["b"][0])
            by = int(pair["b"][1])
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise RepairError(f"第 {idx + 1} 对端点坐标格式错误") from exc

        for x, y in ((ax, ay), (bx, by)):
            if not (0 <= x < width and 0 <= y < height):
                raise RepairError(f"第 {idx + 1} 对端点超出图像范围")
        if (ax, ay) == (bx, by):
            raise RepairError(f"第 {idx + 1} 对的两个端点重合")
        ep = (ax, ay), (bx, by)
        for p in ep:
            if p in seen_endpoints:
                raise RepairError(f"端点 {p} 被多对共用，每段走线的端点必须唯一")
            seen_endpoints.add(p)

        if "region" in pair and pair["region"] is not None:
            rg = pair["region"]
            try:
                rx0, ry0, rx1, ry1 = (int(v) for v in rg)
            except (TypeError, ValueError) as exc:
                raise RepairError(f"第 {idx + 1} 对补线区格式错误") from exc
            if rx1 < rx0 or ry1 < ry0:
                raise RepairError(f"第 {idx + 1} 对补线区坐标顺序错误")
            if rx1 - rx0 + 1 > MAX_REGION_SIDE or ry1 - ry0 + 1 > MAX_REGION_SIDE:
                raise RepairError(
                    f"第 {idx + 1} 对补线区边长超过上限 {MAX_REGION_SIDE}"
                )
            nx0, ny0, nx1, ny1 = rx0, ry0, rx1, ry1
            if not (0 <= nx0 < width and 0 <= ny0 < height
                    and 0 <= nx1 < width and 0 <= ny1 < height):
                raise RepairError(f"第 {idx + 1} 对补线区超出图像范围")
        else:
            # 默认：包住两个端点的最小外接矩形，再扩展为方形
            rx0, ry0 = min(ax, bx), min(ay, by)
            rx1, ry1 = max(ax, bx), max(ay, by)
            w = rx1 - rx0 + 1
            h = ry1 - ry0 + 1
            side = max(w, h)
            if side > MAX_REGION_SIDE:
                raise RepairError(
                    f"第 {idx + 1} 对补线区边长 {side} 超过上限 {MAX_REGION_SIDE}"
                )
            cx = (rx0 + rx1) // 2
            cy = (ry0 + ry1) // 2
            nx0 = cx - (side - 1) // 2
            ny0 = cy - (side - 1) // 2
            nx1 = nx0 + side - 1
            ny1 = ny0 + side - 1
            # 触边时平移回图像内
            if nx0 < 0:
                nx1 -= nx0
                nx0 = 0
            if ny0 < 0:
                ny1 -= ny0
                ny0 = 0
            if nx1 >= width:
                nx0 -= nx1 - (width - 1)
                nx1 = width - 1
            if ny1 >= height:
                ny0 -= ny1 - (height - 1)
                ny1 = height - 1
            if nx0 < 0 or ny0 < 0:
                raise RepairError(f"第 {idx + 1} 对补线区无法放入图像")
        for x, y in ep:
            if not (nx0 <= x <= nx1 and ny0 <= y <= ny1):
                raise RepairError(f"第 {idx + 1} 对端点不在补线区内")

        specs.append(
            PairSpec(idx, (ax, ay), (bx, by), nx0, ny0, nx1, ny1)
        )
    return specs


@dataclass
class PreparedRepair:
    """共用准备阶段的结果：候选与互不影响的连通分量（冲突组）。"""

    specs: list[PairSpec]
    components: list[list[int]]


def prepare_repair(
    specs: list[PairSpec],
    mask: list[bytearray],
    gray: list[bytes] | list[bytearray],
    width: int,
    height: int,
) -> PreparedRepair | SolveResult:
    """枚举候选、构建跨对冲突图并拆分连通分量。

    若某对端点本身无合法候选，返回不可行的 ``SolveResult``；
    否则返回 ``PreparedRepair``，供 ``solve`` / ``stream_repair`` 复用。
    """
    # 1) 逐对枚举并校验候选路径
    for spec in specs:
        if not mask[spec.a[1]][spec.a[0]] or not mask[spec.b[1]][spec.b[0]]:
            raise RepairError(
                f"第 {spec.pair_id + 1} 对端点必须位于原走线（掩膜前景）上"
            )
        # 允许同一 specs 被重复求解（普通请求 / 流式 / 测试对拍）
        spec.candidates = []
        region = (spec.x0, spec.y0, spec.x1, spec.y1)
        for path in enumerate_paths(spec.a, spec.b, *region):
            cand = _build_candidate(
                path, spec.a, spec.b, region, mask, gray, width, height
            )
            if cand is not None:
                spec.candidates.append(cand)
        if not spec.candidates:
            return SolveResult(
                feasible=False,
                reason=f"第 {spec.pair_id + 1} 对端点在补线区内无合法路径",
            )

    n = len(specs)

    # 2) 跨对冲突：重叠或四邻接即不兼容（双向检查）
    conflicts: list[list[list[tuple[int, int]]]] = [
        [[] for _ in s.candidates] for s in specs
    ]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = specs[i], specs[j]
            for ci, cand_i in enumerate(si.candidates):
                fi = cand_i.forbid
                for cj, cand_j in enumerate(sj.candidates):
                    if (fi & cand_j.cellset) or (cand_j.forbid & cand_i.cellset):
                        conflicts[i][ci].append((j, cj))
                        conflicts[j][cj].append((i, ci))

    # 3) 按“对与对之间是否存在任一候选冲突”拆连通分量；
    #    不同分量互不影响，可独立求解
    pair_adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for ci in range(len(specs[i].candidates)):
            for j, _cj in conflicts[i][ci]:
                pair_adjacency[i].add(j)
                pair_adjacency[j].add(i)

    comp_of = [-1] * n
    components: list[list[int]] = []
    for i in range(n):
        if comp_of[i] != -1:
            continue
        cid = len(components)
        stack = [i]
        comp_of[i] = cid
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            for j in pair_adjacency[node]:
                if comp_of[j] == -1:
                    comp_of[j] = cid
                    stack.append(j)
        components.append(sorted(members))

    return PreparedRepair(specs=specs, components=components)


def _component_order(specs: list[PairSpec], members: list[int]) -> list[int]:
    """分量内的求解顺序：候选少的先试，稳定排序。"""
    return sorted(members, key=lambda k: len(specs[k].candidates))


def _search(
    specs: list[PairSpec],
    members: list[int],
    fixed: tuple[int, int] | None = None,
    *,
    need_optimum: bool = True,
) -> int | None:
    """在单个连通分量内做记忆化搜索。

    * ``need_optimum=True``：返回最低可行代价，无解返回 ``None``；
    * ``need_optimum=False``：只确认存在可行组合，存在即返回 0（可提前结束）。

    ``fixed`` 为 ``(pair_id, cand_idx)`` 时强制该对选择指定候选；
    若它不在该分量内则忽略（调用方保证不这样用）。
    """
    order = _component_order(specs, members)
    comp_size = len(order)

    def dfs(
        pos: int,
        occ_cells: frozenset,
        occ_forbid: frozenset,
        memo: dict,
    ):
        if pos == comp_size:
            return 0
        key = (pos, occ_cells, occ_forbid)
        cached = memo.get(key)
        if cached is not None:
            return cached
        pair_id = order[pos]
        cands = specs[pair_id].candidates
        forced = fixed[1] if fixed is not None and fixed[0] == pair_id else None
        best = None
        for ci, cand in enumerate(cands):
            if forced is not None and ci != forced:
                continue
            # 双向冲突：重叠或四邻接（任一侧进入对方禁占区）
            if (cand.cellset & occ_cells) or (cand.cellset & occ_forbid) \
                    or (cand.forbid & occ_cells):
                continue
            sub = dfs(
                pos + 1,
                occ_cells | cand.cellset,
                occ_forbid | cand.forbid,
                memo,
            )
            if sub is None:
                continue
            if not need_optimum:
                # 只确认可同时接通：找到第一条可行组合即可
                memo[key] = 0
                return 0
            value = cand.cost + sub
            if best is None or value < best:
                best = value
        memo[key] = best
        return best

    return dfs(0, frozenset(), frozenset(), {})


def _component_infeasible(members: list[int]) -> SolveResult:
    names = "、".join(str(k + 1) for k in members)
    return SolveResult(
        feasible=False,
        reason=f"第 {names} 对之间存在无法同时满足的冲突，无全局可行方案",
    )


def _supported_candidates(
    specs: list[PairSpec],
    members: list[int],
    pair_id: int,
    comp_min: int,
) -> list[Candidate]:
    """求该对在分量全部最低代价方案中出现过的路径（按路径去重）。"""
    supported: list[Candidate] = []
    for ci, cand in enumerate(specs[pair_id].candidates):
        cond_value = _search(specs, members, (pair_id, ci))
        if cond_value == comp_min:
            supported.append(cand)
    uniq: dict[frozenset, Candidate] = {}
    for cand in supported:
        uniq.setdefault(cand.cellset, cand)
    return list(uniq.values())


def _assemble(
    prep: PreparedRepair,
    supported_by_pair: dict[int, list[Candidate]],
    total_cost: int,
) -> SolveResult:
    """把每对的全部最低代价路径整理成最终结果（与历史响应结构一致）。"""
    n = len(prep.specs)
    decisions = ["review"] * n
    alternatives: list[list[Candidate]] = [[] for _ in range(n)]
    pair_details: list[dict | None] = [None] * n
    applied_cells: list[list[int]] = []

    for members in prep.components:
        for pair_id in _component_order(prep.specs, members):
            supported = supported_by_pair[pair_id]
            alternatives[pair_id] = supported
            if len(supported) == 1:
                decisions[pair_id] = "apply"
                only = supported[0]
                applied_cells.extend(only.as_payload())
                pair_details[pair_id] = {
                    "pair_id": pair_id + 1,
                    "decision": "apply",
                    "turns": only.turns,
                    "cost": only.cost,
                    "new_cells": only.as_payload(),
                }
            else:
                pair_details[pair_id] = {
                    "pair_id": pair_id + 1,
                    "decision": "review",
                    "reason": "存在多条同属最低代价的路径，方案间不一致",
                    "alternative_count": len(supported),
                    "alternatives": [
                        {"turns": c.turns, "cost": c.cost,
                         "new_cells": c.as_payload()}
                        for c in supported
                    ],
                }

    return SolveResult(
        feasible=True,
        min_total_cost=total_cost,
        decisions=decisions,
        applied_cells=applied_cells,
        pair_details=[d for d in pair_details if d is not None],
        alternatives=alternatives,
    )


def solve(
    specs: list[PairSpec],
    mask: list[bytearray],
    gray: list[bytes] | list[bytearray],
    width: int,
    height: int,
) -> SolveResult:
    """按全部约束求最低代价补线方案，并判定每对是否路径一致。"""
    prep = prepare_repair(specs, mask, gray, width, height)
    if isinstance(prep, SolveResult):
        return prep

    supported_by_pair: dict[int, list[Candidate]] = {}
    total_cost = 0

    # 每个连通分量独立做记忆化搜索求全局最低代价
    for members in prep.components:
        comp_min = _search(prep.specs, members)
        if comp_min is None:
            return _component_infeasible(members)
        total_cost += comp_min

        # 判定路径一致性：固定每对的每个候选求条件最优。
        # 条件最优仍等于分量最优 => 该候选出现在某个最低代价方案中。
        for pair_id in members:
            supported_by_pair[pair_id] = _supported_candidates(
                prep.specs, members, pair_id, comp_min
            )

    return _assemble(prep, supported_by_pair, total_cost)


# 流式事件类型：("preview", 载荷) 或 ("result", SolveResult)
StreamEvent = tuple[str, dict | SolveResult]


def stream_repair(
    specs: list[PairSpec],
    mask: list[bytearray],
    gray: list[bytes] | list[bytearray],
    width: int,
    height: int,
    preview_pair_ids: list[int],
    *,
    trace=None,
) -> Iterable[StreamEvent]:
    """流式求解入口：先做共用准备阶段，再交由 ``stream_prepared`` 产出事件。"""
    prep = prepare_repair(specs, mask, gray, width, height)
    yield from stream_prepared(prep, preview_pair_ids, trace=trace)


def stream_prepared(
    prep: PreparedRepair | SolveResult,
    preview_pair_ids: list[int],
    *,
    trace=None,
) -> Iterable[StreamEvent]:
    """流式求解：先确认全局可接通，再发预览对的判定，最后发整图结果。

    参数 ``preview_pair_ids`` 为 0 基的端点对编号，按给定顺序预览。
    ``trace`` 仅为测试/观测钩子，形如 ``trace(phase, detail)``。

    事件顺序：
    1. 先对**全部**连通分量做一次存在性搜索（不算最优），确认所有端点对
       能同时接通；任一分量不可行则只发一个 ``result``（无解 + 原掩膜，
       不发预览）。
    2. 按请求顺序发 ``preview``：先求该对所在分量（同一分量只算一次）并
       列出该对在所有最低总代价方案中的路径；只有一条时才判 ``apply``。
       排在最前的预览对最先求、最先发，其余预览对各自算出后立即发出。
    3. 与预览对没有直接/间接候选冲突的分量（即不含预览对的分量）
       在预览发出后才求最优。
    4. 发 ``result``，内容与普通请求等价。
    """
    if isinstance(prep, SolveResult):
        yield ("result", prep)
        return

    n_pairs = len(prep.specs)
    if not preview_pair_ids:
        raise RepairError("流式请求至少需要一个预览端点对")
    for pair_id in preview_pair_ids:
        if not (0 <= pair_id < n_pairs):
            raise RepairError(f"预览端点对编号 {pair_id + 1} 超出范围 1..{n_pairs}")

    def emit(phase: str, **detail) -> None:
        if trace is not None:
            trace(phase, detail)

    comp_of: dict[int, int] = {}
    for cid, members in enumerate(prep.components):
        for pair_id in members:
            comp_of[pair_id] = cid

    # 1) 先确认全部端点对能同时接通（存在性搜索，不是最优）
    for cid, members in enumerate(prep.components):
        feasible = _search(
            prep.specs, members, need_optimum=False
        )
        emit("feasibility_checked", component=cid, members=members,
             feasible=feasible is not None)
        if feasible is None:
            yield ("result", _component_infeasible(members))
            return

    # 2) 只求与预览对直接/间接冲突的分量（含预览对的分量），按请求顺序预览。
    #    同一分量只算一次；该分量内的其它预览对随分量一起算出，随后立刻预览。
    supported_by_pair: dict[int, list[Candidate]] = {}
    comp_min: dict[int, int] = {}
    solved_cids: set[int] = set()

    def solve_component(cid: int) -> None:
        if cid in solved_cids:
            return
        members = prep.components[cid]
        value = _search(prep.specs, members)
        assert value is not None  # 上一步已确认可行
        comp_min[cid] = value
        for pair_id in members:
            supported_by_pair[pair_id] = _supported_candidates(
                prep.specs, members, pair_id, value
            )
        solved_cids.add(cid)
        emit("preview_component_solved", component=cid, members=members,
             min_cost=value)

    def preview_event(pair_id: int) -> dict:
        supported = supported_by_pair[pair_id]
        decision = "apply" if len(supported) == 1 else "review"
        return {
            "pair_id": pair_id + 1,
            "decision": decision,
            "path_count": len(supported),
            "component_min_cost": comp_min[comp_of[pair_id]],
            "paths": [
                {"turns": c.turns, "cost": c.cost, "new_cells": c.as_payload()}
                for c in supported
            ],
        }

    # 先预览请求里排在最前的那个分量；其余预览分量等到首个预览发出后
    # （仍早于整图结果）；每个预览对一算出就立刻发出。
    emitted: set[int] = set()
    if preview_pair_ids:
        solve_component(comp_of[preview_pair_ids[0]])
    for pair_id in preview_pair_ids:
        solve_component(comp_of[pair_id])
        if pair_id in emitted:
            continue
        emitted.add(pair_id)
        supported = supported_by_pair[pair_id]
        emit("preview", pair_id=pair_id,
             decision="apply" if len(supported) == 1 else "review",
             path_count=len(supported))
        yield ("preview", preview_event(pair_id))

    # 3) 预览之后才求与预览对无冲突的端点组最优
    for cid, members in enumerate(prep.components):
        if cid in solved_cids:
            continue
        value = _search(prep.specs, members)
        assert value is not None  # 上一步已确认可行
        comp_min[cid] = value
        for pair_id in members:
            supported_by_pair[pair_id] = _supported_candidates(
                prep.specs, members, pair_id, value
            )
        emit("deferred_component_solved", component=cid, members=members,
             min_cost=value)

    total_cost = sum(comp_min.values())
    yield ("result", _assemble(prep, supported_by_pair, total_cost))


def apply_cells_to_mask(
    mask: list[bytearray], cells: Iterable[tuple[int, int]]
) -> None:
    """把确定补线的像素写入掩膜副本。"""
    for x, y in cells:
        mask[y][x] = 1
