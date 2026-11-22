"""FastAPI 补线服务入口。

接口:
    GET  /health  健康检查
    POST /repair  上传原图(8 位灰度 PNG)、二值掩膜 PNG 与端点对 JSON，
                  返回补线结果与（可能被修改后的）掩膜 PNG（base64）。

`/repair` 增加可选的流式响应：表单字段 `stream=true` 时返回 NDJSON
（每行一个 JSON 事件），其余表单字段与原请求完全相同。事件顺序为
`preview`（可多条，指定端点对的补线判定）在前、`result`（与普通请求
等价的整图结果）在后；无解时只发一条 `result`（无解 + 原掩膜）。
"""

from __future__ import annotations

import base64
import io
import json
from typing import Iterable

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from .solver import (
    MAX_SIDE,
    RepairError,
    SolveResult,
    apply_cells_to_mask,
    prepare_repair,
    solve,
    stream_prepared,
    validate_pair_inputs,
)

app = FastAPI(
    title="旧式线路底片补线 API",
    description="在二值走线掩膜上，按最低灰度代价补回褪色线段",
    version="1.0.0",
)


async def _load_gray_png(upload: UploadFile, *, is_mask: bool):
    """读取并校验 PNG。灰度图返回逐行字节；掩膜返回逐行 bytearray(0/1)。"""
    raw = await upload.read()
    if not raw:
        raise RepairError(f"文件 {upload.filename} 为空")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # Pillow 抛出的各类解码异常
        raise RepairError(f"文件 {upload.filename} 不是可识别的图像") from exc
    if img.format != "PNG":
        raise RepairError(f"文件 {upload.filename} 必须是 PNG 格式")

    if is_mask:
        if img.mode not in ("1", "L"):
            raise RepairError("掩膜必须是二值(模式 1)或 8 位灰度(模式 L) PNG")
        if img.mode == "1":
            img = img.convert("L")
        raw_values = img.tobytes()
        if any(v not in (0, 255) for v in raw_values):
            bad = sorted(set(raw_values) - {0, 255})[:8]
            raise RepairError(
                f"掩膜必须是二值图像，仅允许像素值 0/255，实际含 {bad}"
            )
        width, height = img.size
        rows = [
            bytearray(1 if v == 255 else 0 for v in raw_values[y * width:(y + 1) * width])
            for y in range(height)
        ]
        return rows, width, height

    if img.mode != "L":
        raise RepairError("原图必须是 8 位灰度 PNG（模式 L）")
    width, height = img.size
    raw_values = img.tobytes()
    rows = [raw_values[y * width:(y + 1) * width] for y in range(height)]
    return rows, width, height


def _mask_to_png(mask: list[bytearray], width: int, height: int) -> bytes:
    buf = bytearray(width * height)
    for y, row in enumerate(mask):
        base = y * width
        for x, value in enumerate(row):
            if value:
                buf[base + x] = 255
    out = Image.frombytes("L", (width, height), bytes(buf))
    io_buf = io.BytesIO()
    out.save(io_buf, format="PNG", optimize=True)
    return io_buf.getvalue()


def _result_payload(
    result: SolveResult,
    mask: list[bytearray],
    width: int,
    height: int,
) -> dict:
    """把求解结果整理成响应体（普通请求与流式 result 事件共用）。

    无解时返回原掩膜与原因；有解时写回确定补线的像素。
    """
    if not result.feasible:
        png_bytes = _mask_to_png(mask, width, height)
        return {
            "status": "no_solution",
            "message": "无法同时接通全部端点对，返回原掩膜",
            "reason": result.reason,
            "mask_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    out_mask = [bytearray(row) for row in mask]
    apply_cells_to_mask(out_mask, [(x, y) for x, y in result.applied_cells])
    png_bytes = _mask_to_png(out_mask, width, height)
    applied_set = {tuple(p) for p in result.applied_cells}
    return {
        "status": "ok",
        "image_size": [width, height],
        "min_total_cost": result.min_total_cost,
        "applied_pair_count": result.decisions.count("apply"),
        "review_pair_count": result.decisions.count("review"),
        "pairs": result.pair_details,
        "applied_cells": sorted(applied_set),
        "mask_png_base64": base64.b64encode(png_bytes).decode("ascii"),
    }


def _parse_stream_flag(raw: str | None) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in ("", "false", "0", "no"):
        return False
    if value in ("true", "1", "yes"):
        return True
    raise RepairError("stream 只能是 true/false")


def _parse_preview_pairs(raw: str | None, pair_count: int) -> list[int]:
    """解析要先看的端点对编号（1 基，与响应中的 pair_id 一致）。

    接受 ``"1,3"`` 或 JSON 数组 ``[1,3]``；顺序即预览顺序，重复编号只保留首次。
    """
    if raw is None or not raw.strip():
        raise RepairError("stream=true 时必须提供 preview_pairs，指明要先看的端点对编号")
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RepairError("preview_pairs 不是合法 JSON 数组") from exc
        if not isinstance(parsed, list):
            raise RepairError("preview_pairs 必须是编号数组")
        values = parsed
    else:
        values = [part for part in text.replace("，", ",").split(",") if part.strip()]

    ids: list[int] = []
    for value in values:
        try:
            pair_id = int(value)
        except (TypeError, ValueError) as exc:
            raise RepairError("preview_pairs 中的编号必须是整数") from exc
        if not (1 <= pair_id <= pair_count):
            raise RepairError(f"preview_pairs 中的编号 {pair_id} 超出端点对范围 1..{pair_count}")
        if pair_id not in ids:
            ids.append(pair_id)
    if not ids:
        raise RepairError("preview_pairs 至少需要一个端点对编号")
    return ids


def _ndjson(events: Iterable[dict]):
    for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"


def stream_repair_events(
    prep,
    preview_pair_ids: list[int],
    mask: list[bytearray],
    width: int,
    height: int,
    *,
    trace=None,
) -> Iterable[dict]:
    """把求解事件转成 NDJSON 事件体（惰性：消费一条才算一条）。"""
    for kind, payload in stream_prepared(prep, preview_pair_ids, trace=trace):
        if kind == "preview":
            yield {"event": "preview", **payload}
        else:
            yield {"event": "result", **_result_payload(payload, mask, width, height)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/repair")
async def repair(
    image: UploadFile = File(..., description="8 位灰度 PNG 原图"),
    mask_file: UploadFile = File(..., alias="mask", description="二值走线掩膜 PNG"),
    pairs: str = Form(
        ...,
        description=(
            "JSON 数组，至多 8 对端点。元素形如 "
            '{"a":[x1,y1],"b":[x2,y2],"region":[x0,y0,x1,y1](可选, 边长<=24)}'
        ),
    ),
    stream: str | None = Form(
        None,
        description="true 时返回 NDJSON 流式响应（先 preview 后 result）",
    ),
    preview_pairs: str | None = Form(
        None,
        description='要先看补线判定的端点对编号（1 基），如 "1,3" 或 "[1,3]"',
    ),
):
    try:
        gray, w1, h1 = await _load_gray_png(image, is_mask=False)
        mask, w2, h2 = await _load_gray_png(mask_file, is_mask=True)
        if (w1, h1) != (w2, h2):
            raise RepairError("原图与掩膜尺寸必须完全相同")
        if w1 > MAX_SIDE or h1 > MAX_SIDE:
            raise RepairError(f"图像每边至多 {MAX_SIDE} 像素")
        try:
            pair_payload = json.loads(pairs)
        except json.JSONDecodeError as exc:
            raise RepairError("pairs 不是合法 JSON") from exc
        if not isinstance(pair_payload, list):
            raise RepairError("pairs 必须是 JSON 数组")

        specs = validate_pair_inputs(pair_payload, w1, h1)
        streaming = _parse_stream_flag(stream)
        preview_ids = (
            _parse_preview_pairs(preview_pairs, len(specs)) if streaming else []
        )

        if not streaming:
            result = solve(specs, mask, gray, w1, h1)
            return JSONResponse(status_code=200, content=_result_payload(
                result, mask, w1, h1
            ))

        # 流式：准备阶段（候选枚举/冲突图）先同步完成，输入错误仍能返回 400；
        # 之后的事件按“存在性确认 -> 预览 -> 独立组最优 -> 整图结果”产出。
        prep = prepare_repair(specs, mask, gray, w1, h1)

        return StreamingResponse(
            _ndjson(stream_repair_events(
                prep, [i - 1 for i in preview_ids], mask, w1, h1
            )),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    except RepairError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
