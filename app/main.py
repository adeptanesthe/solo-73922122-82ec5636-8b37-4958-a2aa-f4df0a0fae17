"""FastAPI 补线服务入口。

接口:
    GET  /health  健康检查
    POST /repair  上传原图(8 位灰度 PNG)、二值掩膜 PNG 与端点对 JSON，
                  返回补线结果与（可能被修改后的）掩膜 PNG（base64）。
"""

from __future__ import annotations

import base64
import io
import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .solver import (
    MAX_SIDE,
    RepairError,
    apply_cells_to_mask,
    solve,
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
        result = solve(specs, mask, gray, w1, h1)

        if not result.feasible:
            # 无法同时接通：返回原掩膜与无解提示
            png_bytes = _mask_to_png(mask, w1, h1)
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_solution",
                    "message": "无法同时接通全部端点对，返回原掩膜",
                    "reason": result.reason,
                    "mask_png_base64": base64.b64encode(png_bytes).decode("ascii"),
                },
            )

        applied = [(x, y) for x, y in result.applied_cells]
        out_mask = [bytearray(row) for row in mask]
        apply_cells_to_mask(out_mask, applied)
        png_bytes = _mask_to_png(out_mask, w1, h1)

        applied_set = {tuple(p) for p in result.applied_cells}
        return {
            "status": "ok",
            "image_size": [w1, h1],
            "min_total_cost": result.min_total_cost,
            "applied_pair_count": result.decisions.count("apply"),
            "review_pair_count": result.decisions.count("review"),
            "pairs": result.pair_details,
            "applied_cells": sorted(applied_set),
            "mask_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        }
    except RepairError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
