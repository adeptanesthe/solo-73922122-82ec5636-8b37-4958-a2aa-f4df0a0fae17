"""FastAPI 接口层测试：文件校验、端到端补线、无解返回原掩膜。"""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def png_gray(pixels, width, height, mode="L"):
    img = Image.frombytes(mode, (width, height), bytes(pixels))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mask_png(mask, width, height):
    flat = bytearray(width * height)
    for y, row in enumerate(mask):
        for x, v in enumerate(row):
            flat[y * width + x] = 255 if v else 0
    return png_gray(flat, width, height, mode="L")


def decode_mask(b64, width, height):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (width, height)
    return img.convert("L")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_repair_end_to_end_applies_unique_path():
    w = h = 24
    gray = bytearray([200] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    a = (4, 6)
    b = (10, 6)
    mask[a[1]][a[0]] = 1
    mask[b[1]][b[0]] = 1
    pairs = [{"a": list(a), "b": list(b)}]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps(pairs)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["applied_pair_count"] == 1
    assert body["review_pair_count"] == 0
    out = decode_mask(body["mask_png_base64"], w, h)
    # 直线上 5 个新增像素都被写入，原端点保留
    for x in range(4, 11):
        assert out.getpixel((x, 6)) == 255
    # 其它位置未被改动
    assert out.getpixel((4, 7)) == 0


def test_repair_review_pair_not_written():
    w = h = 20
    gray = bytearray([100] * (w * h))  # 均匀灰度 -> L 形多路径同代价
    mask = [bytearray(w) for _ in range(h)]
    a, b = (3, 3), (8, 8)
    mask[a[1]][a[0]] = 1
    mask[b[1]][b[0]] = 1
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps([{"a": list(a), "b": list(b)}])},
    )
    body = r.json()
    assert body["status"] == "ok"
    assert body["review_pair_count"] == 1
    assert body["applied_cells"] == []
    out = decode_mask(body["mask_png_base64"], w, h)
    # 待复核 -> 掩膜与输入完全相同（只有两个端点）
    assert out.getpixel(a) == 255 and out.getpixel(b) == 255
    assert out.getpixel((4, 3)) == 0
    # 备选路径已返回供人工复核
    detail = body["pairs"][0]
    assert detail["decision"] == "review"
    assert detail["alternative_count"] >= 2


def test_no_solution_returns_original_mask():
    w = h = 12
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    for x, y in [(1, 4), (5, 4), (1, 5), (5, 5)]:
        mask[y][x] = 1
    pairs = [
        {"a": [1, 4], "b": [5, 4], "region": [1, 4, 5, 4]},
        {"a": [1, 5], "b": [5, 5], "region": [1, 5, 5, 5]},
    ]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps(pairs)},
    )
    body = r.json()
    assert body["status"] == "no_solution"
    out = decode_mask(body["mask_png_base64"], w, h)
    # 返回的掩膜必须与输入逐像素一致
    for y in range(h):
        for x in range(w):
            assert out.getpixel((x, y)) == (255 if mask[y][x] else 0)


def test_rejects_non_binary_mask():
    w = h = 8
    gray = bytearray([100] * (w * h))
    bad_mask = bytearray([0, 128, 255] * 21 + [0])[:w * h]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", png_gray(bad_mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps([{"a": [0, 0], "b": [1, 1]}])},
    )
    assert r.status_code == 400
    assert "二值" in r.json()["message"]


def test_rejects_size_mismatch():
    gray = bytearray([100] * 64)
    mask = [bytearray(4) for _ in range(4)]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, 8, 8), "image/png"),
            "mask": ("mask.png", mask_png(mask, 4, 4), "image/png"),
        },
        data={"pairs": json.dumps([{"a": [0, 0], "b": [1, 1]}])},
    )
    assert r.status_code == 400
    assert "尺寸" in r.json()["message"]


def test_rejects_oversized_image():
    w = h = 513
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps([{"a": [0, 0], "b": [1, 1]}])},
    )
    assert r.status_code == 400
    assert "512" in r.json()["message"]


def test_rejects_non_png():
    buf = io.BytesIO()
    Image.new("L", (4, 4), 100).save(buf, format="JPEG")
    mask = [bytearray(4) for _ in range(4)]
    r = client.post(
        "/repair",
        files={
            "image": ("image.jpg", buf.getvalue(), "image/jpeg"),
            "mask": ("mask.png", mask_png(mask, 4, 4), "image/png"),
        },
        data={"pairs": json.dumps([{"a": [0, 0], "b": [1, 1]}])},
    )
    assert r.status_code == 400
    assert "PNG" in r.json()["message"]


def test_rejects_bad_pairs_json():
    w = h = 8
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": "not-json"},
    )
    assert r.status_code == 400


def test_mode1_binary_mask_accepted():
    # 模式 1（1 位二值）PNG 也应被接受；掩膜只含两个端点
    w = h = 12
    gray = bytearray([150] * (w * h))
    img1 = Image.new("1", (w, h), 0)
    img1.putpixel((4, 5), 1)
    img1.putpixel((8, 5), 1)
    buf = io.BytesIO()
    img1.save(buf, format="PNG")
    pairs = [{"a": [4, 5], "b": [8, 5]}]
    r = client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", buf.getvalue(), "image/png"),
        },
        data={"pairs": json.dumps(pairs)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
