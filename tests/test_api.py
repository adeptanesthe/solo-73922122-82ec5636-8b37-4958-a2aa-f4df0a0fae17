"""FastAPI 接口层测试：文件校验、端到端补线、无解返回原掩膜。"""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.solver import validate_pair_inputs

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


# ---------- 流式响应 ----------

def _post(mask, gray, pairs, w, h, **extra):
    return client.post(
        "/repair",
        files={
            "image": ("image.png", png_gray(gray, w, h), "image/png"),
            "mask": ("mask.png", mask_png(mask, w, h), "image/png"),
        },
        data={"pairs": json.dumps(pairs), **extra},
    )


def _stream_events(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_stream_preview_then_result_matches_plain():
    w = h = 24
    gray = bytearray([200] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[6][4] = mask[6][10] = 1
    pairs = [{"a": [4, 6], "b": [10, 6]}]
    plain = _post(mask, gray, pairs, w, h).json()

    events = _stream_events(_post(
        mask, gray, pairs, w, h, stream="true", preview_pairs="1"
    ))
    assert [e["event"] for e in events] == ["preview", "result"]
    preview, result = events
    assert preview["pair_id"] == 1
    assert preview["decision"] == "apply"
    assert preview["path_count"] == 1
    assert preview["paths"][0]["new_cells"] == [[5, 6], [6, 6], [7, 6], [8, 6], [9, 6]]
    # result 与普通请求等价
    result.pop("event")
    assert result == plain


def test_stream_tie_marks_review_and_lists_paths():
    w = h = 20
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[3][3] = mask[8][8] = 1
    pairs = [{"a": [3, 3], "b": [8, 8]}]
    events = _stream_events(_post(
        mask, gray, pairs, w, h, stream="true", preview_pairs="1"
    ))
    preview = events[0]
    assert preview["decision"] == "review"
    assert preview["path_count"] >= 2
    assert len(preview["paths"]) == preview["path_count"]
    cells = [tuple(c) for p in preview["paths"] for c in p["new_cells"]]
    assert cells and all(mask[y][x] == 0 for x, y in cells)
    # 待复核 -> 整图结果中该对不写入
    assert events[1]["applied_cells"] == []
    assert events[1]["review_pair_count"] == 1


def test_stream_no_solution_sends_only_result_with_original_mask():
    w = h = 12
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    for x, y in [(1, 4), (5, 4), (1, 5), (5, 5)]:
        mask[y][x] = 1
    pairs = [
        {"a": [1, 4], "b": [5, 4], "region": [1, 4, 5, 4]},
        {"a": [1, 5], "b": [5, 5], "region": [1, 5, 5, 5]},
    ]
    events = _stream_events(_post(
        mask, gray, pairs, w, h, stream="true", preview_pairs="1"
    ))
    assert [e["event"] for e in events] == ["result"]
    assert events[0]["status"] == "no_solution"
    assert "冲突" in events[0]["reason"]
    out = decode_mask(events[0]["mask_png_base64"], w, h)
    for y in range(h):
        for x in range(w):
            assert out.getpixel((x, y)) == (255 if mask[y][x] else 0)


def test_stream_independent_group_emitted_after_preview():
    w = h = 24
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    for x, y in [(2, 3), (8, 3), (2, 15), (8, 15)]:
        mask[y][x] = 1
    pairs = [
        {"a": [2, 3], "b": [8, 3], "region": [1, 1, 10, 6]},
        {"a": [2, 15], "b": [8, 15], "region": [1, 13, 10, 18]},
    ]
    stream_events = _stream_events(_post(
        mask, gray, pairs, w, h, stream="true", preview_pairs="1"
    ))
    # 只预览第 1 对，但整图结果包含两对（独立组在预览后补算）
    kinds = [e["event"] for e in stream_events]
    assert kinds == ["preview", "result"]
    result = stream_events[-1]
    assert result["applied_pair_count"] == 2
    plain = _post(mask, gray, pairs, w, h).json()
    result.pop("event")
    assert result == plain


def test_stream_requires_preview_pairs():
    w = h = 8
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[2][2] = mask[2][5] = 1
    r = _post(mask, gray, [{"a": [2, 2], "b": [2, 5]}], w, h, stream="true")
    assert r.status_code == 400
    assert "preview_pairs" in r.json()["message"]


def test_stream_rejects_out_of_range_preview_pair():
    w = h = 8
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[2][2] = mask[2][5] = 1
    r = _post(
        mask, gray, [{"a": [2, 2], "b": [2, 5]}], w, h,
        stream="true", preview_pairs="2",
    )
    assert r.status_code == 400
    assert "超出" in r.json()["message"]


def test_stream_rejects_bad_flag():
    w = h = 8
    gray = bytearray([100] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[2][2] = mask[2][5] = 1
    r = _post(
        mask, gray, [{"a": [2, 2], "b": [2, 5]}], w, h,
        stream="maybe", preview_pairs="1",
    )
    assert r.status_code == 400


def test_stream_false_is_plain_response():
    w = h = 20
    gray = bytearray([150] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[5][3] = mask[5][9] = 1
    r = _post(
        mask, gray, [{"a": [3, 5], "b": [9, 5]}], w, h,
        stream="false", preview_pairs="1",
    )
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["status"] == "ok"


def test_stream_pairs_json_array_accepted():
    w = h = 20
    gray = bytearray([150] * (w * h))
    mask = [bytearray(w) for _ in range(h)]
    mask[5][3] = mask[5][9] = 1
    events = _stream_events(_post(
        mask, gray, [{"a": [3, 5], "b": [9, 5]}], w, h,
        stream="true", preview_pairs="[1]",
    ))
    assert events[0]["pair_id"] == 1


def test_stream_preview_emitted_before_independent_group_solves(monkeypatch):
    """惰性消费预览事件时核对：预览行产出前，独立端点组尚未求最优。"""
    import app.main as main_mod
    import app.solver as solver

    w = h = 24
    gray = [bytearray([100] * w) for _ in range(h)]
    mask = [bytearray(w) for _ in range(h)]
    for x, y in [(2, 3), (8, 3), (2, 15), (8, 15)]:
        mask[y][x] = 1
    payload = [
        {"a": [2, 3], "b": [8, 3], "region": [1, 1, 10, 6]},
        {"a": [2, 15], "b": [8, 15], "region": [1, 13, 10, 18]},
    ]
    specs = validate_pair_inputs(payload, w, h)

    calls = []
    real_search = solver._search

    def spy(specs_arg, members, fixed=None, *, need_optimum=True):
        calls.append((tuple(members), need_optimum))
        return real_search(specs_arg, members, fixed, need_optimum=need_optimum)

    monkeypatch.setattr(solver, "_search", spy)
    prep = solver.prepare_repair(specs, mask, gray, w, h)

    events = main_mod.stream_repair_events(prep, [0], mask, w, h)
    first = next(events)
    calls_at_preview = list(calls)
    assert first["event"] == "preview"
    assert first["pair_id"] == 1
    # 预览前只允许对“含预览对的分量”求最优；独立组（第 2 对）尚未
    assert set(m for m, need in calls_at_preview if need) == {(0,)}
    assert ((1,), True) not in calls_at_preview
    # 存在性确认先于任何最优求解
    assert [need for _, need in calls_at_preview[:2]] == [False, False]

    rest = list(events)
    assert [e["event"] for e in rest] == ["result"]
    assert ((1,), True) in calls
    assert calls.index(((1,), True)) >= len(calls_at_preview)
    assert rest[0]["applied_pair_count"] == 2
