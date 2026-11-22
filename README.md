# 旧式线路底片补线 API

沿二值走线掩膜上的残墨，按最低原图灰度代价补回褪色线段的 FastAPI 服务。

## 规则与实现的对应关系

| 需求 | 实现位置 |
| --- | --- |
| 同尺寸 8 位灰度 PNG + 二值掩膜，每边 ≤ 512 | `app/main.py` 的 `_load_gray_png`（校验 PNG、模式 `L`/`1`、掩膜仅 0/255、尺寸一致、边长上限） |
| 最多 8 对端点，补线区每边 ≤ 24 | `app/solver.py` 的 `validate_pair_inputs`（区域可省略，自动生成包住两端的方形区） |
| 区内横/竖走线，最多拐两次 | `enumerate_paths`：0 转弯直线、1 转弯两条 L、2 转弯 H-V-H（逐列）/ V-H-V（逐行），去重 |
| 不覆盖原走线、仅在本对端点处与原走线四邻接 | `_build_candidate` 逐像素校验掩膜与四邻域 |
| 补线之间不重叠、不四邻接 | 每个候选生成 `forbid` 禁占集（自身+端点+各自四邻域），DFS 做**双向**冲突检查 |
| 灰度总和最小 | `solve` 中按冲突关系拆连通分量，记忆化 DFS 求全局最低代价 |
| 只补所有最低代价方案中路径一致的对 | 逐对固定每个候选求条件最优：条件最优等于全局最优的候选集合大小为 1 才 `apply`，否则 `review` |
| 无法同时接通 | 返回 `status=no_solution`、原因说明与**原掩膜**（逐像素不变） |
| 先确认全部能接通，再预览指定端点对 | `stream_repair`：先对全部分量做存在性搜索，再按请求顺序发 `preview` |
| 与预览对无直接/间接冲突的端点组延后求解 | 连通分量即“冲突组”；不含预览对的分量在预览事件发出后才求最优 |
| 流式结果与普通请求等价 | 两条路径共用 `prepare_repair` / `_supported_candidates` / `_assemble` |
| 无解只发无解结果与原掩膜 | 流式求解在预览前返回时只产出单个 `result` 事件 |

### 代价与一致性判定

- 一条候选的代价 = 其**新增像素**（不含端点，端点本就是原走线）在原图上的灰度值之和。
- DFS 枚举所有合法组合求全局最低代价 `C*`；再对每对端点的每个候选求“强制选它时的最优代价”，
  等于 `C*` 的候选构成该对在全部最低代价方案中的路径集合：
  - 集合只有 1 条路径 → 路径在所有最低代价方案中一致，写入掩膜（`apply`）；
  - 多于 1 条 → 列为待复核（`review`），**不写入**，响应里给出全部备选路径供人工裁决。
- 正确性通过 `tests/fuzz_crosscheck.py` 与一份**完全独立的暴力穷举实现**（itertools 全组合）
  在 2000+ 随机小网格上逐案例核对：可行性、最低代价、每对 apply/review 判定、备选路径集合完全一致。

## 目录结构

```
app/
  solver.py          # 纯函数核心：路径枚举、约束校验、最优化与一致性判定
  main.py            # FastAPI：PNG 校验、请求解析、掩膜写回
tests/
  test_solver.py     # 求解器单元测试（27 例，含流式预览/延后求解）
  test_api.py        # 接口层测试（20 例，含无解返回原掩膜、各类 400、流式响应）
  test_fuzz.py       # fuzz 对拍暴力穷举（200 例随机网格）
  fuzz_crosscheck.py # 独立暴力实现 + 流式一致性核对
examples/
  generate_example.py# 生成演示用灰度图/掩膜/pairs.json
  call_repair.py     # 请求示例，落盘补线掩膜
  generate_stream_example.py # 生成流式演示数据（含独立端点组）
  call_repair_stream.py      # 流式请求示例：先预览再落盘整图掩膜
Dockerfile  docker-compose.yml  requirements*.txt
```

## 用 Docker Compose 启动

```bash
docker compose up --build
# 健康检查
curl http://127.0.0.1:8000/health
```

## 请求示例（能直接得到补线掩膜）

先生成演示数据（仓库已附带生成结果，可跳过）：

```bash
python3 examples/generate_example.py
```

`curl` 调用，把返回的 base64 掩膜解码成 PNG：

```bash
curl -s -X POST http://127.0.0.1:8000/repair \
  -F "image=@examples/film_gray.png;type=image/png" \
  -F "mask=@examples/film_mask.png;type=image/png" \
  -F "pairs=<examples/pairs.json" \
  | python3 -c "import sys,json,base64; b=json.load(sys.stdin); \
open('repaired_mask.png','wb').write(base64.b64decode(b['mask_png_base64'])); \
print(b['status'], 'apply=',b.get('applied_pair_count'),'review=',b.get('review_pair_count'))"
```

或等价的 Python 客户端（自动落盘到 `examples/output/repaired_mask.png`）：

```bash
python3 examples/call_repair.py
```

### 请求字段

- `image`：8 位灰度 PNG（模式 `L`），每边 ≤ 512。
- `mask`：二值 PNG（模式 `1` 或仅含 0/255 的 `L`），与原图同尺寸。
- `pairs`：JSON 数组（表单字符串字段），至多 8 个元素：

```json
[
  {"a": [12, 18], "b": [24, 18], "region": [10, 14, 26, 22]},
  {"a": [44, 14], "b": [54, 26]}
]
```

- `a`/`b` 为端点 `[x, y]`，必须落在掩膜前景上且全局唯一；
- `region` 可选，`[x0, y0, x1, y1]` 含端点、两边各 ≤ 24；省略时自动取包住两端的最小方形。

### 响应

- 成功：`status="ok"`，含 `min_total_cost`、每对 `decision`（`apply`/`review`）、
  新增像素坐标、待复核对的 `alternatives`，以及 `mask_png_base64`（补线后的二值 PNG）。
- 无解：HTTP 200 + `status="no_solution"`，`mask_png_base64` 为**原掩膜**，并带 `reason`。
- 输入非法：HTTP 400 + 中文错误信息。

## 可选流式响应（先预览指定端点对）

原请求不变，额外两个**可选**表单字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `stream` | 否 | `true` 时返回 NDJSON（`application/x-ndjson`，每行一个 JSON 事件） |
| `preview_pairs` | `stream=true` 时必填 | 要先看的端点对编号，1 基，与响应中的 `pair_id` 一致；`"2,1"` 或 `"[2,1]"`，顺序即预览顺序 |

事件顺序与语义：

1. **先确认全部端点对能同时接通**（对整个冲突图做一次存在性搜索，不算最优）。
   任一冲突组无法同时接通 → 只发一条 `result`（`status=no_solution` + 原掩膜），不发预览。
2. **`preview` 事件**（按请求顺序，可多条）：列出该对在**所有最低总代价方案**中的路径
   `paths`，`path_count` 为路径数；**只有 1 条**时 `decision="apply"`，否则 `"review"`。
3. 与预览对**没有直接/间接候选冲突**的端点组（不含预览对的连通分量）在预览发出后才求最优。
4. **`result` 事件**：与普通请求的响应体完全相同（多一个 `"event": "result"` 字段）。

```json
{"event":"preview","pair_id":2,"decision":"review","path_count":22,"component_min_cost":4410,"paths":[{"turns":1,"cost":4410,"new_cells":[[38,11],"..."]}]}
{"event":"result","status":"ok","min_total_cost":5390,"applied_pair_count":2,...}
```

一条含独立端点组的演示请求（`examples/stream_pairs.json` 的第 3 对与前两对无冲突）：

```bash
docker compose up --build -d
# 生成流式演示数据（仓库已附带生成结果，可跳过）
python3 examples/generate_stream_example.py
# 预览第 2、1 对；第 3 对（独立组）在预览之后才求最优
python3 examples/call_repair_stream.py
```

等价的 `curl`：

```bash
curl -sN -X POST http://127.0.0.1:8000/repair \
  -F "image=@examples/stream_gray.png;type=image/png" \
  -F "mask=@examples/stream_mask.png;type=image/png" \
  -F "pairs=<examples/stream_pairs.json" \
  -F "stream=true" -F "preview_pairs=2,1"
```

不传 `stream`（或 `stream=false`）时，响应与旧版本完全一致（`application/json`）。

## 本地开发与自测

```bash
pip install -r requirements-dev.txt
python3 -m pytest                 # 48 个测试
python3 tests/fuzz_crosscheck.py 2000   # 2000 例对拍独立暴力穷举（含流式一致性）
python3 -m uvicorn app.main:app --port 8000
```

## 验证记录

- `pytest`：48 例全过（原 29 例 + 流式求解 9 例 + 流式接口 10 例）。
- `tests/fuzz_crosscheck.py 2000`：除原有可行性 / 最低代价 / apply-review 判定 /
  备选集合外，新增核对**流式结果与普通求解逐字段一致**、预览判定与路径数和最终结果一致、
  无解时无预览事件。
- 流式预览发出时机：`test_stream_preview_emitted_before_independent_group_solves`
  惰性消费第一条预览事件，断言此时只对含预览对的分量求过最优，独立分量尚未求解；
  预览之后再求解独立分量，最终整图结果与普通请求一致。
- 并列最优：`test_stream_preview_lists_all_optimal_paths_and_apply_rule` /
  `test_stream_tie_marks_review_and_lists_paths` 核对均匀灰度下多路径同代价 →
  `decision=review`、列出全部路径；唯一最优 → `apply` 且写入掩膜。
- 无解：`test_stream_no_solution_sends_only_result_with_original_mask` 核对只发一条
  `no_solution` 结果且返回掩膜与输入逐像素一致。
- 实跑：`uvicorn` 启动后 `examples/call_repair_stream.py` 依次收到
  `preview(pair=2, review, 22 条)`、`preview(pair=1, apply, 1 条)`、`result(ok, cost=5390)`，
  落盘掩膜与普通请求的 `mask_png_base64` 完全一致。

## 已知边界 / 未完成项

- 运行环境未安装 Docker，`Dockerfile` 与 `docker-compose.yml` 只做了静态核对，
  未在本机 `docker compose up` 实跑；服务本身已用 `uvicorn` 实跑并通过端到端请求验证。
- 流式的“先发预览”指服务端按事件顺序尽快 flush NDJSON；中间的求解是同步计算，
  预览事件发出前仍会先把“含预览对的分量”算完，因此预览延迟取决于该分量的规模。
- 流式响应若在传输途中客户端断开，剩余分量（含独立组）不再计算；这属于预期行为。
- “最多拐两次”按 Manhattan 三段折线枚举；同行/同列端点在此约束下只有直线路径，
  无法绕行（若直线被挡且区域内无合法路径，该对判无解或待复核，符合规则字面含义）。
- 内存中以逐行字节数组表示图像，512×512 与 8 对规模下实测求解 < 0.4 秒；
  当前未对请求体大小做额外限额（仅约束边长）。
