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
  test_solver.py     # 求解器单元测试（18 例）
  test_api.py        # 接口层测试（10 例，含无解返回原掩膜、各类 400）
  test_fuzz.py       # fuzz 对拍暴力穷举（200 例随机网格）
  fuzz_crosscheck.py # 独立暴力实现
examples/
  generate_example.py# 生成演示用灰度图/掩膜/pairs.json
  call_repair.py     # 请求示例，落盘补线掩膜
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

## 本地开发与自测

```bash
pip install -r requirements-dev.txt
python3 -m pytest                 # 29 个测试
python3 tests/fuzz_crosscheck.py 2000   # 2000 例对拍独立暴力穷举
python3 -m uvicorn app.main:app --port 8000
```

## 已知边界 / 未完成项

- 运行环境未安装 Docker，`Dockerfile` 与 `docker-compose.yml` 已通过 YAML/关键字与语法静态核对，
  但未在本机实际 `docker compose up` 验证；服务本身已用 uvicorn 实跑并通过端到端请求验证。
- “最多拐两次”按 Manhattan 三段折线枚举；同行/同列端点在此约束下只有直线路径，
  无法绕行（若直线被挡且区域内无合法路径，该对判无解或待复核，符合规则字面含义）。
- 内存中以逐行字节数组表示图像，512×512 与 8 对规模下实测求解 < 0.4 秒；
  当前未对请求体大小做额外限额（仅约束边长）。
