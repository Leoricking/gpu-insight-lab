# GPU Insight Lab

**跨廠商 GPU 效能驗證與工作負載診斷平台**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)]()

GPU Insight Lab 是一套以 Python 為基礎的 GPU 效能診斷平台，能夠收集系統與 GPU 遙測資料、
執行 CUDA 基準測試核心、套用以證據為基礎的診斷引擎，並產生多種格式的報告——
全程只需一行 CLI 指令或透過 PySide6 圖形介面操作。

---

## 目錄

1. [功能特色](#功能特色)
2. [快速開始](#快速開始)
3. [安裝方式](#安裝方式)
4. [CLI 使用說明](#cli-使用說明)
5. [GUI 使用說明](#gui-使用說明)
6. [基準測試套件](#基準測試套件)
7. [診斷引擎](#診斷引擎)
8. [GPU Insight 評分](#gpu-insight-評分)
9. [報告格式](#報告格式)
10. [歷史記錄](#歷史記錄)
11. [原生 CUDA 執行檔](#原生-cuda-執行檔)
12. [設定檔](#設定檔)
13. [開發與測試](#開發與測試)
14. [系統架構](#系統架構)
15. [限制與免責聲明](#限制與免責聲明)
16. [授權與商標聲明](#授權與商標聲明)

---

## 功能特色

- **系統與 GPU 資訊收集** — CPU、記憶體、作業系統、透過 pynvml（備援為 nvidia-smi）取得
  NVIDIA GPU 資訊、CUDA/ROCm 工具鏈偵測、PCIe 連線狀態、AMD GPU 存根支援
- **原生 CUDA 基準測試** — 7 個 C++ CUDA 核心（向量加法、歸約、矩陣轉置、樸素與分塊 GEMM、
  記憶體頻寬、串流管線、影像灰階轉換），編譯支援 sm_75 至 sm_90
- **CPU 基準測試** — 以 NumPy 實作的參考版本（向量加法、矩陣乘法、影像灰階），包含暖機
  與多次重複測量的統計數據
- **以證據為基礎的診斷** — 9 條規則，每項發現均包含非空的 `evidence` 字串，引用具體量測值；
  不進行推測
- **GPU Insight 評分** — 0–100 的綜合分數，涵蓋 6 個類別，並附信心評級
- **多格式報告** — JSON、CSV、Markdown、HTML（Jinja2）、Excel（openpyxl，7 個工作表）
- **SQLite 歷史記錄** — 版本化結構描述遷移、Session 比較（含差異百分比）
- **PySide6 圖形介面** — QMainWindow 側邊欄導覽、QThread 工作執行緒（GUI 不會凍結）
- **argparse CLI** — 8 個指令，支援 `--json` 旗標及結束碼 0/1/2
- **優雅降級** — 每個收集器均回傳部分資料；硬體或工具缺失時會產生具資訊量的診斷結果，而非崩潰

---

## 快速開始

```bash
# 安裝（包含所有選用依賴）
pip install -e ".[all]"

# 執行快速基準測試並顯示結果
gpu-insight quick-test

# 取得 JSON 格式的系統資訊
gpu-insight system-info --json

# 執行完整基準測試套件並儲存 HTML 報告
gpu-insight full-test --output-dir ~/gpu_reports

# 啟動圖形介面
gpu-insight gui
```

---

## 安裝方式

### 系統需求

- Python 3.11 或 3.12
- Windows 10/11 或 Linux（Ubuntu 20.04+）
- 建議搭配 NVIDIA GPU 及驅動程式版本 520+（無 GPU 時仍可執行 CPU 基準測試）
- CUDA Toolkit 11.8+ 為選用（僅重新編譯原生核心時需要）

### 從原始碼安裝

```bash
git clone https://github.com/yourusername/gpu-insight-lab.git
cd gpu-insight-lab

# 最小安裝（CLI + 收集器 + 診斷）
pip install -e .

# 完整安裝（新增 GUI、報告、pynvml、Pillow）
pip install -e ".[all]"

# 開發安裝（新增 pytest）
pip install -e ".[all,dev]"
```

### 驗證安裝

```bash
python -m compileall app collectors benchmarks diagnosis storage reports profilers workloads -q
python -m pytest tests/ -v
gpu-insight --version
```

---

## CLI 使用說明

```
usage: gpu-insight [-h] [--version] {system-info,gpu-info,quick-test,full-test,
                                      single-test,diagnose,report,gui} ...
```

### 指令列表

| 指令 | 說明 | 主要選項 |
|------|------|---------|
| `system-info` | 收集並顯示系統與 GPU 資訊 | `--json` |
| `gpu-info` | GPU 專屬資訊（驅動程式、顯示記憶體、時脈） | `--json` |
| `quick-test` | 執行快速基準測試套件 | `--json`、`--no-save`、`--output-dir` |
| `full-test` | 執行完整基準測試套件 | `--json`、`--no-save`、`--output-dir`、`--repeat N` |
| `single-test` | 執行單一指定基準測試 | `--test NAME`、`--repeat N`、`--size N`、`--block-size N` |
| `diagnose` | 對最近一次 Session 執行診斷 | `--session-id ID`、`--json` |
| `report` | 從 Session 產生報告 | `--format json/csv/md/html/excel`、`--output FILE` |
| `gui` | 啟動 PySide6 圖形介面 | — |

### 結束碼

| 代碼 | 意義 |
|------|------|
| 0 | 成功 |
| 1 | 錯誤（收集或基準測試失敗） |
| 2 | 部分成功（部分資料缺失，工具已繼續執行） |

### 使用範例

```bash
# 以 JSON 格式顯示所有 GPU 資訊
gpu-insight gpu-info --json

# 執行 vector_add 基準測試，重複 20 次
gpu-insight single-test --test vector_add --repeat 20

# 從最近一次 Session 產生 HTML 報告
gpu-insight report --format html --output report.html

# 完整測試，不儲存至資料庫，報告儲存至自訂目錄
gpu-insight full-test --no-save --output-dir /tmp/gpu_reports

# 對指定的已儲存 Session 執行診斷
gpu-insight diagnose --session-id abc123
```

---

## GUI 使用說明

```bash
gpu-insight gui
```

PySide6 圖形介面提供：
- **儀表板** — GPU Insight 評分、關鍵指標、快速操作按鈕
- **系統資訊** — 所有收集的系統與 GPU 資料（樹狀檢視）
- **基準測試** — 執行快速/完整/單一基準測試，含進度列
- **診斷** — 依嚴重程度篩選的診斷結果與證據
- **歷史記錄** — Session 時間軸、Session 比較、趨勢圖
- 另有 6 個 Pro/Lab 版功能的預留頁面

所有基準測試均在 QThread 工作執行緒中執行；量測期間 GUI 不會凍結。

---

## 基準測試套件

### 原生 CUDA 核心

| 核心 | 量測項目 | 輸出指標 |
|------|---------|---------|
| `vector_add` | 記憶體頻寬（受限型） | GB/s |
| `reduction` | 平行歸約效率 | GB/s 輸入 |
| `transpose` | 合併記憶體存取 | GB/s |
| `gemm_naive` | 未最佳化 GEMM 基準 | GFLOP/s |
| `gemm_tiled` | 共享記憶體分塊 GEMM | GFLOP/s |
| `memory_bandwidth` | 峰值裝置記憶體頻寬 | GB/s |
| `stream_pipeline` | 非同步並行執行 | GB/s + ms 延遲 |
| `image_grayscale` | 2D 像素處理 | Mpix/s |

### CPU Python 基準測試

| 基準測試 | 實作方式 | 指標 |
|---------|---------|------|
| `cpu_vector_add` | NumPy 逐元素運算 | GB/s |
| `cpu_matrix_multiply` | NumPy `@`（BLAS） | GFLOP/s |
| `cpu_image_grayscale` | NumPy 加權求和 | Mpix/s |

### 統計數據

每項基準測試均會產生：平均值、中位數、最小值、最大值、標準差（母體）、變異係數（CV）。
CV 過高（> 10%）會觸發 `HIGH_VARIANCE` 診斷規則。

### 暖機

在開始量測前會捨棄 3 次暖機迭代。預設重複次數為 10 次。
使用 `--repeat N` 可增加重複次數以獲得更穩定的結果。

---

## 診斷引擎

診斷引擎對每個 Session 套用 9 條以證據為基礎的規則：

| 規則 ID | 類別 | 偵測項目 |
|---------|------|---------|
| `DRIVER_MISSING` | 環境 | 未安裝或無法偵測 NVIDIA 驅動程式 |
| `PCIE_BOTTLENECK` | PCIe/記憶體 | GPU 執行於低於最大值的連線寬度/世代 |
| `THERMAL_THROTTLE` | 散熱/電源 | 溫度 >= 83°C 或有效的降頻原因 |
| `LOW_MEMORY_BANDWIDTH` | 效能 | 頻寬明顯低於該 GPU 型號的預期值 |
| `LOW_COMPUTE_THROUGHPUT` | 效能 | FP32 GFLOP/s 低於該 GPU 型號的預期值 |
| `CORRECTNESS_FAILURE` | 核心正確性 | 基準測試輸出與 CPU 參考值不符 |
| `HIGH_VARIANCE` | 一致性 | 重複測試的變異係數 > 10% |
| `CUDA_TOOLKIT_MISSING` | 環境 | PATH 或 CUDA_HOME 中找不到 nvcc |
| `AMD_NOT_VALIDATED` | 相容性 | 偵測到 AMD GPU；結果為 NOT_VALIDATED |

**證據政策**：每項發現均包含非空的 `evidence` 字串，引用具體量測值。
無法取得足夠資料的規則會回傳 `None`（而非低信心發現）。
完整規則說明請參閱 [docs/DIAGNOSIS_RULES.md](docs/DIAGNOSIS_RULES.md)。

---

## GPU Insight 評分

GPU Insight 評分是涵蓋 6 個類別的 0–100 綜合分數：

| 類別 | 最高分 | 量測項目 |
|------|-------|---------|
| 環境準備度 | 20 | 驅動程式、工具鏈、toolkit 是否存在 |
| GPU 執行環境 | 15 | GPU 已偵測、pynvml 正常、裝置可存取 |
| PCIe / 記憶體 | 20 | 連線寬度/世代達最大值、頻寬符合目標 |
| 核心正確性 | 20 | 所有基準測試的正確性檢查均通過 |
| 效能一致性 | 15 | 重複測試的 CV 較低 |
| 散熱 / 電源 | 10 | 溫度在安全範圍內、無降頻 |

評分信心依可用資料量分為 HIGH、MEDIUM、LOW。
使用缺失硬體資料計算的分數將標示為 LOW 信心。

---

## 報告格式

| 格式 | 指令 | 備註 |
|------|------|------|
| JSON | `--format json` | 完整 Session 資料，機器可讀 |
| CSV | `--format csv` | 基準測試結果表格，與 Excel 相容 |
| Markdown | `--format md` | 人類可讀，適合 GitHub/Confluence |
| HTML | `--format html` | 自給自足（內聯 CSS），不需 CDN |
| Excel | `--format excel` | 7 個工作表：摘要、系統、GPU、基準測試、診斷、評分、原始資料 |

HTML 報告使用 Jinja2 模板，可在隔離網路環境中使用（無外部依賴）。
Excel 報告使用 openpyxl，包含凍結窗格、自動篩選及嚴重程度色彩編碼；
不使用合併儲存格，以便程式存取。

---

## 歷史記錄

基準測試 Session 以 SQLite 儲存於 `~/.gpu_insight_lab/gpu_insight.sqlite`，
啟用 WAL 模式並強制執行外鍵約束。

```bash
# 列出所有已儲存的 Session
gpu-insight report --list-sessions

# 比較兩個 Session（顯示差異百分比）
gpu-insight report --compare SESSION_ID_1 SESSION_ID_2

# 從已儲存的 Session 產生報告
gpu-insight report --session-id SESSION_ID --format html
```

結構描述遷移會自動執行。資料庫版本記錄於 `schema_version` 資料表。

---

## 原生 CUDA 執行檔

原生基準測試執行檔（`gpu_insight_benchmark` / `gpu_insight_benchmark.exe`）
使用 CUDA Events 提供 GPU 端核心計時。Python 透過 subprocess 進行協調並解析 JSON stdout。

### 從原始碼編譯

```bash
cd native
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

需要 CMake 3.18+、CUDA Toolkit 11.8+、C++17 編譯器。

編譯支援的計算能力：sm_75（Turing）、sm_80（Ampere）、sm_86（Ampere）、
sm_89（Ada）、sm_90（Hopper）。

### 獨立執行

```bash
./gpu_insight_benchmark --device-info
./gpu_insight_benchmark --quick --repeat 10
./gpu_insight_benchmark --full --output results.json
./gpu_insight_benchmark --test gemm_tiled --size 1024 --block-size 32
```

Python 在沒有原生執行檔的情況下仍可正常運作（CPU 基準測試繼續執行；
CUDA 核心基準測試顯示為不可用）。

---

## 設定檔

預設設定檔儲存於 `~/.gpu_insight_lab/config.json`：

```json
{
  "output_dir": "~/gpu_insight_lab_output",
  "db_path": "~/.gpu_insight_lab/gpu_insight.sqlite",
  "native_binary_path": null,
  "default_repeat": 10,
  "default_timeout": 120,
  "warmup_runs": 3,
  "save_sessions": true,
  "log_level": "INFO"
}
```

所有設定均可透過 CLI 旗標或環境變數覆寫：

| 環境變數 | 設定鍵 | 預設值 |
|---------|-------|-------|
| `GPU_INSIGHT_OUTPUT_DIR` | `output_dir` | `~/gpu_insight_lab_output` |
| `GPU_INSIGHT_DB_PATH` | `db_path` | `~/.gpu_insight_lab/gpu_insight.sqlite` |
| `GPU_INSIGHT_REPEAT` | `default_repeat` | `10` |
| `GPU_INSIGHT_TIMEOUT` | `default_timeout` | `120` |

---

## 開發與測試

```bash
# 執行所有測試
python -m pytest tests/ -v

# 執行含覆蓋率報告的測試
python -m pytest tests/ --cov=app --cov=collectors --cov=benchmarks \
    --cov=diagnosis --cov=storage --cov=reports -v

# 編譯檢查所有 Python 模組
python -m compileall app collectors benchmarks diagnosis storage reports \
    profilers workloads tests -q

# CLI 冒煙測試
gpu-insight --version
gpu-insight system-info --json
gpu-insight quick-test
python -c "from app.gui.main_window import MainWindow; print('GUI import OK')"
```

### 測試覆蓋範圍

| 測試檔案 | 測試項目 |
|---------|---------|
| `tests/test_smoke.py` | 模組匯入、CLI `--help`、`--version`、JSON 輸出結構 |
| `tests/test_diagnosis.py` | 每條診斷規則、證據政策、資料不足時規則回傳 None |
| `tests/test_storage.py` | 遷移冪等性、儲存/載入來回測試、Session 比較 |

---

## 系統架構

```
app/           PySide6 GUI、argparse CLI、設定、功能、品牌常數
collectors/    系統、NVIDIA、CUDA、PCIe、工具、AMD 資料收集
benchmarks/    CPU 基準測試、原生執行器、Session 協調器、資料結構
diagnosis/     以證據為基礎的規則引擎、評分
storage/       SQLite 持久化儲存與遷移
reports/       JSON、CSV、Markdown、HTML、Excel 輸出
profilers/     Nsight Systems、Nsight Compute、nvidia-smi 監控、ROCm 存根
workloads/     影像批次、媒體管線、LLM 預留位置、自訂指令
native/        CUDA C++ 基準測試核心（獨立執行檔）
tests/         pytest 測試套件
docs/          架構、方法論、規則、HIP 指南、商業藍圖、面試指南
```

完整的資料流程圖與設計決策，請參閱 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 限制與免責聲明

- **以 NVIDIA 為主**：透過 pynvml/nvidia-smi 完整支援 NVIDIA GPU。
  AMD GPU 結果全程標示為 `NOT_VALIDATED`。不支援 Intel GPU。
- **單一 GPU**：每次 Session 僅測試一個 GPU。不涵蓋多 GPU 拓撲（NVLink、SLI）。
- **僅 FP32**：原生基準測試使用 32 位元浮點數。不測量 Tensor Core 效能（FP16/BF16/INT8）。
- **微基準測試**：結果描述合成負載下的硬體能力，非實際應用效能。
  真實 ML 訓練/推論吞吐量涉及許多其他因素。
- **基準測試影響系統狀態**：以完整負載執行 GPU 基準測試會提高 GPU 溫度，
  可能造成降頻。請勿在生產系統執行即時工作負載時使用。
- **診斷僅供參考**：發現結果基於啟發式規則，應用於有限資料。
  這些是調查的起點，而非確定性診斷。
- **Windows 計時**：由於 WDDM 排程，Windows 上 GPU 核心計時的抖動可能高於 Linux。
  Windows 上的結果 CV 可能高於等效的 Linux 執行。

---

## 授權與商標聲明

版權所有 (c) 2026 Rossi。依 [MIT 授權條款](LICENSE) 發布。

「GPU Insight Lab」為未登記商標。使用「GPU Insight Lab」名稱來識別衍生作品、
競爭產品或服務，需取得書面授權。

本軟體為獨立專案，與 NVIDIA Corporation、Advanced Micro Devices（AMD）
及其子公司無任何關聯、背書或贊助關係。「CUDA」為 NVIDIA Corporation 的商標。
「ROCm」與「HIP」為 Advanced Micro Devices, Inc. 的商標。
其他所有商標均為各自所有人的財產。
