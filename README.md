cat > README.md << 'EOF'
# vllm-eagle3-bench

在 vLLM 上评测 **EAGLE3 投机解码（speculative decoding）自适应 k 控制器**的实验框架。

核心思路：**一次压测，无限回放**。只跑一次 `k=3` 的基准压测，把每个 decode step
的接受情况录下来，然后离线“回放”各种 k 调度策略，估算各自的吞吐量与浪费，
不必每次改 k 都重启 vLLM 重跑。

---

## 背景

投机解码用一个小 draft 模型连续猜 `k` 个 token，再由 target 模型一次性验证。
`k` 固定时有两难：

- `k` 太大 → 猜错的 draft 被浪费，verify 成本不变，吞吐反而下降；
- `k` 太小 → 投机收益不足，接近普通自回归解码。

理想做法是**按当前请求的接受率动态调整 k**。但每改一次规则就重启 vLLM、
重跑全量压测，成本极高。本框架把“采集”和“评估”解耦：

1. **采集**：固定 `k=3`、`max-concurrency=1` 跑一遍，hook 记录每个 step 的
   `num_drafts` / `num_accepted_tokens` 等字段；
2. **评估**：在离线日志上模拟任意 k 控制策略，用时间模型估算 `tok/s`、
   draft 浪费、k 的分布。

---

## 工作流
┌──────────────────────────────────────────────────────────────┐
│ 0. 新机器：给 vLLM 打一次 observe hook │
│ python3 install_observe_hook.py │
│ （卸载：python3 install_observe_hook.py --undo） │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ 1. 终端 A：起服务（k=3，EAGLE3） │
│ ./serve_k.sh 3 │
│ 启动日志里 speculative_config 不能出现 stats_reporting_level │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 终端 B：C=1 压测 + 转日志 + 回放（一条龙） │
│ ./run_c1_replay.sh <dataset> │
│ ├─ run_bench.sh vllm bench serve，C=1 │
│ ├─ observe_to_steps.py observe.jsonl → steps.jsonl │
│ └─ replay.py steps.jsonl → 5 种策略对比 │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 已有 steps 时：只重跑回放，不启动 vLLM │
│ ./replay_existing.sh │
│ 或 python3 replay.py --steps ... --rules B3,R1,R2 │
└──────────────────────────────────────────────────────────────┘

text

---

## 目录结构
vllm-eagle3-bench/
├── README.md
├── serve_k.sh # 启动 vLLM，参数 0/1/2/3 = num_speculative_tokens
├── run_bench.sh # 跑单个 vllm bench serve 任务
├── run_c1_replay.sh # 主入口：C=1 压测 → steps → replay
├── replay_existing.sh # 复用已有 steps 重跑 replay
├── install_observe_hook.py # 给 vLLM 打/卸 observe hook
├── observe_to_steps.py # observe.jsonl → steps.jsonl
├── replay.py # 在 steps 上模拟各种 k 控制策略
├── lib.py # JSONL 读写等工具函数
└── convert_datasets.py # 数据集格式转换（alpaca/gsm8k/humaneval/mtbench）

text

运行产物默认在 `/root/autodl-tmp/datasets/week1_results/`，**不在本仓库内**。

---

## 环境要求

| 项 | 版本 / 路径 |
|---|---|
| Python | 3.10+ |
| vLLM | 0.27.1（`observe` 签名与该版本对齐） |
| 虚拟环境 | `/root/autodl-tmp/venv`（脚本里硬编码，按需改） |
| Target 模型 | `/root/autodl-tmp/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master` |
| Draft 模型 | `/root/autodl-tmp/models/Llama-3.1-8B-Instruct-eagle3` |

> 路径都通过环境变量（`MODEL` / `DRAFT` / `TOKENIZER` / `RESULT_DIR`）可覆盖。

---

## 快速开始

### 0. 打 hook（新机器只需一次）

```bash
source /root/autodl-tmp/venv/bin/activate
cd ~/vllm-eagle3-bench
python3 install_observe_hook.py
输出 patched /.../vllm/v1/spec_decode/metrics.py 即成功。

1. 终端 A：起服务
bash
cd ~/vllm-eagle3-bench
./serve_k.sh 3
启动日志里应出现：

text
speculative-config: {"method": "eagle3", "model": "...", "num_speculative_tokens": 3, ...}
且不能出现 stats_reporting_level（vLLM 0.27.1 不支持）。

2. 终端 B：压测 + 回放
bash
source /root/autodl-tmp/venv/bin/activate
cd ~/vllm-eagle3-bench

./run_c1_replay.sh alpaca
./run_c1_replay.sh mtbench
./run_c1_replay.sh humaneval
./run_c1_replay.sh gsm8k
温度 0.7：

bash
T=0.7 ./run_c1_replay.sh alpaca
每个数据集独立存放结果，不会互相覆盖。

3. 已有 steps 时只重跑回放
bash
# 全部数据集
./replay_existing.sh

# 单个数据集 / 自定义策略
python3 replay.py \
  --steps /root/autodl-tmp/datasets/week1_results/steps_alpaca_k3_t0_c1.jsonl \
  --out-dir /root/autodl-tmp/datasets/week1_results/replay_alpaca_custom \
  --rules B3,R1,R2,U
脚本说明
脚本	作用	关键环境变量
serve_k.sh <0|1|2|3>	启动 vLLM。k=0 关闭投机解码	MODEL DRAFT PORT MAX_NUM_SEQS GPU_MEM
run_bench.sh	跑一次 vllm bench serve	DATASET K T C N RESULT_DIR BASE_URL
run_c1_replay.sh <dataset>	主入口，串起 bench → steps → replay	K(默认3) T(默认0) RESULT_DIR
replay_existing.sh	对已有 steps 重跑 replay	RESULT_DIR
install_observe_hook.py	打 hook；--undo 卸载	EAGLE3_STEP_LOG(运行时)
observe_to_steps.py	observe → steps，可按 output_lens 打包回请求	—
replay.py	回放策略，输出 summary csv/json + 每策略请求明细	—
convert_datasets.py	数据集转 vLLM jsonl	—
k 控制策略
replay.py 默认回放 5 种规则，用 --rules 选择（逗号分隔）：

规则	含义	公式
B3	固定 k=3 基线	k_try = 3
R1	简单加减	全接受：k ← min(k+2, 3)；否则 k ← max(k−1, 1)
R2old	旧版：弱窗口就关，且温度高也压 k	先按 R1；temperature ≥ 0.5 时 k ← min(k, 2)；最近 8 步均值 < 1.2 则 k ← 0
R2	新版：仅在高温时才可能关	先按 R1；temperature < 0.5 时保持 k ≥ 1；高温下 k ← min(k, 2)，弱窗口则 k ← 0
U	理想上界（oracle）	k_try = accept_len（实际接受数）
其中 accept_len = 该 step 被接受的 draft token 数（0..k_used）。

每步产出
给定尝试长度 k_try、实际接受数 m：

text
tokens_this_step = 1 + min(k_try, m)     # 1 是 target 的 bonus token
wasted_drafts    = max(k_try - m, 0)     # 多猜的 draft
leftover_unused  = max(m - k_try, 0)     # 本可接受但没猜的
时间模型
text
k_try = 0:  step_time_ms = t_ar
k_try > 0:  step_time_ms = k_try * t_draft + t_verify
默认 t_ar = 17.5 ms、t_draft = 2.1 ms、t_verify = 17.5 ms，可用
--t-ar-ms / --t-draft-ms / --t-verify-ms 覆盖。吞吐估算：

text
est_tok_s = total_tokens / (total_time_ms / 1000)
这是解析估算，不是实测 tok/s；用于在同一份日志上横向比较策略相对优劣。

输出格式
steps.jsonl（每行一个 decode step）
json
{
  "run_id": "alpaca_k3_t0_c1",
  "req_id": "req-0000",
  "dataset": "alpaca",
  "temperature": 0.0,
  "concurrency": 1,
  "step": 0,
  "k_used": 3,
  "accept_len": 3,
  "pos0": 1, "pos1": 1, "pos2": 1
}
replay_summary.csv / .json
字段	含义
rule	策略名
requests / steps	请求数 / 总 step 数
tokens	生成 token 总数（含 bonus）
wasted_drafts	被浪费的 draft token
leftover_unused	本可接受但未猜的 token
est_tok_s	估算吞吐
mean_k	平均尝试长度
off_steps	k=0 的 step 数
waste_per_step	每步平均浪费
每个策略另有 replay_<RULE>_requests.jsonl，含逐请求明细。

数据集
convert_datasets.py 统一转换，输出 vLLM 所需的 {"prompt": "..."} jsonl：

bash
python3 convert_datasets.py alpaca
python3 convert_datasets.py alpaca gsm8k humaneval
python3 convert_datasets.py mtbench_chat     # 内部两步：拆轮次 → 套 chat template
python3 convert_datasets.py all
数据集	输入	输出	bench N
alpaca	datasets/alpaca/raw（HF Dataset）	datasets/alpaca/alpaca_vllm.jsonl	160
gsm8k	datasets/gsm8k/raw（HF Dataset）	datasets/gsm8k/gsm8k_vllm.jsonl	160
humaneval	datasets/humaneval/HumanEval.jsonl	datasets/humaneval/humaneval_vllm.jsonl	164
mtbench_chat	datasets/mt_bench/mt_bench_prompts.jsonl	datasets/mt_bench/mtbench_chat_vllm.jsonl	160
⚠️ run_bench.sh 里 mtbench 的 DATA_PATH 当前指向
/root/autodl-tmp/mt_bench/mtbench_chat_vllm.jsonl（少了 datasets/），
与转换脚本输出路径不一致。若用 convert_datasets.py 生成，请把
run_bench.sh 中该行改为
/root/autodl-tmp/datasets/mt_bench/mtbench_chat_vllm.jsonl，或跑 bench 时
显式传 DATA_PATH=...。

注意事项
vLLM 0.27.1 不支持 stats_reporting_level。serve_k.sh 默认不加；
只在更新版本上通过 ENABLE_SPEC_STATS_LEVEL=1 显式开启。

hook 是原位修改 vLLM 源码（vllm/v1/spec_decode/metrics.py）。
升级 vLLM 后需重新 install_observe_hook.py；切换版本前建议 --undo。

C 必须为 1 才能得到干净的逐步日志。并发 >1 时 observe 只记录 batch 级
汇总，observe_to_steps.py 会跳过这些行并打印 skipped_batch_rows。

没有 output_lens 时，每个 observe 行被当作独立请求（req_id 不连续），
对 k 的整体分布影响不大，但按请求的曲线会失真。run_c1_replay.sh 会自动
把 bench json 传进去，正常流程无需关心。

吞吐是估算值。若要真实 tok/s，直接看 run_bench.sh 输出的 bench json。