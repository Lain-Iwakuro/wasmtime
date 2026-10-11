# M4 三组重测交接

日期：2026-10-11。接手 agent 先读本文件及 [13-context.md](13-context.md)。用户已确定：**只跑 `none`、`cas_csdb`、`ssbb/ranges_cfg`，不跑 SSBB legacy，不做消融。** 同轮重测 baseline 和 CAS-CSDB，以获得直接配对结果。

编译器固定为已复审的 `2648e4a22107d3409c2e0cfac9499c3772dc25a7`。本目录的 Git 提交另记为 `handoff_commit`；构建脚本从仓库本地复制独立的固定源码，不切换操作者的 checkout。无需旧 tar 包或 140 MB 的 compiler bundle。三个 Wasm 和 fluid 输入已在仓库内。

本交接尚未在 M4 原生构建或执行。Linux 验证范围见 [14-validation.md](14-validation.md)。下面的完整负载正确性检查通过后，才进入正式计时。

## 1. 获取和构建

在原生 Apple M4 的 arm64 终端，使用一个新的工作目录，保留旧实验目录及结果：

```bash
git clone --branch experiment/v4-selective-fences --single-branch \
  https://github.com/Lain-Iwakuro/wasmtime.git wasmtime-m4-retest
cd wasmtime-m4-retest/00-mitigation-bench/00-m4-retest
git rev-parse HEAD
python3 -B -S 06-tests.py
bash 02-build.sh
MAC_HOST_TEST_BINARY="$PWD/07-build/03-host" python3 -B -S 06-tests.py
```

若已拉取这条实验分支，可以直接进入上述相对路径；不要把旧实验分支强行 reset，也不要用浅克隆，固定编译器提交必须存在。

依赖：Python 3.10+、原生 Rust、CMake、Xcode Command Line Tools。历史 M4 使用 Rust 1.96.0，优先沿用；若本机不同，保留 manifest 中的实际版本并在回传说明。Cargo 使用固定 `Cargo.lock`，首次可能需下载依赖。`MAC_BENCH_JOBS` 默认为 8，可按内存降低。不要使用 Rosetta；清除 `RUSTFLAGS` 和 `CARGO_ENCODED_RUSTFLAGS`。

构建严格固定：`aarch64-apple-darwin`、release、panic=abort、C API features `cranelift,wat,wasi,threads,parallel-compilation,gc,gc-drc`。SDK 和 host 写入 `07-build/`，编译器副本为 `07-build/05-source/`。任何已有完整或部分 SDK 都不覆盖；需要重建时保留原目录，用新的 Git checkout。

## 2. 正确性和正式运行

保持历史实验的供电与调度条件，关闭占用 CPU 的后台工作，整个实验期间避免睡眠。不新增绑核、QoS 或 root 操作。使用普通原生终端即可。

```bash
python3 -B -S 03-run.py check --matrix retest --out 08-results/00-retest-check \
&& python3 -B -S 03-run.py run --matrix retest --out 08-results/01-retest-20-blocks
```

`check` 运行三个完整工作量 × 三组，共 9 次，只验证正确性，不进入性能统计。`run` 固定 2 个预热 block + 20 个正式 block，每个 block 三组各一次，三个负载共 **198 次**（18 次预热 + 180 次正式）。默认矩阵也已设为 `retest`；保留的 `primary`／`ablation` 仅兼容历史工具，本轮不要使用。

| arm | 编译器配置 | 含义 |
| --- | --- | --- |
| baseline | `mode=none, policy=legacy` | 不加本实验屏障；这里的 policy 只是配置占位 |
| reference | `mode=cas_csdb, policy=legacy` | 每个 CAS 后加 CSDB；policy 同样不启用 v4 fence |
| v4_opt | `mode=ssbb, policy=ranges_cfg` | 选择性 v4 插桩，AArch64 发射 SSBB（`DSB #0`） |

| 负载 | 固定工作量 | workers | 完成检查 |
| --- | --- | --- | --- |
| stress | 400000 对象，50000 × 8 | 8，另有消费主线程 | 释放数、完成标记和所有 worker |
| malloc | `4 100000000` | 4 | 每 worker 25000000 steps、totals 和完成标记 |
| fluidanimate | `4 1845`，36504 粒子 | 4 | 完整输出长度、header、有限值及所有 worker |

所有组使用 [04-inputs](04-inputs/index.md) 的同一份原始 Wasm，不重编负载、不缩短工作量。runner 每次使用新进程。固定 seed 为 `20261009`；前 18 个正式 block 的六种顺序各出现三次，最后两个采用循环排列。20 个 block 为近似平衡：位置次数 6/7/7，有向相邻次数 6–8，不能称为精确平衡。

计时字段为 `exec_ns`（`CLOCK_MONOTONIC`）：主实例准备好后，从调用 `_start` 前到全部 worker join 后。包含 guest 初始化、worker 创建和执行；不含进程启动、主 Engine/模块 JIT/主实例化、结果检查与哈希 I/O。不要用整轮墙钟替代它。

## 3. 完成标准、复算和回传

每个 trial 核验 loaded dylib、模式／策略、规则版本、所有 worker、输出和 SDK／工具／输入哈希。失败就停止，保留原始证据；不续跑、不覆盖、不丢弃失败后拼接样本。排障后用未占用的数字前缀重新跑完整实验，例如 `02-retest-check`、`03-retest-20-blocks`。

```bash
python3 -B -S 10-analysis.py 08-results/01-retest-20-blocks
```

成功标准：check 的 `07-status.json` 为 `complete` 且完成 9 次；正式实验同为 `complete` 且完成 198 次，复算通过。正式 summary 每个负载有三项：CAS-CSDB/none、优化 SSBB/none、优化 SSBB/CAS-CSDB。统计是 **block 内时间比的中位数**，完整 block 重采样 10000 次得到 95% CI，排除预热；直接配对比不能用两个中位数相除替代。

请回传整个 `08-results/`、`07-build/04-build-info.json`、本机 `git rev-parse HEAD`、构建与测试输出，以及是否更换机器／系统／工具链的说明。保留 `07-build/` 原地，复算会核对 SDK 和工具；仅复制结果到别处不等于可在新绝对路径复算。结果目录包含原始 stdout/stderr、每 trial 粒子输出、冻结计划、环境信息、CSV 和证据哈希；不要只回传 summary。

生成目录均由 `.gitignore` 排除。本次不把 build、结果或旧实验数据提交回源码分支。最终论文主表是 M4 CAS-CSDB、M4 优化 SSBB、AMD 优化 LFENCE、AMD SSBD 四个防御配置。该实验衡量插桩开销，尚不能据此声称 M4 性能核上的攻击通道或缓解有效性已经验证。
