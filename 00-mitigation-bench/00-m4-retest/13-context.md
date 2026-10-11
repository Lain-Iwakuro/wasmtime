# 给 M4 执行 agent 的上下文

日期：2026-10-11。此次任务是补齐优化版 v4 软件屏障的 M4 性能结果，使用 Wasmtime C API 和已冻结的三个负载。用户明确不要 M4 legacy 和消融；不要自行增加实验组、改编译器、重编 Wasm 或改工作量。构建或正确性有问题时保存证据，先排障。

## 已确定的比较口径

最终主表的四个防御配置是 M4 CAS-CSDB、M4 优化 SSBB、AMD 优化 LFENCE、AMD 硬件 SSBD。各平台都有无本实验防御的 baseline。AMD 没有 CAS-CSDB 数据；AMD 软件结果是在 SSBD **off** 下，对同轮 `none/off` baseline 比较。M4 SSBS 已被先前探测排除：FEAT_SSBS=0，用户态读写在 P/E 核均 trap，本轮不再测该开关。

虽然已有旧 M4 CAS-CSDB 数据，这轮仍与 none 和优化 SSBB 同批重测，形成三组配对实验。旧结果只保留为背景，不与新 trial 合并、不跨轮直接配对。新 SDK 使用七项 C API features，旧 SDK feature 集更广；这也是必须重新测 none 和 CAS-CSDB 的原因。

## 编译器及已完成的复审

固定提交 `2648e4a22107d3409c2e0cfac9499c3772dc25a7`，祖先基线 `1cea7556c072bc9304a4770afc88de9e0b956ccb`，原 M4 mitigation 提交 `6e77f6ebf6e5ba11ade0b23539c145b8af35eea6`。这些都在当前仓库历史内。本次交接不改 Cranelift 源码。

选择性算法在 regalloc 和 frame layout 之后建模实际访存，使用栈范围、块内同基址固定偏移和 CFG 放置。仅证明与全部未隔离 store 的字节范围不重叠时，才省相应保护；未知访问、调用和回边保守处理。不消除 load，不改原子操作，不把“显然同址”当作豁免。共同策略在 AArch64 发射 SSBB，在 AMD 发射 LFENCE；两 ISA 数量和性能不必相同。CAS-CSDB 仍对全部 CAS 插桩，不应用 v4 范围豁免。

旧选择性实现曾因复杂展开入口从 dirty 开始而新增冗余屏障；`2648e4a221` 已修复，由外层按实际入口状态保护第一条访问，内部必需屏障保留。2026-10-10 独立复审已通过，原交接中“等待复审、禁止启动”的状态已经失效。复审重跑了 273 项 codegen/disassembly 测试；真实 x64 SDK 的 clean/dirty/load-after-call 回归及三个 Wasm × 五配置静态审计通过。none/legacy 的完整 `.text` 与历史 x64 SDK 一致。没有新做 M4 原生执行或 gadget 实验。

AMD 静态计数（legacy → ranges_cfg）：stress 1379→1316，malloc 1272→1212，fluidanimate 18584→18363。静态下降不大，但动态收益与热点执行次数有关；本轮不再做消融来分解来源。

源码入口：`cranelift/codegen/src/machinst/v4_fences.rs`、`machinst/vcode.rs`、`isa/aarch64/inst/ssbb.rs`、`isa/aarch64/inst/emit.rs`；设置 `v4_fence_policy=ranges_cfg`、规则版本 1。

## AMD 已完成结果，仅作背景

最新 AMD EPYC 9554 三组实验：none/off、legacy LFENCE/off、optimized LFENCE/off；2 个预热 + 20 个正式 block，每组工作量与 M4 本轮相同。198 次全部通过，SSBD 均读回 raw=3/off。下表是各自配对时间比的中位数及 95% CI。

| 负载 | legacy/none | optimized/none | optimized/legacy |
| --- | --- | --- | --- |
| stress | 1.328501 [1.287682, 1.356297] | 1.322548 [1.309101, 1.376157] | 1.015965 [0.984236, 1.035448] |
| malloc | 1.146560 [1.143878, 1.148872] | 1.118678 [1.114857, 1.123991] | 0.977401 [0.974086, 0.980983] |
| fluidanimate | 7.395467 [7.382188, 7.409332] | 6.688683 [6.683560, 6.696229] | 0.904075 [0.903496, 0.905438] |

优化相对 legacy：stress 没有明确差异；malloc 时间减少约 2.26%；fluidanimate 减少约 9.59%，仍有很大开销。用户据此决定优化已有作用，不做消融。旧 AMD SSBD 开销为 +6.12%、−1.66%、+26.77%，来自另一轮同机配对实验，不能直接与新 LFENCE trial 组成配对。

来源为服务器 `mitigation-bench/03-v4-selective-fences/07-results/13-three-arm-20-blocks/05-summary.md`；SSBD 来源为 `mitigation-bench/00-linux-ssbd/12-x64-v4-lfence/05-results/12-paired-20-blocks/05-summary.md`。原始证据保留在服务器，没有复制进本交接。

## 历史 M4 及输入来源

历史机器：Apple M4，4 P + 6 E，16 GB，macOS 15.1；Rust 1.96.0。历史 CAS-CSDB slowdown：stress 1.752917×、malloc 1.000643×、fluidanimate 1.155674×；legacy SSBB 为 3.690860×、1.608450×、8.390914×。这些是历史同轮配对结果，不能替代本轮 baseline。

固定输入复制自用户提供的 M4 handoff，SHA-256 见 [04-inputs/04-hashes.json](04-inputs/04-hashes.json)。stress 移植自 Emscripten `test/pthread/test_pthread_malloc_free.cpp`，malloc 来自 `test/benchmark/benchmark_malloc_multithreading/`，Emscripten 来源提交 `5372a12809f8f73c77c1940832eed570b7077d7d`，WASI threads / dlmalloc，memory32。malloc 工作循环 compare-exchange 为零，保留为低 CAS 对照。fluidanimate 为此前接受的 CCGrid Wasm 移植，输入为 simsmall 的 `in_35K.fluid`，没有更换成新的 PARSEC 移植。

原始材料在服务器 `mitigation-bench/00-linux-ssbd/05-input/02-m4-wasmtime-handoff/` 的 `00-manifest.md` 和 `07-fluidanimate/`；本轮不依赖这些服务器绝对路径。当前宿主从上一份选择性 fence 交接直接复制，`01-host.cpp` 未修改，延续 `_start` 到 worker join 的计时边界、Cranelift、has_lse=true、WASI threads、无持久缓存及原有调度。

## 解释边界

- 性能核此前没有建立攻击信道，历史绑核接口返回 EPERM。这里测插桩性能，不能宣称已经在 P 核上验证缓解有效性。E/P 核调度不得从系统快照推断为严格钉核。
- fluidanimate 只校验执行、输出大小、header 和有限值；不要求线程间粒子顺序逐位一致，也不宣称物理数值正确性。
- v4 fence 作用于 Cranelift 生成的 guest/trampoline；预编译 host/runtime 没有同等插桩。AMD SSBD 作用于受控线程，范围不同。
- 选择性 v4 判据针对传统 store bypass，不将其外推为 CAS 成功预测、错误转发或所有 Spectre 变体的完整防御。
- 跨平台 overhead 不能直接解释为 ISA 的纯差异；需分别报告相对本机、同轮 baseline 的值。
