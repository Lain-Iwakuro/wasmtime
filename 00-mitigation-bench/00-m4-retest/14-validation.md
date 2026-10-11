# 交接验证记录

日期：2026-10-11，验证平台 Linux x86-64。此次仅增加 `00-mitigation-bench/` 的交接材料；编译器固定 `2648e4a22107d3409c2e0cfac9499c3772dc25a7`，没有修改 Cranelift 或历史 AMD/M4 实验文件。

## 此次检查

- `python3 -B -S 06-tests.py`：16 项测试，15 项通过、1 项跳过。跳过项需要 `MAC_HOST_TEST_BINARY` 指向原生构建 host；M4 接手命令明确要求补跑。
- 新增三组计划、近似顺序平衡、配对统计和固定 Git 源码测试，先确认旧交接缺少 retest/source-preparation 时失败，再实现并通过。
- 实际计划生成：正确性 9 次；正式 198 次；每负载 2 个预热、20 个正式 block，所有 block 三组齐全，原始工作量不变。
- 三组完整发布 fixture 通过：逐日志/粒子证据复核、冻结、summary 发布及原地复算一致；篡改日志、缺粒子输出、缺 trial 会被拒绝。fixture 是合成数据，不是 M4 性能数据。
- `bash -n 02-build.sh`、全部 Python AST、build/runner `--help` 通过。Linux 上实际调用 build 返回 77，不创建 build；runner 同样在启动 guest 前拒绝非 Mac 平台。
- 用实际 Wasmtime 仓库调用 `12-prepare-source.py`，成功复制独立 checkout，HEAD 精确为固定提交、源码工作树干净、Cargo.lock 和 C API 头文件齐全；原 checkout 未切换。临时副本检查后清理。
- C API `01-host.cpp`、fluid 输出检查器和全部固定输入逐字节等同于原交接；输入 SHA-256 与冻结常量一致。

独立只读审查已通过，未发现 Critical / Important 问题。审查提出的统计 fixture 小问题已修正并重跑：让倍率随 block 变化，明确区分配对比的中位数与两个中位数相除。编译器自身的 273 项测试及 SDK 审计属于此前对 `2648e4a221` 的复审，见 [13-context.md](13-context.md)，本次没有重新编译编译器。

## 留给 M4 的检查

1. 原生 arm64 release SDK / host 构建；记录工具链、SDK、源码与交接提交。
2. 设置 `MAC_HOST_TEST_BINARY` 补跑宿主参数拒绝测试。
3. 三个完整负载 × 三组的 9 次正确性检查。
4. 198 次配对实验、完整证据及 summary 复算。

尚未声称完成以上 Mac 检查，没有新 M4 性能结果，也没有进行攻击 gadget 或硬件缓解有效性验证。本次发布仅交付可执行的重测流程。
