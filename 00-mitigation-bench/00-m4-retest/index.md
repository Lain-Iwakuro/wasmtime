# M4 三组重测

- 00 [00-readme.md](00-readme.md)：接手入口、构建、正确性、198 次正式实验和结果回传。
- 01 [01-host.cpp](01-host.cpp)：历史 C API 宿主，保留线程、计时和 LSE 配置，记录策略与规则版本。
- 02 [02-build.sh](02-build.sh)：原生 Apple arm64 构建固定编译器、SDK、宿主及哈希 manifest。
- 03 [03-run.py](03-run.py)：默认三组 retest，预检、逐 trial 证据和完整完成标记。
- 04 [04-inputs](04-inputs/index.md)：三个历史原始 Wasm 与 fluidanimate 输入；不重编负载。
- 05 [05-source.json](05-source.json)：已复审编译器提交、基线提交、交接仓库和分支。
- 06 [06-tests.py](06-tests.py)：便携协议、配对统计、源代码固定、证据完整性与宿主参数测试。
- 07 `07-build/`：本机生成并忽略的源码、SDK、宿主和 manifest；构建时创建 index。
- 08 `08-results/`：本机生成并忽略的编号运行目录；runner 维护 index 与原始证据。
- 09 [09-support.py](09-support.py)：固定工作量、三组顺序、输入与构建哈希、worker 和输出检查。
- 10 [10-analysis.py](10-analysis.py)：完整 block bootstrap、两项开销和 SSBB/CAS-CSDB 直接配对比。
- 11 [11-output-check.py](11-output-check.py)：fluidanimate 输出长度、header、有限值检查。
- 12 [12-prepare-source.py](12-prepare-source.py)：从当前仓库复制固定提交，不切换现有 checkout。
- 13 [13-context.md](13-context.md)：已定实验范围、编译器复审、AMD 结果、历史 M4 数据与解释边界。
- 14 [14-validation.md](14-validation.md)：此次交接的验证记录和仍需 M4 执行的检查。
