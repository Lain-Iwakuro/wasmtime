# 固定 M4 输入

- 00 [00-pthread-malloc-free.wasm](00-pthread-malloc-free.wasm)：历史 M4 原始 Wasm，400000 对象、8 workers。
- 01 [01-malloc-multithreading.wasm](01-malloc-multithreading.wasm)：历史 M4 原始 Wasm，4 workers、100000000 迭代。
- 02 [02-fluidanimate.wasm](02-fluidanimate.wasm)：历史 CCGrid Wasm，4 workers、1845 帧。
- 03 [03-in_35K.fluid](03-in_35K.fluid)：原 simsmall 输入，36504 粒子。
- 04 [04-hashes.json](04-hashes.json)：四份历史输入的固定 SHA-256；runner 另用冻结常量核验。
