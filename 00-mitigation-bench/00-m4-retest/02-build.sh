#!/usr/bin/env bash
# Native SDK build. Cargo/CMake/SDK output keeps the tools' native layout.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'usage: bash 02-build.sh' \
        'Requires native Apple arm64 macOS, Rust, Python 3.10+, CMake, and Xcode Command Line Tools.' \
        'Builds the reviewed commit in 05-source.json from this Git repository; publishes SDK/host/manifest under 07-build.' \
        'MAC_BENCH_JOBS (default 8) controls Cargo jobs; dependencies may require network access.'
    exit 0
fi
if [[ $# != 0 ]]; then
    printf '%s\n' 'Unexpected arguments; use --help.' >&2
    exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    printf '%s\n' 'Native Apple arm64 macOS required; this script does not cross-build.' >&2
    exit 77
fi
MAC_BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAC_BENCH_BUILD="${MAC_BENCH_ROOT}/07-build"
MAC_BENCH_SRC="${MAC_BENCH_BUILD}/05-source"
MAC_BENCH_REPO="$(cd -- "${MAC_BENCH_ROOT}/../.." && pwd)"
MAC_BENCH_SDK="${MAC_BENCH_BUILD}/02-sdk"
MAC_BENCH_BASE=1cea7556c072bc9304a4770afc88de9e0b956ccb
MAC_BENCH_TARGET=aarch64-apple-darwin
MAC_BENCH_FEATURES=cranelift,wat,wasi,threads,parallel-compilation,gc,gc-drc
MAC_BENCH_JOBS="${MAC_BENCH_JOBS:-8}"
for MAC_BENCH_TOOL in git cargo rustc cmake python3 xcrun install_name_tool codesign; do
    command -v "$MAC_BENCH_TOOL" >/dev/null
done
if [[ ! -f "${MAC_BENCH_ROOT}/05-source.json" ]]; then
    printf '%s\n' 'Missing compiler pin 05-source.json.' >&2
    exit 1
fi
if [[ -e "${MAC_BENCH_BUILD}/04-build-info.json" || -e "${MAC_BENCH_BUILD}/02-sdk" || -e "${MAC_BENCH_BUILD}/03-host" || -e "$MAC_BENCH_SRC" ]]; then
    printf '%s\n' 'A published or partial SDK already exists. Preserve it and use a fresh handoff copy for another build.' >&2
    exit 1
fi
if [[ -n "${RUSTFLAGS:-}" || -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
    printf '%s\n' 'The frozen build requires unset RUSTFLAGS and CARGO_ENCODED_RUSTFLAGS.' >&2
    exit 1
fi
if [[ "$(rustc -vV | sed -n 's/^host: //p')" != "$MAC_BENCH_TARGET" ]]; then
    printf '%s\n' 'Use a native aarch64-apple-darwin Rust toolchain, outside Rosetta.' >&2
    exit 1
fi
if [[ "$(git -C "$MAC_BENCH_REPO" rev-parse --show-toplevel)" != "$MAC_BENCH_REPO" || \
      -n "$(git -C "$MAC_BENCH_REPO" status --porcelain)" ]]; then
    printf '%s\n' 'Use a clean Git checkout containing this handoff.' >&2
    exit 1
fi
MAC_BENCH_HANDOFF_REV="$(git -C "$MAC_BENCH_REPO" rev-parse HEAD)"
MAC_BENCH_REV="$(python3 -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["compiler_commit"])' "${MAC_BENCH_ROOT}/05-source.json")"
python3 -B -S "${MAC_BENCH_ROOT}/12-prepare-source.py"
if [[ -n "$(git -C "$MAC_BENCH_SRC" status --porcelain)" || \
      "$(git -C "$MAC_BENCH_SRC" rev-parse HEAD)" != "$MAC_BENCH_REV" ]]; then
    printf '%s\n' 'Compiler checkout must be clean and exactly match the reviewed commit.' >&2
    exit 1
fi
git -C "$MAC_BENCH_SRC" merge-base --is-ancestor "$MAC_BENCH_BASE" HEAD
mkdir -p "$MAC_BENCH_BUILD"
export CARGO_TARGET_DIR="${MAC_BENCH_BUILD}/00-cargo"
export CARGO_PROFILE_RELEASE_PANIC=abort
MAC_BENCH_CXX="$(xcrun --find clang++)"
MAC_BENCH_CARGO="$(command -v cargo)"
cargo build --manifest-path "${MAC_BENCH_SRC}/Cargo.toml" --locked --release \
    --target "$MAC_BENCH_TARGET" --package wasmtime-c-api --no-default-features \
    --features "$MAC_BENCH_FEATURES" --jobs "$MAC_BENCH_JOBS"
cmake -S "${MAC_BENCH_SRC}/crates/c-api" -B "${MAC_BENCH_BUILD}/01-cmake" \
    -DCMAKE_BUILD_TYPE=Release -DWASMTIME_TARGET="$MAC_BENCH_TARGET" \
    -DWASMTIME_CARGO_BINARY="$MAC_BENCH_CARGO" -DWASMTIME_DISABLE_ALL_FEATURES=ON \
    -DWASMTIME_FEATURE_CRANELIFT=ON -DWASMTIME_FEATURE_WAT=ON \
    -DWASMTIME_FEATURE_WASI=ON -DWASMTIME_FEATURE_THREADS=ON \
    -DWASMTIME_FEATURE_PARALLEL_COMPILATION=ON \
    -DWASMTIME_FEATURE_GC=ON -DWASMTIME_FEATURE_GC_DRC=ON
mkdir -p "${MAC_BENCH_SDK}/include" "${MAC_BENCH_SDK}/lib"
cp -R "${MAC_BENCH_SRC}/crates/c-api/include/." "${MAC_BENCH_SDK}/include/"
cp "${MAC_BENCH_BUILD}/01-cmake/include/wasmtime/conf.h" "${MAC_BENCH_SDK}/include/wasmtime/conf.h"
cp "${CARGO_TARGET_DIR}/${MAC_BENCH_TARGET}/release/libwasmtime.dylib" "${MAC_BENCH_SDK}/lib/"
cp "${CARGO_TARGET_DIR}/${MAC_BENCH_TARGET}/release/libwasmtime.a" "${MAC_BENCH_SDK}/lib/"
# Only Mach-O load metadata changes; the SDK can move together with the host.
install_name_tool -id @rpath/libwasmtime.dylib "${MAC_BENCH_SDK}/lib/libwasmtime.dylib"
codesign --force --sign - "${MAC_BENCH_SDK}/lib/libwasmtime.dylib"
"$MAC_BENCH_CXX" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    -I "${MAC_BENCH_SDK}/include" "${MAC_BENCH_ROOT}/01-host.cpp" \
    -L "${MAC_BENCH_SDK}/lib" -lwasmtime -Wl,-rpath,@loader_path/02-sdk/lib \
    -o "${MAC_BENCH_BUILD}/03-host"
if [[ -n "$(git -C "$MAC_BENCH_SRC" status --porcelain)" || \
      "$(git -C "$MAC_BENCH_SRC" rev-parse HEAD)" != "$MAC_BENCH_REV" || \
      -n "$(git -C "$MAC_BENCH_REPO" status --porcelain)" || \
      "$(git -C "$MAC_BENCH_REPO" rev-parse HEAD)" != "$MAC_BENCH_HANDOFF_REV" ]]; then
    printf '%s\n' 'Compiler source changed during build; no build manifest was published.' >&2
    exit 1
fi
python3 -B -S - "$MAC_BENCH_ROOT" "$MAC_BENCH_REV" "$MAC_BENCH_FEATURES" "$MAC_BENCH_CXX" "$MAC_BENCH_HANDOFF_REV" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
p=Path(sys.argv[1])
def capture(args):return subprocess.check_output(args,text=True).strip()
files=[p/'05-source.json',p/'07-build/02-sdk/lib/libwasmtime.dylib',
       p/'07-build/02-sdk/lib/libwasmtime.a',p/'07-build/03-host']
files += sorted((p/'07-build/02-sdk/include').rglob('*.h'))
files += [f for f in p.iterdir() if f.is_file() and f.suffix in ('.sh','.cpp','.py')]
record={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'source_commit':sys.argv[2],'source_base':'1cea7556c072bc9304a4770afc88de9e0b956ccb',
    'source_clean':True,'source_delivery':'git-pinned-checkout','handoff_commit':sys.argv[5],
    'v4_rule_version':1,'v4_policies':['legacy','frame','ranges','ranges_cfg'],
    'target':'aarch64-apple-darwin','profile':'release','panic':'abort',
    'features':sys.argv[3].split(','),'has_lse':True,'persistent_cache':False,
    'rust':capture(['rustc','--version']),'cargo':capture(['cargo','--version']),
    'cxx':capture([sys.argv[4],'--version']).splitlines()[0],
    'cmake':capture(['cmake','--version']).splitlines()[0],
    'macos':capture(['sw_vers']),'sdk':capture(['xcrun','--show-sdk-version']),
    'sha256':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(p/'07-build/04-build-info.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
index=['# M4 原生构建','',
    '- 00 [00-cargo](00-cargo/index.md)：Cargo 缓存和原生 release 产物，内部使用原生工具布局。',
    '- 01 [01-cmake](01-cmake/index.md)：与七项 C API features 一致的生成头文件。',
    '- 02 [02-sdk](02-sdk/index.md)：M4 原生 C API 头文件、动态库和静态库。',
    '- 03 [03-host](03-host)：沿用历史计时区间的 WASI threads 宿主。',
    '- 04 [04-build-info.json](04-build-info.json)：已审查源码提交、交接提交、工具版本与 SDK 哈希。',
    '- 05 [05-source](05-source/index.md)：固定 2648e4a221 的独立 Git 源码，沿用 Wasmtime 原生布局。']
(p/'07-build/index.md').write_text('\n'.join(index)+'\n')
for name in ('00-cargo','01-cmake','02-sdk'):
    (p/'07-build'/name/'index.md').write_text('# 原生工具布局\n\n此目录及内部使用 Cargo、CMake 或 C API SDK 的既有目录命名。\n')
results=p/'08-results';results.mkdir(exist_ok=True)
if not (results/'index.md').exists():
    (results/'index.md').write_text('# M4 三组重测结果\n\n运行目录按编号追加，成功标记与完整证据由 runner 维护。\n')
print('Native M4 SDK and host built; compiler/handoff/SDK hashes recorded. No benchmark has been run.')
PY
