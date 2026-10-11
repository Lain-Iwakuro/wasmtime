#!/usr/bin/env python3
"""Create a separate, pinned compiler checkout without altering the user's tree."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent


def prepare(repository,destination,pin):
    repository=Path(repository).resolve()
    destination=Path(destination).resolve()
    revision=pin['compiler_commit'];base=pin['compiler_base']
    if not all(re.fullmatch('[0-9a-f]{40}',value) for value in (revision,base)):
        raise ValueError('compiler pins must be full commit IDs')
    if destination.exists():
        raise ValueError('source destination exists; preserve it and use a fresh checkout')
    def git(*args):
        return subprocess.check_output(['git','-C',str(repository),*args],text=True).strip()
    git('cat-file','-e',revision+'^{commit}')
    git('merge-base','--is-ancestor',base,revision)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # No alternates or hardlinks: this snapshot remains usable independently.
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(repository),str(destination)],check=True)
    subprocess.run(['git','-C',str(destination),'checkout','--detach',revision],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:
        raise ValueError('prepared source revision differs from reviewed compiler')
    (destination/'index.md').write_text('# 固定编译器源码\n\n本目录按 Wasmtime 原生仓库布局组织，源码固定为 '+revision+'。\n')
    # Keep experiment metadata outside the compiler's clean tracked/untracked view.
    with (destination/'.git/info/exclude').open('a') as handle:
        handle.write('\n/index.md\n')
    return actual


if __name__=='__main__':
    pin=json.loads((ROOT/'05-source.json').read_text())
    print(prepare(ROOT.parents[1],ROOT/'07-build/05-source',pin))
