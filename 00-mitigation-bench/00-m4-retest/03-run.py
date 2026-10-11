#!/usr/bin/env python3
"""M4 retest: none, CAS-CSDB, and optimized SSBB on the original Wasms."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S=load('ssbd_support_runner','09-support.py')
A=load('ssbd_analysis_runner','10-analysis.py')
PYTHON=[sys.executable,'-B','-S']


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(kind):
    result={'checked_at':now(),'kind':kind,'ok':False,'steps':[]}
    try:
        S.require(kind in ('paired','smoke'),'unknown experiment kind')
        # Refuse on other platforms before touching SDK files or starting guests.
        S.require(S.platform.system()=='Darwin' and S.platform.machine()=='arm64',
                  'native Apple arm64 macOS is required')
        brand=S.capture(['sysctl','-n','machdep.cpu.brand_string'])
        result['steps'].append(brand)
        S.require(brand.get('returncode')==0 and 'Apple M4' in brand['stdout'],
                  'this frozen handoff targets Apple M4')
        result['build']=S.validate_build()
        S.validate_inputs()
        result['machine']=S.machine_snapshot()
        result['ok']=True
    except (ValueError,KeyError,TypeError,OSError,IndexError,subprocess.TimeoutExpired) as exc:
        result['reason']=str(exc) or type(exc).__name__
    return result


def create_output(path):
    S.require(not path.exists(),f'output already exists: {path}')
    match=re.fullmatch(r'(\d{2,})-.+',path.name)
    S.require(match is not None,'output directory must have a numeric prefix, e.g. 00-check')
    prefix=match[1]+'-'
    S.require(not any(p.name.startswith(prefix) for p in path.parent.iterdir()),'output numeric prefix is already used')
    path.mkdir()
    with (path.parent/'index.md').open('a') as index:
        index.write(f'\n- [{path.name}]({path.name}/index.md)：M4 选择性 v4 运行记录；状态见目录内 07-status.json。\n')


def write_index(out):
    descriptions={
        '00-plan.json':'冻结所选矩阵；本轮为三组重测，保留原始工作量和构建哈希。',
        '01-preflight.json':'M4 平台、原生 SDK、固定输入和构建哈希检查。',
        '02-trials.jsonl':'逐次运行及校验信息；失败样本也保留。',
        '03-trials.csv':'逐次运行的表格形式，分析以 JSONL 和原始日志为准。',
        '04-summary.csv':'完整配对实验的 slowdown 中位数与 95% CI。',
        '05-summary.md':'同机同构建各防御相对 baseline 及防御间直接配对时间比；smoke 不产生此文件。',
        '06-logs':'每次进程的原始标准输出和错误输出。',
        '07-status.json':'运行状态、完成数量、阻断原因及原始证据冻结哈希。',
        '08-machine-after.json':'运行结束时的系统、供电和热状态快照。',
        '10-particles':'逐次保存的完整粒子状态和输入副本。',
    }
    lines=['# M4 选择性 v4 屏障运行记录','']
    for name,description in descriptions.items():
        if (out/name).exists():
            target=name+'/index.md' if (out/name).is_dir() else name
            lines.append(f'- [{name}]({target})：{description}')
    (out/'index.md').write_text('\n'.join(lines)+'\n')


def trial_command(job,out):
    cfg=S.WORKLOADS[job['workload']]
    command=[str(S.HOST),'--mode',job['compiler_mode'],'--wasm',cfg['wasm'],
             '--v4-policy',job['v4_policy'],
             '--timeout',str(cfg['timeout']),'--time']
    guest=list(cfg['args'])
    if job['workload']=='fluidanimate':
        command += ['--dir',str(out/'10-particles')+'::/work']
        guest += ['/work/000-input.fluid','/work/'+S.particle_name(job)]
    return command+['--']+guest


CSV_FIELDS=['id','workload','phase','block','position','arm','compiler_mode','v4_policy','exec_ns',
            'workers','work_units','correctness','validation_scope','identity_sha256',
            'returncode','stdout_file','stderr_file','error']


def append_record(out, record):
    with (out/'02-trials.jsonl').open('a') as handle:
        handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    path=out/'03-trials.csv'
    new=not path.exists()
    with path.open('a',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=CSV_FIELDS,extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list)) else value
                         for key,value in record.items()})
        handle.flush()
        os.fsync(handle.fileno())


def execute_trial(out, plan, job):
    S.assert_artifacts(plan)
    cfg=S.WORKLOADS[job['workload']]
    command=trial_command(job,out)
    logs=out/'06-logs'
    stdout_path=logs/f'{2*job["id"]:03d}-{job["workload"]}-{job["arm"]}-stdout.txt'
    stderr_path=logs/f'{2*job["id"]+1:03d}-{job["workload"]}-{job["arm"]}-stderr.txt'
    record={**job,'identity_sha256':plan['identity_sha256'],'command':command,
            'started_at':now(),'correctness':'fail',
            'stdout_file':str(stdout_path.relative_to(out)),
            'stderr_file':str(stderr_path.relative_to(out))}
    started=time.monotonic()
    failure=None
    with stdout_path.open('x') as stdout,stderr_path.open('x') as stderr:
        try:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=cfg['timeout']+30)
            record['returncode']=result.returncode
        except subprocess.TimeoutExpired:
            record['returncode']=124
            failure='outer process timeout'
        except KeyboardInterrupt:
            record['returncode']=130
            failure='interrupted by operator'
        except InterruptedError as exc:
            record['returncode']=143
            failure=str(exc)
    record.update({'wall_seconds':time.monotonic()-started,'ended_at':now(),
                   'stdout_sha256':S.sha256(stdout_path),'stderr_sha256':S.sha256(stderr_path)})
    with (logs/'index.md').open('a') as index:
        for path in (stdout_path,stderr_path):
            index.write(f'- [{path.name}]({path.name})：trial {job["id"]}，{job["workload"]}，{job["phase"]}，arm={job["arm"]}。\n')
    try:
        S.require(failure is None,failure)
        record.update(S.validate_trial(plan,job,stdout_path.read_text(),stderr_path.read_text(),record['returncode']))
        record.update(S.validate_output(out,job))
        S.assert_artifacts(plan)
        if job['workload']=='fluidanimate':
            with (out/'10-particles/index.md').open('a') as index:
                index.write(f'- [{S.particle_name(job)}]({S.particle_name(job)})：trial {job["id"]}，arm={job["arm"]}。\n')
    except (ValueError,KeyError,TypeError,IndexError,OSError,KeyboardInterrupt) as exc:
        failure=str(exc) or type(exc).__name__
        record['error']=failure
        record['correctness']='fail'
    append_record(out,record)
    S.require(failure is None,f'trial {job["id"]} failed: {failure}')
    return record


def run(args):
    out=args.out.resolve()
    create_output(out)
    status={'state':'preparing','started_at':now(),'completed_trials':0}
    plan=None
    try:
        kind='smoke' if args.action=='check' else 'paired'
        status['kind']=kind
        check=preflight(kind)
        S.write_json(out/'01-preflight.json',check)
        if not check['ok']:
            status.update(state='blocked',reason=check['reason'])
            print(f'{kind} run blocked: {check["reason"]}; no benchmark was started',file=sys.stderr)
            return 77
        plan=S.make_plan(kind,args.blocks,args.seed,args.matrix,args.workloads)
        S.write_json(out/'00-plan.json',plan)
        status['plan_sha256']=S.digest_json(plan)
        status['identity_sha256']=plan['identity_sha256']
        status['state']='running'
        S.write_json(out/'07-status.json',status)
        S.prepare_particles(out)
        logs=out/'06-logs'
        logs.mkdir()
        (logs/'index.md').write_text('# 原始进程日志\n\n')
        write_index(out)
        for job in plan['jobs']:
            row=execute_trial(out,plan,job)
            status['completed_trials']+=1
            S.write_json(out/'07-status.json',status)
            print(f'{job["id"]+1}/{len(plan["jobs"])} {job["workload"]} {job["phase"]} '
                  f'block={job["block"]} arm={job["arm"]}: {row["exec_ns"]/1e9:.9f}s; validated',flush=True)
        S.assert_artifacts(plan)
        S.write_json(out/'08-machine-after.json',S.machine_snapshot())
        status['ended_at']=now()
        if kind=='paired':
            status['state']='analyzing'
            S.write_json(out/'07-status.json',status)
            A.analyze_run(out,plan['bootstrap_reps'],completing_status=status)
        else:
            status['evidence_sha256']=S.freeze_evidence(out)
            status['interpretation']=plan['interpretation']
            status['state']='complete'
            S.write_json(out/'07-status.json',status)
        print(f'{kind} complete: {out}',flush=True)
        return 0
    except (ValueError,KeyError,TypeError,OSError,IndexError,subprocess.TimeoutExpired,KeyboardInterrupt,InterruptedError) as exc:
        if status['state']=='complete' and 'summary_sha256' in status:
            print(f'paired results were committed before interruption: {out}',flush=True)
            return 0
        status.update(state='failed',reason=str(exc) or type(exc).__name__)
        print(f'run failed: {status["reason"]}',file=sys.stderr)
        return 1
    finally:
        if status['state']!='complete':
            status['ended_at']=now()
            if plan is not None and not (out/'08-machine-after.json').exists():
                S.write_json(out/'08-machine-after.json',S.machine_snapshot())
            S.write_json(out/'07-status.json',status)
        write_index(out)


def main():
    signal.signal(signal.SIGTERM,A.interrupt_on_sigterm)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--blocks',type=int,choices=(20,),default=20)
    parser.add_argument('--seed',type=int,choices=(20261009,),default=20261009)
    parser.add_argument('--matrix',choices=('retest','primary','ablation'),default='retest',
                        help='retest is the agreed three-arm M4 run; other matrices are historical')
    parser.add_argument('--workloads',nargs='+',choices=tuple(S.WORKLOADS),
                        help='freeze a separate subset run; default all three workloads')
    args=parser.parse_args()
    try:
        return run(args)
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
