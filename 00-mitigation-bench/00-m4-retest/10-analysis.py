"""Complete paired-block analysis; smoke data never produce statistics."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import signal
import statistics

SPEC=importlib.util.spec_from_file_location('ssbd_support_analysis',Path(__file__).with_name('09-support.py'))
S=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def percentile(values, p):
    index=(len(values)-1)*p
    low=int(index)
    high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(index-low)


def summarize(plan, records, reps=10000):
    S.require(plan['kind']=='paired','smoke checks are not performance experiments')
    S.validate_plan(plan)
    S.require(reps==plan['bootstrap_reps']==10000,'protocol requires 10000 bootstrap repetitions')
    S.require(len(records)==len(plan['jobs']),'incomplete run or extra records')
    grouped={}
    seen=set()
    for job,row in zip(plan['jobs'],records):
        S.require(row['id'] not in seen,'duplicate trial')
        seen.add(row['id'])
        S.require(all(row.get(key)==value for key,value in job.items()),'trial differs from planned order or configuration')
        S.require(row['identity_sha256']==plan['identity_sha256'],'mixed artifact/config identities')
        S.require(row['correctness']=='pass','failed trial in run')
        S.require(type(row['exec_ns']) is int and row['exec_ns']>0,'invalid execution time')
        if row['phase']=='warmup':continue
        S.require(row['phase']=='paired','invalid measurement phase')
        values=grouped.setdefault((row['workload'],row['block']),{})
        S.require(row['arm'] not in values,'duplicate arm in block')
        values[row['arm']]=row['exec_ns']
    result=[]
    for workload in plan['identity'].get('selected_workloads',list(S.WORKLOADS)):
        blocks=[grouped[(workload,block)] for block in range(1,plan['blocks']+1)]
        configurations=plan['identity']['arms']
        S.require(all(set(block)==set(configurations) for block in blocks),'incomplete full-arm block')
        # Both comparisons resample exactly the same complete block indices.
        seed=int(S.digest_json([plan['seed'],workload,'v4-complete-block-bootstrap'])[:16],16)
        rng=random.Random(seed)
        draws=[[rng.randrange(len(blocks)) for _ in blocks] for _ in range(reps)]
        comparisons=[(arm,'baseline',arm) for arm in configurations if arm!='baseline']
        if 'v4_legacy' in configurations:
            comparisons += [('v4_opt','v4_legacy','v4_opt/legacy')]
        else:
            comparisons += [('v4_opt','reference','v4_opt/reference')]
        for numerator,denominator,arm in comparisons:
            ratios=[block[numerator]/block[denominator] for block in blocks]
            medians=sorted(statistics.median([ratios[index] for index in draw]) for draw in draws)
            point=statistics.median(ratios)
            low,high=percentile(medians,.025),percentile(medians,.975)
            result.append({'workload':workload,'arm':arm,'n_blocks':len(blocks),
                'denominator':denominator,'numerator':numerator,
                'median_baseline_seconds':statistics.median([block[denominator] for block in blocks])/1e9,
                'median_arm_seconds':statistics.median([block[numerator] for block in blocks])/1e9,
                'slowdown':point,'ci95_low':low,'ci95_high':high,
                'overhead_percent':100*(point-1),'overhead_ci95_low':100*(low-1),
                'overhead_ci95_high':100*(high-1),'time_reduction_percent':100*(1-point),
                'time_reduction_ci95_low':100*(1-high),'time_reduction_ci95_high':100*(1-low),
                'bootstrap_reps':reps,'bootstrap_seed':seed})
    return result


def analyze_run(directory, reps=10000, completing_status=None):
    directory=Path(directory)
    status=json.loads((directory/'07-status.json').read_text())
    if completing_status is None:
        S.require(status['state']=='complete','run is not complete')
    else:
        S.require(status['state']=='analyzing' and status==completing_status,'run is not ready for analysis')
        status=completing_status
    plan=json.loads((directory/'00-plan.json').read_text())
    S.validate_plan(plan)
    S.assert_artifacts(plan)
    if completing_status is None:S.verify_frozen(directory,status)
    S.require(status['completed_trials']==len(plan['jobs']),'incomplete trial count')
    S.require(S.digest_json(plan)==status['plan_sha256'],'plan hash mismatch')
    S.require(S.digest_json(plan['identity'])==plan['identity_sha256'],'plan identity mismatch')
    records=[json.loads(line) for line in (directory/'02-trials.jsonl').read_text().splitlines()]
    S.require(len(records)==len(plan['jobs']),'incomplete run or extra records')
    for index,row in enumerate(records):
        S.require(row.get('id')==index,'trial record order mismatch')
        job=plan['jobs'][index]
        expected_stdout=f'06-logs/{2*index:03d}-{job["workload"]}-{job["arm"]}-stdout.txt'
        expected_stderr=f'06-logs/{2*index+1:03d}-{job["workload"]}-{job["arm"]}-stderr.txt'
        S.require(row.get('stdout_file')==expected_stdout and row.get('stderr_file')==expected_stderr,'wrong raw log path')
        S.audit_output(directory,row)
        stdout_path=directory/row['stdout_file']
        stderr_path=directory/row['stderr_file']
        S.require(S.sha256(stdout_path)==row['stdout_sha256'] and S.sha256(stderr_path)==row['stderr_sha256'],'raw log hash mismatch')
        job=plan['jobs'][row['id']]
        actual=S.validate_trial(plan,job,stdout_path.read_text(),stderr_path.read_text(),row['returncode'])
        S.require(all(row.get(key)==value for key,value in actual.items()),'record does not match raw evidence')
    rows=summarize(plan,records,reps)
    outputs=render_summary(rows)
    if 'summary_sha256' in status:
        # An already committed result is immutable. Offline re-analysis verifies
        # the original publication instead of risking a partial overwrite.
        for name,content in outputs.items():
            S.require(S.sha256(directory/name)==status['summary_sha256'][name],'published summary hash mismatch')
            S.require((directory/name).read_bytes()==content.encode(),'re-analysis differs from published summary')
    else:
        status['evidence_sha256']=S.freeze_evidence(directory)
        publish_summary(directory,outputs,status)
    return rows


def render_summary(rows):
    csv_output=io.StringIO(newline='')
    writer=csv.DictWriter(csv_output,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    lines=['# M4 选择性 v4 屏障配对开销','',
           '同机同 SDK 构建，每个 block 包含全部实验臂各一次。',
           '报告 block 内时间比的中位数；95% CI 对完整 block 重采样 10000 次。',
           '配置由冻结计划决定；三组重测为 baseline=none、reference=CAS-CSDB、v4_opt=SSBB ranges_cfg。',
           '覆盖范围：软件防御插入 Cranelift 生成的 guest/trampoline 代码；host/runtime 不受同等插桩。',
           '每负载排除 2 个预热 block，纳入 20 个完整正式 block。',
           '原始 M4 Wasm 和工作量；fluidanimate 只验证执行与输出完整性，不要求跨运行数值等价。',
           '本结果衡量插桩性能，不代表在本机建立并验证了 Spectre-v4 攻击通道。','',
           '| workload | comparison | blocks | ratio | 95% CI | overhead | time reduction |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["workload"]} | {row["numerator"]}/{row["denominator"]} | {row["n_blocks"]} | {row["slowdown"]:.6f}× | [{row["ci95_low"]:.6f}, {row["ci95_high"]:.6f}] | {row["overhead_percent"]:+.3f}% | {row["time_reduction_percent"]:+.3f}% |')
    return {'04-summary.csv':csv_output.getvalue(),'05-summary.md':'\n'.join(lines)+'\n'}


def publish_summary(directory, outputs, status):
    """Stage both outputs, then publish the complete marker last.

    SIGINT/SIGTERM are deferred only during the short rename/rollback section.
    Readers must require the complete marker and matching summary hashes. A
    SIGKILL may leave uncommitted files, but cannot publish an early complete.
    """
    S.require(not any((directory/name).exists() for name in outputs),'uncommitted summary files already exist')
    stage=directory/'09-pending-summary'
    stage.mkdir()
    committed=False
    status['state']='analyzing'
    S.write_json(directory/'07-status.json',status)
    try:
        (stage/'index.md').write_text('# 未提交的统计文件\n\n完成标记发布前不得当作结果使用。\n')
        pending=[]
        for index,(name,content) in enumerate(outputs.items()):
            path=stage/f'{index:02d}-summary{Path(name).suffix}'
            path.write_text(content)
            pending.append((path,directory/name))
        complete={**status,'state':'complete','ended_at':datetime.now(timezone.utc).isoformat(),
                  'summary_sha256':{target.name:S.sha256(source) for source,target in pending}}
        marker=stage/'02-complete.json'
        S.write_json(marker,complete)
        old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
        try:
            try:
                for source,target in pending:
                    source.replace(target)
                marker.replace(directory/'07-status.json')
                status.clear()
                status.update(complete)
                committed=True
            except BaseException:
                for name in outputs:
                    (directory/name).unlink(missing_ok=True)
                raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
    except BaseException as exc:
        if committed and isinstance(exc,(KeyboardInterrupt,InterruptedError)):
            return  # The publication finished before the deferred signal.
        if not committed:
            for name in outputs:
                (directory/name).unlink(missing_ok=True)
            status.update(state='failed',reason=str(exc) or type(exc).__name__)
            S.write_json(directory/'07-status.json',status)
        raise
    finally:
        shutil.rmtree(stage)


def interrupt_on_sigterm(signum, frame):
    raise InterruptedError(f'terminated by signal {signum}')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupt_on_sigterm)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--reps',type=int,default=10000)
    args=parser.parse_args()
    try:
        print(json.dumps(analyze_run(args.directory,args.reps),indent=2))
    except (ValueError,KeyError,OSError,IndexError) as exc:
        parser.exit(1,f'analysis refused: {exc}\n')
