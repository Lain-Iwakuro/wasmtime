#!/usr/bin/env python3
"""Portable protocol checks; no Mac build or timing is performed here."""
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent

def load(filename):
    path=ROOT/filename
    if not path.is_file():
        raise AssertionError('portable implementation missing: '+filename)
    spec=importlib.util.spec_from_file_location('test_'+filename,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class PortableProtocol(unittest.TestCase):
    def setUp(self):
        self.s=load('09-support.py')
        self.a=load('10-analysis.py')
        self.r=load('03-run.py')

    def plan(self,matrix='primary',kind='paired'):
        identity={'artifacts':{},'workloads':self.s.WORKLOADS,'arms':self.s.arms(matrix),
                  'matrix':matrix,'v4_rule_version':1,'scheduling':'original-m4-no-affinity'}
        return {'kind':kind,'blocks':20 if kind=='paired' else 0,'seed':20261009,
                'bootstrap_reps':10000,'identity':identity,
                'identity_sha256':self.s.digest_json(identity),'library_path':str(self.s.LIB),
                'host_path':str(self.s.HOST),'jobs':self.s.jobs(kind,20,20261009,matrix)}

    def evidence(self,job):
        count=self.s.WORKLOADS[job['workload']]['workers'];mode=job['compiler_mode']
        lines=['loaded-lib='+str(self.s.LIB),
               f'mode={mode} setting={mode} has_lse=true strategy=cranelift',
               f'v4-policy={job["v4_policy"]} v4-rule-version=1',
               'engine-pulley=0','guest-stop: kind=return code=0','result: ok',
               'timer=CLOCK_MONOTONIC',f'spawn-count={count} workers-joined={count}',
               'thread-entry=yes shared-memory-imports=1','exec_ns=12345']
        lines += [f'worker-stop: tid={tid} kind=return code=0' for tid in range(1,count+1)]
        if job['workload']=='stress':
            stdout='Test finished successfully!\nfreed 400000 workers 8 blocks 50000\n'
        elif job['workload']=='malloc':
            stdout='Workers: 4\n'+('thread started\nthread exiting with total -512 steps 25000000\n'*4)+'Done.\n'
        else:
            stdout='PARSEC Benchmark Suite\nNumber of cells: 36900\nNumber of particles: 36504\nLoading file "/work/000-input.fluid"...\nSaving file "/work/'+self.s.particle_name(job)+'"...\n'
        return stdout,'\n'.join(lines)+'\n'

    def test_williams_literal_position_and_directed_pair_balance(self):
        for matrix,count in [('primary',5),('ablation',4)]:
            rows=self.s.orders(20,20261009,matrix);names=list(self.s.arms(matrix))
            self.assertEqual(len(rows),20)
            for position in range(len(names)):
                self.assertEqual(Counter(row[position] for row in rows),dict.fromkeys(names,count))
            pairs=Counter((a,b) for row in rows for a,b in zip(row,row[1:]))
            self.assertEqual(pairs,{(a,b):count for a in names for b in names if a!=b})
            self.assertNotEqual(rows,self.s.orders(20,20261010,matrix))

    def test_retest_has_three_complete_arms_and_unchanged_work(self):
        plan=self.plan('retest')
        self.s.validate_plan(plan)
        self.assertEqual(len(plan['jobs']),198)
        self.assertEqual(len(self.s.jobs('smoke',matrix='retest')),9)
        self.assertEqual(plan['identity']['arms'],{
            'baseline':{'compiler_mode':'none','v4_policy':'legacy'},
            'reference':{'compiler_mode':'cas_csdb','v4_policy':'legacy'},
            'v4_opt':{'compiler_mode':'ssbb','v4_policy':'ranges_cfg'}})
        for workload in ('stress','malloc','fluidanimate'):
            measured=[j for j in plan['jobs'] if j['workload']==workload and j['phase']=='paired']
            self.assertEqual(len(measured),60)
            for block in range(1,21):
                self.assertEqual({j['arm'] for j in measured if j['block']==block},
                                 {'baseline','reference','v4_opt'})
        for job in plan['jobs']:
            cmd=self.r.trial_command(job,Path('/tmp/00-retest'))
            self.assertEqual(cmd[cmd.index('--mode')+1],job['compiler_mode'])
            self.assertEqual(cmd[cmd.index('--v4-policy')+1],job['v4_policy'])
            if job['workload']=='malloc':self.assertEqual(cmd[-3:],['--','4','100000000'])
            elif job['workload']=='fluidanimate':self.assertEqual(cmd[-4:-1],['4','1845','/work/000-input.fluid'])

    def test_retest_order_is_near_balanced_and_reproducible(self):
        for seed in range(20261009,20261018):
            rows=self.s.orders(20,seed,'retest')
            self.assertEqual(rows,self.s.orders(20,seed,'retest'))
            self.assertEqual(len(rows),20)
            self.assertEqual(sorted(Counter(tuple(row) for row in rows[:18]).values()),[3]*6)
            for position in range(3):
                self.assertEqual(sorted(Counter(row[position] for row in rows).values()),[6,7,7])
            pairs=Counter((a,b) for row in rows for a,b in zip(row,row[1:]))
            self.assertEqual(len(pairs),6)
            self.assertTrue(all(6<=count<=8 for count in pairs.values()))

    def test_retest_statistics_exclude_warmup_and_compare_csdb_directly(self):
        plan=self.plan('retest');records=[]
        for job in plan['jobs']:
            # Drifting baseline makes ratios of unpaired medians inappropriate.
            base=10 if job['block']%2 else 1000
            elapsed={'baseline':base,'reference':200 if job['block']%2 else 1100,
                     'v4_opt':int(base*1.5)}[job['arm']]
            if job['phase']=='warmup':elapsed=999999
            records.append({**job,'identity_sha256':plan['identity_sha256'],
                            'correctness':'pass','exec_ns':elapsed})
        rows=self.a.summarize(plan,records)
        self.assertEqual(len(rows),9)
        for row in rows:
            self.assertEqual(row['n_blocks'],20)
            want={('reference','baseline'):(10.55,1.1,20),
                  ('v4_opt','baseline'):(1.5,1.5,1.5),
                  ('v4_opt','reference'):((15/200+1500/1100)/2,15/200,1500/1100)}[
                      (row['numerator'],row['denominator'])]
            self.assertEqual((row['slowdown'],row['ci95_low'],row['ci95_high']),want)
        with self.assertRaises(ValueError):self.a.summarize(plan,records[:-1])

    def test_prepare_source_uses_pinned_commit_and_preserves_checkout(self):
        prepare=load('12-prepare-source.py')
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
            (repo/'code').write_text('reviewed\n');git('add','code');git('commit','-m','reviewed')
            pinned=git('rev-parse','HEAD')
            (repo/'code').write_text('newer\n');git('commit','-am','newer')
            newer=git('rev-parse','HEAD');(repo/'code').write_text('uncommitted\n')
            dest=Path(tmp)/'build'/'05-source'
            prepare.prepare(repo,dest,{'compiler_commit':pinned,'compiler_base':pinned})
            self.assertEqual((dest/'code').read_text(),'reviewed\n')
            self.assertEqual(git('rev-parse','HEAD'),newer)
            self.assertEqual((repo/'code').read_text(),'uncommitted\n')
            with self.assertRaises(ValueError):prepare.prepare(repo,dest,{'compiler_commit':pinned,'compiler_base':pinned})
            with self.assertRaises((ValueError,subprocess.CalledProcessError)):
                prepare.prepare(repo,Path(tmp)/'missing',{'compiler_commit':'0'*40,'compiler_base':pinned})
            self.assertFalse((Path(tmp)/'missing').exists())

    def test_workload_and_arm_commands_preserve_original_work(self):
        for matrix,number in [('primary',264),('ablation',330)]:
            plan=self.plan(matrix);self.assertEqual(len(plan['jobs']),number)
            for job in plan['jobs']:
                command=self.r.trial_command(job,Path('/tmp/00-run'))
                self.assertEqual(command[command.index('--mode')+1],job['compiler_mode'])
                self.assertEqual(command[command.index('--v4-policy')+1],job['v4_policy'])
                self.assertNotIn('--cpus',command)
                if job['workload']=='malloc':self.assertEqual(command[-3:],['--','4','100000000'])
                elif job['workload']=='fluidanimate':self.assertEqual(command[-4:-1],['4','1845','/work/000-input.fluid'])
                else:self.assertEqual(command[-1],'--')
        self.assertEqual(self.s.arms()['reference'],{'compiler_mode':'cas_csdb','v4_policy':'legacy'})
        self.assertEqual(self.s.WORKLOADS['fluidanimate']['workers'],4)

    def test_all_evidence_and_invalid_library_policy_thread_exit_are_checked(self):
        plan=self.plan()
        for workload in self.s.WORKLOADS:
            for arm in self.s.arms():
                job=next(j for j in plan['jobs'] if j['workload']==workload and j['arm']==arm)
                stdout,stderr=self.evidence(job)
                self.assertEqual(self.s.validate_trial(plan,job,stdout,stderr,0)['exec_ns'],12345)
                bad=[stderr.replace('loaded-lib='+str(self.s.LIB),'loaded-lib=/tmp/other.dylib'),
                     stderr.replace('v4-rule-version=1','v4-rule-version=2'),
                     stderr.replace('workers-joined='+str(self.s.WORKLOADS[workload]['workers']),'workers-joined=0'),
                     stderr.replace('worker-stop: tid=1 kind=return code=0','worker-stop: tid=1 kind=trap code=0'),
                     stderr+'exec_ns=777\n',stderr+'result: ok\n']
                for broken in bad:
                    with self.assertRaises(ValueError):self.s.validate_trial(plan,job,stdout,broken,0)
                with self.assertRaises(ValueError):self.s.validate_trial(plan,job,stdout,stderr,1)
                with self.assertRaises(ValueError):self.s.validate_trial(plan,job,'',stderr,0)

    def test_frozen_inputs_and_file_integrity(self):
        self.s.validate_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'00-out.fluid'
            p.write_bytes(struct.pack('<fi',392,36504)+bytes(36504*9*4))
            self.assertEqual(self.s.V.inspect_output(p)['particles'],36504)
            p.write_bytes(struct.pack('<fi',392,36503)+bytes(36504*9*4))
            with self.assertRaises(ValueError):self.s.V.inspect_output(p)
            p.write_bytes(struct.pack('<fi',392,36504)+struct.pack('<f',float('nan'))+bytes(36504*9*4-4))
            with self.assertRaises(ValueError):self.s.V.inspect_output(p)

    def test_whole_block_bootstrap_and_opt_legacy_direct_pair(self):
        plan=self.plan();records=[]
        for job in plan['jobs']:
            base=10 if job['block']%2 else 1000
            t=base*{'baseline':1,'v4_legacy':2,'v4_opt':1.5,'reference':3}[job['arm']]
            if job['arm']=='v4_legacy':t=200 if job['block']%2 else 1100
            records.append({**job,'identity_sha256':plan['identity_sha256'],'correctness':'pass','exec_ns':int(t)})
        rows=self.a.summarize(plan,records)
        self.assertEqual(len(rows),12)
        for row in rows:
            self.assertEqual(row['n_blocks'],20)
            self.assertEqual(row['slowdown'],{'v4_legacy':10.55,'v4_opt':1.5,'reference':3,'v4_opt/legacy':(15/200+1500/1100)/2}[row['arm']])
            if row['arm']=='v4_legacy':self.assertEqual((row['ci95_low'],row['ci95_high']),(1.1,20))
        for rows in [records[:-1],records+[records[0]]]:
            with self.assertRaises(ValueError):self.a.summarize(plan,rows)
        broken=copy.deepcopy(records);broken[0]['correctness']='fail'
        with self.assertRaises(ValueError):self.a.summarize(plan,broken)
        with self.assertRaises(ValueError):self.a.summarize(self.plan(kind='smoke'),records)

    def test_changed_artifact_and_raw_evidence_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'00-artifact';p.write_text('old')
            identity={'artifacts':{str(p):self.s.sha256(p)}}
            plan={'identity':identity,'identity_sha256':self.s.digest_json(identity)}
            self.s.assert_artifacts(plan);p.write_text('new')
            with self.assertRaises(ValueError):self.s.assert_artifacts(plan)
            frozen=self.s.freeze_evidence(tmp)
            self.assertEqual(p.stat().st_mode&0o222,0)
            self.s.verify_frozen(tmp,{'evidence_sha256':frozen})
            p.chmod(0o644);p.write_text('changed')
            with self.assertRaises(ValueError):self.s.verify_frozen(tmp,{'evidence_sha256':frozen})

    def test_existing_output_and_incomplete_analysis_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'00-existing';out.mkdir();(out/'00-marker').write_text('keep')
            with self.assertRaises(ValueError):self.r.create_output(out)
            self.assertEqual((out/'00-marker').read_text(),'keep')
            self.s.write_json(out/'07-status.json',{'state':'failed'})
            with self.assertRaisesRegex(ValueError,'not complete'):self.a.analyze_run(out)
            self.assertFalse((out/'04-summary.csv').exists())

    def test_nonmac_preflight_starts_no_trial(self):
        if sys.platform=='darwin':self.skipTest('Linux portable guard test')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'00-nonmac'
            result=subprocess.run([sys.executable,'-B','-S',str(ROOT/'03-run.py'),'run','--out',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,77,result.stderr)
            status=json.loads((out/'07-status.json').read_text())
            self.assertEqual((status['state'],status['completed_trials']),('blocked',0))
            self.assertFalse((out/'02-trials.jsonl').exists())

    def test_subset_ablation_keeps_full_five_arm_blocks(self):
        plan=self.plan('ablation');plan['identity']['selected_workloads']=['fluidanimate']
        plan['identity_sha256']=self.s.digest_json(plan['identity'])
        plan['jobs']=[job for job in plan['jobs'] if job['workload']=='fluidanimate']
        for number,job in enumerate(plan['jobs']):job['id']=number
        self.s.validate_plan(plan)
        self.assertEqual(len(plan['jobs']),110)
        records=[{**job,'identity_sha256':plan['identity_sha256'],'correctness':'pass',
                  'exec_ns':{'baseline':10,'v4_legacy':20,'v4_frame':18,'v4_ranges':16,'v4_opt':15}[job['arm']]}
                 for job in plan['jobs']]
        rows=self.a.summarize(plan,records)
        self.assertEqual({row['workload'] for row in rows},{'fluidanimate'})
        self.assertEqual(len(rows),5)

    def complete_fixture(self,out):
        # The requested three-arm fixture exercises all production publication gates.
        plan=self.plan('retest');plan['identity']['selected_workloads']=['fluidanimate']
        plan['identity_sha256']=self.s.digest_json(plan['identity'])
        plan['jobs']=[job for job in plan['jobs'] if job['workload']=='fluidanimate']
        for number,job in enumerate(plan['jobs']):job['id']=number
        self.s.write_json(out/'00-plan.json',plan)
        self.s.write_json(out/'01-preflight.json',{'kind':'paired','ok':True})
        self.s.write_json(out/'08-machine-after.json',{'fixture':True})
        self.s.prepare_particles(out)
        logs=out/'06-logs';logs.mkdir()
        fluid=struct.pack('<fi',392,36504)+bytes(36504*9*4)
        records=[]
        for job in plan['jobs']:
            stdout,stderr=self.evidence(job)
            paths=[logs/f'{2*job["id"]:03d}-{job["workload"]}-{job["arm"]}-stdout.txt',
                   logs/f'{2*job["id"]+1:03d}-{job["workload"]}-{job["arm"]}-stderr.txt']
            for path,content in zip(paths,(stdout,stderr)):path.write_text(content)
            (out/'10-particles'/self.s.particle_name(job)).write_bytes(fluid)
            row={**job,'identity_sha256':plan['identity_sha256'],'returncode':0,
                 'command':self.r.trial_command(job,out),
                 'stdout_file':str(paths[0].relative_to(out)),'stdout_sha256':self.s.sha256(paths[0]),
                 'stderr_file':str(paths[1].relative_to(out)),'stderr_sha256':self.s.sha256(paths[1]),
                 **self.s.validate_trial(plan,job,stdout,stderr,0),**self.s.validate_output(out,job)}
            records.append(row)
        (out/'02-trials.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in records))
        status={'state':'analyzing','completed_trials':66,
                'plan_sha256':self.s.digest_json(plan),'identity_sha256':plan['identity_sha256']}
        self.s.write_json(out/'07-status.json',status)
        return plan,status

    def test_complete_publication_is_immutable_and_raw_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan,status=self.complete_fixture(out)
            rows=self.a.analyze_run(out,completing_status=status)
            self.assertEqual(len(rows),3);self.assertEqual(status['state'],'complete')
            before={name:(out/name).read_bytes() for name in ('04-summary.csv','05-summary.md','07-status.json')}
            self.assertEqual(self.a.analyze_run(out),rows)
            for name,data in before.items():self.assertEqual((out/name).read_bytes(),data)
            raw=out/f'06-logs/000-fluidanimate-{plan["jobs"][0]["arm"]}-stdout.txt'
            raw.chmod(0o644);raw.write_text('tampered\n')
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):self.a.analyze_run(out)
            for name,data in before.items():self.assertEqual((out/name).read_bytes(),data)

    def test_missing_fluid_output_cannot_publish_pass_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan,status=self.complete_fixture(out)
            (out/'10-particles'/self.s.particle_name(plan['jobs'][0])).unlink()
            with self.assertRaises(FileNotFoundError):self.a.analyze_run(out,completing_status=status)
            self.assertFalse((out/'04-summary.csv').exists())
            self.assertFalse((out/'05-summary.md').exists())

    def test_host_rejects_invalid_combinations_before_loading_guest(self):
        binary=os.environ.get('MAC_HOST_TEST_BINARY')
        if not binary:self.skipTest('set MAC_HOST_TEST_BINARY to compiled host for parser checks')
        for mode,policy in [('none','frame'),('cas_csdb','ranges_cfg'),('ssbb','unknown')]:
            result=subprocess.run([binary,'--mode',mode,'--v4-policy',policy,'--wasm','/missing.wasm'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn('loaded-lib=',result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
