"""Portable M4 protocol and evidence, with no Linux topology or SSBD imports."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'07-build'
HOST=BUILD/'03-host'
LIB=BUILD/'02-sdk/lib/libwasmtime.dylib'
MANIFEST=BUILD/'04-build-info.json'
INPUT_FILE=ROOT/'04-inputs/03-in_35K.fluid'
INPUT_SHA256='0a649e52c616ee52ef24038b1d4dae316520f517c122f45fa16755b761ada5a6'
FEATURES=['cranelift','wat','wasi','threads','parallel-compilation','gc','gc-drc']
BASE='1cea7556c072bc9304a4770afc88de9e0b956ccb'
COMPILER_COMMIT='2648e4a22107d3409c2e0cfac9499c3772dc25a7'

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

V=load('m4_fluid_integrity','11-output-check.py')

def require(condition,message):
    if not condition:raise ValueError(message)

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,value):
    # Replace atomically, so interruptions cannot publish a partial marker.
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)

def one(pattern,text,label):
    matches=list(re.finditer(pattern,text,re.M))
    require(len(matches)==1,'missing or duplicate '+label)
    return matches[0]

ARMS={
    'baseline':{'compiler_mode':'none','v4_policy':'legacy'},
    'v4_legacy':{'compiler_mode':'ssbb','v4_policy':'legacy'},
    'v4_opt':{'compiler_mode':'ssbb','v4_policy':'ranges_cfg'},
    'reference':{'compiler_mode':'cas_csdb','v4_policy':'legacy'},
}
ABLATION_ARMS={
    'baseline':ARMS['baseline'],'v4_legacy':ARMS['v4_legacy'],
    'v4_frame':{'compiler_mode':'ssbb','v4_policy':'frame'},
    'v4_ranges':{'compiler_mode':'ssbb','v4_policy':'ranges'},'v4_opt':ARMS['v4_opt'],
}
RETEST_ARMS={name:ARMS[name] for name in ('baseline','reference','v4_opt')}
WORKLOADS={
    'stress':{'wasm':str(ROOT/'04-inputs/00-pthread-malloc-free.wasm'),'args':[],
              'workers':8,'units':400000,'timeout':1800,
              'sha256':'d14f3e10fca626860da7703abae863d3023836a6a04c01ce38906c371b3007b3'},
    'malloc':{'wasm':str(ROOT/'04-inputs/01-malloc-multithreading.wasm'),'args':['4','100000000'],
              'workers':4,'units':100000000,'timeout':1800,
              'sha256':'6fcffb5c98dc89703b599856ab56362f3fee7ce1f436b926211526d2cb67e5f5'},
    'fluidanimate':{'wasm':str(ROOT/'04-inputs/02-fluidanimate.wasm'),'args':['4','1845'],
              'workers':4,'units':1845,'particles':36504,'timeout':1800,
              'sha256':'fc94c58be6916eada4f6573bbe6317a8398476d01b922768ff435cb796b7bcb0'},
}

def arms(matrix='primary'):
    matrices={'retest':RETEST_ARMS,'primary':ARMS,'ablation':ABLATION_ARMS}
    require(matrix in matrices,'unknown measurement matrix')
    return matrices[matrix]

def orders(blocks,seed,matrix='primary'):
    """Complete-arm blocks; three arms use the AMD runner's near balance."""
    require(blocks in (2,20),'protocol permits 2 warmup or 20 measured blocks')
    rng=random.Random(seed);names=list(arms(matrix));n=len(names)
    if n==3:
        # Every permutation appears three times in the first 18 blocks.
        # Two cyclic rows finish 20: positions 6/7/7, directed neighbors 6..8.
        result=[]
        for _ in range(3):
            round_rows=[list(row) for row in itertools.permutations(names)]
            rng.shuffle(round_rows);result.extend(round_rows)
        labels=names[:];rng.shuffle(labels)
        result.extend([labels,labels[1:]+labels[:1]])
        return result[:blocks]
    row=[0,1,3,2] if n==4 else [0,1,4,2,3]
    template=[[(value+offset)%n for value in row] for offset in range(n)]
    if n%2:template += [list(reversed(row)) for row in template[:]]
    result=[]
    for _ in range(20//len(template)):
        labels=names[:];rng.shuffle(labels)
        round_rows=[[labels[index] for index in row] for row in template]
        rng.shuffle(round_rows);result += round_rows
    return result[:blocks]

def jobs(kind,blocks=20,seed=20261009,matrix='primary'):
    require(kind in ('paired','smoke'),'unknown experiment kind')
    result=[]
    for offset,workload in enumerate(WORKLOADS):
        configurations=arms(matrix)
        stages=[('smoke',[list(configurations)])] if kind=='smoke' else [
            ('warmup',orders(2,seed+offset*2+1,matrix)),
            ('paired',orders(blocks,seed+offset*2,matrix))]
        for phase,sequences in stages:
            for block,arm_order in enumerate(sequences,1):
                for position,arm in enumerate(arm_order,1):
                    result.append({'id':len(result),'workload':workload,'phase':phase,
                        'block':block,'position':position,'arm':arm,**configurations[arm]})
    return result

def capture(command):
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=15)
        return {'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'command':command,'error':str(exc)}

def machine_snapshot():
    result={'system':platform.system(),'machine':platform.machine(),'platform':platform.platform(),
            'scheduling':'Original host std::thread scheduling; no affinity or QoS changes.'}
    if platform.system()=='Darwin':
        result['commands']=[capture(cmd) for cmd in [
            ['sw_vers'],['sysctl','machdep.cpu.brand_string','hw.memsize','hw.ncpu','hw.physicalcpu','hw.logicalcpu'],
            ['pmset','-g','batt'],['pmset','-g','therm'],['uptime']]]
    return result

def validate_inputs():
    for cfg in WORKLOADS.values():require(sha256(cfg['wasm'])==cfg['sha256'],'original M4 Wasm hash mismatch')
    require(sha256(INPUT_FILE)==INPUT_SHA256,'original M4 fluid input hash mismatch')

def validate_build():
    manifest=json.loads(MANIFEST.read_text())
    require(manifest['target']=='aarch64-apple-darwin' and manifest['profile']=='release','wrong native build target/profile')
    require(manifest['features']==FEATURES,'C API feature set differs')
    require(manifest['v4_rule_version']==1 and manifest['v4_policies']==['legacy','frame','ranges','ranges_cfg'],'unsupported v4 rules')
    require(manifest['source_base']==BASE and manifest['source_commit']==COMPILER_COMMIT,'wrong compiler source identity')
    require(manifest['source_delivery']=='git-pinned-checkout' and manifest['source_clean'] is True,'unreviewed/dirty compiler source')
    require(manifest['has_lse'] is True and manifest['persistent_cache'] is False,'wrong LSE/cache configuration')
    required=['03-host','02-sdk/lib/libwasmtime.dylib','02-sdk/lib/libwasmtime.a']
    for relative in required:require('07-build/'+relative in manifest['sha256'],'missing SDK/host hash')
    require('05-source.json' in manifest['sha256'],'missing compiler pin hash')
    for name,expected in manifest['sha256'].items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT),'build manifest artifact path escapes handoff')
        require(sha256(path)==expected,'build manifest mismatch: '+name)
    return manifest

def assert_artifacts(plan):
    require(digest_json(plan['identity'])==plan['identity_sha256'],'plan identity hash mismatch')
    for name,expected in plan['identity']['artifacts'].items():
        require(sha256(ROOT/name)==expected,'artifact changed: '+name)

def make_plan(kind,blocks=20,seed=20261009,matrix='primary',workloads=None):
    require(blocks==20,'formal protocol requires exactly 20 measured blocks')
    require(seed==20261009,'frozen experiment seed must be 20261009')
    selected=list(WORKLOADS) if workloads is None else workloads
    require(bool(selected) and len(set(selected))==len(selected) and all(w in WORKLOADS for w in selected),'invalid workload selection')
    validate_inputs();manifest=validate_build();hashes=dict(manifest['sha256'])
    required=[MANIFEST,INPUT_FILE,ROOT/'05-source.json']+[Path(cfg['wasm']) for cfg in WORKLOADS.values()]
    required += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.cpp','.sh')]
    for path in required:hashes[str(path.relative_to(ROOT))]=sha256(path)
    identity={'artifacts':hashes,'workloads':WORKLOADS,'arms':arms(matrix),'matrix':matrix,
        'selected_workloads':selected,'v4_rule_version':1,'scheduling':'original-m4-no-affinity',
        'build':manifest,'fluid_validation':{'kind':V.POLICY,'particles':V.PARTICLES,'bytes':V.OUTPUT_BYTES,
        'limitation':'Execution/output integrity only; no cross-run numerical or physical-correctness claim.'}}
    plan={'schema':1,'kind':kind,'blocks':blocks if kind=='paired' else 0,'seed':seed,'bootstrap_reps':10000,
        'library_path':str(LIB),'host_path':str(HOST),'identity':identity,'identity_sha256':digest_json(identity),
        'jobs':[job for job in jobs(kind,blocks,seed,matrix) if job['workload'] in selected],
        'machine':machine_snapshot(),'interpretation':matrix+' complete-block paired performance' if kind=='paired' else 'SMOKE ONLY: correctness; no performance summary.'}
    # IDs remain dense when an independently frozen ablation selects fluidanimate.
    for number,job in enumerate(plan['jobs']):job['id']=number
    assert_artifacts(plan);return plan

def validate_plan(plan):
    require(plan['kind'] in ('paired','smoke'),'unknown experiment kind')
    require(plan['blocks']==(20 if plan['kind']=='paired' else 0),'wrong block count')
    require(plan['seed']==20261009 and plan['bootstrap_reps']==10000,'wrong frozen seed/bootstrap count')
    identity=plan['identity'];matrix=identity['matrix']
    require(identity['workloads']==WORKLOADS and identity['arms']==arms(matrix),'workload or arm protocol changed')
    require(identity['v4_rule_version']==1 and identity['scheduling']=='original-m4-no-affinity','v4 rule/scheduling changed')
    selected=identity.get('selected_workloads',list(WORKLOADS))
    require(bool(selected) and len(set(selected))==len(selected) and all(w in WORKLOADS for w in selected),'invalid workload selection')
    expected=[job for job in jobs(plan['kind'],20,plan['seed'],matrix) if job['workload'] in selected]
    for number,job in enumerate(expected):job['id']=number
    require(plan['jobs']==expected,'planned jobs do not match fixed balanced protocol')
    require(plan['host_path']==str(HOST) and plan['library_path']==str(LIB),'host/SDK path mismatch')

def particle_name(job):
    return f'{job["id"]+1:03d}-fluidanimate-{job["arm"]}.fluid'

def prepare_particles(out):
    folder=out/'10-particles';folder.mkdir();shutil.copyfile(INPUT_FILE,folder/'000-input.fluid')
    (folder/'index.md').write_text('# 粒子状态\n\n- 000 [000-input.fluid](000-input.fluid)：固定 M4 simsmall 输入副本。\n')

def validate_output(out,job):
    if job['workload']!='fluidanimate':return {}
    folder=Path(out)/'10-particles'
    require(sha256(folder/'000-input.fluid')==INPUT_SHA256,'input copy changed')
    path=folder/particle_name(job);result=V.inspect_output(path)
    return {'particle_file':str(path.relative_to(out)),'particle_sha256':sha256(path),'particle_validation':result}

def audit_output(out,record):
    if record['workload']!='fluidanimate':
        require('particle_file' not in record,'unexpected particle output');return
    require(record.get('particle_file')=='10-particles/'+particle_name(record),'wrong particle path')
    actual=validate_output(out,record)
    require(all(record.get(key)==value for key,value in actual.items()),'particle evidence differs from saved output')

def validate_trial(plan,job,stdout,stderr,returncode):
    require(returncode==0,f'host exited {returncode}')
    cfg=WORKLOADS[job['workload']];count=cfg['workers'];arm=arms(plan['identity']['matrix'])[job['arm']]
    require(job['compiler_mode']==arm['compiler_mode'] and job['v4_policy']==arm['v4_policy'],'arm/compiler/policy mismatch')
    loaded=one(r'^loaded-lib=(.+)$',stderr,'loaded library')[1]
    require(Path(loaded).resolve()==Path(plan['library_path']).resolve(),'wrong loaded library')
    mode=job['compiler_mode']
    setting=one(r'^mode=(\w+) setting=(\w+) has_lse=(\w+) strategy=(\w+)$',stderr,'compiler setting')
    require(setting.groups()==(mode,mode,'true','cranelift'),'compiler setting mismatch')
    policy=one(r'^v4-policy=(\w+) v4-rule-version=(\d+)$',stderr,'v4 policy')
    require(policy.groups()==(job['v4_policy'],'1'),'v4 policy/rule mismatch')
    for line in ['engine-pulley=0','guest-stop: kind=return code=0','result: ok','timer=CLOCK_MONOTONIC',
                 f'spawn-count={count} workers-joined={count}','thread-entry=yes shared-memory-imports=1']:
        require(stderr.splitlines().count(line)==1,'missing or duplicate evidence: '+line)
    stops=re.findall(r'^worker-stop: tid=(\d+) kind=(\w+) code=(-?\d+)$',stderr,re.M)
    require(len(stops)==count and {int(x[0]) for x in stops}==set(range(1,count+1)),'missing or duplicate worker evidence')
    require(all(kind=='return' and code=='0' for _,kind,code in stops),'worker exit failure')
    elapsed=int(one(r'^exec_ns=(\d+)$',stderr,'execution time')[1]);require(elapsed>0,'nonpositive execution time')
    if job['workload']=='stress':
        require(stdout.count('Test finished successfully!')==1 and stdout.count('freed 400000 workers 8 blocks 50000')==1,'stress correctness output mismatch')
    elif job['workload']=='malloc':
        require(stdout.count('Workers: 4')==1 and stdout.count('thread started')==4 and
                stdout.count('thread exiting with total -512 steps 25000000')==4 and stdout.count('Done.')==1,'malloc correctness output mismatch')
    else:
        for line in ['PARSEC Benchmark Suite','Number of cells: 36900','Number of particles: 36504',
                     'Loading file "/work/000-input.fluid"...',f'Saving file "/work/{particle_name(job)}"...']:
            require(stdout.splitlines().count(line)==1,'fluidanimate stdout mismatch: '+line)
    return {'exec_ns':elapsed,'workers':count,'work_units':cfg['units'],'compiler_mode':mode,
        'v4_policy':job['v4_policy'],'correctness':'pass','worker_stops':[list(stop) for stop in stops],
        'validation_scope':V.POLICY if job['workload']=='fluidanimate' else 'm4-workload-completion-v1'}

def freeze_evidence(directory):
    hashes={}
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.name=='index.md':continue
        relative=str(path.relative_to(directory))
        if relative in ('07-status.json','04-summary.csv','05-summary.md') or relative.startswith('09-pending-summary/'):continue
        hashes[relative]=sha256(path);path.chmod(0o444)
    return hashes

def verify_frozen(directory,status):
    require(bool(status.get('evidence_sha256')),'missing frozen evidence')
    for name,expected in status['evidence_sha256'].items():
        path=Path(directory)/name
        require(path.resolve().is_relative_to(Path(directory).resolve()),'evidence path escapes run directory')
        require(sha256(path)==expected,'frozen evidence changed: '+name)
