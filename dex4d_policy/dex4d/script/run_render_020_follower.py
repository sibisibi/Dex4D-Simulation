"""020 replay-render follower. Scans completed benchmark units and renders each
unit's MEDIAN-TP trajectory (sorted index n//2, upper median, ties broken by
trajectory index) with render_replay_020.py, up to --max_procs concurrent
processes round-robined over --gpus. Rerun-safe: existing mp4s are skipped.
A failed render is recorded in render_failed.txt and the scan continues.

    python script/run_render_020_follower.py --once      # one pass over done units
    python script/run_render_020_follower.py             # keep following the queue
"""

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys
import time

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)
from eval_benchmark_020 import resolve_render_urdf  # noqa: E402

RUNS = '/home/nas5/sibeenkim/work/_020-diverse-eval/runs_dex4d'
VIDEOS = '/home/nas5/sibeenkim/work/_020-diverse-eval/videos_dex4d'


def median_traj(result):
    tp = result['per_traj_task_progress_pct']
    order = sorted(range(len(tp)), key=lambda i: (tp[i], i))
    return order[len(tp) // 2]


def pending_jobs():
    jobs = []
    for eval_json in sorted(glob.glob(f'{RUNS}/*/*/benchmark_eval.json')):
        with open(eval_json) as f:
            result = json.load(f)
        safe = osp.basename(osp.dirname(eval_json))
        traj = median_traj(result)
        out = osp.join(VIDEOS, f'{safe}_dex4d_traj{traj}.mp4')
        if osp.exists(out):
            continue
        jobs.append({
            'npz': osp.join(osp.dirname(eval_json), 'poses.npz'),
            'traj': traj,
            'urdf': resolve_render_urdf(result['key'], result['set'], REPO),
            'out': out,
            'name': f"{result['set']}/{safe}",
        })
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--max_procs', type=int, default=4)
    p.add_argument('--gpus', default='1,2')
    p.add_argument('--poll_s', type=int, default=60)
    args = p.parse_args()
    gpus = args.gpus.split(',')

    os.makedirs(VIDEOS, exist_ok=True)
    running = []  # (proc, job, log_file)
    strikes = {}  # out path -> abort count; 3 strikes blacklists for this run
    failed = set()
    slot = 0
    while True:
        for proc, job, lf in running[:]:
            if proc.poll() is not None:
                lf.close()
                running.remove((proc, job, lf))
                ok = proc.returncode == 0 and osp.exists(job['out']) and osp.getsize(job['out']) > 0
                if not ok:
                    # IsaacGym startup aborts are probabilistic on this host,
                    # retry up to 3 times before recording the failure
                    strikes[job['out']] = strikes.get(job['out'], 0) + 1
                    if strikes[job['out']] >= 3:
                        failed.add(job['out'])
                        with open(osp.join(VIDEOS, 'render_failed.txt'), 'a') as f:
                            f.write(job['name'] + '\n')
                        print(f"[FAIL] {job['name']} (rc {proc.returncode}, 3 strikes)")
                    else:
                        print(f"[retry] {job['name']} (rc {proc.returncode}, strike {strikes[job['out']]})")
                else:
                    print(f"[done] {job['out']}")
        launched = 0
        if len(running) < args.max_procs:
            active = {j['out'] for _, j, _ in running}
            for job in pending_jobs():
                if len(running) >= args.max_procs:
                    break
                if job['out'] in active or job['out'] in failed:
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[slot % len(gpus)])
                slot += 1
                lf = open(job['out'] + '.log', 'w')
                proc = subprocess.Popen(
                    [sys.executable, osp.join(REPO, 'render_replay_020.py'),
                     '--npz', job['npz'], '--traj', str(job['traj']),
                     '--object_urdf', job['urdf'], '--out', job['out']],
                    cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=env)
                running.append((proc, job, lf))
                launched += 1
                print(f"[start] {job['name']} traj {job['traj']} on gpu {env['CUDA_VISIBLE_DEVICES']}")
        if args.once and not running and launched == 0 and \
                all(j['out'] in failed for j in pending_jobs()):
            break
        time.sleep(5 if running else args.poll_s)
    print('[follower] all pending renders done')


if __name__ == '__main__':
    main()
