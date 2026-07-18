"""020 benchmark driver, one UniDexGrasp or DexToolBench unit per launch.

Wraps train.py --test: builds a one-object cfg_env yaml carrying the unit's goal
bank, launches the stage-3 teacher on 10 envs (env i = bank trajectory i), then
verifies the run against the bank and writes benchmark_eval.json.

The success gate is SimToolReal's, applied inside the env (see
tasks/xarm6_leap_hand_ap2ap.py _benchmark_post_reward): max corner distance of a
fixed box <= 0.015 m for one step, 600-step per-goal budget, episode ends the
moment the 10th goal lands. Bank z maps +0.22 (SimToolReal table top 0.38,
Dex4D table top 0.6; both frames center the table at x,y origin).

    python eval_benchmark_020.py --key core/bottle-...@008 --set udg-seen \
        --out_dir /home/nas5/sibeenkim/work/_020-diverse-eval/runs_dex4d/udg-seen/<safe_key>
"""

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import yaml

SCALE_BY_SUFFIX = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}
BANK_DIR = '/home/nas5/sibeenkim/work/_020-diverse-eval/goal_banks'
DTB_ASSETS = '/home/nas5/sibeenkim/work/simtoolreal-020/assets/urdf/dextoolbench'
Z_SHIFT = 0.22
K_GOALS = 10
N_TRAJ = 10
# IsaacGym's VHACD path corrupts the heap on this host's glibc (aborts with
# free(): invalid pointer or corrupted size right after the decomposition
# lines). Benchmark units therefore load a generated per-piece urdf built from
# the shipped coacd_convex_piece_*.obj files with VHACD off: one convex hull
# per coacd piece, the true decomposition, the same visual mesh.


def generate_pieces_urdf(code, suffix, out_dir, repo_dir):
    coacd_dir = osp.abspath(osp.join(repo_dir, '..', 'assets', 'meshdatav3_scaled', code, 'coacd'))
    pieces = sorted(glob.glob(osp.join(coacd_dir, 'coacd_convex_piece_*.obj')))
    assert pieces, coacd_dir
    scale = SCALE_BY_SUFFIX[suffix]
    vis = osp.join(coacd_dir, f'decomposed_{suffix}.obj')
    assert osp.exists(vis), vis
    # pieces are at canonical scale, decomposed_<suffix>.obj is pre-scaled
    # (verified: decomposed_008 extents = 0.08 x decomposed extents exactly)
    lines = ['<robot name="root">', '  <link name="link_001">',
             '    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>',
             f'      <mesh filename="{vis}" scale="1 1 1"/>',
             '    </geometry></visual>']
    for p in pieces:
        lines += ['    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry>',
                  f'      <mesh filename="{p}" scale="{scale} {scale} {scale}"/>',
                  '    </geometry></collision>']
    lines += ['  </link>', '</robot>']
    path = osp.join(out_dir, 'object_pieces.urdf')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def safe_key(key):
    return key.replace('/', '__').replace('@', '_s')


def dtb_object_dir(key):
    matches = glob.glob(f'{DTB_ASSETS}/*/{key}')
    assert len(matches) == 1, f'{key}: expected one DTB object dir, got {matches}'
    return matches[0]


def resolve_render_urdf(key, set_name, repo_dir):
    '''Object urdf for the replay renderer, same asset the eval loaded.'''
    if set_name == 'dtb':
        d = dtb_object_dir(key)
        return osp.join(d, f'{key}_decomposed.urdf')
    code, suffix = key.split('@')
    return osp.abspath(osp.join(repo_dir, '..', 'assets', 'meshdatav3_scaled', code, 'coacd', f'coacd_{suffix}.urdf'))


def build_cfg(key, set_name, out_dir, repo_dir):
    with open(osp.join(repo_dir, 'cfg', 'xarm6_leap_hand_ap2ap.yaml')) as f:
        cfg = yaml.safe_load(f)
    del cfg['env']['object_code_dict_file']
    if set_name == 'dtb':
        d = dtb_object_dir(key)
        urdf = osp.join(d, f'{key}_decomposed.urdf')
        mesh = osp.join(d, f'{key}.obj')
        assert osp.exists(urdf), urdf
        assert osp.exists(mesh), mesh
        cfg['env']['object_code_dict'] = {f'dtb/{key}': [1.0]}
        cfg['env']['benchmark_object_urdf'] = urdf
        cfg['env']['benchmark_keypoint_mesh'] = mesh
        cfg['env']['benchmark_zero_feat'] = True
    else:
        code, suffix = key.split('@')
        cfg['env']['object_code_dict'] = {code: [SCALE_BY_SUFFIX[suffix]]}
        cfg['env']['benchmark_object_urdf'] = generate_pieces_urdf(code, suffix, out_dir, repo_dir)
        # reference keypoint source, byte-identical extraction
        cfg['env']['benchmark_keypoint_mesh'] = osp.abspath(osp.join(
            repo_dir, '..', 'assets', 'meshdatav3_scaled', code, 'coacd', f'decomposed_{suffix}.obj'))
    bank_path = osp.join(BANK_DIR, f'{safe_key(key)}.json')
    assert osp.exists(bank_path), bank_path
    cfg['env']['benchmark_bank_json'] = bank_path
    cfg['env']['benchmark_output_dir'] = out_dir
    cfg_path = osp.join(out_dir, 'cfg_env.yaml')
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, bank_path


def verify_and_score(key, set_name, out_dir, bank_path):
    with open(osp.join(out_dir, 'native_summary.json')) as f:
        summary = json.load(f)
    with open(bank_path) as f:
        bank = json.load(f)
    d = np.load(osp.join(out_dir, 'poses.npz'))

    k = summary['k_goals']
    assert k == K_GOALS, k
    goals_reached = summary['goals_reached']
    ep_len = summary['ep_len']

    # start poses match the bank rows: x,y exact, z = bank + 0.22, quat wxyz->xyzw
    sp = np.array(bank['start_pos'], dtype=np.float32)
    sq = np.array(bank['start_quat_wxyz'], dtype=np.float32)
    sa = d['start_applied']
    assert np.allclose(sa[:, 0:2], sp[:, 0:2], atol=1e-6), 'start x,y mismatch vs bank'
    assert np.allclose(sa[:, 2] - sp[:, 2], Z_SHIFT, atol=1e-6), 'start z shift is not +0.22'
    assert np.allclose(sa[:, 3:7], sq[:, [1, 2, 3, 0]], atol=1e-6), 'start quat mismatch vs bank'

    # goals advance through the bank sequence in order
    seq_pos = np.array(bank['pos'], dtype=np.float32)
    seq_pos[:, :, 2] += Z_SHIFT
    for i in range(N_TRAJ):
        gi, si = d[f'goal_{i}'], d[f'successes_{i}']
        assert len(gi) == ep_len[i] == len(si) == len(d[f'joint_{i}']) == len(d[f'obj_{i}']), f'traj {i} length mismatch'
        assert int(si[-1]) == goals_reached[i], f'traj {i} successes tail vs summary'
        assert np.all(np.diff(si) >= 0), f'traj {i} successes not monotone'
        # goal active at step t was placed for index successes[t-1] (advance lands
        # on the next step's pre_physics), step 0 holds goal 0
        active = np.minimum(np.concatenate([[0], si[:-1]]), k - 1)
        assert np.allclose(gi[:, 0:3], seq_pos[i, active], atol=1e-5), f'traj {i} goal sequence out of order'

    tp = [gr / k * 100.0 for gr in goals_reached]
    result = {
        'key': key,
        'set': set_name,
        'condition': 'dex4d-native',
        'n_trajectories': N_TRAJ,
        'traj_indices': list(range(N_TRAJ)),
        'k_goals': k,
        'per_traj_goals_reached': goals_reached,
        'per_traj_task_progress_pct': tp,
        'mean_task_progress_pct': float(np.mean(tp)),
        # per_traj_steps: episode length under the benchmark rules, i.e. the step
        # of the 10th achieved goal, or the step the 600-step per-goal clock ran out
        'per_traj_steps': ep_len,
    }
    if set_name == 'dtb':
        result['pc_feat'] = 'zeroed, encoder weights unreleased'
    with open(osp.join(out_dir, 'benchmark_eval.json'), 'w') as f:
        json.dump(result, f, indent=1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True)
    p.add_argument('--set', required=True, choices=['udg-seen', 'udg-unseen', 'dtb'])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_dir', default='example_models/teacher_policy_stage_3/model_34000_best.pt')
    args = p.parse_args()

    repo_dir = osp.dirname(osp.abspath(__file__))
    out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path, bank_path = build_cfg(args.key, args.set, out_dir, repo_dir)

    cmd = [sys.executable, '-u', 'train.py', '--task=XArm6LeapHandAP2AP', '--algo=ppo', '--seed=0',
           '--rl_device=cuda:0', '--sim_device=cuda:0', '--headless', '--test',
           f'--model_dir={args.model_dir}', f'--num_envs={N_TRAJ}', f'--cfg_env={cfg_path}']
    log_path = osp.join(out_dir, 'train.log')
    with open(log_path, 'w') as lf:
        subprocess.run(cmd, cwd=repo_dir, stdout=lf, stderr=subprocess.STDOUT, check=True)

    result = verify_and_score(args.key, args.set, out_dir, bank_path)
    print(f"[020] {args.key} ({args.set}): goals {result['per_traj_goals_reached']}, "
          f"mean TP {result['mean_task_progress_pct']:.1f}%, steps {result['per_traj_steps']}")


if __name__ == '__main__':
    main()
