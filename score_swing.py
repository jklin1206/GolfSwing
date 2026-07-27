"""
Golf Swing Score - booth demo runner.

Ties the new SwingScorer into the existing pose-estimation pipeline:

    video --(MediaPipe_class)--> CSV --(DataProcessor)--> 3 key frames
          --(SwingScorer)--> single golf-style score (lower is better)

Usage:
    python score_swing.py <folder_with_video_and_csv>

If no folder is given it defaults to the bundled 'test_classes' sample and also
prints a small demo leaderboard so the ranking behaviour is visible end to end.
"""
import sys
import copy
from process_swing import DataProcessor, SwingScorer


def score_folder(folder_path):
    """Full pipeline for one participant folder -> (score_dict, data_processor)."""
    dp = DataProcessor(folder_path)
    dp.load_data()
    dp.preprocess_data()                 # -> 3 rows: address / top / contact
    result = SwingScorer(dp).score()
    return result, dp


def _perturb(dp, **col_deltas):
    """Clone a processed swing and nudge columns per phase to fake an amateur fault.

    col_deltas maps 'column@phase_index' -> delta, e.g. arm_angle@0=-35 bends the
    lead arm 35 degrees at address. Used only to populate the demo leaderboard.
    """
    clone = copy.deepcopy(dp)
    for spec, delta in col_deltas.items():
        col, idx = spec.split('@')
        clone.data.loc[int(idx), col] = clone.data[col].iloc[int(idx)] + delta
    return clone


def demo_leaderboard(base_dp):
    """Score the real swing plus a few synthetic amateurs and rank them (low wins)."""
    entries = [('Pro (real sample)', SwingScorer(base_dp).score()['total'])]

    # Synthetic amateurs: same swing, specific faults injected. Clearly not real people.
    faults = {
        'Amateur A (bent arm)':       dict(arm_angle_at={'0': -40, '2': -35}),
        'Amateur B (early hip + sway)': dict(pelvis_at={'1': -45}, nose_sway_top=60),
        'Amateur C (chicken wing)':   dict(arm_angle_at={'2': -55}, knee_at={'2': -30}),
    }

    for name, f in faults.items():
        deltas = {}
        for idx, d in f.get('arm_angle_at', {}).items():
            deltas[f'arm_angle@{idx}'] = d
        for idx, d in f.get('pelvis_at', {}).items():
            deltas[f'pelvis_angle@{idx}'] = d
        for idx, d in f.get('knee_at', {}).items():
            deltas[f'knee_angle@{idx}'] = d
        if 'nose_sway_top' in f:                 # move the head at the top of the backswing
            deltas['nose_x@1'] = f['nose_sway_top']
        clone = _perturb(base_dp, **deltas)
        entries.append((name, SwingScorer(clone).score()['total']))

    entries.sort(key=lambda e: e[1])             # lower score first
    return entries


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else 'test_classes'

    result, dp = score_folder(folder)
    print(SwingScorer(dp).scorecard_text(player_name=folder))

    if len(sys.argv) <= 1:                        # default sample -> also show a leaderboard
        print("\n==== DEMO LEADERBOARD (lower score wins, like golf) ====")
        board = demo_leaderboard(dp)
        for rank, (name, score) in enumerate(board, 1):
            print(f"  {rank}. {name:<32} {score:6.1f}")
        print("\n  (Pro swing is the one real video; amateurs A-C are synthetic "
              "perturbations\n   of it, included only to show the score ranks swings low-is-better.)")
