#!/usr/bin/env python3

# runs on PC not on cluster because it needs to read the PsychoPy log files
#!/usr/bin/env python3
"""
Score one-back accuracy for the Experiment 2 tool localizer (PsychoPy).

Ground truth comes from the .log file, NOT the .csv:
  - The CSV tool/non_tool/scramble columns are the base randomized condition
    lists. The runtime one-back repeats are inserted by the script and appear
    ONLY in the log's "stim: image = '...'" lines.
  - A target = a stimulus whose image is identical to the immediately preceding
    image WITHIN the same block.
  - Fixation blocks loop a single placeholder image and are excluded.
  - Log stim events and CSV stim rows are 1:1 in presentation order, so
    button-press responses (key_resp_2.keys) are joined positionally.

Usage:
  python score_oneback.py SUB.log SUB.csv [more pairs...]      # explicit pairs
  python score_oneback.py --dir /path/to/data                  # auto-pair by stem
Outputs a per-subject per-condition table and writes oneback_accuracy.csv.
"""
import re, sys, os, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm

CONDITIONS = ['tool', 'non_tool', 'scramble']

def parse_log(log_path):
    """Return DataFrame: one row per displayed stimulus, in order."""
    blk_re = re.compile(
        r"New trial \(rep=\d+, index=\d+\): OrderedDict\(\[\('block_type', '(\w+)'\)\]\)")
    img_re = re.compile(r"stim: image = '([^']+)'")
    rows, btype, blk, pos, prev = [], None, -1, 0, None
    with open(log_path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            m = re.match(r'([0-9.]+)\s+\w+\s+(.*)', ln)
            if not m:
                continue
            t, msg = float(m.group(1)), m.group(2)
            b = blk_re.search(msg)
            if b:
                btype, blk, pos, prev = b.group(1), blk + 1, 0, None
                continue
            im = img_re.search(msg)
            if im:
                img = im.group(1)
                rows.append(dict(block=blk, btype=btype, pos=pos, img=img,
                                 t_log=t, target=(pos > 0 and img == prev)))
                prev, pos = img, pos + 1
    return pd.DataFrame(rows)

def score_subject(log_path, csv_path, lag=0):
    """lag: a press counts as a hit if it lands on the target trial or up to
    `lag` trials later, within the same block. Consumed presses cannot also
    count as false alarms. lag=0 reproduces strict same-trial scoring."""
    log = parse_log(log_path)
    csv = pd.read_csv(csv_path)
    csv = csv[csv['stim.started'].notna()].reset_index(drop=True)
    if len(csv) != len(log):
        raise ValueError(f"row mismatch: log {len(log)} vs csv {len(csv)} "
                         f"({os.path.basename(log_path)})")
    if not (log['btype'].values == csv['block_type'].values).all():
        raise ValueError(f"block order mismatch ({os.path.basename(log_path)})")
    log['press'] = csv['key_resp_2.keys'].notna().values

    sub = os.path.basename(log_path).split('_')[0]
    run = int(csv['run'].dropna().iloc[0]) if 'run' in csv.columns and csv['run'].notna().any() else None

    task = log[log['btype'] != 'fixation'].sort_values(['block', 'pos']).reset_index(drop=True)
    n = len(task)
    press = task['press'].values
    target = task['target'].values
    block = task['block'].values
    consumed = np.zeros(n, bool)        # presses claimed by a hit
    hit_target = np.zeros(n, bool)      # targets detected within their window
    for i in range(n):
        if not target[i]:
            continue
        for j in range(i, min(i + lag + 1, n)):
            if block[j] != block[i]:
                break
            if press[j] and not consumed[j]:
                consumed[j] = True
                hit_target[i] = True
                break
    false_alarm = press & ~consumed     # unclaimed presses

    out = []
    for c in CONDITIONS:
        m = (task['btype'] == c).values
        nt = int((target & m).sum())
        nn = int((~target & m).sum())
        H = int((hit_target & m).sum())
        M = nt - H
        FA = int((false_alarm & m).sum())
        CR = nn - FA
        ntr = int(m.sum())
        hr = H / nt if nt else np.nan
        far = FA / nn if nn else np.nan
        acc = (H + CR) / ntr if ntr else np.nan
        hrc, farc = (H + 0.5) / (nt + 1), (FA + 0.5) / (nn + 1)
        dpr = norm.ppf(hrc) - norm.ppf(farc)
        out.append(dict(subject=sub, run=run, cond=c, n_trials=ntr, targets=nt,
                        hits=H, misses=M, FA=FA, CR=CR,
                        hit_rate=round(hr, 3), FA_rate=round(far, 4),
                        accuracy=round(acc, 4), dprime=round(dpr, 2)))
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='*', help='alternating LOG CSV paths')
    ap.add_argument('--dir', help='directory; auto-pair .log/.csv by filename stem')
    ap.add_argument('--lag', type=int, default=0,
                    help='hit tolerance in trials (press on target row or up to '
                         'LAG rows later, within block). Default 0 = strict.')
    ap.add_argument('-o', '--out', default='oneback_accuracy.csv')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='subject IDs to drop entirely, e.g. '
                         '--exclude spaceloc1005 spaceloc1006')
    args = ap.parse_args()

    jobs = []
    if args.dir:
        for lg in sorted(glob.glob(os.path.join(args.dir, '*.log'))):
            cv = lg[:-4] + '.csv'
            if os.path.exists(cv):
                jobs.append((lg, cv))
    elif args.pairs:
        p = args.pairs
        # accept either "L C L C" or all logs then matching csvs
        if all(x.endswith(('.log', '.csv')) for x in p) and len(p) % 2 == 0:
            jobs = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
    if not jobs:
        ap.error('provide LOG CSV pairs or --dir')

    excluded = set(args.exclude)
    if excluded:
        kept = [(lg, cv) for lg, cv in jobs
                if os.path.basename(lg).split('_')[0] not in excluded]
        for lg, cv in jobs:
            if os.path.basename(lg).split('_')[0] in excluded:
                print(f"EXCLUDE {os.path.basename(lg)}", file=sys.stderr)
        jobs = kept

    results = []
    for lg, cv in jobs:
        try:
            results.append(score_subject(lg, cv, lag=args.lag))
        except Exception as e:
            print(f"SKIP {os.path.basename(lg)}: {e}", file=sys.stderr)
    if not results:
        sys.exit('no subjects scored')

    metrics = ['hit_rate', 'FA_rate', 'accuracy', 'dprime']
    allr = pd.concat(results, ignore_index=True)            # one row per run
    # per-subject: average across that subject's runs (each subject weighted once)
    per_sub = (allr.groupby(['subject', 'cond'], as_index=False)[metrics]
               .mean().round(3))
    # group: average across subjects
    group = per_sub.groupby('cond')[metrics].mean().round(3)

    allr.to_csv(args.out, index=False)
    per_sub.to_csv(args.out.replace('.csv', '_per_subject.csv'), index=False)

    pd.set_option('display.width', 160)
    print(f"(lag = {args.lag} trial{'s' if args.lag != 1 else ''})\n")
    print('=== per-run detail ===')
    print(allr.to_string(index=False))
    print('\n=== per-subject (runs averaged) ===')
    print(per_sub.to_string(index=False))
    print('\n=== group means by condition (across subjects) ===')
    print(group.to_string())

if __name__ == '__main__':
    main()