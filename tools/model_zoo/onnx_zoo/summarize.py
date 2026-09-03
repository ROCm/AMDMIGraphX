#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""Create a GitHub-readable summary from ONNX zoo accuracy and perf logs."""

import argparse
import html
import math
import re
from pathlib import Path
from urllib.parse import quote

DTYPES = ('fp32', 'fp16', 'int8')
PERF_THRESHOLD = 5.0


class RawMarkdown(str):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--gpu')
    parser.add_argument('--build-url')
    parser.add_argument('--baseline-root', type=Path)
    return parser.parse_args()


def read(path):
    return path.read_text(errors='replace') if path.exists() else ''


def variant(stem):
    if stem.endswith('_fp16'):
        return stem[:-5], 'fp16'
    if 'int8' in stem.lower() or 'qdq' in stem.lower():
        return stem, 'int8'
    return stem, 'fp32'


def log_stems(folder):
    return {
        path.stem
        for suffix in ('*.out', '*.err') for path in folder.glob(suffix)
    }


def accuracy_result(folder, stem):
    output_path = folder / (stem + '.out')
    error_path = folder / (stem + '.err')
    attempted = output_path.exists() or error_path.exists()
    output = read(output_path)
    summaries = list(
        re.finditer(
            r'^Test [^\n]+ has (?P<cases>\d+) cases:[ \t]*\n'
            r'^[ \t]*Passed:[ \t]*(?P<passed>\d+)[ \t]*\n'
            r'^[ \t]*Failed:[ \t]*(?P<failed>\d+)[ \t]*\n'
            r'^[ \t]*Worst deviation:[ \t]*(?P<max_diff>\S+)[ \t]*'
            r'\((?P<tol_frac>\S+)[ \t]+of tolerance\)[ \t]*$', output,
            re.MULTILINE))
    summary = summaries[-1] if summaries else None
    if summary and int(summary['passed']) + int(summary['failed']) != int(
            summary['cases']):
        summary = None
    if re.search(r'^SKIPPED:', output, re.MULTILINE):
        status = 'skipped'
    elif summary:
        status = 'pass' if int(summary['failed']) == 0 else 'fail'
    else:
        status = 'error' if attempted else 'missing'
    return {
        'status': status,
        'cases': summary['cases'] if summary else '',
        'passed': summary['passed'] if summary else '',
        'max_diff': summary['max_diff'] if summary else '',
        'tol_frac': summary['tol_frac'] if summary else '',
    }


def perf_result(folder, stem):
    output_path = folder / (stem + '.out')
    error_path = folder / (stem + '.err')
    attempted = output_path.exists() or error_path.exists()
    output = read(output_path)
    error = read(error_path)
    report = output + '\n' + error
    rate = re.search(r'^Rate:\s*(\S+)', report, re.MULTILINE)
    total_time = re.search(r'^Total time:', report, re.MULTILINE)
    batch = re.search(r'^Batch size:\s*(\d+)', report, re.MULTILINE)
    if re.search(r'^SKIPPED:', output, re.MULTILINE):
        status = 'skipped'
    elif rate and total_time:
        status = 'complete'
    else:
        status = 'error' if attempted else 'missing'
    return {
        'status': status,
        'rate': rate.group(1) if rate else '',
        'batch': batch.group(1) if batch else '',
    }


def markdown_link(status, folder, stem, label=None, prefix=None):
    suffix = '.err' if status in ('error', 'missing') else '.out'
    path = folder / (stem + suffix)
    if not path.exists():
        path = folder / (stem + ('.out' if suffix == '.err' else '.err'))
    if not path.exists():
        return label or status
    return RawMarkdown('[{}]({}/{})'.format(label or status, prefix
                                            or folder.name, quote(path.name)))


def markdown_escape(value):
    if isinstance(value, RawMarkdown):
        return value
    value = ''.join(char if char >= ' ' else ' ' for char in str(value))
    value = html.escape(value, quote=False)
    for char in ('\\', '|', '`', '*', '_', '~', '[', ']'):
        value = value.replace(char, '\\' + char)
    return value


def status_icon(status):
    return {
        'pass': ':white_check_mark:',
        'fail': ':red_circle:',
        'error': ':x:',
        'missing': ':x:',
        'skipped': ':heavy_minus_sign:',
    }[status]


def accuracy_cell(result, folder, stem):
    label = {
        'pass': 'PASSED',
        'fail': 'FAILED',
        'error': 'ERROR',
        'missing': 'MISSING',
        'skipped': 'SKIPPED',
    }[result['status']]
    link = markdown_link(result['status'], folder, stem, label)
    details = []
    if result['cases']:
        details.append('{}/{} cases'.format(result['passed'], result['cases']))
    if result['max_diff']:
        details.append('max diff {}'.format(result['max_diff']))
    if result['tol_frac']:
        details.append('{}x tolerance'.format(result['tol_frac']))
    return RawMarkdown('{}{}'.format(
        link, ': ' + ', '.join(details) if details else ''))


def find_baseline(root, run_id):
    if root is None or not root.is_dir():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and path.name != run_id and (path / 'perf').is_dir()
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def formatted_rate(result):
    try:
        rate = float(result['rate'])
    except ValueError:
        return result['rate']
    return '{:,.2f}'.format(rate)


def compare_perf(current, baseline):
    if current['status'] != 'complete' or baseline['status'] != 'complete':
        return 'error', '', ':x:'
    if current['batch'] and baseline[
            'batch'] and current['batch'] != baseline['batch']:
        return 'error', '', ':x:'
    try:
        current_rate = float(current['rate'])
        baseline_rate = float(baseline['rate'])
    except ValueError:
        return 'error', '', ':x:'
    if not math.isfinite(current_rate) or not math.isfinite(
            baseline_rate) or baseline_rate <= 0:
        return 'error', '', ':x:'
    diff = (current_rate - baseline_rate) * 100 / baseline_rate
    if diff <= -PERF_THRESHOLD:
        return 'regress', '{:.2f}%'.format(diff), ':red_circle:'
    if diff >= PERF_THRESHOLD:
        return 'pass', '{:.2f}%'.format(diff), ':high_brightness:'
    return 'pass', '{:.2f}%'.format(diff), ':white_check_mark:'


def generate(args):
    accuracy_dir = args.logs / 'accuracy'
    perf_dir = args.logs / 'perf'
    baseline_dir = find_baseline(args.baseline_root, args.run_id)
    baseline_perf_dir = baseline_dir / 'perf' if baseline_dir else None

    def sort_key(stem):
        model, dtype = variant(stem)
        return model.lower(), DTYPES.index(dtype)

    stems = sorted(log_stems(accuracy_dir) | log_stems(perf_dir), key=sort_key)
    if not stems:
        raise RuntimeError(
            'No accuracy or performance logs found in {}'.format(args.logs))

    rows = []
    for stem in stems:
        model, dtype = variant(stem)
        current_perf = perf_result(perf_dir, stem)
        old_perf = perf_result(baseline_perf_dir,
                               stem) if baseline_perf_dir else {
                                   'status': 'missing',
                                   'rate': '',
                                   'batch': ''
                               }
        comparison, diff, perf_icon = compare_perf(current_perf, old_perf)
        rows.append({
            'model': model,
            'dtype': dtype,
            'stem': stem,
            'accuracy': accuracy_result(accuracy_dir, stem),
            'perf': current_perf,
            'old_perf': old_perf,
            'comparison': comparison,
            'diff': diff,
            'perf_icon': perf_icon,
        })

    lines = ['# ONNX Model Zoo results', '']
    metadata = []
    if args.run_id:
        metadata.append('Run **{}**'.format(markdown_escape(args.run_id)))
    if args.gpu:
        metadata.append('GPU **{}**'.format(markdown_escape(args.gpu)))
    if baseline_dir:
        metadata.append('baseline **{}**'.format(
            markdown_escape(baseline_dir.name)))
    if args.build_url:
        metadata.append('[Jenkins build]({})'.format(args.build_url))
    if metadata:
        lines += [' · '.join(metadata), '']

    accuracy_pass = sum(row['accuracy']['status'] == 'pass' for row in rows)
    accuracy_fail = sum(row['accuracy']['status'] == 'fail' for row in rows)
    accuracy_error = len(rows) - accuracy_pass - accuracy_fail
    perf_pass = sum(row['comparison'] == 'pass' for row in rows)
    perf_regress = sum(row['comparison'] == 'regress' for row in rows)
    perf_error = len(rows) - perf_pass - perf_regress

    lines += [
        '## Summary',
        '',
        '| Check | Pass | Fail | Regress | Error |',
        '|:------|-----:|-----:|--------:|------:|',
        '| Accuracy | {} | {} | — | {} |'.format(accuracy_pass, accuracy_fail,
                                                 accuracy_error),
        '| Performance | {} | — | {} | {} |'.format(perf_pass, perf_regress,
                                                    perf_error),
        '',
        'Performance regressions are rate drops of at least {:.0f}%. '
        'Rate gains of at least {:.0f}% are highlighted :high_brightness:.'.
        format(PERF_THRESHOLD, PERF_THRESHOLD),
        '',
    ]

    lines += [
        '## Accuracy',
        '',
        '| Test | Status | Result |',
        '|:-----|:------:|:-------|',
    ]
    for row in rows:
        accuracy = row['accuracy']
        test = '{} ({})'.format(row['model'], row['dtype'])
        lines.append('| {} | {} | {} |'.format(
            markdown_escape(test), status_icon(accuracy['status']),
            accuracy_cell(accuracy, accuracy_dir, row['stem'])))

    lines += [
        '',
        '## Performance',
        '',
        '| Test | Batch | New Rate ({}) | Old Rate ({}) | Diff | Status |'.
        format(
            args.run_id.rsplit('-', 1)[-1] if args.run_id else 'new',
            baseline_dir.name.rsplit('-', 1)[-1] if baseline_dir else 'none'),
        '|:-----|------:|-------------:|-------------:|-----:|:------:|',
    ]
    baseline_prefix = '../{}/perf'.format(quote(
        baseline_dir.name)) if baseline_dir else None
    for row in rows:
        perf = row['perf']
        old_perf = row['old_perf']
        test = '{} ({})'.format(row['model'], row['dtype'])
        label = formatted_rate(
            perf) if perf['status'] == 'complete' else perf['status'].upper()
        old_label = formatted_rate(
            old_perf
        ) if old_perf['status'] == 'complete' else old_perf['status'].upper()
        rate = markdown_link(perf['status'], perf_dir, row['stem'], label)
        old_rate = markdown_link(
            old_perf['status'], baseline_perf_dir, row['stem'], old_label,
            baseline_prefix) if baseline_perf_dir else 'N/A'
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            markdown_escape(test),
            markdown_escape(perf['batch'] or old_perf['batch']), rate,
            old_rate, markdown_escape(row['diff']), row['perf_icon']))

    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    output = args.logs / 'README.md'
    output.write_text(generate(args))
    print('Wrote {}'.format(output))


if __name__ == '__main__':
    main()
