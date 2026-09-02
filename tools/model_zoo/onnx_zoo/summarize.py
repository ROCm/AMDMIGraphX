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
import re
from pathlib import Path
from urllib.parse import quote

DTYPES = ('fp32', 'fp16', 'int8')


class RawMarkdown(str):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--gpu')
    parser.add_argument('--build-url')
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
    median = re.search(r'Median:\s*([0-9.eE+-]+)ms', report)
    compile_time = re.search(r'Compilation time:\s*([0-9.eE+-]+)ms', report)
    if re.search(r'^SKIPPED:', output, re.MULTILINE):
        status = 'skipped'
    elif rate and median:
        status = 'complete'
    else:
        status = 'error' if attempted else 'missing'
    return {
        'status': status,
        'rate': rate.group(1) if rate else '',
        'median': median.group(1) if median else '',
        'compile': compile_time.group(1) if compile_time else '',
    }


def markdown_link(status, folder, stem, label=None):
    suffix = '.err' if status in ('error', 'missing') else '.out'
    path = folder / (stem + suffix)
    if not path.exists():
        path = folder / (stem + ('.out' if suffix == '.err' else '.err'))
    if not path.exists():
        return label or status
    return RawMarkdown('[{}]({}/{})'.format(label or status, folder.name,
                                            quote(path.name)))


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
        'complete': ':white_check_mark:',
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


def summary_line(label, successful, total):
    failed = total - successful
    if failed == 0:
        return 'All {} results passed :white_check_mark:'.format(label)
    noun = 'result' if failed == 1 else 'results'
    return '{} {} {} need attention :red_circle:'.format(failed, label, noun)


def generate(args):
    accuracy_dir = args.logs / 'accuracy'
    perf_dir = args.logs / 'perf'

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
        rows.append({
            'model': model,
            'dtype': dtype,
            'stem': stem,
            'accuracy': accuracy_result(accuracy_dir, stem),
            'perf': perf_result(perf_dir, stem),
        })

    lines = ['# ONNX Model Zoo results', '']
    metadata = []
    if args.run_id:
        metadata.append('Run **{}**'.format(markdown_escape(args.run_id)))
    if args.gpu:
        metadata.append('GPU **{}**'.format(markdown_escape(args.gpu)))
    if args.build_url:
        metadata.append('[Jenkins build]({})'.format(args.build_url))
    if metadata:
        lines += [' · '.join(metadata), '']

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

    accuracy_passed = sum(row['accuracy']['status'] == 'pass' for row in rows)
    lines += [
        '',
        summary_line('accuracy', accuracy_passed, len(rows)),
        '',
        '## Performance',
        '',
        '| Test | Rate (inf/s) | Median (ms) | Compile (ms) | Status |',
        '|:-----|-------------:|------------:|-------------:|:------:|',
    ]
    for row in rows:
        perf = row['perf']
        test = '{} ({})'.format(row['model'], row['dtype'])
        label = perf['rate'] if perf['status'] == 'complete' else perf[
            'status'].upper()
        rate = markdown_link(perf['status'], perf_dir, row['stem'], label)
        lines.append('| {} | {} | {} | {} | {} |'.format(
            markdown_escape(test), rate, markdown_escape(perf['median']),
            markdown_escape(perf['compile']), status_icon(perf['status'])))

    perf_completed = sum(row['perf']['status'] == 'complete' for row in rows)
    lines += ['', summary_line('performance', perf_completed, len(rows))]

    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    output = args.logs / 'README.md'
    output.write_text(generate(args))
    print('Wrote {}'.format(output))


if __name__ == '__main__':
    main()
