"""Execute every research notebook cell without requiring a Jupyter installation."""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import base64
import io
import json
import os
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT / 'FX_Technical_Research_Complete.ipynb')
    arguments = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'reproduced' / '.matplotlib'))
    output_path = ROOT / 'reproduced' / (arguments.notebook.stem + '.executed.ipynb')
    output_path.parent.mkdir(exist_ok=True)
    validation_path = ROOT / 'reproduced' / 'notebook_validation_summary.json'
    started_utc = datetime.now(timezone.utc).isoformat()
    validation_path.write_text(json.dumps({'execution_status': 'running', 'started_utc': started_utc,
                                          'validation_complete': False}, indent=2), encoding='utf-8')
    import matplotlib.pyplot as plt
    notebook = json.loads(arguments.notebook.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    active_outputs = []

    def display(value):
        data = {'text/plain': [str(value)]}
        if hasattr(value, 'to_html'):
            data['text/html'] = [value.to_html()]
        active_outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': data})

    def show(*arguments, **keywords):
        for figure_number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(figure_number).savefig(buffer, format='png', bbox_inches='tight')
            active_outputs.append({'output_type': 'display_data', 'metadata': {},
                                   'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}})
        plt.close('all')

    namespace['display'] = display
    plt.show = show
    execution_count = 0
    failed = False
    for cell_index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    for cell_index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        execution_count += 1
        active_outputs = []
        stream = io.StringIO()
        started = time.perf_counter()
        print(f'Cell {cell_index}: running', flush=True)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                exec(compile(''.join(cell['source']), f'{arguments.notebook.name}:cell_{cell_index}', 'exec'), namespace)
        except Exception as error:
            failed = True
            active_outputs.append({'output_type': 'error', 'ename': type(error).__name__,
                                   'evalue': str(error), 'traceback': traceback.format_exc().splitlines()})
            traceback.print_exc()
        cell['execution_count'] = execution_count
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)}]
                           if stream.getvalue() else []) + active_outputs
        print(f'Cell {cell_index}: {"FAILED" if failed else "passed"} ({time.perf_counter() - started:.1f}s)', flush=True)
        if failed:
            break
    notebook['metadata']['reconstruction_execution'] = {
        'success': not failed, 'executed_code_cells': execution_count,
        'method': 'Sequential Python execution, shared namespace, Agg figures; no original frozen engine claim.'}
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    summary = json.loads(validation_path.read_text(encoding='utf-8')) if not failed else {}
    summary.update({'execution_status': 'failed' if failed else 'passed', 'started_utc': started_utc,
                    'finished_utc': datetime.now(timezone.utc).isoformat(),
                    'executed_code_cells': execution_count,
                    'validation_complete': not failed and 'filter_reconciliation_checks_pass' in summary})
    if failed:
        summary['failed_cell'] = cell_index
        summary['error'] = active_outputs[-1]['evalue']
    validation_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Executed notebook: {output_path}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
