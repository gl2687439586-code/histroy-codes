"""Validate this code-only archive without imports, inference, services or network."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    catalog = json.loads((root/'catalog/files.json').read_text(encoding='utf-8'))['files']
    errors = []
    indexed = {r['path'] for r in catalog}
    for row in catalog:
        p = root/row['path']
        if not p.is_file():
            errors.append('Missing file: '+row['path']); continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['published_sha256']:
            errors.append('Hash mismatch: '+row['path'])
        if not row.get('role') or not (root/row['doc_path']).is_file():
            errors.append('Missing file-level explanation: '+row['path'])
        if p.suffix == '.py':
            try: ast.parse(p.read_text(encoding='utf-8'))
            except SyntaxError as exc: errors.append('Syntax error: '+row['path']+': '+str(exc))
    for folder in ['experiments','analysis','third_party','tools']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.sh'} and p.relative_to(root).as_posix() not in indexed:
                errors.append('Unindexed code: '+p.relative_to(root).as_posix())
    # Original third-party notices may refer to external dependencies. Check generated navigation.
    docs = [root/'README.md']+list((root/'docs').rglob('*.md'))
    docs += [p for name in ['experiments','analysis','third_party'] for p in (root/name).rglob('README.md')]
    for p in docs:
        for target in re.findall(r'\]\(<?([^>\n]+?)>?\)',p.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:')): continue
            if not (p.parent/target).exists():
                errors.append('Broken link: '+p.relative_to(root).as_posix()+' -> '+target)
    forbidden_ext = {'.jsonl','.gz','.zip','.npz','.parquet','.safetensors','.bin','.pt','.pth','.faiss','.index'}
    for p in root.rglob('*'):
        if '.git' in p.parts or not p.is_file(): continue
        if p.suffix in forbidden_ext or p.name.startswith('.env'):
            errors.append('Runtime asset or secret container must not be tracked: '+p.relative_to(root).as_posix())
    result = dict(passed=not errors,indexed_files=len(catalog),
                  python_files=sum(r['path'].endswith('.py') for r in catalog),
                  shell_files=sum(r['path'].endswith('.sh') for r in catalog),
                  errors=errors,scope='Static archive checks only; no inference or runtime claim.')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not errors else 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
