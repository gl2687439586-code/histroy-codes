"""Inspect user-supplied archive as data only; never execute its contents."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

BASE = Path(__file__).resolve().parent
ARCHIVE = Path('C:/Users/RESEARCH_USER/Desktop/过程结果/hotpot_matched500_RAW_20261004_184407_2500821.tar.gz')
INPUT = BASE / 'input'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def short(value, depth=0):
    if isinstance(value, str):
        return value if len(value) < 250 else value[:150] + ' ... [chars=%s]' % len(value)
    if isinstance(value, dict):
        return {k: short(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return {'length': len(value), 'first': short(value[0], depth + 1)} if value else []
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    archive = parser.parse_args().archive
    expected = Path(str(archive) + '.sha256').read_text().split()[0]
    actual = sha(archive)
    assert actual == expected, 'Archive SHA256 mismatch'
    print('ARCHIVE_SHA256_VERIFIED', actual)
    INPUT.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len({x.name for x in members}) == len(members), 'Duplicate archive path'
        for member in members:
            assert member.isfile(), member.name
            target = (INPUT / member.name).resolve()
            assert INPUT.resolve() in target.parents, member.name
            target.parent.mkdir(exist_ok=True, parents=True)
            if not target.exists():
                with tar.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    manifest = json.loads((INPUT / 'RAW_EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(INPUT / item['archive_name']) == item['sha256'], item['archive_name']
    print('ALL_MEMBER_CHECKSUMS_VERIFIED', len(manifest['files']))
    report = json.loads((INPUT / 'RAW_PREFLIGHT_REPORT.json').read_text(encoding='utf-8'))
    for name in ['old500', 'new500', 'matched500']:
        print(name, json.dumps(report[name], ensure_ascii=False, indent=2))
    for label in ['old500', 'new500']:
        for rel in ['roots/roots.merged.jsonl.gz', 'trajectories/trajectories.merged.jsonl.gz', 'analysis/scored_records.jsonl.gz']:
            with gzip.open(INPUT / label / rel, 'rt', encoding='utf-8') as f:
                row = json.loads(next(f))
            print('SCHEMA', label, rel, json.dumps(short(row), ensure_ascii=False, indent=2))
    print('CONFIG', (INPUT / 'source_metadata/new500/FULL7405_CONFIG.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
