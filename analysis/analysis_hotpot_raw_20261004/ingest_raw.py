"""Validate and unpack only regular data members of the uploaded raw archive."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ARCHIVE = Path(r'C:\Users\RESEARCH_USER\Desktop\过程结果\hotpot_500_and7405_RAW_20261004_163430_2267929.tar.gz')
DEST = Path(__file__).resolve().parent / 'input'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    expected = Path(str(ARCHIVE) + '.sha256').read_text(encoding='utf-8-sig').split()[0]
    actual = sha(ARCHIVE)
    assert actual == expected, (actual, expected)
    DEST.mkdir(parents=True, exist_ok=True)
    inventory = []
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts
            assert p.parts[0] in {'old500', 'full7405', 'RAW_EXPORT_MANIFEST.json', 'RAW_PREFLIGHT_REPORT.json', 'README.txt'}
            target = DEST.joinpath(*p.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            inventory.append({'name': member.name, 'bytes': member.size})
    manifest = json.loads((DEST / 'RAW_EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = DEST / item['archive_name']
        assert path.stat().st_size == item['bytes']
        assert sha(path) == item['sha256'], item['archive_name']
    result = {'archive_sha256': actual, 'archive_bytes': ARCHIVE.stat().st_size,
              'member_hashes_verified': len(manifest['files']), 'members': inventory}
    (DEST.parent / 'INGEST_VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print((DEST / 'RAW_PREFLIGHT_REPORT.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
