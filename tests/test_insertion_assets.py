"""Shot manifests use one relocatable asset root."""
import json
from pathlib import Path

import pytest

pytest.importorskip("warp", reason="insertion requires the optional --with-apps dependencies")

from examples.insertion.insert import resolve_assets

ROOT = Path(__file__).resolve().parents[1]


def test_paper_shots_only_reference_relative_assets():
    shots = ROOT / 'examples/insertion/shots'
    found = []
    for path in shots.rglob('*.json'):
        shot = json.loads(path.read_text())
        if not isinstance(shot, dict) or 'objects' not in shot:
            continue
        found.append(path)
        for obj in shot['objects']:
            if 'manifest' in obj:
                asset = Path(obj['manifest'])
                assert not asset.is_absolute() and '..' not in asset.parts and '$' not in str(asset)
                assert asset.parts[0] == 'models'
        if 'camera' in shot:
            assert (path.parent / shot['camera']).is_file()
    assert found


def test_asset_directory_can_be_relocated(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/mesh.json').write_text('{}')
    (tmp_path / 'texture.png').touch()
    shot = {'objects': [{'manifest': 'models/mesh.json', 'materials': {'part': {'maps': {'albedo': 'texture.png'}}}}]}
    resolved = resolve_assets(shot, tmp_path)
    assert resolved['objects'][0]['manifest'] == str(tmp_path / 'models/mesh.json')
    assert resolved['objects'][0]['materials']['part']['maps']['albedo'] == str(tmp_path / 'texture.png')
    with pytest.raises(FileNotFoundError, match='asset missing'):
        resolve_assets(shot, tmp_path / 'missing')
    for invalid in ['../outside.json', '/absolute.json', '$FALCON/manifest.json']:
        with pytest.raises(ValueError, match='relative to --assets'):
            resolve_assets({'objects': [{'manifest': invalid}]}, tmp_path)
