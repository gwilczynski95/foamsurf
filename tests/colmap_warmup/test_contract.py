from pathlib import Path
import numpy as np
import pytest
import torch
import pycolmap
from foamsurf_data.colmap import ColmapDataset
from training.warmup_pcd import sample_pcd
from training.warmup_sampling import MixedRaySampler
from training.sampler import RandomCameraRaySampler
from training.end_to_end import EndToEndConfig

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted')


@pytest.mark.parametrize('resolution', [4, 16, 32])
def test_real_intrinsics_match_pycolmap(resolution):
    d = ColmapDataset(DATA, resolution=resolution, device='cpu')
    r = pycolmap.Reconstruction(str(DATA/'sparse/0'))
    assert len(d.train)==25 and len(d.test)==4
    for camera in [d.train[0], d.test[-1]]:
        record = next(i for i in r.images.values() if Path(i.name).stem == camera.name)
        calibration = pycolmap.Camera(r.cameras[record.camera_id].todict())
        calibration.rescale(camera.width, camera.height)
        indices = np.array([0, camera.width//2, camera.width*camera.height-1])
        pixels = np.stack([indices % camera.width, indices//camera.width], -1)+.5
        xy = calibration.cam_from_img(pixels)
        dirs = np.concatenate([xy, np.ones((len(xy),1))], -1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        expected = dirs @ record.cam_from_world.inverse().matrix()[:,:3].T
        actual = camera.rays().directions.reshape(-1,3)[indices].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    sampler = MixedRaySampler(d, 16, (1,1,1))
    indices = torch.tensor([0, d.width*d.height-1, d.width*d.height+3])
    batch = sampler.batch_from_indices(indices)
    expected = torch.stack([d.train[int(i)//(d.width*d.height)].rays().directions.reshape(-1,3)[int(i)%(d.width*d.height)] for i in indices])
    torch.testing.assert_close(batch.rays.directions, expected)
    assert torch.all(batch.alpha==1)
    single = RandomCameraRaySampler(d,16,(1,1,1)).next()
    torch.testing.assert_close(single.rays.directions, single.camera.rays().directions.reshape(-1,3)[single.rays.pixel_indices])


def test_pcd_reference_distribution_and_rng():
    points=torch.arange(300,dtype=torch.float32).reshape(100,3)
    torch.manual_seed(42)
    sites,raw,mask=sample_pcd(points)
    torch.manual_seed(42)
    outer=torch.randn(5000,3)*10
    inner=points[torch.randint(100,(90,))]
    inner=inner+torch.randn_like(inner)*.01
    density=torch.cat([torch.rand(90,1),torch.full((5000,1),-.5)])
    torch.testing.assert_close(sites,torch.cat([inner,outer]))
    torch.testing.assert_close(raw,density)
    assert mask.sum()==5000 and not mask[:90].any()
    assert sites.max()>1.5  # no LEGO clamp
    with pytest.raises(ValueError): sample_pcd(torch.zeros(3,3))


def test_configs_keep_lego_unchanged():
    c=EndToEndConfig.load(ROOT/'configs/end_to_end_iui3.yaml')
    assert c.profile.site_initialization=='pcd'
    assert c.warmup.model.medium_bbox_min is None
    assert c.warmup.dataset.gaussian_ground_truth_path is None
    old=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml')
    assert old.profile.site_initialization=='random'
    assert old.warmup.model.medium_bbox_min==[-1.5]*3
