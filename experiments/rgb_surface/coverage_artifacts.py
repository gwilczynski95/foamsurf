"""Gaussian-only coverage artifacts, independent of the training background."""
import csv
import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluation.metrics import psnr


def black_gaussians(image, alpha, background):
    return image - (1-alpha)*image.new_tensor(background)


def coverage_metrics(alpha):
    return {'alpha_mean': float(alpha.mean()),
            **{f'coverage_gt_{t:g}': float((alpha > t).float().mean()) for t in (.1, .5, .9)}}


def labeled_panel(tiles, labels):
    w, h = tiles[0].size
    panel = Image.new('RGB', (w*len(tiles), h+28), 'white')
    draw = ImageDraw.Draw(panel)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        panel.paste(tile, (i*w, 28))
        draw.text((i*w+8, 7), label, fill='black')
    return panel


def install_coverage(evaluator):
    """Observe existing G-only renders; no extra full-resolution rendering."""
    original_evaluate = evaluator.evaluate

    def evaluate(iteration):
        captured = []
        rows = []
        original_gaussians = evaluator.renderer.render_gaussians

        def capture(*args, **kwargs):
            result = original_gaussians(*args, **kwargs)
            index = len(rows)
            camera = evaluator.dataset.test[index]
            black = black_gaussians(result.image, result.alpha, evaluator.config.background).clamp(0, 1)
            row = dict(iteration=iteration, view_name=camera.name, **coverage_metrics(result.alpha))
            row['gaussian_black_psnr'] = float(psnr(black, camera.ground_truth(tuple(evaluator.config.background))))
            rows.append(row)
            if index < evaluator.config.validation.image_views:
                captured.append((index, result.alpha.detach().cpu().numpy(),
                                 (black.detach().cpu().numpy()*255).round().astype(np.uint8)))
            return result

        evaluator.renderer.render_gaussians = capture
        try:
            summary, images, histograms = original_evaluate(iteration)
        finally:
            evaluator.renderer.render_gaussians = original_gaussians
        directory = evaluator.output_dir/'evaluation'/f'iteration_{iteration:06d}'
        with (directory/'coverage.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for key in rows[0]:
            if key not in ('iteration', 'view_name'):
                summary[f'eval/gaussian_coverage/{key}'] = float(np.mean([r[key] for r in rows]))
        for index, alpha, black in captured:
            camera = evaluator.dataset.test[index]
            prefix = f'{index:04d}_{camera.name}'
            np.savez_compressed(directory/f'{prefix}_gaussian_alpha.npz', alpha=alpha)
            a = alpha[..., 0].clip(0, 1)
            Image.fromarray((255*a).round().astype(np.uint8)).save(directory/f'{prefix}_gaussian_alpha.png')
            Image.fromarray(black).save(directory/f'{prefix}_gaussian_only_black.png')
            target = camera.ground_truth(tuple(evaluator.config.background)).detach().cpu().numpy()
            gt = (target.clip(0, 1)*255).round().astype(np.uint8)
            # Red pixels mean A_G <= .1; includes open water, not a semantic floor mask.
            holes = gt.copy()
            mask = a <= .1
            holes[mask] = (gt[mask]*.3+np.array([255, 0, 0])*.7).round().astype(np.uint8)
            tiles = [Image.fromarray(gt), Image.open(directory/f'{prefix}_hybrid.png').convert('RGB'),
                     Image.fromarray(black), Image.fromarray((a*255).round().astype(np.uint8)).convert('RGB'),
                     Image.fromarray(holes), Image.open(directory/f'{prefix}_voronoi_only.png').convert('RGB')]
            panel = labeled_panel(tiles, ['GT', 'HYBRID', 'GAUSSIAN ONLY (BLACK)', 'ALPHA GAUSSIANS',
                                          'RED: A_G <= 0.1 (ALL PIXELS)', 'VORONOI ONLY (TRAIN BG)'])
            panel.save(directory/f'{prefix}_coverage_panel.png')
            images[f'coverage/view_{index:02d}'] = np.asarray(panel)
        return summary, images, histograms

    evaluator.evaluate = evaluate
