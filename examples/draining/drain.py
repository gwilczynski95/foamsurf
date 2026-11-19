"""Inference-only draining of the learned Voronoi medium; checkpoints are never changed.

Clips integration segments to water, keeping both extinction and backscatter.
This is an appearance edit, not a fluid or refractive free-surface simulation.
"""
import argparse
from pathlib import Path
import json
import subprocess
import shutil
import numpy as np
import torch
from PIL import Image
import render


def water_intervals(rays, up, level, near, far):
    height = rays[:, :3] @ up
    slope = torch.nn.functional.normalize(rays[:, 3:], dim=-1) @ up
    crossing = (level - height) / torch.where(slope.abs() > 1e-8, slope, torch.ones_like(slope))
    start = torch.where(slope < 0, crossing.clamp(near, far), torch.full_like(crossing, near))
    end = torch.where(slope > 0, crossing.clamp(near, far), torch.full_like(crossing, far))
    dry_parallel = (slope.abs() <= 1e-8) & (height > level)
    end = torch.where(dry_parallel, start, end)
    return torch.stack((start, end), -1).contiguous()


def final_level_rise(frame, total_frames, fps, height, seconds=5.):
    count=min(total_frames,max(2,round(seconds*fps)))
    start=total_frames-count
    x=float(np.clip((frame-start)/(count-1),0,1))
    return height*x*x*(3-2*x)


@torch.no_grad()
def main():
    from evaluation.evaluator import _rays
    from training.camera_isp import pixel_grid
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=192)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--post-rise-hold', type=float, default=0., help='extra seconds after the final level rise, without shifting its timing')
    p.add_argument('--final-rise', type=float, default=0., help='final mean-level rise as a fraction of visible reef height range')
    p.add_argument('--rise-seconds', type=float, default=5., help='duration of final mean-level rise')
    p.add_argument('--final-wave-scale', type=float, default=1., help='late multiplier of surface height and slopes after simulation')
    p.add_argument('--surface-dynamics', action='store_true', help='simulate shallow-water waves reflecting from the reef')
    p.add_argument('--impact-strength', type=float, default=3., help='breaking effect intensity (1 is the original subtle effect)')
    p.add_argument('--end-hold', type=float, default=0., help='extra seconds of waves at the final level')
    p.add_argument('--breaking-waves', action='store_true', help='grow late waves, reef contact foam and spray')
    p.add_argument('--water-surface', action='store_true', help='trace waves, dielectric reflection and refraction')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--reference-levels', type=Path, help='exact previous frame levels: hold 102, stop at 153')
    p.add_argument('--schedule', type=Path, help='previous frame_stats.json to equalize the rate of visual change')
    p.add_argument('--view', help='camera name, e.g. MTN_5903')
    p.add_argument('--direction', choices=['surface', 'camera-up'], default='surface')
    a = p.parse_args()
    if a.final_rise < 0 or a.rise_seconds <= 0 or a.post_rise_hold < 0:
        p.error("final rise must be nonnegative and rise duration positive")
    if a.impact_strength <= 0 or a.end_hold < 0 or a.final_wave_scale <= 0:
        p.error("impact strength must be positive and end hold nonnegative")
    if a.surface_dynamics:
        a.breaking_waves=True
        if a.preview:
            p.error("surface dynamics requires consecutive frames; omit --preview")
    if a.breaking_waves:
        a.water_surface=True
    if a.frames < 2 or a.fps <= 0 or a.width < 2 or a.width % 2:
        p.error('frames >= 2, fps > 0 and positive even width >= 2 are required')
    if a.reference_levels and (a.frames-1)/a.fps <= 2/.44:
        p.error('reference-level animation must exceed 4.55 seconds to include the pause')
    if not a.preview and shutil.which('ffmpeg') is None:
        p.error('ffmpeg is required on PATH')
    a.out.mkdir(parents=True, exist_ok=True)
    t = render.load_run(a.run, 'last', scene=a.scene)
    cameras = t.dataset.test + t.dataset.train
    camera = t.dataset.test[0] if a.view is None else next((c for c in cameras if c.name == a.view), None)
    if camera is None:
        raise ValueError(f"Unknown camera {a.view}; available: {[c.name for c in cameras]}")
    record = render.camera_record(camera)
    scale = a.width / record['width']
    record['width'] = a.width
    record['height'] = 2 * round(record['height'] * scale / 2)
    for key in ['fx', 'fy', 'cx', 'cy']:
        record[key] *= scale
    camera = render.camera_from_record(record, 'cuda')
    rays = _rays(camera).reshape(-1, 6)
    r = t.renderer
    g = t.model.gaussians.render_scene(camera.origin)
    m = t.model.medium.render_scene()
    r.set_gaussian_geometry(g)
    surface = r.render_gaussians(rays, g, background=(0.,0.,0.))
    valid = (surface.surface_depth.reshape(-1) > 0) & (surface.alpha.reshape(-1) > .7)
    depths = surface.surface_depth.reshape(-1)[valid]
    points = rays[valid,:3] + rays[valid,3:] * depths[:,None]
    if len(points) < 32:
        raise ValueError("Too few opaque surface hits to fit a plane; select another --view")
    center = points.median(0).values
    distance = (points-center).norm(dim=-1)
    points = points[distance <= torch.quantile(distance,.9)]
    center = points.mean(0)
    _, eigenvectors = torch.linalg.eigh((points-center).T @ (points-center) / len(points))
    up = eigenvectors[:,0]
    if torch.dot(up,camera.origin-center) < 0:
        up = -up
    if a.direction == 'camera-up':
        up = torch.nn.functional.normalize(-torch.stack([c.down for c in t.dataset.train]).mean(0), dim=0)
    heights = points @ up
    reef_low, reef_high = torch.quantile(heights, torch.tensor([.005,.995],device='cuda'))
    # Include every ray endpoint: the natural plane now reaches full wet/dry states.
    endpoint_heights = torch.cat((rays[:,:3] @ up + r.near*(rays[:,3:] @ up),
                                  rays[:,:3] @ up + r.far*(rays[:,3:] @ up)))
    high = endpoint_heights.max() + 1e-4
    low = endpoint_heights.min() - 1e-4
    # Spend most time around the visible reef; smoothly traverse the distant tail.
    knots = np.array([float(high),float(reef_high),float(reef_low),float(low)])
    knots = np.minimum.accumulate(knots)
    level_center = float((reef_high+reef_low)/2)
    level_scale = max(float(reef_high-reef_low), .01)
    curve_knots = np.arcsinh((knots-level_center)/level_scale)
    grid = pixel_grid(camera.height, camera.width, 'cuda').reshape(-1, 2)
    light = t.evaluator.water_light
    def draw():
        result = r.render(rays, gaussians=g, voronoi=m, background=(.48,.60,.70))
        raw = light(result.image, result.backscatter, rays) if light else result.image
        if t.camera_isp:
            raw = t.camera_isp.apply(raw, grid, (camera.height, camera.width), None, camera.origin)
        return raw.reshape(camera.height, camera.width, 3).clamp(0,1)
    # Endpoint invariants verify actual CUDA clipping, not a crossfade.
    r.water_intervals = None
    baseline = draw()
    r.water_intervals = torch.tensor([r.near,r.far],device='cuda').expand(len(rays),2).contiguous()
    torch.testing.assert_close(draw(), baseline, atol=1e-6, rtol=1e-5)
    r.water_intervals = torch.zeros((len(rays),2),device='cuda')
    dry = draw()
    from dataclasses import replace
    dry_medium = replace(m, attenuation=torch.zeros_like(m.attenuation), backscatter=torch.zeros_like(m.backscatter))
    original_medium = m
    m = dry_medium
    r.water_intervals = None
    torch.testing.assert_close(draw(), dry, atol=1e-6, rtol=1e-5)
    m = original_medium
    rise_end_frames=a.frames+round(a.end_hold*a.fps)
    total_frames=rise_end_frames+round(a.post_rise_hold*a.fps)
    indices = [0,a.frames//4,a.frames//2,3*a.frames//4,total_frames-1] if a.preview else range(total_frames)
    schedule = json.loads(a.schedule.read_text()) if a.schedule else None
    if schedule:
        costs = np.array([max(x["mean_absolute_change"] or 0., .0002) for x in schedule[1:]])
        cumulative = np.r_[0., np.cumsum(costs)]
        cumulative /= cumulative[-1]
        schedule_levels = np.array([x["level"] for x in schedule])
    water = None
    if a.water_surface:
        from examples.draining.water_surface import WaterSurface
        water = WaterSurface(t,g,m,rays,up,max(float(reef_high-reef_low),.1),surface.surface_depth)
    if water is not None and a.breaking_waves:
        from examples.draining.reef_interaction import ReefInteraction
        water.duration=a.frames/a.fps
        water.final_wave_scale=a.final_wave_scale
        water.impact_strength=a.impact_strength
        water.breaking=ReefInteraction(water,camera)
        if a.surface_dynamics:
            from examples.draining.surface_dynamics import ShallowWater
            proxy=water.breaking
            water.dynamics=ShallowWater(proxy.terrain,proxy.reliable,proxy.lower,proxy.extent,water.scale)
    reference_levels = None
    if a.reference_levels:
        reference_levels = np.array([x["level"] for x in json.loads(a.reference_levels.read_text())])
        if len(reference_levels) < 154 or not np.isfinite(reference_levels).all() or np.any(np.diff(reference_levels) > 1e-6):
            p.error("reference levels need >=154 finite, nonincreasing samples from the same scene/view")
    frame_stats = []
    previous = None
    for i in indices:
        progress = np.clip((i/(a.frames-1)-.10)/.80,0,1)
        progress = progress*progress*(3-2*progress)
        phase = min(int(progress * 3), 2)
        # Three monotone, smooth phases with exact matching endpoints.
        phase_progress = np.clip(progress * 3 - phase, 0, 1)
        phase_progress = phase_progress**2*(3-2*phase_progress)
        coordinate = curve_knots[phase]*(1-phase_progress) + curve_knots[phase+1]*phase_progress
        level = level_center + level_scale*np.sinh(coordinate)
        if schedule:
            level = np.interp(progress, cumulative, schedule_levels)
        if reference_levels is not None:
            time = i/a.fps
            duration = (a.frames-1)/a.fps
            fraction = min(time/duration,1.)
            if fraction < .04:
                source_frame = 0.
            elif fraction < .40:
                x = (fraction-.04)/.36
                source_frame = 102*(x*x*(3-2*x))
            elif fraction < .40+2/duration:
                source_frame = 102.
            elif fraction < .84:
                x = (fraction-.40-2/duration)/(.44-2/duration)
                source_frame = 102+51*(x*x*(3-2*x))
            else:
                source_frame = 153.
            level = np.interp(source_frame,np.arange(len(reference_levels)),reference_levels)
        rise=final_level_rise(i,rise_end_frames,a.fps,a.final_rise*float(reef_high-reef_low),a.rise_seconds)
        level+=rise
        r.water_intervals = water_intervals(rays, up, level, r.near, r.far)
        if water is None:
            frame = draw()
        else:
            raw = water.draw(float(level),i/a.fps)
            if t.camera_isp:
                raw = t.camera_isp.apply(raw,grid,(camera.height,camera.width),None,camera.origin)
            frame = raw.reshape(camera.height,camera.width,3).clamp(0,1)
        image = render.to_png('full', frame)
        values = image.astype(np.float32)/255
        frame_stats.append({'frame':i,'level':float(level),'level_rise':rise,'mean_rgb':values.mean((0,1)).tolist(),
                            'mean_absolute_change':None if previous is None else float(np.abs(values-previous).mean())})
        if water is not None and a.breaking_waves:
            frame_stats[-1].update(wave_multiplier=water.wave_multiplier,wave_strength=water.ocean.strength,foam_max=float(water.breaking.foam.max()),spray_particles=len(water.breaking.life))
        if a.surface_dynamics:
            frame_stats[-1].update(simulation_height_max=water.dynamics.amplitude,simulation_steps=water.dynamics.steps,solid_fraction=float((~water.dynamics.wet).float().mean()))
        previous = values
        Image.fromarray(image).save(a.out / f'{i:04d}.png')
        print(f'frame {i+1}/{total_frames}', flush=True)
    (a.out/'settings.json').write_text(json.dumps({'camera':record,'up':up.tolist(),'direction':a.direction,'level_knots':knots.tolist(),'level_start':float(high),'level_end':float(low),'frames':total_frames,'motion_frames':a.frames,'impact_strength':a.impact_strength,'final_wave_scale':a.final_wave_scale,'final_rise':a.final_rise,'rise_seconds':a.rise_seconds,'post_rise_hold':a.post_rise_hold,'rise_end_frames':rise_end_frames,'rise_height':a.final_rise*float(reef_high-reef_low),'end_hold':a.end_hold,'fps':a.fps,'endpoint_checks':'passed','water_surface':a.water_surface,'breaking_waves':a.breaking_waves,'surface_dynamics':a.surface_dynamics,'reference_levels':str(a.reference_levels),'pause_source_frame':102 if a.reference_levels else None,'end_source_frame':153 if a.reference_levels else None,'limitations':'One dielectric bounce and procedural environment with water surface enabled; no fluid dynamics or dry-surface relighting.'},indent=2))
    (a.out/'frame_stats.json').write_text(json.dumps(frame_stats,indent=2))
    if not a.preview:
        subprocess.run([shutil.which('ffmpeg') or 'ffmpeg','-y','-framerate',str(a.fps),'-i',str(a.out/'%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'draining.mp4')],check=True)

if __name__ == '__main__':
    main()
