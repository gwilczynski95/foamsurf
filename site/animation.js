/* A procedural interpretation of the paper's Voronoi medium and Gaussian surfels.
   The hero is illustrative, not a scientific rendering of a reconstructed scene. */
(() => {
  'use strict';
  const medium = document.querySelector('#medium');
  const wordmark = document.querySelector('#wordmark');
  const toggle = document.querySelector('#motion-toggle');
  const background = medium?.getContext('2d');
  const ink = wordmark?.getContext('2d');
  if (!background || !ink) { if (toggle) toggle.hidden = true; return; }
  toggle.hidden = false;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = reduced.matches, visible = true, raf = 0, previous = 0, elapsed = 0;
  let width = 1, height = 1, textWidth = 1, textHeight = 1, sites = [], splats = [];
  const pointer = {x: 0, y: 0};
  let state = 4317;
  function random() { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; }
  const sprite = document.createElement('canvas');
  sprite.width = sprite.height = 64;
  const brush = sprite.getContext('2d');
  const gradient = brush.createRadialGradient(32, 32, 0, 32, 32, 31);
  gradient.addColorStop(0, 'rgba(196,48,62,.98)');
  gradient.addColorStop(.32, 'rgba(218,66,76,.90)');
  gradient.addColorStop(.65, 'rgba(232,96,101,.60)');
  gradient.addColorStop(1, 'rgba(234,106,112,0)');
  brush.fillStyle = gradient; brush.fillRect(0, 0, 64, 64);

  function sizeCanvas(canvas, ctx) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [rect.width, rect.height];
  }
  function resize() {
    [width, height] = sizeCanvas(medium, background);
    [textWidth, textHeight] = sizeCanvas(wordmark, ink);
    state = 4317;
    const cols = width < 700 ? 6 : 10, rows = 6;
    sites = [];
    for (let y = -1; y < rows; y++) for (let x = -1; x < cols + 1; x++) {
      sites.push({x: (x + .3 + random() * .65) * width / cols,
        y: (y + .25 + random() * .7) * height / rows,
        phase: random() * Math.PI * 2, speed: .10 + random() * .13, shade: random()});
    }
    const mask = document.createElement('canvas');
    mask.width = Math.ceil(textWidth); mask.height = Math.ceil(textHeight);
    const m = mask.getContext('2d', {willReadFrequently: true});
    let fontSize = Math.min(textHeight * .91, textWidth * .22);
    m.font = `900 ${fontSize}px Arial, sans-serif`;
    fontSize *= Math.min(1, textWidth * .94 / m.measureText('FoamSurf').width);
    m.font = `900 ${fontSize}px Arial, sans-serif`;
    m.textAlign = 'center'; m.textBaseline = 'middle';
    m.fillText('FoamSurf', textWidth / 2, textHeight * .54);
    const pixels = m.getImageData(0, 0, mask.width, mask.height).data;
    const step = Math.max(3.1, fontSize / 27);
    splats = [];
    for (let y = step; y < mask.height - step; y += step) {
      for (let x = step; x < mask.width - step; x += step) {
        const px = Math.round(x + (random() - .5) * step * .65);
        const py = Math.round(y + (random() - .5) * step * .65);
        if (pixels[(py * mask.width + px) * 4 + 3] < 140) continue;
        splats.push({x: px, y: py, rx: step * (.9 + random() * .55), ry: step * (.38 + random() * .32),
          angle: -.8 + random() * 1.2, phase: random() * 6.28, alpha: .68 + random() * .3});
      }
    }
    draw();
    document.documentElement.classList.add('wordmark-ready');
  }
  // Clip a convex polygon to the half-plane nearer to site a than to site b.
  function clip(polygon, a, b) {
    const nx = b.x - a.x, ny = b.y - a.y;
    const c = (b.x * b.x + b.y * b.y - a.x * a.x - a.y * a.y) * .5;
    const output = [];
    for (let i = 0; i < polygon.length; i++) {
      const p = polygon[i], q = polygon[(i + 1) % polygon.length];
      const dp = p[0] * nx + p[1] * ny - c, dq = q[0] * nx + q[1] * ny - c;
      if (dp <= 0) output.push(p);
      if ((dp <= 0) !== (dq <= 0)) {
        const t = dp / (dp - dq);
        output.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
      }
    }
    return output;
  }
  function draw() {
    const t = elapsed / 1000;
    background.clearRect(0, 0, width, height);
    const points = sites.map(s => ({...s,
      x: s.x + Math.sin(t * s.speed + s.phase) * 14 + pointer.x * 5,
      y: s.y + Math.cos(t * s.speed * .8 + s.phase) * 12 + pointer.y * 4}));
    for (const a of points) {
      let polygon = [[0, 0], [width, 0], [width, height], [0, height]];
      for (const b of points) { if (a !== b) polygon = clip(polygon, a, b); if (!polygon.length) break; }
      if (!polygon.length) continue;
      background.beginPath(); background.moveTo(...polygon[0]);
      for (const p of polygon.slice(1)) background.lineTo(...p);
      background.closePath();
      background.fillStyle = `rgba(94,158,108,${.035 + a.shade * .12})`;
      background.fill(); background.lineWidth = .7;
      background.strokeStyle = 'rgba(66,128,83,.20)'; background.stroke();
      background.fillStyle = 'rgba(68,129,83,.28)';
      background.beginPath(); background.arc(a.x, a.y, 1.35, 0, Math.PI * 2); background.fill();
    }
    ink.clearRect(0, 0, textWidth, textHeight);
    const assemble = reduced.matches ? 1 : Math.min(1, .5 + elapsed / 1800);
    for (const s of splats) {
      const drift = Math.sin(t * .45 + s.phase);
      ink.save();
      ink.translate(s.x + drift * .8 + (1 - assemble) * Math.sin(s.phase) * 15,
        s.y + Math.cos(t * .35 + s.phase) * .7 + (1 - assemble) * Math.cos(s.phase) * 9);
      ink.rotate(s.angle + drift * .16);
      ink.globalAlpha = s.alpha;
      ink.drawImage(sprite, -s.rx, -s.ry, s.rx * 2, s.ry * 2);
      ink.restore();
    }
  }
  function tick(now) {
    if (paused || !visible || document.hidden) { raf = 0; previous = 0; return; }
    if (!previous || now - previous >= 1000 / 24) {
      elapsed += previous ? Math.min(now - previous, 80) : 0;
      previous = now; draw();
    }
    raf = requestAnimationFrame(tick);
  }
  function start() { if (!raf && !paused && visible && !document.hidden) raf = requestAnimationFrame(tick); }
  function updateToggle() {
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.textContent = paused ? 'Resume motion  ▷' : 'Pause motion  Ⅱ';
  }
  toggle.addEventListener('click', () => { paused = !paused; updateToggle(); start(); });
  reduced.addEventListener('change', () => { paused = reduced.matches; updateToggle(); draw(); start(); });
  document.addEventListener('visibilitychange', start);
  document.querySelector('.hero').addEventListener('pointermove', e => {
    if (paused || reduced.matches || e.pointerType === 'touch') return;
    const rect = medium.getBoundingClientRect();
    pointer.x = (e.clientX - rect.left) / width - .5;
    pointer.y = (e.clientY - rect.top) / height - .5;
  });
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; start(); }).observe(medium);
  new ResizeObserver(resize).observe(document.querySelector('.hero-content'));
  updateToggle(); resize(); start();
})();
