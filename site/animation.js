/* A procedural interpretation of the paper's Voronoi medium and Gaussian surfels.
   The hero is illustrative, not a scientific rendering of a reconstructed scene. */
(() => {
  'use strict';
  const medium = document.querySelector('#medium');
  const nearMedium = document.querySelector('#medium-front');
  const foreground = nearMedium?.getContext('2d');
  const wordmark = document.querySelector('#wordmark');
  const toggle = document.querySelector('#motion-toggle');
  const background = medium?.getContext('2d');
  const ink = wordmark?.getContext('2d');
  if (!background || !foreground || !ink) { if (toggle) toggle.hidden = true; return; }
  toggle.hidden = false;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = reduced.matches, visible = true, raf = 0, previous = 0, elapsed = 0;
  let width = 1, height = 1, textWidth = 1, textHeight = 1, layers = [], splats = [], titleTop = 0;
  const pointer = {x: 0, y: 0};
  let state = 4317;
  function random() { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; }
  const sprite = document.createElement('canvas');
  sprite.width = sprite.height = 64;
  const brush = sprite.getContext('2d');
  const gradient = brush.createRadialGradient(32, 32, 0, 32, 32, 31);
  gradient.addColorStop(0, 'rgba(180,35,49,.98)');
  gradient.addColorStop(.32, 'rgba(205,48,62,.94)');
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
    sizeCanvas(nearMedium, foreground);
    [textWidth, textHeight] = sizeCanvas(wordmark, ink);
    titleTop = wordmark.getBoundingClientRect().top - medium.getBoundingClientRect().top;
    state = 4317;
    // Separate tessellations form distant, middle and near sheets of water.
    // Moving their sites recomputes every shared Voronoi vertex, not just its dot.
    layers = [
      {cols: width < 700 ? 7 : 12, rows: 6, depth: .35, amplitude: 22},
      {cols: width < 700 ? 5 : 8, rows: 5, depth: .65, amplitude: 34},
      {cols: width < 700 ? 3 : 5, rows: 4, depth: 1, amplitude: 46}
    ].map((layer, index) => {
      const sites = [];
      for (let y = -1; y < layer.rows; y++) for (let x = -1; x < layer.cols + 1; x++) {
        sites.push({x: (x + .25 + random() * .7) * width / layer.cols,
          y: (y + .2 + random() * .75) * height / layer.rows,
          phase: random() * Math.PI * 2, speed: .18 + random() * .2 + index * .045,
          shade: random()});
      }
      return {...layer, sites};
    });
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
    const step = Math.max(3.5, fontSize / 22);
    splats = [];
    for (let y = step; y < mask.height - step; y += step) {
      for (let x = step; x < mask.width - step; x += step) {
        const px = Math.round(x + (random() - .5) * step * .65);
        const py = Math.round(y + (random() - .5) * step * .65);
        if (pixels[(py * mask.width + px) * 4 + 3] < 140) continue;
        splats.push({x: px, y: py, rx: step * (1.05 + random() * .85), ry: step * (.48 + random() * .4),
          angle: -1.1 + random() * 2.2, phase: random() * 6.28, alpha: .55 + random() * .45,
          speed: .45 + random() * .55});
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
  function drawLayer(ctx, layer, t, near = false) {
    const points = layer.sites.map(s => ({...s,
      x: s.x + Math.sin(t * s.speed + s.phase) * layer.amplitude + pointer.x * 18 * layer.depth,
      y: s.y + Math.cos(t * s.speed * .8 + s.phase) * layer.amplitude * .8 + pointer.y * 14 * layer.depth}));
    for (const a of points) {
      let polygon = [[0, 0], [width, 0], [width, height], [0, height]];
      for (const b of points) { if (a !== b) polygon = clip(polygon, a, b); if (!polygon.length) break; }
      if (!polygon.length) continue;
      ctx.beginPath(); ctx.moveTo(...polygon[0]);
      for (const p of polygon.slice(1)) ctx.lineTo(...p);
      ctx.closePath();
      ctx.fillStyle = near
        ? `rgba(137,183,143,${.025 + Math.pow(a.shade, 3) * .32})`
        : `rgba(94,158,108,${(.035 + a.shade * .11) * layer.depth})`;
      ctx.fill(); ctx.lineWidth = near ? 1 : .65;
      ctx.strokeStyle = `rgba(66,128,83,${near ? .22 : .19 * layer.depth})`; ctx.stroke();
      // Mark the moving polygon corners, rather than the Voronoi sites.
      ctx.fillStyle = `rgba(57,120,74,${near ? .42 : .28 * layer.depth})`;
      for (const p of polygon) {
        ctx.beginPath(); ctx.arc(p[0], p[1], near ? 1.8 : 1.1, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  function draw() {
    const t = elapsed / 1000;
    background.clearRect(0, 0, width, height);
    drawLayer(background, layers[0], t);
    drawLayer(background, layers[1], t);
    foreground.clearRect(0, 0, width, height);
    drawLayer(foreground, layers[2], t, true);
    // The near layer veils portions of the wordmark, then dissolves before the copy.
    foreground.save();
    foreground.globalCompositeOperation = 'destination-in';
    const veil = foreground.createLinearGradient(0, 0, 0, height);
    const titleBottom = (titleTop + textHeight) / height;
    veil.addColorStop(0, 'rgba(0,0,0,.6)');
    veil.addColorStop(Math.max(0, titleTop / height), '#000');
    veil.addColorStop(Math.min(1, titleBottom - .04), '#000');
    veil.addColorStop(Math.min(1, titleBottom + .10), 'transparent');
    veil.addColorStop(1, 'transparent');
    foreground.fillStyle = veil; foreground.fillRect(0, 0, width, height);
    foreground.restore();
    ink.clearRect(0, 0, textWidth, textHeight);
    const assemble = reduced.matches ? 1 : Math.min(1, .5 + elapsed / 1800);
    for (const s of splats) {
      const drift = Math.sin(t * s.speed + s.phase);
      const movement = Math.max(2, Math.min(5, textWidth / 200));
      ink.save();
      ink.translate(s.x + drift * movement + (1 - assemble) * Math.sin(s.phase) * 15,
        s.y + Math.cos(t * s.speed * .8 + s.phase) * movement * .75 + (1 - assemble) * Math.cos(s.phase) * 9);
      ink.rotate(s.angle + drift * .38);
      ink.globalAlpha = s.alpha * (.85 + .15 * Math.sin(t * s.speed * .6 + s.phase));
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
