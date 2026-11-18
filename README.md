# FoamSurf project page

The `page` branch contains a standalone, buildless project website in `site/`.
The release worktree and training code are independent of this page.
All page content is English. No hosting or public publication has been performed.

## Local preview

```bash
python -m http.server 8000 --directory site --bind 127.0.0.1
```

Open http://127.0.0.1:8000 locally. On a remote machine, forward the port with SSH;
a server's loopback address is not a public preview URL.

## Design

An original Canvas 2D animation renders three independently moving Voronoi layers
and a wordmark built from red elliptical Gaussian sprites. Polygon vertices move as
the tessellations evolve. The nearest translucent layer overlaps the title; all layers
fade into a sage-green page. The Gaussians drift, rotate and vary in size and opacity. The paper method figure supplies the visual language. FlaRe's project page
(https://kbyrski.github.io/FlaRe/) is a layout/animation reference; its JavaScript and
styles have not been copied. The hero is an artistic illustration, not a physical simulator.

No external fonts, scripts, CDN dependencies or build toolchain are needed. Motion
respects the OS reduced-motion preference, can be paused, and stops when the hero is
offscreen or the tab is hidden. HTML text provides a no-JavaScript wordmark fallback.

## Replace placeholders

Edit `site/index.html` to add:
- Confirmed author names, affiliations and profile links.
- The arXiv/PDF URL, final code release URL, dataset links and citation.
- A method film in place of `.film-placeholder`.
- Application video elements in place of the static figure previews and video statuses.

Unavailable resources are labeled spans, not fake links or controls. No publication date,
acceptance status, author identity or arXiv identifier is invented. The abstract is shortened
from the manuscript; figures are resized WebP exports of the existing paper illustrations.
For reference, originals reside in the separate paper workspace; all files needed to serve
the page are copied into `site/assets/`.

## Optional GitHub Pages publication

`.github/workflows/project-page.yml` is manual-only. When publication is wanted, push the
`page` branch, choose GitHub Actions as the repository's Pages source, and run the
"Publish project page" workflow **on the page branch**. It uploads `site/` directly.
Merely committing or pushing this branch does not trigger deployment.
