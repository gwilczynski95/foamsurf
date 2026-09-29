"""Polska prezentacja statusowa FoamSurf (PDF + PNG slajdów)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
IUI = ROOT / "outputs/rgb_smooth/513211"
LEGO = ROOT / "outputs/radfoam_prior/radfoam_prior_run30_502753/evaluation/iteration_030000"
REPORT = ROOT / "docs/reports/rgb_disentanglement_2026_09"

BG = "#0c1425"
CARD = "#17243a"
CARD2 = "#1c2d46"
WHITE = "#f6f8fc"
MUTED = "#b7c4d7"
CYAN = "#59d4e8"
BLUE = "#6ca6ff"
GREEN = "#78d8aa"
YELLOW = "#ffcd69"
RED = "#ff8996"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": BG})


def slide():
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, value, size=14, color=WHITE, weight="normal", va="top", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   va=va, ha=kw.pop("ha", "left"), linespacing=1.3, **kw)


def card(ax, x, y, w, h, fill=CARD, edge=None, radius=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.15,rounding_size={radius}",
                 linewidth=1.2 if edge else 0, edgecolor=edge or fill,
                 facecolor=fill))


def arrow(ax, x1, y1, x2, y2, color=CYAN, lw=2.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=lw, color=color))


def heading(ax, n, title, subtitle=None):
    txt(ax, 5, 53.5, f"FOAMSURF  /  STAN PROJEKTU  /  {n:02d}", 9, CYAN, "bold")
    txt(ax, 5, 49.7, title, 25, WHITE, "bold")
    if subtitle:
        txt(ax, 5, 44.9, subtitle, 11.5, MUTED)
    ax.plot([5, 95], [3.3, 3.3], color="#34435a", lw=0.8)
    txt(ax, 5, 2.5, "29.09.2026  ·  dane: repozytorium FoamSurf", 8, MUTED)
    txt(ax, 95, 2.5, f"{n}/10", 8, MUTED, ha="right")


def image(ax, path, x, y, w, h, crop=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    iw, ih = im.size
    box_ratio, img_ratio = w / h, iw / ih
    if img_ratio > box_ratio:
        dw = h * img_ratio
        x0 = x + (w - dw) / 2
        extent = (x0, x0 + dw, y, y + h)
    else:
        dh = w / img_ratio
        y0 = y + (h - dh) / 2
        extent = (x, x + w, y0, y0 + dh)
    artist = ax.imshow(im, extent=extent, zorder=3, interpolation="nearest")
    artist.set_clip_path(Rectangle((x, y), w, h, transform=ax.transData))
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.8, edgecolor="#728298",
                           facecolor="none", zorder=4))


def labeled_box(ax, x, y, w, h, label, body, accent=CYAN):
    card(ax, x, y, w, h, fill=CARD2)
    ax.add_patch(Rectangle((x, y), .65, h, facecolor=accent, linewidth=0))
    txt(ax, x+2.2, y+h-2.5, label, 12.5, accent, "bold")
    txt(ax, x+2.2, y+h-7, body, 10.2, WHITE)


def make_deck():
    pages = []

    # 1 — Teza i uczciwy status.
    fig, ax = slide()
    txt(ax, 5, 52, "FOAMSURF", 13, CYAN, "bold")
    txt(ax, 5, 46, "Powierzchnie Gaussami,\nmedium komórkami Voronoi", 27, WHITE, "bold")
    txt(ax, 5, 31, "Stan projektu i najbliższy eksperyment", 15, MUTED)
    card(ax, 5, 11, 48, 14, fill=CARD2)
    txt(ax, 8, 22, "Wynik dziś", 12, CYAN, "bold")
    txt(ax, 8, 17.7, "LEGO: rozdzielenie działa.  IUI3: dno nadal ma dziury\nw renderze samych Gaussów.", 12.5, WHITE)
    image(ax, IUI / "run_13/joint/evaluation/iteration_030000/0002_MTN_5911_hybrid.png",
          59, 13, 36, 31)
    txt(ax, 59, 11, "IUI3 Red Sea · silne TV · wynik hybrydowy", 9, MUTED)
    txt(ax, 5, 4, "29.09.2026 · 10 slajdów", 9, MUTED)
    pages.append(fig)

    # 2 — Cel i architektura.
    fig, ax = slide(); heading(ax, 2, "Co ma modelować każda reprezentacja?")
    labeled_box(ax, 5, 21, 43, 20, "GAUSSY  ·  CIAŁA STAŁE",
                "Ostre granice, geometria dna i obiektów,\ntekstura oraz efekty zależne od widoku.", BLUE)
    labeled_box(ax, 52, 21, 43, 20, "VORONOI  ·  MEDIUM",
                "Tłumienie światła, backscatter, mętność\ni płynne zmiany barwy w wodzie.", GREEN)
    arrow(ax, 48.5, 30.8, 51.5, 30.8, YELLOW, 2)
    card(ax, 5, 8, 90, 8.5, fill="#23334b")
    txt(ax, 8, 14.4, "Kryterium sukcesu", 11, YELLOW, "bold")
    txt(ax, 8, 11.1, "Dno ma być kompletne w „tylko Gaussy”; woda nie może odtwarzać struktury dna.", 12, WHITE)
    pages.append(fig)

    # 3 — Schemat blokowy renderera i równania składu RGB.
    fig, ax = slide(); heading(ax, 3, "Renderer: od promienia do piksela",
                              "Zdarzenia są uporządkowane według głębokości; rachunek prowadzimy osobno dla RGB.")
    labeled_box(ax, 5, 34, 26, 9, "WEJŚCIE", "Promień + kolor tła", CYAN)
    labeled_box(ax, 37, 34, 26, 9, "ZDARZENIA", "Gaussy i odcinki Voronoi", BLUE)
    labeled_box(ax, 69, 34, 26, 9, "CAŁKOWANIE", "Kolejność wzdłuż promienia", GREEN)
    arrow(ax, 31, 38.5, 36.5, 38.5); arrow(ax, 63, 38.5, 68.5, 38.5)
    card(ax, 5, 17, 43, 15, fill=CARD2)
    txt(ax, 7.5, 29.5, "DIRECT: powierzchnia + tłumione tło", 11.5, BLUE, "bold")
    txt(ax, 7.5, 25.2, r"$D=\sum_k T_G(t_k)T_D(t_k)\alpha_k\mathbf{c}_k$", 14.5)
    txt(ax, 7.5, 20.7, r"$\qquad+T_G(t_f)T_D(t_f)\mathbf{C}_{\rm bg}$", 14.5)
    card(ax, 52, 17, 43, 15, fill=CARD2)
    txt(ax, 54.5, 29.5, "BACKSCATTER: wkład odcinków medium", 11.5, GREEN, "bold")
    txt(ax, 54.5, 25.2, r"$S=\sum_i T_G(s_i)T_B(s_i)\mathbf{B}_i$", 14.5)
    txt(ax, 54.5, 20.7, r"$\qquad\cdot(1-e^{-\mathbf{b}_i\ell_i})$", 14.5)
    arrow(ax, 82, 33.5, 77, 32.5, GREEN, 1.5)
    txt(ax, 50, 14.9, r"$\mathbf{I}=D+S$", 17, YELLOW, "bold", ha="center")
    txt(ax, 5, 11.8, r"$T_D\!\leftarrow\!T_D e^{-\mathbf{a}_i\ell_i}$   "
        r"$T_B\!\leftarrow\!T_B e^{-\mathbf{b}_i\ell_i}$   "
        r"$T_G\!\leftarrow\!T_G(1-\alpha_k)$", 11.3, WHITE)
    txt(ax, 5, 8.7, r"$\ell_i$: długość komórki; $\mathbf{a}_i$: efektywne tłumienie; "
        r"$\mathbf{b}_i,\mathbf{B}_i$: tempo i kolor backscatter; "
        r"$\alpha_k,\mathbf{c}_k$: opacity i kolor trafienia G.", 9.3, MUTED)
    txt(ax, 5, 5.8, r"$T_D,T_B,T_G$: transmitancje do bieżącego miejsca; "
        r"$t_f$: koniec promienia. Direct zawiera tło, więc nie jest renderem „tylko Gaussy”.", 9.3, MUTED)
    pages.append(fig)

    # 4 — Schemat blokowy treningu.
    fig, ax = slide(); heading(ax, 4, "Pipeline treningowy: jeden obraz na iterację",
                              "W pokazanych runach IUI3 jeden batch obejmuje jeden pełny obraz.")
    labeled_box(ax, 5, 31, 27, 12, "DANE TRENINGOWE", "25 zdjęć RGB + kalibracja\ni pozy COLMAP", CYAN)
    labeled_box(ax, 37, 31, 27, 12, "WYBÓR KAMERY", "1 kamera na krok; losowa\nkolejność bez powtórzeń", BLUE)
    labeled_box(ax, 69, 31, 26, 12, "BATCH", "Wszystkie piksele tej kamery\n→ promienie + docelowe RGB", GREEN)
    arrow(ax, 32, 37, 36.5, 37); arrow(ax, 64, 37, 68.5, 37)
    labeled_box(ax, 5, 13, 27, 12, "PARAMETRY", "Gaussy: geometria, kolor, α\nVoronoi: sites, a, b, B", BLUE)
    labeled_box(ax, 37, 13, 27, 12, "RENDERER", "Model + pełny obraz promieni\n→ przewidziane RGB", CYAN)
    labeled_box(ax, 69, 13, 26, 12, "LOSS + ADAM", "RGB + regularyzacje\n→ gradienty i adaptacja", GREEN)
    arrow(ax, 32, 19, 36.5, 19); arrow(ax, 64, 19, 68.5, 19)
    arrow(ax, 82, 30.5, 52, 25.5, YELLOW, 1.4)
    arrow(ax, 81, 12.5, 19, 12.5, YELLOW, 1.4)
    txt(ax, 5, 9.4, "W IUI3 obraz ma 1383 × 917 = 1 268 211 pikseli, czyli tyle promieni na krok;", 10.1, WHITE)
    txt(ax, 5, 6.6, "4 widoki testowe służą tylko do ewaluacji. Parametr rays_per_batch=640000 nie ogranicza pełnego obrazu.", 9.7, MUTED)
    pages.append(fig)

    # 5 — Stałe cele i mechanizmy adaptacji.
    fig, ax = slide(); heading(ax, 5, "Straty bazowe i pruning Gaussów",
                              "Pruning jest decyzją dyskretną, nie różniczkowalną funkcją kosztu.")
    card(ax, 5, 25, 43, 17, fill=CARD2)
    txt(ax, 7.5, 39.2, "01  ·  REKONSTRUKCJA RGB", 11.5, CYAN, "bold")
    txt(ax, 7.5, 34.8, r"$\mathcal{L}_{\rm RGB}=0.8\langle|\hat I-I|\rangle$", 13.1)
    txt(ax, 7.5, 30.8, r"$\qquad+0.2(1-\mathrm{SSIM}(\hat I,I))$", 13.1)
    txt(ax, 7.5, 27.6, "Dopasowanie koloru i struktury całego obrazu.", 9.6, MUTED)
    card(ax, 52, 25, 43, 17, fill=CARD2)
    txt(ax, 54.5, 39.2, "02  ·  BINARYZACJA OPACITY", 11.5, BLUE, "bold")
    txt(ax, 54.5, 34.8, r"$\mathcal{L}_{\rm bin}=\lambda_{\rm bin}(t)$", 13.1)
    txt(ax, 54.5, 30.8, r"$\qquad\cdot N_G^{-1}\sum_j\alpha_j(1-\alpha_j)$", 13.1)
    txt(ax, 54.5, 27.6, "Minima przy α=0/1; λ rośnie cosinusowo 1–4k do 0,001.", 9.25, MUTED)
    card(ax, 5, 5.7, 43, 17, fill=CARD2)
    txt(ax, 7.5, 19.8, "03  ·  REGULARYZACJA SKALI", 11.5, GREEN, "bold")
    txt(ax, 7.5, 15.4, r"$\mathcal{L}_{\rm scale}=10^{-3}N_G^{-1}$", 13.1)
    txt(ax, 7.5, 11.4, r"$\qquad\cdot\sum_j\|\mathbf{s}_j\|_2$", 13.1)
    txt(ax, 7.5, 8.2, "Ogranicza rozmiary Gaussów; nie wymusza pokrycia dna.", 9.6, MUTED)
    card(ax, 52, 5.7, 43, 17, fill=CARD2)
    txt(ax, 54.5, 19.8, "04  ·  OPACITY PRUNING (BEZ GRADIENTU)", 11.1, YELLOW, "bold")
    txt(ax, 54.5, 15.4, r"$\alpha_j<\tau(t),\quad\tau:0.005\to0.10$", 12.9)
    txt(ax, 54.5, 11.8, "Co 250 kroków do 30k; usuwa słabe Gaussy.", 9.8)
    txt(ax, 54.5, 8.2, "Densyfikacja Gaussów kończy się przy 15k.", 9.6, MUTED)
    pages.append(fig)

    # 6 — Eksperymentalne straty: TV, overlap, opacity, nowy overlap.
    fig, ax = slide(); heading(ax, 6, "Straty badane dla rozdzielenia reprezentacji",
                              "Wagi strat zmieniają się harmonogramem; „nowy O” nie ma jeszcze wyniku treningowego.")
    card(ax, 5, 24.6, 43, 17.4, fill=CARD2)
    txt(ax, 7.5, 39.3, "S  ·  GRAPH TV / HUBER", 11.5, GREEN, "bold")
    txt(ax, 7.5, 35, r"$\mathcal{L}_S=\sum_{q\in\{a,b,B\}}w_q$", 12.9)
    txt(ax, 7.5, 31.2, r"$\qquad\cdot\langle H_{\delta_q}(q_i'-q_j')\rangle_{(i,j),c}$", 12.9)
    txt(ax, 7.5, 27.8, "a,b w log; B wprost. Wagi (0,1; 0,1; 1),\nδ=(0,1; 0,1; 0,05). Średnia po krawędziach i RGB.", 8.9, MUTED)
    card(ax, 52, 24.6, 43, 17.4, fill=CARD2)
    txt(ax, 54.5, 39.3, "O  ·  STARY OVERLAP", 11.5, YELLOW, "bold")
    txt(ax, 54.5, 35, r"$\mathcal{L}_O^{\rm old}=\lambda_O\langle C_rM_rF_rQ_r\rangle_r$", 12.6)
    txt(ax, 54.5, 31.2, "C: skupienie backscatter w głębi; M: jego siła;", 9.5)
    txt(ax, 54.5, 28.5, "F: udział blisko G; Q: pewność głębokości G.\nGradient tylko do b i B, nie do a ani sites.", 9.3, MUTED)
    card(ax, 5, 5.5, 43, 17.2, fill=CARD2)
    txt(ax, 7.5, 19.8, "G  ·  GAUSSIAN OPACITY", 11.5, RED, "bold")
    txt(ax, 7.5, 15.5, r"$\mathcal{L}_G=\lambda_G\frac{\sum_{r:A_r>0.1}(1-A_r)}{\#\{r:A_r>0.1\}}$", 12.2)
    txt(ax, 7.5, 10.5, "A: akumulacja opacity G; maska >0,1 bez gradientu.\nWzmacnia aktywne piksele; nie wypełnia dziur.", 9.4, MUTED)
    card(ax, 52, 5.5, 43, 17.2, fill=CARD2)
    txt(ax, 54.5, 19.8, "O  ·  NOWY LOKALNY OVERLAP", 11.1, CYAN, "bold")
    txt(ax, 54.5, 15.5, r"$\mathcal{L}_O^{\rm new}=\lambda_O\langle Q_r(1-e^{-\int_{W_r}k(x)\,dx})\rangle_r$", 11.4)
    txt(ax, 54.5, 11.5, r"$k=\mathrm{mean}_{RGB}(a+b+bB)/(3s)$", 11.3)
    txt(ax, 54.5, 8.4, "Okno przy trafieniu G; gradient do a,b,B i sites;\nG odłączone. Czarne tło: osobny czynnik sweepu.", 8.9, MUTED)
    pages.append(fig)

    # 7 — LEGO proof of concept.
    fig, ax = slide(); heading(ax, 7, "LEGO: dowód, że podział jest możliwy",
                              "Zewnętrzny prior RadFoam → klasyfikacja komórek → Gaussy na powierzchni → trening 30k.")
    labels = ["OBRAZ REFERENCYJNY", "HYBRYDA", "TYLKO GAUSSY", "TYLKO VORONOI"]
    files = ["0000_r_000_panel.png", "0000_r_000_hybrid.png",
             "0000_r_000_gaussian_only.png", "0000_r_000_voronoi_only.png"]
    for j, (lab, name) in enumerate(zip(labels, files)):
        x = 5+j*23
        txt(ax, x, 41, lab, 10.2, CYAN if j<2 else YELLOW, "bold")
        image(ax, LEGO/name, x, 17, 20, 20, crop=(0,0,800,800) if j==0 else None)
    card(ax, 5, 6, 90, 7.5, fill=CARD2)
    txt(ax, 8, 11.2, "34,824 dB hybryda", 14, GREEN, "bold")
    txt(ax, 38, 11.2, "Gaussy: 99,45% dominacji na LEGO", 11.5, WHITE)
    txt(ax, 38, 8.2, "Medium: 99,71% dominacji na chmurach", 11.5, WHITE)
    txt(ax, 5, 4.3, "Ważne: wynik na scenie syntetycznej i z priorem zewnętrznym; nie jest to wynik IUI3.", 9, MUTED)
    pages.append(fig)

    # 8 — PSNR z protokołem.
    fig, ax = slide(); heading(ax, 8, "IUI3 Red Sea: PSNR i jego ograniczenia",
                              "FoamSurf: ukończone runy RGB, średnia finalna po 3 seedach (30k iteracji).")
    vals = [("Kontrola",21.05,BLUE), ("Overlap + TV ×3",22.36,CYAN),
            ("Silne TV",24.45,GREEN), ("WaterSplatting · publikacja",29.386,YELLOW)]
    for i, (name,v,c) in enumerate(vals):
        y=35.5-i*7.2
        txt(ax, 5, y+2.6, name, 12.5, WHITE, "bold")
        ax.add_patch(Rectangle((39,y),51,3.6,facecolor="#29384d",linewidth=0))
        ax.add_patch(Rectangle((39,y),51*v/32,3.6,facecolor=c,linewidth=0))
        txt(ax, 92, y+2.8, f"{v:.2f} dB", 11.5, c, "bold")
    card(ax, 5, 6.3, 90, 7, fill="#293646")
    txt(ax, 8, 10.8, "Uwaga metodologiczna", 10.8, YELLOW, "bold")
    txt(ax, 8, 8, "29,386 dB pochodzi z tabeli publikacji WaterSplatting; split, preprocessing i ustawienia nie są tu zweryfikowane 1:1.", 9.6, WHITE)
    txt(ax, 5, 4.1, "Źródło WaterSplatting: arxiv.org/html/2408.08206, tabela IUI3 Red Sea.", 8.6, MUTED)
    pages.append(fig)

    # 9 — Realna diagnostyka disentanglementu.
    fig, ax = slide(); heading(ax, 9, "IUI3: PSNR rośnie, lecz dno nie należy do Gaussów",
                              "Ten sam widok MTN_5911; porównanie kontroli i silnego TV (seed 43).")
    names = ["REFERENCJA", "HYBRYDA", "TYLKO GAUSSY", "TYLKO VORONOI"]
    stems = [None, "hybrid", "gaussian_only", "voronoi_only"]
    for j,name in enumerate(names): txt(ax, 5+j*23, 41.7, name, 10.6, CYAN if j<2 else YELLOW, "bold")
    for i,(run,label,color) in enumerate([(1,"KONTROLA",BLUE),(13,"SILNE TV",GREEN)]):
        y=25-i*14.2
        txt(ax, 5, y+13.5, label, 9.5, color, "bold")
        p = IUI/f"run_{run}/joint/evaluation/iteration_030000"
        for j,stem in enumerate(stems):
            x=5+j*23
            if stem is None:
                image(ax,p/"0002_MTN_5911_panel.png",x,y,20.5,11,crop=(0,0,1383,917))
            else:
                image(ax,p/f"0002_MTN_5911_{stem}.png",x,y,20.5,11)
    card(ax, 5, 4.5, 90, 5, fill="#3b3040")
    txt(ax, 8, 7.7, "Wniosek: „direct” ≠ „tylko Gaussy” — direct zawiera również tłumione tło. Dziury w G-only są rzeczywiste.", 10.8, WHITE)
    pages.append(fig)

    # 10 — Następne testy.
    fig, ax = slide(); heading(ax, 10, "Co dalej: wymusić kompletne dno bez utraty medium")
    labeled_box(ax, 5, 27, 27, 16, "01  ·  SWEEP 72 RUNÓW",
                "Tło: białe/czarne\nOverlap: 0 / 0,03 / 0,10\nTV: 0 / 1 / 3 / 10\n3 seedy; test GPU w kolejce", CYAN)
    labeled_box(ax, 37, 27, 27, 16, "02  ·  MIARY PODZIAŁU",
                "Maska dna i wody\nPokrycie dna w G-only\nZanik struktury w V-only\nPSNR osobno", GREEN)
    labeled_box(ax, 69, 27, 26, 16, "03  ·  DALSZY MODEL",
                "Sprawdzić harmonogram\nadaptacji 15k / 30k\nPotem różniczkowalna\nbramka komórka–Gauss", YELLOW)
    card(ax, 5, 10, 90, 11, fill=CARD2)
    txt(ax, 8, 17.5, "Warunek decyzji", 12.5, CYAN, "bold")
    txt(ax, 8, 13.5, "Nie wybieramy modelu tylko po PSNR: wymagamy pełnego dna w G-only i gładkiego medium w V-only.", 12.2, WHITE)
    txt(ax, 5, 5.2, "Nowy sweep jeszcze nie dał wyników: zadanie testowe i treningi są w kolejce Slurm.", 9.5, MUTED)
    pages.append(fig)

    pdf = OUT/"FoamSurf_stan_projektu_PL.pdf"
    with PdfPages(pdf) as pp:
        for i, fig in enumerate(pages, 1):
            pp.savefig(fig, dpi=150, facecolor=BG)
            fig.savefig(OUT/f"slajd_{i:02d}.png", dpi=130, facecolor=BG)
            plt.close(fig)
    print(pdf)


if __name__ == "__main__":
    make_deck()
