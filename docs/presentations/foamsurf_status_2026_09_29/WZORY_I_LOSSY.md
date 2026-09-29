# Notatki do slajdów 3–6

Wzory dotyczą ukończonych runów RGB IUI3 (513211, 513646). „Nowy overlap” jest przygotowany, ale nie ma jeszcze wyniku treningowego. Wszystkie parametry medium oznaczone pogrubieniem są wektorami RGB; działania na nich są kanałowe.

## Renderer

Na promieniu przeplatane są odcinki Voronoi (komórka \(i\), długość \(\ell_i\)) i trafienia Gaussów (indeks \(k\), głębokość \(t_k\)). Komórka ma efektywne tłumienie \(\mathbf a_i\), współczynnik narastania backscatter \(\mathbf b_i\) oraz kolor źródłowy \(\mathbf B_i\). Gauss ma opacity \(\alpha_k\) i kolor \(\mathbf c_k\).

\[
\mathbf T_D\leftarrow\mathbf T_D e^{-\mathbf a_i\ell_i},\quad
\mathbf T_B\leftarrow\mathbf T_B e^{-\mathbf b_i\ell_i},\quad
T_G\leftarrow T_G(1-\alpha_k).
\]

\[
\mathbf D=\sum_k T_G(t_k)\mathbf T_D(t_k)\alpha_k\mathbf c_k
+T_G(t_f)\mathbf T_D(t_f)\mathbf C_{\mathrm{bg}},
\]
\[
\mathbf S=\sum_i T_G(s_i)\mathbf T_B(s_i)\mathbf B_i
 \left(1-e^{-\mathbf b_i\ell_i}\right),\quad
\hat{\mathbf I}=\mathbf D+\mathbf S.
\]

Tu \(s_i\) jest początkiem odcinka, a \(t_f\) końcem promienia. Wkład scatter liczy się przy transmisji na początku odcinka, a direct Gaussa po poprzedzającym go odcinku. **Direct zawiera też tłumione tło** — nie jest renderem samych Gaussów. Kod referencyjny: renderer/foamsurf/rgb_medium.py, funkcja reference_segments.

## Sampling treningowy

training/sampler.py wybiera jedną kamerę treningową na iterację. W prezentowanych runach kamery są tasowane bez powtórzeń w cyklu. Ponieważ full_image_batch=true, z tej jednej kamery pobierane są wszystkie piksele, czyli 1383×917 = 1 268 211 promieni na krok. 25 widoków jest w train, 4 w test; test nie bierze udziału w optymalizacji. Wartość rays_per_batch=640000 nie ogranicza batcha, gdy pełny obraz jest włączony.

## Straty bazowe

Rekonstrukcja: \(\mathcal L_{\rm RGB}=0.8\operatorname{mean}|\hat I-I|+0.2[1-\operatorname{SSIM}(\hat I,I)]\). SSIM jest liczony na całym obrazie. MSE jest mierzone dla PSNR, a nie używane jako ta strata (training/losses.py).

Binaryzacja: \(\mathcal L_{\rm bin}=\lambda_{\rm bin}(t)N_G^{-1}\sum_j\alpha_j(1-\alpha_j)\). Minimum jest zarówno przy opacity 0, jak i 1; loss sam nie wybiera nieprzezroczystości. Waga rośnie cosinusowo od 1k do 4k do 0.001 (training/disentanglement.py i konfiguracja runu 13).

Skala: \(\mathcal L_{\rm scale}=0.001N_G^{-1}\sum_j\|\mathbf s_j\|_2\). Ogranicza rozmiar Gaussów, ale nie tworzy brakującej geometrii (training/trainer.py).

Opacity pruning to **nie loss**. Próg rośnie cosinusowo z 0.005 do 0.10; usuwanie co 250 kroków od 1k do 30k (z limitem usuwanej frakcji). Densyfikacja Gaussów kończy się przy 15k.

## Straty eksperymentalne

Graph TV / Huber (S): dla nieskierowanych sąsiedztw \(\mathcal E\), przy wadze każdej krawędzi równej 1,
\[
\mathcal L_S=r(t)\sum_{q\in\{a,b,B\}}w_q\,
\operatorname{mean}_{(i,j)\in\mathcal E,c}
H_{\delta_q}(q'_{i,c}-q'_{j,c}),
\]
gdzie \(q'=\log(q+10^{-6})\) dla \(a,b\), a \(q'=B\). Huber: \(H_\delta(z)=z^2/2\) dla \(|z|\le\delta\), w przeciwnym razie \(\delta(|z|-\delta/2)\). Silne TV ma wagi \((0.1,0.1,1)\), delty \((0.1,0.1,0.05)\) i rampę 250 kroków. Wygładza aktywowane parametry optyczne, bez gradientu przez dyskretną topologię grafu (experiments/rgb_smooth/loss.py).

Stary overlap (O): na 128 dodatkowych promieniach próbkowanych w 1024 binach głębokości obliczana jest masa backscatter \(m_{r,s}\), zależna od \(T_G T_B\mathbf B(1-e^{-\mathbf b\Delta})\). Z niej pochodzą: \(C_r\) — koncentracja w głębi; \(M_r\) — siła całego backscatter; \(F_r\) — jego frakcja blisko średniej głębokości Gaussów; \(Q_r=A_r e^{-\operatorname{Var}_G/h^2}\) — pewność zwartej powierzchni Gaussów. Minimalizowano \(\mathcal L_O^{\rm old}=\lambda_O(t)\operatorname{mean}_r(C_rM_rF_rQ_r)\). Gradient trafia tylko do \(\mathbf b,\mathbf B\), a **nie** do \(\mathbf a\), sites ani Gaussów; łatwo zmniejszyć karę przez osłabienie źródła backscatter (experiments/rgb_disentangle/losses.py).

Gaussian opacity (G): \(A_r\) to łączna alpha obrazu z samych Gaussów na 128 pomocniczych promieniach. Dla maski \(m_r=\mathbf1[A_r>0.1]\), odłączonej od gradientu,
\[
\mathcal L_G=\lambda_G(t)
\frac{\sum_r m_r(1-A_r)}{\max(1,\sum_r m_r)}.
\]
Gdy maska jest pusta, strata wynosi zero. Zwiększa opacity tam, gdzie Gaussy już są, ale nie wymusza wypełnienia pustych promieni (experiments/rgb_disentangle/losses.py).

Nowy lokalny overlap (O, **jeszcze bez wyników**): z trafień G obliczamy średnią głębokość \(\mu_r\), wariancję i odłączoną pewność \(Q_r=A_r e^{-\operatorname{Var}_G/h^2}\). W oknie \(W_r\) długości \(2h\) wokół \(\mu_r\), \(h=0.02L\), całkujemy pomocnicze pole:
\[
k_i=\frac{\operatorname{mean}_{RGB}(\mathbf a_i+\mathbf b_i+\mathbf b_i\mathbf B_i)}{3s},\quad
\tau_r=\int_{W_r}k(x)\,dx,\quad
\mathcal L_O^{\rm new}=\lambda_O(t)\operatorname{mean}_r[Q_r(1-e^{-\tau_r})].
\]
To **proxy konfliktu**, nie nowy fizyczny parametr wody. Gradient trafia do \(\mathbf a,\mathbf b,\mathbf B\) odwiedzonych komórek i do sites przez długości odcinków; Gaussy są odłączone. Nieodwiedzone komórki nie dostają gradientu (experiments/rgb_surface/loss.py i renderer.py).

Czarne tło jest osobnym ustawieniem renderera, **nie** lossem. Zeruje składnik \(T_GT_D\mathbf C_{\rm bg}\) z direct, nie zmieniając referencyjnych zdjęć RGB IUI3.
