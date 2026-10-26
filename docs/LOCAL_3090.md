# Uruchamianie na stacji z RTX 3090 (bez Slurma)

## Raz, po `git pull`

```bash
cd /home/grzegos/projects/phd/foamsurf/foamsurf
git pull                      # branch dev/medium_fix
./build_3090.sh               # renderer + topologia pod sm_86 (kernele zmieniły się od c9d96d9)
python -m pytest -q tests/    # GPU testy też, w tym audyt gradientu renderera
```

## Zestawy wariantów jak na klastrze

`scripts/run_local.sh` puszcza zadania `scripts/train_seathru.sbatch` po kolei na jednym GPU.
Ścieżka do datasetu jest wymagana:

```bash
export FS_DATASETS=/sciezka/do/SeathruNeRF_dataset     # folder z IUI3-RedSea-Undistorted, ..., Panama-Undisorted
FS_VARIANT_SET=diag FS_SLIM=0 FS_RUN_ID=diag scripts/run_local.sh 0 15
```

Domyślne ścieżki stacji: `FS_ENV=/home/grzegos/.conda/envs/foamsurf`, `FS_DRIVER=/usr/lib/x86_64-linux-gnu`,
`TORCH_CUDA_ARCH_LIST=8.6`; każdą można nadpisać. Wyniki: `outputs/seathru/<FS_RUN_ID>/`, logi:
`slurm_logs/seathru_<FS_RUN_ID>_<zadanie>.{out,err}`. Padnięte zadanie jest logowane, a następne startuje.

## Zestaw `diag`: co ciągnie medium w stronę przezroczystości

16 runów po 10 000 iteracji (JG i Curaçao, seed 42): `_minit` oraz `_minit` z wyłączonym jednym
elementem (`_nooverlap`, `_nochroma`, `_nocover`, `_noclear`, `_nobin`, `_nosmooth`) i `_plain_minit`
(wszystko wyłączone). `FS_SLIM=0` zostawia rendery z każdej ewaluacji (co 500).

Pomiar tłumienia przed dnem względem WaterSplatting (referencja WS jest wpisana w skrypt):

```bash
python scripts/medium_attenuation.py --datasets "$FS_DATASETS" \
    --runs outputs/seathru/diag/* --iterations 500 2000 5000 10000 --output outputs/seathru/diag/attenuation.json
```

Kontekst i dotychczasowe liczby: `outputs/seathru/report_findings_2026-10-26.html` (na klastrze).
