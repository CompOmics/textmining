# File-level disambiguation: HAMLET lists, MLMarker chooses (2026-09-14)

`scripts/validate_disambiguation.py` (truth), `scripts/draw_disambiguation_figure.py`
(`figure_file_level_disambiguation.{png,svg}`), `scripts/draw_concordance_figure.py`
(`figure_hamlet_vs_mlmarker.{png,svg}`, panels a-c).

## Ground truth (`disambiguation_truth.csv`)

| project | files | truth source | HAMLET list | in MLMarker training |
|---|---|---|---|---|
| PXD006401 | 14 | SDRF (the 10 kidney files are mouse and excluded) | tonsil, kidney | no |
| PXD048734 | 58 | `Type.xlsx`, tissue of origin of each cancer sample | 7 tissues | no |
| PXD020192 | 92 | cached SDRF, 40 organism parts | 11 tissues | yes |
| PXD010154 | 40 | cached SDRF, 28 tissues | 28 tissues | yes |
| PXD000561 | 85 | PRIDE SDRF (`sdrf(1).tsv`), 30 organism parts | 14 tissues | yes |

PXD010296 was removed: its "spleen" files are rat (`20180521_ratB*`,
`Spleen*`; HAMLET organism "rattus norvegicus; homo sapiens"), scored by a
human model; `build_run_labels.py` counted the ambiguous organism as human,
which is a bug to fix there. The lymph-node files never had a run-level
label, so nothing was compared on them.

## Result (`disambiguation_validation.csv`, 287 usable files)

| | files | truth in MLMarker's 34 classes | truth in HAMLET's list | chance 1/k | MLMarker alone | HAMLET list + MLMarker choice | choice, conf >= 0.3 |
|---|---|---|---|---|---|---|---|
| held-out (PXD006401, PXD048734) | 72 | 86 % | 86 % | 0.21 | 0.81 | **0.84** | 1.00 (9 files) |
| training (PXD020192, PXD010154, PXD000561) | 215 | 76 % | 58 % | 0.07 | 0.85 | 0.86 | 0.94 (93 files) |
| all | 287 | 78 % | 65 % | 0.11 | 0.83 | 0.86 | 0.94 (102 files) |

Accuracies are on files whose true tissue is in HAMLET's list, which is the
only case where "choose from the list" is defined. Where the true tissue is
an MLMarker class that HAMLET's list missed (39 files, 32 of them in
PXD020192 whose SDRF has 40 organism parts against HAMLET's 11), MLMarker
alone still names it in 54 %: the classifier can extend the list, not only
choose from it, but at half the reliability.

What neither can resolve: 22 % of files have a true tissue outside
MLMarker's classes (breast 12, gallbladder 5, NK / T cells 11, platelets,
spinal cord, retina, pancreas, trachea, cervix ...). These stay at HAMLET's
project-level list.

Per project: PXD006401 14/14; PXD048734 79 % of 48 (the 10 breast files
cannot be placed); PXD010154 92 % of 37; PXD000561 81 % of 53; PXD020192
88 % of 34 in-list files plus 32 files HAMLET's list missed.

## More projects?

Of the 15 human multi-tissue projects, 5 now have file-level truth. Three
more have usable coverage but no SDRF in the repository cache (PXD010271
brain/liver/ovary, 112 usable files; PXD005693 colon/liver; PXD048647
colon/rectum); they could be added if PRIDE has an SDRF or the run names can
be decoded (PXD010271's `61928_PT_S1_...` cannot). The remaining six have no
usable coverage (PXD050416, PXD055814, PXD056300, PXD068664) or are
mouse/human mixtures (PXD058971, PXD056689).
