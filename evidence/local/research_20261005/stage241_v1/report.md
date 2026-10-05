# Stage bank: 241 episodes, origin model.raw_nn 42.90, INSID3 rule 54.47, FoRIS pre-CRF 58.61, complete FoRIS 59.07

Rebuilt FoRIS pre-CRF differs from the cached one in 72736 pixels over 29 episodes.

## Every mask as an edit of the origin

| mask | mIoU | additions alone | deletions alone | added true | added false | deleted false | deleted true |
|---|---:|---:|---:|---:|---:|---:|---:|
| foris.pre-vote | 59.36 | 48.54 | 53.38 | 4,440,645 | 4,238,558 | 13,015,588 | 1,991,547 |
| native | 59.07 | 48.78 | 52.77 | 4,137,295 | 3,355,996 | 11,744,438 | 1,576,922 |
| foris.pre-bg | 58.71 | 48.30 | 52.90 | 4,027,770 | 3,317,143 | 12,093,549 | 1,536,298 |
| foris.pre | 58.61 | 48.59 | 52.68 | 4,124,377 | 3,489,447 | 11,529,841 | 1,445,729 |
| foris.pre-penalty | 57.88 | 48.53 | 52.40 | 4,454,265 | 4,126,907 | 11,048,668 | 1,167,451 |
| foris.pre-delta | 57.35 | 48.18 | 51.97 | 4,052,560 | 3,494,617 | 11,054,259 | 1,204,766 |
| foris.s4_penalty | 57.35 | 48.18 | 51.97 | 4,052,560 | 3,494,617 | 11,054,259 | 1,204,766 |
| foris.pre-prior | 56.51 | 47.56 | 51.87 | 3,569,156 | 3,335,368 | 11,078,522 | 1,516,485 |
| foris.s3 | 56.36 | 48.34 | 51.18 | 4,694,125 | 4,527,590 | 9,954,564 | 884,417 |
| foris@never.pre | 55.64 | 46.99 | 51.35 | 2,460,910 | 2,265,493 | 10,370,822 | 1,815,126 |
| foris@never.s4_penalty | 55.48 | 46.97 | 51.28 | 2,583,751 | 2,215,929 | 9,719,652 | 1,224,918 |
| insid3.final | 54.47 | 47.15 | 49.99 | 3,528,540 | 2,706,411 | 12,385,849 | 2,849,100 |
| foris@never.s3 | 54.34 | 47.24 | 50.28 | 3,363,141 | 3,193,917 | 8,175,299 | 716,431 |
| foris.s3_vote | 52.20 | 46.61 | 49.08 | 3,894,491 | 5,074,523 | 8,304,233 | 859,060 |
| foris.s2 | 51.94 | 44.70 | 51.70 | 4,983,309 | 10,646,776 | 11,110,518 | 1,425,115 |
| foris@never.s3_vote | 48.91 | 45.03 | 47.33 | 2,704,226 | 4,674,629 | 5,587,620 | 574,344 |
| foris@never.s2 | 48.81 | 43.14 | 50.13 | 4,232,518 | 13,819,688 | 8,650,038 | 1,377,280 |
| insid3.candidates | 44.29 | 42.66 | 45.18 | 2,381,049 | 6,416,258 | 4,955,579 | 902,412 |
| insid3.vote | 44.29 | 42.66 | 45.18 | 2,381,049 | 6,416,258 | 4,955,579 | 902,412 |
| model.raw_nn | 42.90 | 42.90 | 42.90 | 0 | 0 | 0 | 0 |
| insid3.seed | 42.51 | 45.54 | 39.56 | 2,199,371 | 1,943,056 | 15,518,506 | 10,087,758 |
| model.raw_mean | 39.56 | 38.87 | 44.64 | 3,827,664 | 12,964,867 | 4,183,912 | 780,389 |

## Chain: insid3

| step | mIoU | added true | added false | deleted false | deleted true |
|---|---|---:|---:|---:|---:|
| model.raw_nn -> insid3.vote | 42.90 -> 44.29 | 2,381,049 | 6,416,258 | 4,955,579 | 902,412 |
| insid3.vote -> insid3.candidates | 44.29 -> 44.29 | 0 | 0 | 0 | 0 |
| insid3.candidates -> insid3.seed | 44.29 -> 42.51 | 1,910,580 | 1,703,516 | 16,739,645 | 11,277,604 |
| insid3.seed -> insid3.final | 42.51 -> 54.47 | 8,567,827 | 3,896,012 | 0 | 0 |

## Chain: foris

| step | mIoU | added true | added false | deleted false | deleted true |
|---|---|---:|---:|---:|---:|
| model.raw_nn -> foris.s2 | 42.90 -> 51.94 | 4,983,309 | 10,646,776 | 11,110,518 | 1,425,115 |
| foris.s2 -> foris.s3_vote | 51.94 -> 52.20 | 1,183,214 | 5,104,440 | 7,870,408 | 1,705,977 |
| foris.s3_vote -> foris.s3 | 52.20 -> 56.36 | 1,221,895 | 1,563,428 | 3,760,692 | 447,618 |
| foris.s3 -> foris.s4_penalty | 56.36 -> 57.35 | 4,435 | 53,441 | 2,186,109 | 966,349 |
| foris.s4_penalty -> foris.pre | 57.35 -> 58.61 | 305,013 | 707,325 | 1,188,077 | 474,159 |
| foris.pre -> native | 58.61 -> 59.07 | 387,079 | 539,481 | 887,529 | 505,354 |

## Family, chosen on three folds and read on the fourth

- `family.nested` 57.35: -1.73 [-2.80, +0.05] vs native; -1.26 [-2.23, +0.69] vs foris.pre.control; +2.88 [+0.23, +5.49] vs insid3.final.control; +14.44 [+11.99, +18.62] vs model.raw_nn.control; -1.44 [-2.20, +0.25] vs family.threshold_only.control; -1.54 [-2.42, +0.35] vs family.lines_only.control; -1.41 [-2.25, +0.21] vs family.foris_terms_only.control. Chosen: {'0': 'random823', '1': 'random823', '2': 'random218', '3': 'random71'}
- `family.threshold_only.control` 58.79: -0.29 [-1.24, +0.47] vs native; +0.18 [-0.64, +1.06] vs foris.pre.control; +4.32 [+1.19, +6.50] vs insid3.final.control; +15.88 [+13.31, +19.14] vs model.raw_nn.control; -0.09 [-1.06, +0.92] vs family.lines_only.control; +0.04 [-0.87, +0.68] vs family.foris_terms_only.control. Chosen: {'0': 'foris:t=0.55', '1': 'foris:t=0.6', '2': 'foris:t=0.55', '3': 'foris:t=0.55'}
- `family.lines_only.control` 58.88: -0.19 [-1.41, +0.66] vs native; +0.28 [-0.73, +1.23] vs foris.pre.control; +4.41 [+1.21, +6.51] vs insid3.final.control; +15.98 [+13.22, +19.35] vs model.raw_nn.control; +0.09 [-0.92, +1.06] vs family.threshold_only.control; +0.13 [-0.78, +0.58] vs family.foris_terms_only.control. Chosen: {'0': 'foris:foris.vote x0', '1': 'foris:foris.vote x0.25', '2': 'foris:foris.vote x0', '3': 'foris:foris.prior x2'}
- `family.foris_terms_only.control` 58.75: -0.32 [-1.15, +0.51] vs native; +0.15 [-0.54, +1.10] vs foris.pre.control; +4.28 [+1.29, +6.49] vs insid3.final.control; +15.85 [+13.30, +19.34] vs model.raw_nn.control; -0.04 [-0.68, +0.87] vs family.threshold_only.control; -0.13 [-0.58, +0.78] vs family.lines_only.control. Chosen: {'0': 'random783', '1': 'foris:foris.vote x0.25', '2': 'random783', '3': 'foris:foris.prior x2'}

Best point in sample (optimistic): random823 59.92.

## Component lines (all episodes, no selection)

| point | mIoU | vs FoRIS pre-CRF | vs INSID3 rule |
|---|---:|---:|---:|
| foris.pre | 58.61 | -0.00 | +4.14 |
| insid3.final | 54.47 | -4.14 | +0.00 |
| model.raw_nn | 42.90 | -15.70 | -11.57 |
| model.raw_mean | 39.56 | -19.05 | -14.91 |
| foris:foris.fg x0 | 53.45 | -5.16 | -1.02 |
| foris:foris.fg x0.25 | 56.15 | -2.46 | +1.68 |
| foris:foris.fg x0.5 | 57.56 | -1.05 | +3.09 |
| foris:foris.fg x0.75 | 58.42 | -0.18 | +3.95 |
| foris:foris.fg x1.5 | 58.64 | +0.04 | +4.17 |
| foris:foris.fg x2 | 58.57 | -0.04 | +4.10 |
| foris:foris.fg x3 | 57.27 | -1.34 | +2.80 |
| foris:foris.bg x0 | 58.71 | +0.10 | +4.24 |
| foris:foris.bg x0.25 | 58.95 | +0.34 | +4.48 |
| foris:foris.bg x0.5 | 58.90 | +0.30 | +4.43 |
| foris:foris.bg x0.75 | 58.86 | +0.25 | +4.39 |
| foris:foris.bg x1.5 | 58.07 | -0.54 | +3.60 |
| foris:foris.bg x2 | 56.93 | -1.68 | +2.46 |
| foris:foris.bg x3 | 54.45 | -4.16 | -0.02 |
| foris:foris.vote x0 | 59.36 | +0.75 | +4.89 |
| foris:foris.vote x0.25 | 59.37 | +0.76 | +4.90 |
| foris:foris.vote x0.5 | 59.12 | +0.52 | +4.65 |
| foris:foris.vote x0.75 | 58.88 | +0.28 | +4.41 |
| foris:foris.vote x1.5 | 57.78 | -0.82 | +3.31 |
| foris:foris.vote x2 | 55.65 | -2.95 | +1.18 |
| foris:foris.vote x3 | 51.82 | -6.78 | -2.65 |
| foris:foris.prior x0 | 56.51 | -2.09 | +2.04 |
| foris:foris.prior x0.25 | 57.61 | -0.99 | +3.14 |
| foris:foris.prior x0.5 | 58.05 | -0.55 | +3.58 |
| foris:foris.prior x0.75 | 58.33 | -0.28 | +3.86 |
| foris:foris.prior x1.5 | 59.12 | +0.52 | +4.65 |
| foris:foris.prior x2 | 59.28 | +0.68 | +4.81 |
| foris:foris.prior x3 | 58.64 | +0.03 | +4.17 |
| foris:foris.penalty x0 | 57.88 | -0.72 | +3.41 |
| foris:foris.penalty x0.25 | 58.16 | -0.44 | +3.69 |
| foris:foris.penalty x0.5 | 58.35 | -0.25 | +3.88 |
| foris:foris.penalty x0.75 | 58.50 | -0.10 | +4.03 |
| foris:foris.penalty x1.5 | 58.73 | +0.12 | +4.26 |
| foris:foris.penalty x2 | 58.67 | +0.06 | +4.20 |
| foris:foris.penalty x3 | 58.67 | +0.07 | +4.21 |
| foris:foris.delta x0 | 57.35 | -1.26 | +2.88 |
| foris:foris.delta x0.25 | 57.89 | -0.72 | +3.42 |
| foris:foris.delta x0.5 | 58.29 | -0.32 | +3.82 |
| foris:foris.delta x0.75 | 58.53 | -0.07 | +4.06 |
| foris:foris.delta x1.5 | 58.70 | +0.09 | +4.23 |
| foris:foris.delta x2 | 58.53 | -0.08 | +4.06 |
| foris:foris.delta x3 | 58.02 | -0.58 | +3.55 |
| foris:t=0.3 | 48.09 | -10.51 | -6.38 |
| foris:t=0.35 | 51.96 | -6.65 | -2.51 |
| foris:t=0.4 | 54.92 | -3.68 | +0.45 |
| foris:t=0.45 | 57.44 | -1.16 | +2.97 |
| foris:t=0.55 | 59.08 | +0.47 | +4.61 |
| foris:t=0.6 | 58.79 | +0.19 | +4.32 |
| foris:t=0.65 | 57.66 | -0.94 | +3.19 |
| foris:t=0.7 | 55.49 | -3.11 | +1.02 |
| foris:render=mask | 57.54 | -1.07 | +3.07 |
| foris:fixed_point a=4 b=0.5 | 31.08 | -27.53 | -23.39 |
| foris:fixed_point a=8 b=0.5 | 52.70 | -5.91 | -1.77 |
| foris:fixed_point a=16 b=0.5 | 57.64 | -0.96 | +3.17 |
| foris:fixed_point a=8 b=0.4 | 43.35 | -15.26 | -11.12 |
| foris:fixed_point a=8 b=0.6 | 56.92 | -1.68 | +2.45 |
| foris:fixed_point a=16 b=0.4 | 52.78 | -5.83 | -1.69 |
| foris:fixed_point a=16 b=0.6 | 58.62 | +0.01 | +4.15 |
| insid3:insid3.lcross ^0 | 50.35 | -8.25 | -4.12 |
| insid3:insid3.lcross ^0.5 | 54.68 | -3.93 | +0.21 |
| insid3:insid3.lcross ^1.5 | 51.34 | -7.27 | -3.13 |
| insid3:insid3.lcross ^2 | 47.51 | -11.10 | -6.96 |
| insid3:insid3.lcross ^3 | 30.35 | -28.25 | -24.12 |
| insid3:insid3.lintra ^0 | 51.87 | -6.73 | -2.60 |
| insid3:insid3.lintra ^0.5 | 54.70 | -3.90 | +0.23 |
| insid3:insid3.lintra ^1.5 | 53.24 | -5.36 | -1.23 |
| insid3:insid3.lintra ^2 | 52.78 | -5.82 | -1.69 |
| insid3:insid3.lintra ^3 | 48.85 | -9.76 | -5.62 |
| insid3:insid3.larea ^0 | 41.82 | -16.78 | -12.65 |
| insid3:insid3.larea ^0.5 | 54.72 | -3.88 | +0.25 |
| insid3:insid3.larea ^1.5 | 53.61 | -4.99 | -0.86 |
| insid3:insid3.larea ^2 | 52.62 | -5.99 | -1.85 |
| insid3:insid3.larea ^3 | 51.80 | -6.81 | -2.67 |
| insid3:t=0.05 | 48.84 | -9.77 | -5.63 |
| insid3:t=0.1 | 53.53 | -5.08 | -0.94 |
| insid3:t=0.15 | 54.22 | -4.39 | -0.25 |
| insid3:t=0.25 | 53.37 | -5.24 | -1.10 |
| insid3:t=0.3 | 50.46 | -8.14 | -4.01 |
| insid3:t=0.4 | 47.69 | -10.92 | -6.78 |
| insid3:render=field | 53.16 | -5.44 | -1.30 |
| model.raw_nn:t=-0.1 | 25.25 | -33.35 | -29.22 |
| model.raw_nn:t=-0.05 | 33.75 | -24.86 | -20.72 |
| model.raw_nn:t=-0.02 | 40.23 | -18.37 | -14.23 |
| model.raw_nn:t=0.02 | 41.20 | -17.41 | -13.27 |
| model.raw_nn:t=0.05 | 32.18 | -26.43 | -22.29 |
| model.raw_nn:t=0.1 | 15.63 | -42.97 | -38.84 |
| model.raw_mean:t=-0.1 | 26.70 | -31.91 | -27.77 |
| model.raw_mean:t=-0.05 | 33.70 | -24.91 | -20.77 |
| model.raw_mean:t=-0.02 | 37.34 | -21.27 | -17.13 |
| model.raw_mean:t=0.02 | 42.15 | -16.46 | -12.32 |
| model.raw_mean:t=0.05 | 44.79 | -13.82 | -9.68 |
| model.raw_mean:t=0.1 | 45.93 | -12.68 | -8.54 |
| foris+insid3.lcross x0.1 | 58.69 | +0.09 | +4.22 |
| foris+insid3.lcross x0.25 | 58.60 | -0.01 | +4.13 |
| foris+insid3.lcross x0.5 | 58.17 | -0.44 | +3.70 |
| foris+insid3.lcross x1 | 56.59 | -2.01 | +2.12 |
| foris+insid3.lintra x0.1 | 58.82 | +0.22 | +4.35 |
| foris+insid3.lintra x0.25 | 58.77 | +0.16 | +4.30 |
| foris+insid3.lintra x0.5 | 58.61 | +0.01 | +4.14 |
| foris+insid3.lintra x1 | 57.53 | -1.08 | +3.06 |
| foris+insid3.larea x0.1 | 58.49 | -0.12 | +4.02 |
| foris+insid3.larea x0.25 | 57.94 | -0.66 | +3.48 |
| foris+insid3.larea x0.5 | 56.35 | -2.25 | +1.88 |
| foris+insid3.larea x1 | 52.91 | -5.70 | -1.56 |
| foris+model.raw_nn x0.1 | 58.39 | -0.21 | +3.92 |
| foris+model.raw_nn x0.25 | 57.39 | -1.21 | +2.92 |
| foris+model.raw_nn x0.5 | 55.21 | -3.39 | +0.74 |
| foris+model.raw_nn x1 | 50.22 | -8.39 | -4.25 |
| foris+model.raw_mean x0.1 | 58.56 | -0.04 | +4.09 |
| foris+model.raw_mean x0.25 | 58.41 | -0.19 | +3.94 |
| foris+model.raw_mean x0.5 | 57.94 | -0.67 | +3.47 |
| foris+model.raw_mean x1 | 56.35 | -2.25 | +1.88 |
