"""Does one forward pass over both photographs give the query tokens what two separate passes do not? New encodings.

Every method in the repository encodes reference and query apart and compares last-layer tokens afterwards. Here the
two square inputs are placed side by side and pass the frozen encoder once; the left and right halves are read as
reference and query tokens and go through the same evidence fields and the same instrument as inmask_evidence.py.
All arms are raw O24 unit tokens from this script at the same input side, device and code path (no positional debias):

  separate.*   reference and query encoded alone (control)
  joint.*      both in one pass
  masked.*     both in one pass, reference background pixels at the mean colour (zero after normalisation); --masked
  shift.*      how the reference's presence moved each query token: (joint - separate) toward the reference target
               minus toward the reference background; for masked, toward the visible target

--side is the side of each input. 768 gives a 1536 x 768 canvas of 4608 tokens, close to one ordinary 1024 input, and
fits the MPS process limit; 1024 gives the 2048 x 1024 canvas of 8192 tokens (about 15 s per pass on CPU, beyond an
11 GiB MPS limit). The control is encoded at the same side, so the comparison does not mix in a change of resolution.

  python3 evidence/local/false_alarm_floor_20261010/joint_encoding.py fields --cohort a/<cohort> --name joint_coco --limit 3 --side 768 --device mps
  python3 evidence/local/false_alarm_floor_20261010/inmask_evidence.py evaluate --cohort a/<cohort> --name joint_coco

Checked on the writing machine with a stand-in encoder only (--encoder fake): folders, halves, grids and fields.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inmask_evidence as base


class Fake(torch.nn.Module):
    """Stand-in with the encoder's call and output shape; for checking this file without weights."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.patch = torch.nn.Conv2d(3, 1024, 16, 16)

    def get_intermediate_layers(self, x, n=1, reshape=True):
        return [self.patch(x)]


def build(args):
    if args.encoder == 'fake':
        return Fake().eval()
    sys.path.insert(0, str(base.REPO / 'src'))
    from ics.data import TimmDINOv3
    return TimmDINOv3(str(Path(args.assets) / 'demo4_cache/models/dinov3-vitl16-timm')).eval().to(args.device)


def joint_fields(args):
    encoder, grid = build(args), args.side // 16
    transform = base.official_transform(args.assets, args.side)

    @torch.inference_mode()
    def tokens(x):                                            # (1, 3, H, W) -> (H / 16, W / 16, 1024) unit rows
        maps = encoder.get_intermediate_layers(x.to(args.device), n=1, reshape=True)[0]
        if args.device == 'mps':
            torch.mps.synchronize()
        return F.normalize(maps[0].float().cpu().permute(1, 2, 0), dim=2)

    def halves(x):
        both = tokens(x)
        assert tuple(both.shape) == (grid, 2 * grid, 1024), f'The encoder returned {tuple(both.shape)} for the side-by-side canvas'
        return both[:, :grid].reshape(-1, 1024), both[:, grid:].reshape(-1, 1024)

    def compute(source, rows, described, i):
        reference, mask, query = source.images(rows[i])
        xr, xq = transform(reference)[None].float(), transform(query)[None].float()
        assert tuple(xr.shape) == tuple(xq.shape) == (1, 3, args.side, args.side)
        coverage = torch.from_numpy(base.token_mass(mask, grid) / np.maximum(base.token_mass(np.ones_like(mask), grid), 1)).float()
        candidate = base.candidate_tokens(source.prediction(rows[i]), grid)
        started = time.monotonic()
        Rs, Qs = tokens(xr).reshape(-1, 1024), tokens(xq).reshape(-1, 1024)
        apart = time.monotonic() - started
        started = time.monotonic()
        Rj, Qj = halves(torch.cat((xr, xq), 3))
        together = time.monotonic() - started
        arms = [('separate', Rs, Qs), ('joint', Rj, Qj)]
        if args.masked:
            visible = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], (args.side, args.side), mode='nearest')
            arms.append(('masked', *halves(torch.cat((xr * visible, xq), 3))))
        fields, note = {}, {}
        for name, R, Q in arms:
            found, note = base.evidence(R, Q, coverage, candidate)
            fields.update({f'{name}.{k}': v for k, v in found.items() if not k.startswith('aux.')})
        target = base.target_tokens(coverage)
        background = (coverage < .05) & ~target
        if background.any():
            toward = F.normalize(Rj[target].mean(0), dim=0) - F.normalize(Rj[background].mean(0), dim=0)
            fields['shift.joint'] = ((Qj - Qs) @ toward).double().numpy()
        if args.masked:
            fields['shift.masked'] = ((arms[2][2] - Qs) @ F.normalize(arms[2][1][target].mean(0), dim=0)).double().numpy()
        return fields, dict(note, side=args.side, separate_seconds=round(apart, 3), joint_seconds=round(together, 3),
                            reference_tokens_kept=float((Rj * Rs).sum(1).mean()), query_tokens_kept=float((Qj * Qs).sum(1).mean()))

    base.make_fields(args, compute=compute, grid=grid)


if __name__ == '__main__':
    parser = base.arguments(['fields'])
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--side', type=int, default=768, help='side of each input in pixels, a multiple of 16')
    parser.add_argument('--masked', action='store_true', help='add the arm with the reference background blanked')
    parser.add_argument('--encoder', default='dinov3', choices=['dinov3', 'fake'])
    args = parser.parse_args()
    assert args.side % 16 == 0
    torch.set_num_threads(args.threads)
    joint_fields(args)
