"""Dedicated inverse projector for image errors and hair-region evidence."""

import torch
from . import _C


@torch.no_grad()
def project_errors(error_map, means3D, opacities, scales, rotations, *,
                   viewmatrix, projmatrix, image_height,
                   image_width, scale_modifier=1.0, raw_inputs=True,
                   tight_bbox=True, tile_culling=True, load_balance=True,
                   tile_culling_threshold=4, warp_culling=True,
                   separate_sort=True, hair_vote_map=None, frontmost_only=False,
                   debug=False):
    """Project image errors and optional hair evidence using render visibility.

    Returns per-primitive raw weighted error sums and visibility weights. With
    ``hair_vote_map`` of shape ``[2,H,W]`` also returns ``[P,2]`` hair evidence
    and confidence sums. It needs no SH, Gabor, background or autograd inputs.
    ``frontmost_only`` votes on the closest actual surfel intersection and skips
    errors; ordinary voting uses the renderer's alpha/transmittance weights.
    """
    empty = means3D.new_empty((0,))
    if frontmost_only:
        error_map = None
    if error_map is None and hair_vote_map is None:
        raise ValueError("Provide an error map or hair vote map")
    if error_map is not None:
        if error_map.ndim == 3:
            error_map = error_map.mean(dim=0)
        error_map = error_map.contiguous()
    args = (
        empty if error_map is None else error_map, means3D, opacities, scales,
        rotations, viewmatrix, projmatrix,
        int(image_height), int(image_width), float(scale_modifier), bool(raw_inputs),
        bool(tight_bbox), bool(tile_culling), bool(load_balance),
        int(tile_culling_threshold), bool(warp_culling), bool(separate_sort),
        empty if hair_vote_map is None else hair_vote_map.contiguous(),
        bool(frontmost_only), bool(debug),
    )
    _, errors, effects, votes = _C.project_errors(*args)
    return (errors, effects, votes) if hair_vote_map is not None else (errors, effects)

