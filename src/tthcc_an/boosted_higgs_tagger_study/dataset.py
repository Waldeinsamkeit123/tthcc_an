from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np


def select_candidates(jets: ak.Array, args: Any) -> ak.Array:
    mask = (jets.pt >= args.pt_min) & (np.abs(jets.eta) <= args.eta_max)
    jets = jets[mask]

    if args.candidate_strategy in {"mass_window_all_jets", "mass_window_leading_pt"}:
        if args.msd_window_inclusive:
            msd_mask = (
                (jets.msoftdrop >= args.msd_window_low)
                & (jets.msoftdrop <= args.msd_window_high)
            )
        else:
            msd_mask = (
                (jets.msoftdrop > args.msd_window_low)
                & (jets.msoftdrop < args.msd_window_high)
            )
        jets = jets[msd_mask]

    if args.candidate_strategy in {"all_jets", "mass_window_all_jets"}:
        return jets

    ordering = ak.argsort(jets.pt, axis=1, ascending=False)
    jets = jets[ordering]
    return ak.singletons(ak.firsts(jets))
