#!/usr/bin/env python
"""
Convert the original SalsaNext training checkpoint into a weights-only file.

The original checkpoint (pretrained/SalsaNext, ~108 MB) also stores the optimizer and
learning-rate scheduler, so it can only be opened by full unpickling. This reads it with
a restricted unpickler that rebuilds tensors and plain containers only (no code from the
file runs), keeps the model weights (~26 MB) and saves them in a format that loads with
torch.load(weights_only=True).

Example:
  python scripts/convert_weights.py --checkpoint pretrained/SalsaNext --out weights/salsanext_weights.pth
"""

import argparse
import os
import pickle
import types

import torch

ALLOWED = {("collections", "OrderedDict"), ("collections", "defaultdict"), ("__builtin__", "dict"),
           ("builtins", "dict"), ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
           ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype"),
           ("_codecs", "encode")}


class _Inert:
    """Stand-in for any non-tensor object in the checkpoint (optimizer, scheduler)."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED or (module == "torch" and name.endswith("Storage")):
            return super().find_class(module, name)
        return type(name, (_Inert,), {})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="original SalsaNext checkpoint")
    p.add_argument("--out", required=True, help="weights-only output (.pth)")
    args = p.parse_args()

    restricted = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, __name__="restricted_pickle",
                                       load=lambda f, **kw: _RestrictedUnpickler(f, **kw).load())
    ckpt = torch.load(args.checkpoint, map_location="cpu", pickle_module=restricted, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v.clone() for k, v in state.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)

    back = torch.load(args.out, map_location="cpu", weights_only=True)
    assert back.keys() == state.keys() and all(torch.equal(back[k], state[k]) for k in state)
    n = sum(v.numel() for v in state.values() if v.is_floating_point())
    print(f"Saved {args.out}: {len(state)} tensors, {n:,} parameters, "
          f"{os.path.getsize(args.out) / 2**20:.1f} MB (loads with weights_only=True)")


if __name__ == "__main__":
    main()
