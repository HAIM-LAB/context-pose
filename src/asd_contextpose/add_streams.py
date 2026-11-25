"""Copy prepared NPZs and manifests, adding per-step feature streams.

Each stream is an array in a per-clip feature cache laid out like
`video_features` (and the EmotionCLIP features): `[3, steps, D]` for the
pre/event/post regions, beside a `tracked [3, steps]` flag. It is written as
`pre_<name>`/`event_<name>`/`post_<name>` [steps, D + 1] float32 (the features,
then the flag), the layout `train --stream NAME:D` reads. Clips without a
cache entry fail, so the copy covers exactly the source clips.

    python -m asd_contextpose.add_streams --data data/vitpose_face_det_eclip \
        --features dataset/VideoFeatures --streams vst hoghof \
        --output data/vitpose_face_det_video
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from .model import REGIONS

SPLITS = ("train", "val", "test")


def stream_arrays(cache: Path, names: list[str], steps: int) -> dict[str, np.ndarray]:
    with np.load(cache, allow_pickle=False) as data:
        flag = data["tracked"].astype(np.float32)[..., None]
        out = {}
        for name in names:
            values = data[name].astype(np.float32)
            if values.shape[:2] != (len(REGIONS), steps):
                raise ValueError(f"{cache}: {name} has shape {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{cache}: {name} has non-finite values")
            for region, array in zip(REGIONS, np.concatenate([values, flag], -1)):
                out[f"{region}_{name}"] = array
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="prepared data dir")
    parser.add_argument("--features", type=Path, required=True, help="feature cache")
    parser.add_argument("--streams", nargs="+", required=True, help="cache arrays")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.data.resolve():
        parser.error("--output must differ from --data")
    count = 0
    for split in SPLITS:
        manifest = args.data / f"{split}.csv"
        if not manifest.exists():
            continue
        (args.output / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(manifest, args.output / f"{split}.csv")
        for path in sorted((args.data / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as sample:
                arrays = dict(sample)
            steps = arrays["event_pose"].shape[0]
            # The feature cache is split by the clip's original folder.
            caches = [args.features / s / path.name for s in ("train", "test")]
            cache = next((c for c in caches if c.exists()), None)
            if cache is None:
                raise FileNotFoundError(f"no {args.features} features for {path.name}")
            arrays.update(stream_arrays(cache, args.streams, steps))
            np.savez(args.output / split / path.name, **arrays)
            count += 1
    print(f"{args.output}: {count} clips with {', '.join(args.streams)}")


if __name__ == "__main__":
    main()
