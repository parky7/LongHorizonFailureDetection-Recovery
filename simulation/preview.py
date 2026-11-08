"""Export an annotated RGB video and RGB-D skill-boundary contact sheet."""
import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.episode) as file:
        steps = file["steps"]
        camera = next(iter(steps["000000/observations"]))
        boundaries = sorted(file["skills"].values(), key=lambda b: b.attrs["start_step"])
        samples = [int(b.attrs["end_step_exclusive"]) - 1 for b in boundaries
                   if b.attrs["end_step_exclusive"] > b.attrs["start_step"]]
        columns = min(3, len(samples))
        sheet = Image.new("RGB", (512 * columns, 296 * ((len(samples) + columns - 1) // columns)))
        with imageio.get_writer(args.output / "trajectory.mp4", fps=int(file.attrs["control_freq"]),
                                macro_block_size=1) as video:
            for i, step in enumerate(steps.values()):
                rgb = Image.fromarray(step[f"observations/{camera}/rgb"][()]).resize((256, 256))
                canvas = Image.new("RGB", (512, 296))
                canvas.paste(rgb, (0, 40))
                depth = step[f"observations/{camera}/depth"][()][..., 0].astype(float)
                # Fixed 0--2 m scale across all frames; invalid depth is black.
                gray = np.uint8(np.clip(depth / 2000, 0, 1) * 255)
                canvas.paste(Image.fromarray(gray).convert("RGB").resize((256, 256)), (256, 40))
                label = f"step={i} skill={step.attrs['skill']} | RGB / depth (0-2 m)"
                ImageDraw.Draw(canvas).text((8, 8), label, fill="white")
                ImageDraw.Draw(canvas).text((8, 24), f"oracle grasped={bool(step['oracle/is_grasped'][()])} task_success={bool(step['oracle/success'][()])}", fill="white")
                video.append_data(np.asarray(canvas))
                if i in samples:
                    index = samples.index(i)
                    sheet.paste(canvas, ((index % columns) * 512, (index // columns) * 296))
        sheet.save(args.output / "skill_boundaries.png")
    print(args.output / "trajectory.mp4")
    print(args.output / "skill_boundaries.png")


if __name__ == "__main__":
    main()
