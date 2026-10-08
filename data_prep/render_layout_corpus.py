"""Render an existing JSON layout corpus without modifying its training data."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env_generator import GridEnv, render_env


def render_one(task):
    # Audit guide:
    # Render walls, both colored goals, and both starting positions for a saved layout.
    # Coordinate placement follows the JSON row/column convention. A visual check can
    # reveal swapped starts/goals that a distribution plot would miss.
    #
    source, destination = map(Path, task)
    d = json.loads(source.read_text())
    grid = np.asarray(d["grid"])
    env = GridEnv(grid=(grid == 1).astype(np.int8),
        ego_start=tuple(d["ego_start"]), partner_start=tuple(d["partner_start"]),
        red_goal=tuple(d["red_goal"]), blue_goal=tuple(d["blue_goal"]),
        seed=d["seed"], wall_density=d["metadata"]["sampled_wall_probability"],
        metadata=d["metadata"])
    render_env(env, destination)
    with Image.open(destination) as im:
        im.verify()
    return source.stem


def digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def contact_sheet(files, destination, columns, thumb_size=180):
    # Audit guide:
    # Combine existing layout pictures into a labeled overview. The labels identify
    # source layouts for manual review; rendering does not alter training geometry.
    #
    rows = (len(files) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 25) + 45), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf", 14)
    draw.text((12, 12), "E = ego (yellow), P = partner (green); red/blue squares = goals", fill="black", font=font)
    for i, file in enumerate(files):
        x, y = (i % columns) * thumb_size, 45 + (i // columns) * (thumb_size + 25)
        with Image.open(file) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))
        draw.text((x + 10, y + thumb_size), file.stem, fill="black", font=font)
    canvas.save(destination)


def main():
    # Audit guide:
    # Choose saved layouts, write individual renders and contact sheets, and record file
    # digests. This is an inspection utility; it does not regenerate or filter the
    # corpus.
    #
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    layouts = sorted((corpus / "layouts/train").glob("*.json"))
    if not layouts:
        raise ValueError("No training layout JSON files found")
    protected = layouts + [corpus / "manifest.json", corpus / "train_layouts.npz"]
    before = digest(protected)
    renders = corpus / "renders/train"
    sheets = corpus / "renders/contact_sheets"
    renders.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(renders / (p.stem + ".png"))) for p in layouts]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(render_one, tasks), 1):
            if i % 100 == 0 or i == len(tasks):
                print(f"Rendered {i}/{len(tasks)} layouts", flush=True)
    images = [renders / (p.stem + ".png") for p in layouts]
    contact_sheet(images[:16], corpus / "renders/preview.png", columns=4, thumb_size=220)
    for page, offset in enumerate(range(0, len(images), 64), 1):
        contact_sheet(images[offset:offset + 64], sheets / f"page_{page:02d}.png", columns=8)
    cards = []
    for p in layouts:
        d = json.loads(p.read_text())
        m = d["metadata"]
        cards.append(f'<figure><a href="train/{p.stem}.png"><img loading="lazy" src="train/{p.stem}.png" alt="{html.escape(p.stem)}"></a>'
                     f'<figcaption><strong>{html.escape(p.stem)}</strong><br>'
                     f'Ego R/B: {m["ego_to_red"]}/{m["ego_to_blue"]} steps · Partner R/B: {m["partner_to_red"]}/{m["partner_to_blue"]} steps<br>'
                     f'Walls: {m["realized_wall_density"]:.1%}</figcaption></figure>')
    gallery = '<!doctype html><meta charset="utf-8"><title>Balanced 1096 layouts</title><style>body{font:15px system-ui;margin:24px;background:#fafafa}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}figure{margin:0;padding:12px;background:white;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}figcaption{line-height:1.6}</style>'
    gallery += f'<h1>{len(layouts):,} training layouts</h1><p>E = ego (yellow circle); P = partner (green circle). Red and blue squares are goals; dark cells are walls.</p><main>' + "\n".join(cards) + '</main>'
    (corpus / "renders/index.html").write_text(gallery)
    assert digest(protected) == before, "Training data changed during rendering"
    (corpus / "renders/render_manifest.json").write_text(json.dumps(dict(
        n_rendered=len(images), n_contact_sheets=(len(images) + 63) // 64,
        renderer="data_prep.env_generator.render_env", input_sha256=before,
        training_data_unchanged=True), indent=2) + "\n")
    print(f"Saved gallery and renders under {corpus / 'renders'}", flush=True)


if __name__ == "__main__":
    main()
