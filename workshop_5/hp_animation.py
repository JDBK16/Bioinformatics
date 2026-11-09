"""
Workshop 6, Extension (b): animate simulated annealing on the HP lattice.

Left panel : the current fold drawn on a fixed 2n x 2n window of the square
             lattice (x and y from -n to n), so the chain can never leave
             the frame: H residues are red, P residues white, and H-H
             contacts are dashed lines.
Right panel: the current and best energies against iteration, with the
             temperature on a second axis and a cursor marking the frame.

Command line:
    python hp_animation.py                                  # 20-mer, GIF
    python hp_animation.py HHPPHPPHPPHPPHPPHPPHPPHH --iters 100000 --out sa.mp4
    python hp_animation.py --module hp_folding              # use YOUR code

Needs matplotlib (plus pillow for .gif, or ffmpeg for .mp4).
"""

import argparse
import importlib
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

H_COLOUR = "#c8402f"
P_COLOUR = "#ffffff"
EDGE = "#333333"
CHAIN = "#555555"
BEST = "#c8402f"
CURRENT = "#9aa0a6"
TEMP = "#2a6fb0"


def contact_segments(sequence, coords):
    """Line segments joining every pair of non-consecutive, adjacent H's."""
    where = {c: i for i, c in enumerate(coords)}
    segs = []
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for nb in ((x + 1, y), (x, y + 1)):          # each pair counted once
            j = where.get(nb)
            if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                segs.append([(x, y), nb])
    return segs


def animate_sa(sequence, n_iter=100000, n_frames=200, fps=20, seed=3,
               save="sa_animation.gif", module="hp_folding_solution",
               hold_seconds=3, optimum=None, **sa_kwargs):
    """Run simulated annealing on `sequence` and animate it.

    n_frames    : roughly how many snapshots to show (sampled evenly).
    save        : output filename (.gif or .mp4), or None to just return
                  the animation object (e.g. for Jupyter).
    module      : which implementation to use; it must provide
                  simulated_annealing(..., history=[], history_every=k)
                  and generate_coords().
    optimum     : known optimal energy, drawn as a reference line.
    hold_seconds: how long to show the best fold found at the end.
    Extra keyword arguments (T0, alpha, T_min, ...) go to
    simulated_annealing.

    Returns the matplotlib FuncAnimation.
    """
    hp = importlib.import_module(module)
    n = len(sequence)

    history = []
    every = max(1, n_iter // n_frames)
    try:
        result = hp.simulated_annealing(
            sequence, n_iter=n_iter, rng=random.Random(seed),
            history=history, history_every=every, **sa_kwargs)
    except TypeError as err:
        if "history_every" not in str(err):
            raise
        # older signature without history_every: record everything we get,
        # then thin it out below
        history.clear()
        result = hp.simulated_annealing(
            sequence, n_iter=n_iter, rng=random.Random(seed),
            history=history, **sa_kwargs)
    best_e, best_t = result[0], result[1]

    if not history:
        raise ValueError(
            f"{module}.simulated_annealing did not record anything in "
            "`history`. It must append (iteration, current_energy, "
            "best_energy, T, current_turns) as it runs.")
    if len(history[0]) < 5:
        raise ValueError(
            f"{module}.simulated_annealing records history entries with "
            f"{len(history[0])} values, e.g. {history[0]!r}, but the "
            "animation also needs the current fold: append\n"
            "    (iteration, current_energy, best_energy, T, "
            "''.join(turns))\n"
            "If you are using hp_folding_solution.py, replace it with the "
            "latest version.")
    if len(history) > 2 * n_frames:            # thin out to ~n_frames
        step = len(history) // n_frames
        history = history[::step] + ([history[-1]]
                                     if (len(history) - 1) % step else [])
    # copy the turns in case the caller appended a list it later mutated
    history = [(h[0], h[1], h[2], h[3], "".join(h[4])) for h in history]

    its = [h[0] for h in history]
    cur_e = [h[1] for h in history]
    bst_e = [h[2] for h in history]
    temps = [h[3] for h in history]
    # frames: one per snapshot, then the best fold held at the end
    frames = [(k, h[4], False) for k, h in enumerate(history)]
    frames += [(len(history) - 1, best_t, True)] * int(hold_seconds * fps)

    # ---------------------------------------------------------- figure
    fig, (ax, ax_e) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.09, top=0.89,
                        wspace=0.16)
    fig.suptitle(f"Simulated annealing, HP model, n = {n}\n{sequence}",
                 fontsize=11, family="monospace")

    # lattice: 2n x 2n window centred on residue 0, so any fold fits
    ax.set_xlim(-n - 0.5, n + 0.5)
    ax.set_ylim(-n - 0.5, n + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(-n, n + 1))
    ax.set_yticks(range(-n, n + 1))
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.grid(color="#e2e2e2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_color("#bbbbbb")

    # size beads and bonds relative to one lattice cell (in points)
    box = ax.get_position()                  # axes box in figure fraction
    side_in = min(box.width * fig.get_figwidth(),
                  box.height * fig.get_figheight())
    cell = side_in * 72 / (2 * n + 1)        # points per lattice step
    node = (0.55 * cell) ** 2                # bead diameter = 55% of a step
    bond = max(1.0, 0.16 * cell)

    chain_line, = ax.plot([], [], "-", color=CHAIN, lw=bond, zorder=2,
                          solid_capstyle="round")
    contacts = LineCollection([], colors=H_COLOUR, linestyles=(0, (2, 1.5)),
                              lw=max(1.0, 0.1 * cell), zorder=1)
    ax.add_collection(contacts)
    edge_w = max(0.6, 0.06 * cell)
    h_dots = ax.scatter([], [], s=node, c=H_COLOUR, edgecolors=EDGE,
                        linewidths=edge_w, zorder=3)
    p_dots = ax.scatter([], [], s=node, c=P_COLOUR, edgecolors=EDGE,
                        linewidths=edge_w, zorder=3)
    start = ax.scatter([], [], s=node * 2.2, facecolors="none",
                       edgecolors="#222222", linewidths=edge_w * 1.5,
                       zorder=4)
    label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                    fontsize=11, family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc",
                              alpha=0.9))
    ax.scatter([], [], s=60, c=H_COLOUR, edgecolors=EDGE, label="H")
    ax.scatter([], [], s=60, c=P_COLOUR, edgecolors=EDGE, label="P")
    ax.plot([], [], "--", color=H_COLOUR, label="H–H contact")
    ax.scatter([], [], s=120, facecolors="none", edgecolors="#222222",
               label="residue 0")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # energy / temperature trace
    ax_e.step(its, cur_e, where="post", color=CURRENT, lw=1,
              label="current E")
    ax_e.step(its, bst_e, where="post", color=BEST, lw=2, label="best E")
    if optimum is not None:
        ax_e.axhline(optimum, color="#222222", ls=":", lw=1,
                     label=f"optimum ({optimum})")
    lo = min(bst_e + ([optimum] if optimum is not None else []))
    ax_e.set_ylim(lo - 1, max(cur_e) + 1)
    ax_e.set_xlim(0, its[-1])
    ax_e.set_xlabel("iteration")
    ax_e.set_ylabel("energy")
    ax_e.grid(alpha=0.3)
    ax_t = ax_e.twinx()
    ax_t.plot(its, temps, color=TEMP, lw=1.2, label="temperature T")
    ax_t.set_ylabel("temperature T", color=TEMP)
    ax_t.tick_params(axis="y", colors=TEMP)
    ax_t.set_ylim(0, max(temps) * 1.05)
    cursor = ax_e.axvline(0, color="#222222", lw=1)
    lines = ax_e.get_legend_handles_labels()
    lines_t = ax_t.get_legend_handles_labels()
    ax_e.legend(lines[0] + lines_t[0], lines[1] + lines_t[1],
                loc="upper right", fontsize=9, framealpha=0.9)

    def update(frame):
        k, turns, final = frame
        coords = hp.generate_coords(turns)
        xs, ys = zip(*coords)
        chain_line.set_data(xs, ys)
        h_dots.set_offsets([c for c, r in zip(coords, sequence) if r == "H"]
                           or [[float("nan")] * 2])
        p_dots.set_offsets([c for c, r in zip(coords, sequence) if r == "P"]
                           or [[float("nan")] * 2])
        start.set_offsets([coords[0]])
        segs = contact_segments(sequence, coords)
        contacts.set_segments(segs)
        cursor.set_xdata([its[k]])
        if final:
            label.set_text(f"BEST FOUND\nE = {best_e}\n"
                           f"contacts = {len(segs)}")
        else:
            label.set_text(f"iter {its[k]:>7,}\nT    = {temps[k]:.3f}\n"
                           f"E    = {cur_e[k]}\nbest = {bst_e[k]}")
        return chain_line, h_dots, p_dots, start, contacts, cursor, label

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                         blit=False)
    if save:
        if save.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer = PillowWriter(fps=fps)
        anim.save(save, writer=writer, dpi=90)
        print(f"saved {save}: {len(frames)} frames, best E = {best_e}")
        plt.close(fig)
    return anim


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="HPHPPHHPHPPHPHHPPHPH")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=None,
                    help="cooling factor (default: cool over the whole run)")
    ap.add_argument("--optimum", type=int, default=None)
    ap.add_argument("--out", default="sa_animation.gif")
    ap.add_argument("--module", default="hp_folding_solution",
                    help="module providing simulated_annealing")
    args = ap.parse_args()

    known = {"HPHPPHHPHPPHPHHPPHPH": -9, "HHPPHPPHPPHPPHPPHPPHPPHH": -9,
             "PPHPPHHPPPPHHPPPPHHPPPPHH": -8}
    kwargs = {} if args.alpha is None else {"alpha": args.alpha}
    matplotlib.use("Agg")
    animate_sa(args.sequence.upper(), n_iter=args.iters,
               n_frames=args.frames, fps=args.fps, seed=args.seed,
               save=args.out, module=args.module,
               optimum=args.optimum or known.get(args.sequence.upper()),
               **kwargs)
