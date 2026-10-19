"""Render the Acrobot environment as an animated 3D simulation, for the documentation site.

Run::

    python examples/apply_drpp_to_your_own_env/make_acrobot_gif.py      # about 60 seconds

Writes ``public/docs/assets/acrobot_simulation.gif``.

Why a simulation render and not another plot
--------------------------------------------
The front page's "put it on a system of mine" branch used to show covariance ellipses, which say
what a prediction *is* but not what it is *for*. The reader that branch is aimed at already has an
environment; the useful thing to show them is an environment -- a real one, off the shelf, moving.

So this draws `Acrobot-v1` as it is: two links and a joint, in three dimensions, swinging under
the same fixed policy the example uses. The gold marker is a **second body drawn from the
predictor's mean state**: where the model expected the tip to be, one step ahead. It is a mean,
not a distribution -- the distribution is the subject of the other figure.

Honest about what it is
-----------------------
A **re-render in matplotlib**, not a capture of gymnasium's own pygame window: no extra
dependency, deterministic, and it animates headlessly. The geometry is read from the environment
itself, so the motion is the environment's motion.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ppsds.dynamics.identification import encode_action
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import NominalPredictor

EXAMPLE_ROOT = Path(__file__).resolve().parent
#: ``parents[1]``: this file sits at ``examples/<example>/make_acrobot_gif.py``. A script one
#: directory deeper needs ``parents[2]`` -- count the levels; a wrong index raises nothing and
#: writes somewhere plausible instead.
DEFAULT_OUT = EXAMPLE_ROOT.parents[1] / 'public' / 'docs' / 'assets' / 'acrobot_simulation.gif'

ENV_ID = 'Acrobot-v1'
SEED = 601
STEPS = 120
#: One frame every N steps: the swing is slow, and every step would be a slow render of a jittery
#: animation.
STRIDE = 3
FPS = 12
WIDTH = 800
COLORS = 64

#: Frames are rendered this large and cropped afterwards. The crop is what removes the margin --
#: see :func:`content_box` -- so the render has to be comfortably bigger than the result.
RENDER_DPI = 200

#: What counts as "not background", per channel. Small, because the palette here is all pale greys
#: on white: a threshold that ignored them would crop the floor grid away and leave the arm hanging
#: in a void.
CROP_TOLERANCE = 6
#: How much background to leave around the content, as a fraction of its longer side. Not zero:
#: a figure cropped flush to the drawn pixels reads as clipped rather than as framed.
CROP_MARGIN = 0.03

#: Acrobot's own link lengths. Overridden from the environment when it exposes them, because a
#: figure that draws a different link length than the environment is drawing a different machine.
FALLBACK_GEOMETRY = {'link_length_1': 1.0, 'link_length_2': 1.0}


def fixed_policy(obs) -> int:
    """The example's fixed "pump energy" heuristic: torque along the angular velocity."""
    return 2 if obs[4] + obs[5] > 0 else 0


def tip_of(state: np.ndarray, geometry: dict) -> np.ndarray:
    """The tip position ``(x, y)`` for a state, using Acrobot's own kinematics.

    The angles start at "straight down" and increase counter-clockwise, so the first link runs to
    ``L1 * (sin t1, -cos t1)`` and the second continues at ``t1 + t2``. The state carries
    ``(cos t1, sin t1, cos t2, sin t2, dt1, dt2)``, so the angles come back out with ``atan2`` --
    which is also why the environment stores them that way: no wrap-around to get wrong.
    """
    theta1 = np.arctan2(state[1], state[0])
    theta2 = np.arctan2(state[3], state[2])
    l1, l2 = geometry['link_length_1'], geometry['link_length_2']
    joint = np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)])
    return joint + np.array([l2 * np.sin(theta1 + theta2), -l2 * np.cos(theta1 + theta2)])


def joint_of(state: np.ndarray, geometry: dict) -> np.ndarray:
    theta1 = np.arctan2(state[1], state[0])
    return np.array([geometry['link_length_1'] * np.sin(theta1),
                     -geometry['link_length_1'] * np.cos(theta1)])


def rollout():
    """One episode: the states, and the model's one-step-ahead mean at each step."""
    import gymnasium as gym

    env = gym.make(ENV_ID)
    try:
        geometry = dict(FALLBACK_GEOMETRY)
        for attr, key in (('LINK_LENGTH_1', 'link_length_1'), ('LINK_LENGTH_2', 'link_length_2')):
            value = getattr(env.unwrapped, attr, None)
            if value is not None:
                geometry[key] = float(value)

        # A stand-in nominal model: "the state will not move, give or take a little". It is
        # deliberately not a fitted model -- this GIF is about the environment, and a fitted
        # model's mean would track the truth so closely that the gold marker would say nothing.
        nominal = NominalPredictor(
            f_bar=lambda x, u: x,
            mu_bar=np.zeros(6),
            Sigma_bar=np.diag([0.02, 0.02, 0.02, 0.02, 0.35, 0.35]),
        )

        obs, _ = env.reset(seed=SEED)
        states = [np.asarray(obs, dtype=float)]
        predicted = []
        for _ in range(STEPS):
            action = fixed_policy(obs)
            u = encode_action(env.action_space, action)
            nominal.update(obs, u)
            predicted.append(nominal.predict().mean.copy())
            obs, _r, terminated, truncated, _i = env.step(action)
            states.append(np.asarray(obs, dtype=float))
            if terminated or truncated:
                break
        return np.array(states), np.array(predicted), geometry
    finally:
        env.close()


#: How high the pivot sits above the floor. The arm hangs from it: Acrobot swings in a **vertical
#: plane**, so the plane the kinematics live in maps to the scene's ``(x, z)``, with ``y`` across
#: the scene. Drawing it flat -- as the first version of this file did, with the arm's plane laid
#: on the floor -- renders a pendulum lying on the ground, which is not a thing.
PIVOT_HEIGHT = 2.4
#: How far the floor grid reaches along the arm's own plane -- the direction the arm swings in.
#: A little wider than the arm's own reach (``LINK_LENGTH_1 + LINK_LENGTH_2``): a floor that stops
#: exactly where the arm does reads as a box the arm is trapped in.
SPAN = 2.6
#: How far the floor reaches *across* that plane. The arm never leaves ``y = 0``, so this is only
#: there to say "the post stands on a floor".
FLOOR_DEPTH = 1.8
#: Just above the pivot: nothing is ever drawn higher than that, and every unit above it is a band
#: of empty canvas.
CEILING = PIVOT_HEIGHT + 0.2


def draw(ax, state, prediction, trail, geometry) -> None:
    """One frame: the floor, the post, the wake, the two links, and the model's expectation."""
    def lift(plane_point):
        """Arm-plane ``(x, y)`` -> scene ``(x, 0, z)``, hanging from the pivot. ``y`` runs *down*
        in the arm's own frame, so it subtracts from the pivot height."""
        return plane_point[0], 0.0, PIVOT_HEIGHT + plane_point[1]

    ax.clear()
    joint = lift(joint_of(state, geometry))
    tip = lift(tip_of(state, geometry))
    base = (0.0, 0.0, PIVOT_HEIGHT)

    # The floor, and the post that holds the pivot up. A rectangle rather than a square: see
    # `FLOOR_DEPTH`.
    for t in np.linspace(-SPAN, SPAN, 13):
        ax.plot([t, t], [-FLOOR_DEPTH, FLOOR_DEPTH], [0, 0], color='#cfd8dc', lw=0.5)
    for t in np.linspace(-FLOOR_DEPTH, FLOOR_DEPTH, 7):
        ax.plot([-SPAN, SPAN], [t, t], [0, 0], color='#cfd8dc', lw=0.5)
    ax.plot([0, 0], [0, 0], [0, PIVOT_HEIGHT], color='#b0bec5', lw=2.0)

    if len(trail) > 1:
        track = np.array([lift(p) for p in trail])
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color='#b0bec5', lw=1.2)

    for start, end, colour, width in ((base, joint, '#37474f', 9), (joint, tip, '#546e7a', 7)):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colour,
                lw=width, solid_capstyle='round', zorder=4)
    ax.scatter(*zip(base), color='#263238', s=90, depthshade=False)
    ax.scatter(*zip(joint), color='#455a64', s=70, depthshade=False)
    ax.scatter(*zip(tip), color='#1e88e5', s=140, depthshade=False)

    expected = lift(tip_of(prediction, geometry))
    ax.scatter(*zip(expected), color='#f9a825', s=110, depthshade=False,
               edgecolor='#795548', linewidth=0.8)

    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-FLOOR_DEPTH, FLOOR_DEPTH)
    ax.set_zlim(0, CEILING)
    # Sized to the ranges above, so the box the projection draws is the box the scene needs -- a
    # taller or deeper one is empty volume, and empty volume is what a wide margin is made of.
    ax.set_box_aspect((2 * SPAN, 2 * FLOOR_DEPTH, CEILING))
    # Low enough that the floor reads as a ground plane rather than as a tile seen from above, and
    # high enough that the arm is not simply laid along the horizon.
    ax.view_init(elev=14, azim=-62)
    ax.set_axis_off()
    # 3-D axes leave a wide margin of their own; `content_box` trims what is left of it.
    ax.set_position([-0.16, -0.16, 1.32, 1.32])


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='where to write the GIF (default: %(default)s)')
    parser.add_argument('--keep-frames', type=Path, default=None,
                        help='write the PNG frames here instead of a temporary directory')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    states, predicted, geometry = rollout()
    trail = []

    def frames_into(directory: Path) -> int:
        # Wider than tall, and deliberately: the scene is a floor with a narrow machine standing
        # on it, so a square canvas is mostly empty at the sides. This is also what the front
        # page's card wants -- three images of comparable shape (see `public/docs/index.md`).
        figure = plt.figure(figsize=(8.0, 5.0))
        ax = figure.add_subplot(111, projection='3d')
        written = 0
        for k in range(0, len(predicted), STRIDE):
            trail.append(tip_of(states[k], geometry))
            draw(ax, states[k], predicted[k], trail, geometry)
            figure.savefig(directory / f'{written:03d}.png', dpi=RENDER_DPI)
            written += 1
        plt.close(figure)
        return written

    if args.keep_frames:
        args.keep_frames.mkdir(parents=True, exist_ok=True)
        frames_into(args.keep_frames)
        count = _assemble(args.keep_frames, args.out)
    else:
        with tempfile.TemporaryDirectory(prefix='ppsds-acrobot-') as tmp:
            frames_into(Path(tmp))
            count = _assemble(Path(tmp), args.out)

    size = args.out.stat().st_size / 1024
    print(f'wrote {args.out}  ({count} frames, {size:.0f} KiB)')
    return 0


def content_box(frames: list[Path], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Where the drawing actually is, over **every** frame at once.

    A 3-D axes leaves a wide margin of its own -- ``set_axis_off`` removes the ticks and the panes
    but not the projection's own inset -- and this scene does not fill it: the arm is a thin thing
    hanging in the middle of a square canvas. Measured on one frame of the first version, about
    half the image was background. Cropping is both simpler and more honest than tuning the
    projection until it happens to fill the canvas, and it cannot be thrown off by a change to the
    scene's extent.

    The **union**, not each frame's own box: the arm swings across most of the plane, so cropping
    frame by frame would make the whole scene twitch once per frame. Nothing here depends on a
    font stack -- the axes are off, so the content is lines and markers and nothing else -- which
    is what keeps the committed GIF's size reproducible on another machine.
    """
    import numpy as np
    from PIL import Image

    width, height = size
    left, top, right, bottom = width, height, 0, 0
    for path in frames:
        with Image.open(path) as raw:
            pixels = np.asarray(raw.convert('RGB'), dtype=np.int16)
        drawn = np.abs(pixels - 255).max(axis=2) > CROP_TOLERANCE
        rows = np.flatnonzero(drawn.any(axis=1))
        columns = np.flatnonzero(drawn.any(axis=0))
        if rows.size == 0:
            continue
        top = min(top, int(rows[0]))
        bottom = max(bottom, int(rows[-1]) + 1)
        left = min(left, int(columns[0]))
        right = max(right, int(columns[-1]) + 1)

    if right <= left or bottom <= top:
        raise SystemExit('every frame came out blank; the scene is not drawing')

    margin = round(CROP_MARGIN * max(right - left, bottom - top))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(width, right + margin), min(height, bottom + margin)
    # Even dimensions, the same reason the recorder clips to even numbers: an odd-sized GIF frame
    # upsets some decoders for no reason.
    return left, top, right - (right - left) % 2, bottom - (bottom - top) % 2


def _assemble(frames_dir: Path, out_path: Path) -> int:
    """PNGs -> one looping GIF, cropped to the content. Same palette settings as
    ``tools/make_demo_gif.py``, for the same reason: a small palette is the whole of the file size
    on a soft gradient."""
    from PIL import Image

    frames = sorted(frames_dir.glob('[0-9][0-9][0-9].png'))
    if not frames:
        raise SystemExit(f'no frames in {frames_dir}')
    with Image.open(frames[0]) as first:
        box = content_box(frames, first.size)

    images = []
    for path in frames:
        with Image.open(path) as raw:
            image = raw.convert('RGB').crop(box)
        height = round(image.height * WIDTH / image.width / 2) * 2
        image = image.resize((WIDTH, height), Image.LANCZOS)
        images.append(image.quantize(colors=COLORS, method=Image.MEDIANCUT, dither=Image.NONE))

    images[0].save(out_path, format='GIF', save_all=True, append_images=images[1:],
                   duration=round(1000 / FPS), loop=0, optimize=True)
    return len(images)


if __name__ == '__main__':
    raise SystemExit(main())
