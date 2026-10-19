/**
 * The table's own dynamics, in JavaScript -- read straight from the exported drift matrix.
 *
 * Every table in this game is linear, and the exporter **verifies** that before writing the
 * matrix out (`experiments/pmf_betting/export_game_assets.py`), so a puck's next position is one
 * matrix product and the page can draw it. The nominal dynamics are *given* in this setting --
 * that is what makes a bet a bet on the noise -- so the page shows them rather than making the
 * player infer them from the wake.
 *
 * Nothing here is a second implementation of a library number: the matrix and the noise
 * description come from the assets. What the page derives from them is pinned to ppsds' own
 * dynamics by the exported `drift` self-check cases (`tests/test_game_page.py`).
 */

/** Where the drift alone takes a point next: `A x`. */
export function nominalNext(preset, point) {
  const m = preset.drift_matrix;
  return [
    m[0][0] * point[0] + m[0][1] * point[1],
    m[1][0] * point[0] + m[1][1] * point[1],
  ];
}

/** The next `steps` positions the drift alone would take, starting from `start` (inclusive). */
export function nominalPath(preset, start, steps) {
  const path = [[start[0], start[1]]];
  for (let i = 0; i < steps; i += 1) path.push(nominalNext(preset, path[path.length - 1]));
  return path;
}

/**
 * Where the drift's **character** is drawn from, for the table card's preview.
 *
 * A fan, not one path: the Long Table's start `x0` is the origin, which its drift leaves exactly
 * where it is, so a single path from `x0` would be a motionless dot that looks broken. Four
 * starts, chosen with distinct distances from the long table's rail and distinct radii, give
 * each table a picture that says what it does -- lines that fan out and speed up, circles, a
 * spiral -- and none of them depends on where the puck happens to be right now, so the card
 * cannot leak this round's aim.
 */
export function fanStarts() {
  return [
    [0.15, 1.0],
    [1.0, 0.35],
    [-0.45, -0.8],
    [-1.0, -0.15],
  ];
}

/** The box the safe layout covers: the support itself when bounded, a window when not. */
export function supportBox(preset) {
  return preset.support.kind === 'bounded'
    ? { low: preset.support.low, high: preset.support.high }
    : null;
}

/**
 * The box the "cover the table" button tiles.
 *
 * Bounded noise: its own support, so the coverage is a guarantee. Unbounded noise: a window of
 * `safe[eps].window_sigma` standard deviations, which the exporter records -- a choice, not a
 * guarantee, which is why the badge reports a probability there instead of a worst case.
 */
export function safeBox(preset, eps) {
  const box = supportBox(preset);
  if (box !== null) return box;
  const summary = preset.safe[String(eps)] || {};
  const windowSigma = summary.window_sigma === undefined ? 3 : summary.window_sigma;
  const mean = preset.noise.mean;
  const sigma = [Math.sqrt(preset.noise.cov[0][0]), Math.sqrt(preset.noise.cov[1][1])];
  return {
    low: [mean[0] - windowSigma * sigma[0], mean[1] - windowSigma * sigma[1]],
    high: [mean[0] + windowSigma * sigma[0], mean[1] + windowSigma * sigma[1]],
  };
}

/** How wide a view has to be to hold the puck, the prediction and the stacks together. */
export function viewSpan(here, predicted, eps) {
  return Math.max(
    3.0,
    0.5 * Math.max(Math.abs(predicted[0] - here[0]), Math.abs(predicted[1] - here[1])) + 2 * eps + 1.5,
  );
}

/**
 * The view: the box that has to be on screen, and how much room shows it.
 *
 * The centre is the middle of the puck, the prediction and the stacks' **resting** places -- not
 * of wherever the stacks are while they glide. That is what makes a round change read as the hand
 * sliding across a table that stays put, rather than the whole felt lurching and the hand catching
 * up to it.
 *
 * The span is clamped. The floor is `viewSpan`, so an empty table frames exactly as it did before
 * there was anything to place; the ceiling stops a stack dropped near the edge of the screen from
 * ratcheting the view outwards a little further every round.
 *
 * `tower` is the height of the tallest stack in the hand, and it is the one thing that may lift
 * the span through that ceiling: a weight is a height, so a stack the player made tall is the
 * player's own doing rather than a ratchet, and the camera has to back off until it fits. Lower
 * the stack and the frame comes back in.
 *
 * `past` is where the ball has already landed this run. The puck never moves on its own -- the
 * system moves and the puck is where it went -- so the trail behind it is the only place that
 * motion is visible, and a frame fitted to the hand alone lets the trail walk off the edge as the
 * system drifts. Keeping it on screen is what makes a run look like one system moving rather than
 * a table that keeps teleporting.
 *
 * The trail **widens** the frame; it does not move it. The centre stays on the hand, because the
 * hand is where the player is looking, and the span grows only as far as the same `2 * floor`
 * ceiling allows -- a run that wanders further than that shows as much trail as fits and no more.
 * Widening is bounded, so a long run zooms out once and holds instead of creeping outwards every
 * round; centring on the trail instead would eventually walk the puck itself off the screen, which
 * is the one thing the frame exists to prevent. `[]` -- a run with no rounds behind it -- frames
 * exactly as it did before this argument existed.
 */
export function frameBox(here, predicted, resting, eps, tower = 0, past = []) {
  const floor = viewSpan(here, predicted, eps);
  const points = [here, predicted, ...resting];
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const centre = [0.5 * (Math.min(...xs) + Math.max(...xs)), 0.5 * (Math.min(...ys) + Math.max(...ys))];
  const half = 0.5 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const hand = Math.min(2 * floor, Math.max(floor, half + 2 * eps + 1.0));

  let span = hand;
  for (const point of past) {
    const reach = Math.max(Math.abs(point[0] - centre[0]), Math.abs(point[1] - centre[1]));
    span = Math.max(span, Math.min(2 * floor, reach + 2 * eps + 1.0));
  }
  return { centre, span: Math.max(span, TOWER_ROOM * tower) };
}

/** The smoothstep both the easing paths share: `3t² - 2t³`, clamped to `[0, 1]`. */
export function smoothstep(t) {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

/** Where a point lands when it is pulled onto a lattice anchored at `base`. */
export function snapToLattice(point, base, step) {
  if (!(step > 0)) return [point[0], point[1]];
  return [
    base[0] + Math.round((point[0] - base[0]) / step) * step,
    base[1] + Math.round((point[1] - base[1]) / step) * step,
  ];
}

/**
 * The index of the stack a point is on, or `null`.
 *
 * The infinity norm, because every payout region on this page is a box: a click "on" a stack means
 * inside its own square, not inside a disc drawn over it. One distance rule for all three callers
 * (picking, placing and hovering), so they cannot disagree about what is under the pointer.
 */
export function nearestIndex(point, positions, reach) {
  let best = null;
  positions.forEach((at, index) => {
    const distance = Math.max(Math.abs(at[0] - point[0]), Math.abs(at[1] - point[1]));
    if (distance <= reach && (best === null || distance < best.distance)) best = { index, distance };
  });
  return best === null ? null : best.index;
}

// -- stacks: a height, and the lattice it sits on ----------------------------------------------
//
// A stack is an atom with a **continuous weight** -- not a count of stacks. Only proportions are
// observable (the score is `log(share that landed)`), so the weight's absolute size means nothing
// and the page never asks the player to think about it: what is drawn is the height, and what is
// written on the stack is the share it works out to.
//
// The three constants below are one decision each, and both renderers read them here so a tower
// cannot mean two heights depending on whether the CDN answered.

/** A stack's drawn radius, as a fraction of the payout radius. Small enough that the discs read as
 *  a pile on a square rather than as a lid on it -- and, with the footprint below, small enough
 *  that no two stacks can ever touch. */
export const DISC_RADIUS_EPS = 0.28;
/** The same radius' floor, as a fraction of the view span: on a table whose payout radius is tiny
 *  (the Whirlpool's 0.1) a radius proportional to epsilon alone would be a sub-pixel dot. */
export const DISC_RADIUS_MIN = 0.018;
/** How many fine cells a payout cell is cut into when the snap is off. Three, so a player can put
 *  a stack off the payout grid without being able to put one on top of another. */
export const FINE_DIVISIONS = 3;

export function discRadius(eps, span) {
  return Math.max(DISC_RADIUS_MIN * span, DISC_RADIUS_EPS * eps);
}

/** The smallest distance two stack centres may keep: their own diameter. */
export function discFootprint(eps, span) {
  return 2 * discRadius(eps, span);
}

/**
 * How far from a stack's centre a press still counts as touching it.
 *
 * A mouse gets the stack's own footprint -- the square a tap would resolve into. A finger gets
 * half again as much, because a fingertip covers more of the screen than a cursor and a miss on a
 * touch screen is not recoverable by looking more carefully. Neither reaches the next stack: the
 * lattice is never finer than the footprint, so the worst a generous reach can do is pick the
 * neighbour you were closer to.
 */
export function grabReach(eps, span, touch = false) {
  return (touch ? 1.5 : 1) * discFootprint(eps, span);
}

/**
 * The lattice a stack lands on.
 *
 * On: the payout lattice, `2 eps` apart, so the payout squares tile and every stack owns a cell.
 * Off: `2 eps / 3`, so a hand can be shifted by a third of a cell -- and, because two discs are
 * `2 r` apart at the very least, still never overlap.
 *
 * Never finer than the footprint. On a table whose payout radius is smaller than a disc, a lattice
 * of its own would be a lattice of refusals: the player would tap, and the tap would be rejected
 * as too close to a neighbour. Coarsening to the footprint is the honest answer -- when the cells
 * are smaller than the pieces, you cannot have one piece per cell.
 */
export function latticeStep(eps, span, snapping) {
  const cell = snapping ? 2 * eps : (2 * eps) / FINE_DIVISIONS;
  return Math.max(discFootprint(eps, span), cell);
}

/** A disc's height, as a fraction of its radius. Flat, so a stack reads as a pile of discs. */
export const DISC_ASPECT = 0.34;/** The vertical gap between discs, as a multiple of one disc's height. Without the gap the discs
 *  merge into one cylinder and the height stops being countable at a glance. */
export const DISC_GAP = 1.35;
/** The tower's scale: how many discs one unit of weight draws. A height *is* a weight, times
 *  this -- a stack twice as heavy is twice as tall, and what one stack does never moves another. */
export const DISCS_PER_WEIGHT = 3;
/**
 * How much of the view span a tower of a given height needs.
 *
 * A tower stands up out of the felt, so a frame that only measured how *wide* the hand is would
 * let a tall stack walk off the top of the picture. The camera looks down at the table from
 * `elevation`, so a tower of height `H` covers less of the frame than `H` does; solving the
 * frustum for the height at which a vertical tower touches the top edge gives about `0.85 H`,
 * and this rounds that up to leave the tower some air above it. See `frameBox`.
 */
export const TOWER_ROOM = 1.0;
/** The bottom of the slider. A stack is taken off the table by the Remove button, never by
 *  squeezing its height to nothing -- a slider that deletes what it is sliding is a slider nobody
 *  dares drag to the end. */
export const WEIGHT_MIN = 0.02;
/**
 * The slider's range, as a *ratio* rather than a ceiling.
 *
 * Only the ratio between stacks decides what a round pays, so no weight is more correct than
 * another and there is nothing to stop at -- a hard top is what made the height control feel
 * broken. The slider is therefore logarithmic: its handle is the log of the weight, so dragging
 * right keeps multiplying and never runs out. 2^14 spans "barely on this spot" to "sixteen
 * thousand times that", which is far past the point where one stack is the whole bet.
 */
export const WEIGHT_SPAN = 2 ** 14;
/** The top of that range, in weight units. Not a limit the player meets -- only a number that is
 *  still a number, so the export stays finite. */
export const WEIGHT_CEILING = WEIGHT_MIN * WEIGHT_SPAN;
/** One step of the slider's own coordinate, which is `0..1` rather than a weight. */
export const WEIGHT_STEP = 0.001;

/** The slider's coordinate for a weight, and back. `t` is a share of the logarithmic range. */
export function weightFromSlider(t) {
  const bounded = Math.min(1, Math.max(0, t));
  return WEIGHT_MIN * WEIGHT_SPAN ** bounded;
}

export function sliderFromWeight(w) {
  const bounded = Math.min(WEIGHT_CEILING, Math.max(WEIGHT_MIN, w));
  return Math.log(bounded / WEIGHT_MIN) / Math.log(WEIGHT_SPAN);
}

/**
 * What a stack of weight `w` draws: how many whole discs, and how much of one more.
 *
 * The partial top disc is where a continuous weight becomes pixels -- squash it to `partial` of
 * its height and the tower grows smoothly instead of jumping a whole disc at a time.
 */
export function discPlan(w, radius) {
  const units = Math.max(0, w) * DISCS_PER_WEIGHT;
  const full = Math.floor(units + 1e-9);
  const thickness = radius * DISC_ASPECT;
  return {
    full,
    partial: units - full,
    thickness,
    step: thickness * DISC_GAP,
  };
}

/** How tall a stack draws, in state units -- what a label above it has to clear. */
export function stackHeight(w, radius) {
  const { full, partial, thickness, step } = discPlan(w, radius);
  return full * step + partial * thickness;
}

/** Whether two stacks are the same, to within a hair. Used to tell "unchanged" from "edited". */
const sameStack = (a, b) => Math.abs(a.x - b.x) < 1e-9
  && Math.abs(a.y - b.y) < 1e-9 && Math.abs(a.w - b.w) < 1e-9;

/**
 * Whether two hands are the same hand, stack for stack.
 *
 * This is what a quick layout asks before it undoes itself: "is the table still holding exactly
 * what I put there?" A hand that has been touched since is a different hand, and pressing the
 * button again means "lay it out again", not "take it back".
 */
export function sameHand(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return a.every((stack, index) => sameStack(stack, b[index]));
}

/** The narrowest a card is allowed to get before the column stops being a column. */
export const CARD_MIN = 300;
/**
 * The table's own floor, so the panel can never squeeze the game out of the window.
 *
 * Measured rather than guessed: with the felt narrower than about 560px the hand buttons wrap out
 * of their row, and under 470 the score chart in the corner lands on the switches. 700 leaves the
 * margin, and it is the floor for the *drag* only -- the default width is a card, whatever the
 * window is doing.
 */
export const STAGE_MIN = 700;

/** Where the panel's edge lands when it is dragged to `px`, given the room it sits in. */
export function panelWidth(px, benchWidth) {
  const widest = Math.max(CARD_MIN, benchWidth - STAGE_MIN);
  return Math.min(widest, Math.max(CARD_MIN, px));
}

/** The weights as shares of the hand, for the labels and the panel. `[]` on an empty table. */
export function shares(weights) {
  const total = weights.reduce((sum, w) => sum + w, 0);
  return total > 0 ? weights.map((w) => w / total) : weights.map(() => 0);
}

/**
 * How visible the ghost of a landing is.
 *
 * A landing *is* the next state of the system -- `x' = A x + w` -- so the ball comes to rest exactly
 * where the system goes next, and it never moves again on its own: the rounds catch up to it. The
 * ghosts are the rounds before that: the trail of places the ball has been, which is the trail the
 * system walked. Without them the ball would look like it got where it is by itself.
 *
 * The newest ghost is the one the ball is *standing in*, so it is the ones behind it that read;
 * from there they fade with age without ever quite going. `fresh` is the one number here with a
 * second job, since the second-newest is the first one anybody sees.
 *
 * `index` counts from the oldest, which is how the rounds are recorded.
 */
export const LANDING_ALPHA = { fresh: 0.34, oldest: 0.09 };

export function landingFade(index, count) {
  if (count <= 1) return LANDING_ALPHA.fresh;
  const age = (count - 1 - index) / (count - 1); // 0 for the newest landing, 1 for the oldest
  return LANDING_ALPHA.fresh + (LANDING_ALPHA.oldest - LANDING_ALPHA.fresh) * age;
}

// -- the wake ----------------------------------------------------------------------------------

/** How long a step has to be to earn an arrowhead, as a fraction of the view span. */
export const ARROW_MIN = 0.08;
/** How many arrowheads a wake draws at most. A fifty-round run would otherwise sprout fifty. */
export const ARROW_LIMIT = 16;

/**
 * Where to put arrowheads along a path, and which way each one points.
 *
 * The most recent joints win, because the near end of the wake is the part a player is reading;
 * a segment shorter than `ARROW_MIN` of the span gets none, because an arrowhead on a two-pixel
 * step is a blob rather than a direction.
 */
export function arrowJoints(points, span, limit = ARROW_LIMIT) {
  if (!Array.isArray(points) || points.length < 2) return [];
  const joints = [];
  for (let i = points.length - 1; i > 0 && joints.length < limit; i -= 1) {
    const from = points[i - 1];
    const to = points[i];
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const length = Math.hypot(dx, dy);
    if (length < ARROW_MIN * span) continue;
    joints.push({ at: [to[0], to[1]], dir: [dx / length, dy / length] });
  }
  return joints.reverse();
}

/** What the landing law is, in words -- formatting the exported numbers, not inventing any. */
export function noiseProse(preset) {
  const noise = preset.noise;
  const short = (value) => String(Number(value.toFixed(3)));
  if (noise.kind === 'unpublished') {
    return 'This table does not publish its landing law. Read the landings and work it out.';
  }
  if (noise.kind === 'blocks') {
    const heaviest = noise.weights.indexOf(Math.max(...noise.weights));
    const [low, high] = noise.boxes[heaviest];
    const share = Math.round(noise.weights[heaviest] * 100);
    return `Landings come in blocks. ${share}% of them fall in the ${short(low[0])}..${short(high[0])} `
      + `by ${short(low[1])}..${short(high[1])} band, and the rest are spread around it.`;
  }
  if (noise.kind === 'uniform') {
    const width = noise.high[0] - noise.low[0];
    const height = noise.high[1] - noise.low[1];
    const where = Math.abs(width - height) < 1e-9
      ? `a square ${short(width)} wide`
      : `a ${short(width)} by ${short(height)} box`;
    return `Every landing is uniform in ${where}: no spot is more likely than another.`;
  }
  if (noise.kind === 'gaussian') {
    const across = Math.sqrt(noise.cov[0][0]);
    const up = Math.sqrt(noise.cov[1][1]);
    const spread = Math.abs(across - up) < 1e-9
      ? `sigma ${short(across)} on each axis`
      : `sigma ${short(across)} across and ${short(up)} up`;
    return `Every landing wobbles away from the prediction, ${spread}.`;
  }
  return 'The landing follows the table\'s own law.';
}
