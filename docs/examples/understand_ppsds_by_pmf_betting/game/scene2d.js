/**
 * The flat table and the small charts: everything 2-D the page draws.
 *
 * The flat table is used two ways.
 *
 * 1. **As the fallback** when the 3-D scene cannot start -- no WebGL, or a CDN that does not
 *    answer. The game has to keep playing: the whole point of the page is that a reader can try
 *    the idea, and a blank canvas is not an idea.
 * 2. **As the table card's preview**, where it draws the drift's character as a small fan of
 *    nominal paths. A second WebGL context for a 60-pixel picture would be silly.
 *
 * Every entry point takes data and draws it; none decides anything. `game.js` owns the state,
 * `coverage.js` owns the arithmetic, and this file owns pixels.
 */

import { CHART, PALETTE, feltOf } from './theme.js';
import { DISC_ASPECT, arrowJoints, landingFade } from './world.js';

/** How big an arrowhead is, as a fraction of the canvas' short side. An open "→" reads smaller
 *  than the solid wedge it replaced, so it is drawn a little larger to carry the same weight. */
const ARROW_SIZE = 0.024;
/** How far each barb trails behind the tip, in radians. Wide, so the head is unmistakably open. */
const BARB_BACK = 2.5;

/** The ball, in pixels: half the fraction the 3-D table uses, because this canvas maps `2 span`
 *  across its short side. It is the only solid thing on the felt, and its ghosts are drawn at the
 *  same size -- a ghost of the ball that is not the ball's size is a ghost of something else. */
const ballSize = (width) => Math.max(4, 0.009 * width);

/** A scratch canvas sized to its CSS box at the device's pixel ratio. */
export function fitCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

/**
 * The flat table.
 *
 * The surface the shell talks to is the same one the 3-D scene exposes -- `render`, `resize`,
 * `pick`, `zoom`, `orbit` -- so `game.js` never branches on which renderer it got.
 */
export function createFlatScene(canvas) {
  const ctx = canvas.getContext('2d');
  let zoom = 1;
  let pan = [0, 0];
  let last = { span: 1, centre: [0, 0] };
  // The centre the picture is actually showing, easing toward the view's exactly as the 3-D
  // camera's target does. `pick` reads *this* rather than the view's centre, because otherwise a
  // tap would be answered by a picture the player is not looking at.
  let shown = null;
  // The span eases too. It is what turns "the frame changed" into "the camera moved" -- and the
  // easing happens while the table is empty (`view.handAlpha`), so there are no two frames on
  // screen to compare.
  let shownSpan = null;

  function render(view) {
    const { width, height } = fitCanvas(canvas);
    if (shown === null) shown = [view.centre[0], view.centre[1]];
    shown = [
      shown[0] + (view.centre[0] - shown[0]) * 0.18,
      shown[1] + (view.centre[1] - shown[1]) * 0.18,
    ];
    shownSpan = shownSpan === null ? view.span : shownSpan + (view.span - shownSpan) * 0.18;
    const span = shownSpan * zoom;
    const centre = [shown[0] + pan[0], shown[1] + pan[1]];
    last = { span, centre };
    const scale = Math.min(width, height) / (2 * span);
    const geom = { scale, radius: view.radius * scale, handAlpha: view.handAlpha };
    const toPixel = (point) => [
      width / 2 + (point[0] - centre[0]) * scale,
      height / 2 - (point[1] - centre[1]) * scale,
    ];
    const radius = view.eps * scale;
    const felt = feltOf(view.presetKey);

    ctx.clearRect(0, 0, width, height);
    drawFelt(ctx, width, height, felt);
    drawGrid(ctx, width, height, span);
    // The fill sits under the hand; its boundary is drawn again after the stacks, so a stack can
    // hide the tint but never the edge.
    if (view.support) drawSupport(ctx, view.support, toPixel, scale, false);

    if (view.history.show) drawHistory(ctx, view, toPixel, Math.min(width, height));

    // The hand this round did not keep: it stays where it stood and fades, so a layout being
    // cleared each round is something the player watches rather than infers.
    for (const stack of view.leaving || []) {
      const [x, y] = toPixel(stack.at);
      ctx.globalAlpha = 0.45 * stack.alpha;
      ctx.beginPath();
      ctx.ellipse(x, y - geom.radius * 0.2, geom.radius, geom.radius * DISC_ASPECT, 0, 0, 2 * Math.PI);
      ctx.fillStyle = PALETTE.stack;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Everything from here on is the hand, and the hand has its own visibility: it leaves before
    // the camera moves and arrives after it, so nothing is ever sliding while the camera swings.
    geom.handAlpha = view.handAlpha;

    // Every stack's neighbourhood, so the coverage is visible rather than inferred: where two
    // squares meet, the stacks behind them share a landing. The squares may overlap -- the discs
    // may not, which is why the discs are sized from the footprint and these from epsilon.
    ctx.lineWidth = Math.max(1, 1.2 * (width / 620));
    ctx.globalAlpha = view.handAlpha;
    if (view.eps > 0) {
      for (const stack of view.stacks) {
        const [x, y] = toPixel(stack.at);
        square(ctx, x, y, radius, PALETTE.eps, PALETTE.epsEdge);
      }
    }

    for (const stack of view.stacks) {
      drawStack(ctx, stack, toPixel(stack.at), view.settleT, geom);
    }
    ctx.globalAlpha = 1;
    // Boundary over the hand, fill under it: the same split as the 3-D table.
    if (view.support) drawSupport(ctx, view.support, toPixel, scale, true);
    if (view.nominalPath) drawNominalPath(ctx, view.nominalPath, toPixel, view.span, Math.min(width, height));
    if (view.predicted) drawPrediction(ctx, toPixel(view.predicted), width);
    // The one solid thing on the felt, at every moment: where it landed, or in the air on the way.
    // There is no separate marker for the system's state -- a landing *is* the next state, so the
    // state is where the ball is.
    if (view.ball) drawBall(ctx, view.ball, toPixel, width);
  }

  return {
    render,
    resize() {},
    pick(offsetX, offsetY) {
      const ratio = canvas.width / Math.max(1, canvas.clientWidth);
      const x = offsetX * ratio;
      const y = offsetY * ratio;
      const scale = Math.min(canvas.width, canvas.height) / (2 * last.span);
      return [
        last.centre[0] + (x - canvas.width / 2) / scale,
        last.centre[1] - (y - canvas.height / 2) / scale,
      ];
    },
    zoom(deltaY) {
      this.zoomBy(deltaY > 0 ? 1.08 : 0.93);
    },
    /** A pinch hands in a ratio: below one means closer. */
    zoomBy(factor) {
      zoom = Math.min(3, Math.max(0.4, zoom * factor));
    },
    /** On the flat table, dragging the felt pans it -- there is no camera to turn. */
    orbit(dx, dy) {
      const scale = Math.min(canvas.width, canvas.height) / (2 * last.span);
      pan = [pan[0] - dx / scale, pan[1] + dy / scale];
    },
    pan: () => pan,
    project(point) {
      return this.projectAt(point, 0);
    },
    /** The same, for a point `lift` above the felt -- a label sits on top of a tower, and a tower
     *  is as tall as its weight. */
    projectAt(point, lift = 0) {
      const ratio = canvas.width / Math.max(1, canvas.clientWidth);
      const scale = Math.min(canvas.width, canvas.height) / (2 * last.span);
      return {
        x: (canvas.width / 2 + (point[0] - last.centre[0]) * scale) / ratio,
        y: (canvas.height / 2 - (point[1] - last.centre[1]) * scale - lift * scale) / ratio,
      };
    },
  };
}

/** One definition of "which square does this stack own", used by both renderers. */
export function square(ctx, x, y, radius, fill, edge) {
  ctx.fillStyle = fill;
  ctx.fillRect(x - radius, y - radius, 2 * radius, 2 * radius);
  ctx.strokeStyle = edge;
  ctx.strokeRect(x - radius, y - radius, 2 * radius, 2 * radius);
}

function drawFelt(ctx, width, height, felt) {
  const wash = ctx.createRadialGradient(
    width * 0.5, height * 0.42, Math.min(width, height) * 0.05,
    width * 0.5, height * 0.5, Math.max(width, height) * 0.75,
  );
  wash.addColorStop(0, felt.top);
  wash.addColorStop(1, felt.edge);
  ctx.fillStyle = wash;
  ctx.fillRect(0, 0, width, height);
}

function drawGrid(ctx, width, height, span) {
  // A grid whose spacing is a power of two in state units, so it stays legible as the view
  // follows the action instead of crawling under a fixed pixel size.
  const step = Math.pow(2, Math.round(Math.log2(span / 4)));
  const extent = span * 4;
  const scale = Math.min(width, height) / (2 * span);
  ctx.strokeStyle = PALETTE.grid;
  ctx.lineWidth = 1;
  for (let value = -extent; value <= extent; value += step) {
    const x = width / 2 + value * scale;
    const y = height / 2 - value * scale;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
}

/**
 * The noise's support, in the noise frame (around `view.support.anchor`).
 *
 * A block law's fill alpha is the band's weight -- that is what the bands are for. The Gaussian
 * ellipse is a chosen 3σ window, not a support; the caption outside the canvas says so. The
 * renderer draws exactly what the view model handed it, including `null` for a table that keeps
 * its law.
 */
function drawSupport(ctx, support, toPixel, scale, edgesOnly) {
  const anchor = support.anchor;
  ctx.save();
  ctx.lineWidth = Math.max(1, 1.3);
  ctx.strokeStyle = PALETTE.supportEdge;
  if (support.kind === 'ellipse') {
    if (!edgesOnly) { ctx.restore(); return; }
    const [x, y] = toPixel([anchor[0] + support.centre[0], anchor[1] + support.centre[1]]);
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.ellipse(x, y, support.rx * scale, support.ry * scale, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
    return;
  }
  const boxes = support.kind === 'box'
    ? [{ low: support.low, high: support.high, weight: 1 }]
    : support.bands;
  for (const box of boxes) {
    const [x0, y0] = toPixel([anchor[0] + box.low[0], anchor[1] + box.low[1]]);
    const [x1, y1] = toPixel([anchor[0] + box.high[0], anchor[1] + box.high[1]]);
    const left = Math.min(x0, x1);
    const top = Math.min(y0, y1);
    const width = Math.abs(x1 - x0);
    const height = Math.abs(y1 - y0);
    if (edgesOnly) {
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(left, top, width, height);
    } else {
      ctx.globalAlpha = 0.06 + 0.14 * box.weight;
      ctx.fillStyle = PALETTE.support;
      ctx.fillRect(left, top, width, height);
    }
  }
  ctx.restore();
}

/** One "→": two strokes meeting at the tip, drawn at `angle` radians in screen space.
 *
 *  Stroked rather than filled, and open rather than closed: a solid wedge and its mirror read the
 *  same at a glance, and the direction of travel is the entire content of this mark. The caller
 *  sets `strokeStyle` and `lineWidth` -- the head is the same weight as the line it sits on. */
export function arrowHead(ctx, x, y, angle, size) {
  ctx.beginPath();
  [+BARB_BACK, -BARB_BACK].forEach((turn) => {
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle + turn) * size, y + Math.sin(angle + turn) * size);
  });
  ctx.stroke();
}

function drawHistory(ctx, view, toPixel, short) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(232, 236, 243, 0.20)';
  for (const path of view.history.paths) {
    ctx.beginPath();
    path.forEach((point, i) => {
      const [x, y] = toPixel(point);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Arrowheads, so the wake says which way the puck has been going. Without them a path of
    // near-equal steps is a scribble: the joints are the only place a direction is readable.
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(232, 236, 243, 0.30)';
    for (const joint of arrowJoints(path, view.span)) {
      const [x, y] = toPixel(joint.at);
      // Screen y runs the other way from state y, so the angle is mirrored.
      arrowHead(ctx, x, y, Math.atan2(-joint.dir[1], joint.dir[0]), short * ARROW_SIZE);
    }
  }
  // Every landing keeps the ball that made it: a ghost at the ball's own size, fading with age. The
  // puck stands on the newest one -- a landing *is* the next state -- so this trail is the trail of
  // the thing the puck follows. (It was a fixed five-pixel dot in the ball's colour, which is
  // neither the ball's size nor its shape, and read as a mark of some other kind.)
  ctx.fillStyle = PALETTE.landed;
  view.history.landings.forEach((point, index) => {
    const [x, y] = toPixel(point);
    ctx.globalAlpha = landingFade(index, view.history.landings.length);
    ctx.beginPath(); ctx.arc(x, y, ballSize(short), 0, 2 * Math.PI); ctx.fill();
  });
  ctx.globalAlpha = 1;
  // The draws so far, re-centred on this round's prediction: the noise's habits, which is what a
  // player is actually betting on. Dashed because they are a hint, not this round's outcome -- and
  // absent when the prediction is hidden, because there is then no centre to re-centre on and the
  // raw offsets would pile up at the origin.
  const anchor = view.history.show ? view.history.anchor : null;
  if (anchor) {
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = 'rgba(232, 236, 243, 0.35)';
    view.history.draws.forEach((draw) => {
      const [x, y] = toPixel([anchor[0] + draw[0], anchor[1] + draw[1]]);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    });
    ctx.restore();
  }
}

function drawNominalPath(ctx, path, toPixel, span, short) {
  ctx.save();
  ctx.setLineDash([6, 7]);
  ctx.strokeStyle = 'rgba(245, 196, 81, 0.45)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach((point, i) => {
    const [x, y] = toPixel(point);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(245, 196, 81, 0.55)';
  for (const joint of arrowJoints(path, span)) {
    const [x, y] = toPixel(joint.at);
    arrowHead(ctx, x, y, Math.atan2(-joint.dir[1], joint.dir[0]), short * ARROW_SIZE);
  }
  ctx.restore();
}

function drawStack(ctx, stack, at, settleT, geom) {
  const [x, y] = at;
  const colour = stack.state === 'bet' ? PALETTE.stack
    : stack.state === 'hit' ? PALETTE.hit : PALETTE.miss;
  const { full, partial, thickness, step } = stack.plan;
  const disc = thickness * geom.scale;
  const gap = step * geom.scale;
  // The tower's height *is* the weight: whole discs, plus one squashed to the fraction the weight
  // asks for. That partial disc is what makes a slider drag grow the pile smoothly.
  const count = full + (partial > 0 ? 1 : 0);
  const radius = geom.radius * (1 + 0.14 * (stack.selected ? stack.pressT : 0));
  // Winning stacks rise and glow, losing ones sink and go quiet -- so the settlement is legible
  // without relying on the green/red pair alone.
  const lift0 = stack.state === 'hit' ? 5 * settleT : stack.state === 'miss' ? -3 * settleT : 0;
  const alpha = stack.state === 'miss' ? 1 - 0.45 * settleT : 1;
  const pulse = stack.state === 'hit' ? 1 + 0.25 * settleT * (1 + Math.sin(settleT * Math.PI * 6)) / 2 : 1;

  ctx.globalAlpha = alpha * geom.handAlpha;
  for (let i = 0; i < count; i += 1) {
    const top = i === count - 1;
    const height = top && partial > 0 ? disc * partial : disc;
    const centre = y - (i * gap + height / 2) + lift0;
    ctx.beginPath();
    ctx.ellipse(x, centre, radius * pulse, Math.max(1, height / 2) * pulse, 0, 0, 2 * Math.PI);
    ctx.fillStyle = top ? colour : shade(colour, 0.72);
    ctx.fill();
    ctx.strokeStyle = 'rgba(10, 12, 18, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  // No share here: the number lives in the labels overlay, one implementation for both renderers,
  // so the flat table and the 3-D one cannot disagree about what a height works out to. What is
  // written here is the *payout* -- and on the felt below the stack, not above it, because above
  // is where the share label sits. Colour alone is a red/green trap, so the payout is spelled out.
  if (stack.state === 'hit' && settleT > 0.15) {
    ctx.fillStyle = PALETTE.safe;
    ctx.font = `600 ${Math.max(12, radius * 0.8)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`+${Math.round(stack.share * 100)}%`, x, y + radius * 0.7 + 14 / (1 + settleT));
  }
  if (stack.hover) {
    // Dimmer than the selection ring and undashed: one ring says "a click lands here", two say
    // "you are holding this one".
    ctx.beginPath();
    ctx.arc(x, y, radius + 3, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(232, 236, 243, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  if (stack.selected) {
    // White, like the 3-D ring: gold is the prediction, cyan is the payout radius, green and red
    // are the settlement. White is the one colour on this table that means "you are holding this".
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(x, y, radius + 6, 0, 2 * Math.PI);
    ctx.strokeStyle = PALETTE.puck;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }
  ctx.globalAlpha = 1;
}

function drawPrediction(ctx, at, width) {
  const [x, y] = at;
  const size = Math.max(8, 0.026 * width);
  ctx.strokeStyle = PALETTE.predicted;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, size, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - size * 1.6, y); ctx.lineTo(x - size * 0.6, y);
  ctx.moveTo(x + size * 0.6, y); ctx.lineTo(x + size * 1.6, y);
  ctx.moveTo(x, y - size * 1.6); ctx.lineTo(x, y - size * 0.6);
  ctx.moveTo(x, y + size * 0.6); ctx.lineTo(x, y + size * 1.6);
  ctx.stroke();
}

function drawBall(ctx, ball, toPixel, width) {
  const [x, y] = toPixel(ball.at);
  const lift = (ball.height / Math.max(1e-6, ball.height + 1)) * 0.35 * Math.min(width, width);
  const size = ballSize(width);
  ctx.beginPath();
  ctx.ellipse(x, y + (ball.landed ? 0 : size * 0.6), size * (ball.landed ? 1 : 0.7), size * 0.5, 0, 0, 2 * Math.PI);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y - lift, size, 0, 2 * Math.PI);
  ctx.fillStyle = PALETTE.landed;
  ctx.fill();
  ctx.strokeStyle = '#1b1207';
  ctx.lineWidth = Math.max(1, size * 0.25);
  ctx.stroke();
}

/** Darken a hex colour, for the underside of a stack. */
function shade(hex, factor) {
  const value = parseInt(hex.slice(1), 16);
  const parts = [(value >> 16) & 255, (value >> 8) & 255, value & 255]
    .map((channel) => Math.round(channel * factor));
  return `rgb(${parts[0]}, ${parts[1]}, ${parts[2]})`;
}

/**
 * The table card's preview: the drift's character, as a fan of nominal paths.
 *
 * Four paths rather than one, and from fixed demo starts rather than from `x0`: the Long Table's
 * `x0` is a fixed point of its own drift, so one path from there would be a motionless dot. The
 * fan also shows *how* the table moves -- lines that speed up with distance, circles, a spiral --
 * which a single path from one place cannot.
 */
export function drawFanPreview(canvas, preset, phase, starts, pathOf) {
  const steps = 14;
  const paths = starts.map((start) => pathOf(preset, start, steps));
  const { width, height } = fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const felt = feltOf(preset.key);

  // The window is fixed by where the pucks *start*, not by how far they get. Fitting to the paths
  // would be fitting to the runaway: the Long Table's paths are straight lines that travel further
  // the further off the rail they start, so their bounding box is mostly empty space and the fan
  // collapses into one white band. Clipped to a window the same size as the starts, the four lines
  // read as what they are -- same direction, different speeds.
  const low = [Math.min(...starts.map((p) => p[0])), Math.min(...starts.map((p) => p[1]))];
  const high = [Math.max(...starts.map((p) => p[0])), Math.max(...starts.map((p) => p[1]))];
  const centre = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])];
  // The window is set by the table's own two lengths: how far the puck is from the centre (so a
  // circle fits) and how far one step carries it (so a run of steps is visible). A single constant
  // cannot do both -- the Roulette table's circles would shrink to a dot at the width the Long
  // Table's lines need.
  const reach = Math.max(...starts.map((p) => Math.hypot(p[0], p[1])), 1e-6);
  const step = Math.max(...paths.map((path) => Math.hypot(
    path[1][0] - path[0][0], path[1][1] - path[0][1])), 1e-6);
  const span = Math.max(2.4 * step, 1.35 * reach);
  const scale = Math.min(width, height) / (2 * span);
  const toPixel = (point) => [
    width / 2 + (point[0] - centre[0]) * scale,
    height / 2 - (point[1] - centre[1]) * scale,
  ];

  ctx.clearRect(0, 0, width, height);
  drawFelt(ctx, width, height, felt);

  const inside = (point) => Math.max(Math.abs(point[0] - centre[0]), Math.abs(point[1] - centre[1])) <= span;
  paths.forEach((path) => {
    ctx.beginPath();
    path.forEach((point, i) => {
      if (!inside(point)) return;
      const [x, y] = toPixel(point);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'rgba(246, 249, 255, 0.30)';
    ctx.lineWidth = Math.max(1, width / 150);
    ctx.stroke();

    const [sx, sy] = toPixel(path[0]);
    ctx.beginPath(); ctx.arc(sx, sy, Math.max(1.5, width / 90), 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(246, 249, 255, 0.55)';
    ctx.fill();

    const index = Math.min(path.length - 1, Math.floor(phase * path.length));
    if (!inside(path[index])) return;
    const [px, py] = toPixel(path[index]);
    ctx.beginPath(); ctx.arc(px, py, Math.max(1.6, width / 72), 0, 2 * Math.PI);
    ctx.fillStyle = PALETTE.puck;
    ctx.fill();
  });
}

/**
 * The running average score against the par line.
 *
 * One series and one reference, so one axis and no legend: the heading says what the line is and
 * the reference is labelled where it sits. The stake is *not* on this chart -- it is a different
 * quantity in different units, and two scales on one frame is the mistake this page refuses.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object[]} rounds settled rounds, each with `utility`
 * @param {number|null} par the predictability upper bound for this table and radius
 * @param {number|null} hover index of the round under the pointer, for the crosshair
 * @param {boolean} compact the corner-of-the-table version: the same two lines, with the reading
 *   furniture (grid lines and the par caption) left off, because at 220 by 62 they are the whole
 *   picture and there is no room for the words that explain it
 */
export function drawScoreChart(canvas, rounds, par, hover = null, compact = false) {
  const { width, height, ratio } = fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.font = `${(compact ? 10 : 11) * ratio}px ui-sans-serif, system-ui, sans-serif`;
  const pad = (compact ? 5 : 8) * ratio;

  if (rounds.length === 0) {
    ctx.fillStyle = PALETTE.dim;
    ctx.fillText(compact ? 'your running average' : 'your running average will be drawn here',
      pad, height / 2);
    return;
  }
  const running = [];
  let sum = 0;
  rounds.forEach((item, i) => {
    sum += Number.isFinite(item.utility) ? item.utility : -12;
    running.push(sum / (i + 1));
  });
  const values = par === null ? running : running.concat([par]);
  const low = Math.min(...values) - 0.4;
  const high = Math.max(...values) + 0.4;
  const y = (value) => height - pad - ((value - low) / (high - low)) * (height - 2 * pad);
  const x = (i) => pad + (i / Math.max(1, running.length - 1)) * (width - 2 * pad);

  if (!compact) {
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    for (const value of [low + 0.4, (low + high) / 2, high - 0.4]) {
      ctx.beginPath(); ctx.moveTo(0, y(value)); ctx.lineTo(width, y(value)); ctx.stroke();
    }
  }

  if (par !== null) {
    ctx.strokeStyle = CHART.reference;
    ctx.setLineDash([5 * ratio, 4 * ratio]);
    ctx.lineWidth = 2 * ratio;
    ctx.beginPath(); ctx.moveTo(0, y(par)); ctx.lineTo(width, y(par)); ctx.stroke();
    ctx.setLineDash([]);
    if (!compact) {
      ctx.fillStyle = CHART.reference;
      ctx.fillText('par — the best any hand could average', pad, y(par) - 8 * ratio);
    }
  }

  ctx.strokeStyle = CHART.series;
  ctx.lineWidth = 2 * ratio;
  ctx.beginPath();
  running.forEach((value, i) => (i === 0 ? ctx.moveTo(x(i), y(value)) : ctx.lineTo(x(i), y(value))));
  ctx.stroke();

  // A dot on the latest point, with a ring of the surface under it. Without it a run whose score
  // *is* par -- which is exactly what the safe hand achieves -- draws a line hidden under the par
  // line, and the chart looks broken at the moment it is being most informative.
  const last = running.length - 1;
  ctx.beginPath(); ctx.arc(x(last), y(running[last]), 4 * ratio, 0, 2 * Math.PI);
  ctx.fillStyle = CHART.series;
  ctx.fill();
  ctx.lineWidth = 2 * ratio;
  ctx.strokeStyle = PALETTE.panel;
  ctx.stroke();

  if (hover !== null && hover < running.length) {
    ctx.strokeStyle = PALETTE.pale;
    ctx.lineWidth = 1 * ratio;
    ctx.beginPath(); ctx.moveTo(x(hover), pad); ctx.lineTo(x(hover), height - pad); ctx.stroke();
    ctx.beginPath(); ctx.arc(x(hover), y(running[hover]), 4 * ratio, 0, 2 * Math.PI);
    ctx.fillStyle = CHART.series;
    ctx.fill();
  }
}

/** Where along the chart a pointer is: the round index, for the crosshair and the tooltip. */
export function scoreIndexAt(canvas, offsetX, count) {
  if (count === 0) return null;
  const ratio = canvas.width / Math.max(1, canvas.clientWidth);
  const pad = 8 * ratio;
  const usable = canvas.width - 2 * pad;
  const share = Math.min(1, Math.max(0, (offsetX * ratio - pad) / usable));
  return Math.round(share * Math.max(1, count - 1));
}

