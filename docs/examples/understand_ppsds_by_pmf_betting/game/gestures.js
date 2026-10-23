/**
 * Which gesture a pointer stream means. Pure: no DOM, no timers, no geometry.
 *
 * The shell feeds this file events and gets back *intents* ("place a stack", "turn the table",
 * "select that stack"), and it is the only thing in the page that decides what a touch means.
 * That split is not tidiness: thresholds are where a touch interface lives or dies, and a
 * threshold that can only be checked by playing on a phone does not get checked.
 *
 * Three rules came out of actually playing the thing, and all three are the reason this file
 * exists:
 *
 * * **Displacement is measured from where the pointer went down, not between events.** The first
 *   version compared each event to the previous one, so the 3-8 px of jitter in a finger tap
 *   counted as movement: on a phone, tapping the felt often turned the table instead of placing a
 *   stack, and the player had no way to tell why.
 * * **Touching a stack selects it and does nothing else.** Adding to a stack on tap is one
 *   mis-touch away from a different hand, and it cannot be undone by the player's own fingers; the
 *   height changes from the toolbar, where it takes a deliberate press. That promise used to be
 *   kept by the toolbar and broken by the tap itself: the tap selected the stack *and* placed a
 *   new one in the same cell, underneath it. Which is why `pointerUp` now reports what it hit.
 * * **In Select mode a tap toggles membership on the way up, not down.** A drag on a selected
 *   stack is the gesture that moves the whole group, so it cannot also be the gesture that
 *   deselects the stack it started on. A press records the candidate; only a tap that never became
 *   a drag flips it.
 *
 * The caller decides *what* was hit (`hit`) -- picking needs the renderer's projection -- and this
 * file decides what the sequence of hits and movements means.
 */

/** Movement, in CSS pixels, that separates a tap from a drag. Fingers are less precise than mice. */
export const SLOP = { mouse: 4, touch: 12 };

/** How much of a pinch's ratio reaches the camera. Below 1, a spread has to work for it. */
export const PINCH_GAIN = 0.8;

/**
 * A pinch hands in the *previous* and current finger distance and gets back the zoom factor this
 * one event should apply.
 *
 * The ratio is between two samples, never between the current sample and where the pinch began.
 * That is the whole point of the function: the first version kept the starting distance and
 * multiplied by `start / now` on every move, so a pinch that opened 10% in two steps zoomed by
 * 1.1 × 1.1. A gesture that accelerates the longer you hold it reads as "too fast and imprecise",
 * which is exactly how it was reported from a phone. Multiplying per-event ratios is the only
 * form that composes: `pinchZoom(a, b) * pinchZoom(b, c) === pinchZoom(a, c)`.
 */
export function pinchZoom(previous, current) {
  return previous > 0 && current > 0 ? (previous / current) ** PINCH_GAIN : 1;
}

/** How much of a wheel pixel reaches the zoom. Small: a trackpad sends dozens of events per gesture. */
export const WHEEL_GAIN = 0.0018;

/**
 * A wheel event's magnitude, not just its sign.
 *
 * `zoom(deltaY > 0 ? 1.08 : 0.93)` treats a trackpad's 2 px delta and a mouse wheel's 120 px notch
 * as the same step, so the trackpad flew and the wheel crawled. The delta is clamped because a
 * flick can report thousands of pixels in one event.
 */
export function wheelZoom(deltaY) {
  const step = Math.max(-60, Math.min(60, Number(deltaY) || 0));
  return Math.exp(step * WHEEL_GAIN);
}

/** A fresh gesture state. One per stage; the shell keeps it for the life of the page. */
export function createGestureState() {
  return {
    pointers: {}, // pointerId -> { x, y, startX, startY, touch }
    mode: 'idle', // 'idle' | 'pending' | 'select-pending' | 'drag' | 'orbit' | 'pinch' | 'marquee'
    target: null, // index of the stack a drag is carrying
    moved: 0, // the largest distance from the start point, in CSS pixels
    selectMode: false, // the mode captured when this gesture started
    shift: false, // Shift was held when the pointer went down: a desktop drag pans, not turns
  };
}

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const count = (state) => Object.keys(state.pointers).length;
const slopOf = (state) => (state.pointers[Object.keys(state.pointers)[0]]?.touch ? SLOP.touch : SLOP.mouse);
const rectOf = (point, x, y) => ({ x0: point.startX, y0: point.startY, x1: x, y1: y });

/**
 * A pointer went down.
 *
 * @param {object} state the gesture state (mutated)
 * @param {{id: number, x: number, y: number, touch?: boolean, hit?: number|null,
 *          shift?: boolean, selectMode?: boolean}} event
 * @returns {{kind: string, [key: string]: any}} the intent
 */
export function pointerDown(state, event) {
  const point = { x: event.x, y: event.y, startX: event.x, startY: event.y, touch: Boolean(event.touch) };
  state.pointers[event.id] = point;
  state.moved = 0;
  state.selectMode = Boolean(event.selectMode);
  state.shift = Boolean(event.shift);

  if (count(state) >= 2) {
    // A second finger means zooming, and it cancels whatever the first one was doing: a pinch that
    // turns into a drag (or leaves a stack half-moved, or a marquee half-drawn) is the classic
    // multi-touch wart.
    const wasDragging = state.mode === 'drag' && state.target !== null;
    const target = state.target;
    state.mode = 'pinch';
    state.target = null;
    return wasDragging ? { kind: 'release', index: target } : { kind: 'none' };
  }

  state.target = event.hit === null || event.hit === undefined ? null : event.hit;
  if (state.target === null) {
    state.mode = 'pending';
    return { kind: 'none' };
  }
  // Selecting on the way down, not on the way up: the ring has to appear under the finger, and
  // "did that register?" is exactly the doubt a touch interface cannot afford. Select mode is the
  // exception -- there a press may still become a group drag, so membership waits for the tap.
  state.mode = state.selectMode ? 'select-pending' : 'pending';
  return state.selectMode
    ? { kind: 'none' }
    : { kind: 'select', index: state.target, additive: Boolean(event.shift) };
}

/**
 * A pointer moved.
 *
 * @returns {{kind: string, [key: string]: any}}
 */
export function pointerMove(state, event) {
  const point = state.pointers[event.id];
  if (!point) return { kind: 'none' };
  const previous = { x: point.x, y: point.y };
  point.x = event.x;
  point.y = event.y;

  if (state.mode === 'pinch') return pinch(state);

  const fromStart = distance(point, { x: point.startX, y: point.startY });
  state.moved = Math.max(state.moved, fromStart);
  if (state.moved <= slopOf(state)) return { kind: 'none' }; // still a tap, whatever else happens

  if (state.mode === 'marquee') {
    return { kind: 'marquee', rect: rectOf(point, point.x, point.y) };
  }
  if (state.mode === 'pending') {
    if (state.target === null) {
      if (state.selectMode) {
        state.mode = 'marquee';
        return { kind: 'marquee-start', rect: rectOf(point, point.x, point.y) };
      }
      state.mode = 'orbit';
      // The first turn of the table starts from *here*, so entering an orbit does not jump the view
      // by the slop distance.
      return { kind: 'none' };
    }
    state.mode = 'drag';
    return { kind: 'drag-start', index: state.target };
  }
  if (state.mode === 'select-pending') {
    // The press became a drag, which is how a selected group is moved; it is not a toggle.
    state.mode = 'drag';
    return { kind: 'drag-start', index: state.target };
  }
  if (state.mode === 'drag') {
    return { kind: 'drag', index: state.target, dx: point.x - previous.x, dy: point.y - previous.y };
  }
  if (state.mode === 'orbit') {
    // Same drag, two meanings: turning the table shows the towers, moving it gets the hand out
    // from under the tour card. Shift is the desktop's way of asking for the second one.
    return {
      kind: 'orbit', dx: point.x - previous.x, dy: point.y - previous.y, shift: state.shift,
    };
  }
  return { kind: 'none' };
}

/**
 * A pointer came up.
 *
 * A tap is reported **with what it landed on**, because the two halves of "tap the felt" mean
 * different things and only this file knows which one happened: a tap on bare felt places a stack,
 * and a tap on a stack does not touch the hand at all. Leaving the hit out of the report is what
 * used to put a second stack underneath the first one -- a tap selected a stack on the way down and
 * then placed a new one on the way up, in the same cell, on top of it.
 *
 * @returns {{kind: string, [key: string]: any}}
 */
export function pointerUp(state, event) {
  const point = state.pointers[event.id];
  delete state.pointers[event.id];
  if (!point) return { kind: 'none' };

  if (state.mode === 'pinch') {
    // After a pinch, lifting one finger does *not* hand the drag back to the other: a pinch
    // followed by a stray drag would carry a stack across the table.
    if (count(state) > 0) return { kind: 'none' };
    reset(state);
    return { kind: 'none' };
  }

  const moved = distance(point, { x: point.startX, y: point.startY });
  const tapped = moved <= slopOf(state);
  const mode = state.mode;
  const target = state.target;
  const x = event.x === undefined ? point.x : event.x;
  const y = event.y === undefined ? point.y : event.y;
  reset(state);

  if (mode === 'marquee') {
    return tapped
      ? { kind: 'clear-selection' }
      : { kind: 'marquee-end', rect: rectOf(point, x, y) };
  }
  if (mode === 'select-pending') {
    return tapped && target !== null
      ? { kind: 'select', index: target, additive: true }
      : { kind: 'none' };
  }
  if (tapped && mode === 'pending') return { kind: 'place', index: target };
  return { kind: 'none' };
}

/**
 * The system took the pointer away (a notification, an edge gesture).
 *
 * @returns {{kind: string, [key: string]: any}}
 */
export function pointerCancel(state, event = {}) {
  const mode = state.mode;
  const target = mode === 'drag' ? state.target : null;
  if (event.id !== undefined) delete state.pointers[event.id];
  if (count(state) > 0) {
    state.mode = 'pinch';
    state.target = null;
    return mode === 'marquee' ? { kind: 'marquee-cancel' } : { kind: 'none' };
  }
  reset(state);
  if (mode === 'marquee') return { kind: 'marquee-cancel' };
  return target === null ? { kind: 'cancel' } : { kind: 'release', index: target };
}

/**
 * The two fingers' spread *and* their midpoint, for a pinch.
 *
 * The midpoint is what makes two-finger pan possible: the spread decides the zoom, and where the
 * pair as a whole has moved decides how far the table slides. Reporting only the distance is what
 * left the table unpannable -- the one gesture a phone has for "move this out of the way".
 */
function pinch(state) {
  const ids = Object.keys(state.pointers);
  if (ids.length < 2) return { kind: 'none' };
  const [a, b] = ids.map((id) => state.pointers[id]);
  return {
    kind: 'pinch',
    distance: distance(a, b),
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  };
}

function reset(state) {
  state.mode = 'idle';
  state.target = null;
  state.moved = 0;
  state.selectMode = false;
  state.shift = false;
}
