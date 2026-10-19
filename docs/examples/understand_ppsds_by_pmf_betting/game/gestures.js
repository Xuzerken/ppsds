/**
 * Which gesture a pointer stream means. Pure: no DOM, no timers, no geometry.
 *
 * The shell feeds this file events and gets back *intents* ("place a stack", "turn the table",
 * "select that stack"), and it is the only thing in the page that decides what a touch means.
 * That split is not tidiness: thresholds are where a touch interface lives or dies, and a
 * threshold that can only be checked by playing on a phone does not get checked.
 *
 * Two rules came out of actually playing the thing, and both are the reason this file exists:
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
 *
 * The caller decides *what* was hit (`hit`) -- picking needs the renderer's projection -- and this
 * file decides what the sequence of hits and movements means.
 */

/** Movement, in CSS pixels, that separates a tap from a drag. Fingers are less precise than mice. */
export const SLOP = { mouse: 4, touch: 12 };

/** How long a press on a stack has to last to remove it. */
export const LONGPRESS_MS = 500;

/** A fresh gesture state. One per stage; the shell keeps it for the life of the page. */
export function createGestureState() {
  return {
    pointers: {}, // pointerId -> { x, y, startX, startY, touch }
    mode: 'idle', // 'idle' | 'pending' | 'drag' | 'orbit' | 'pinch' | 'spent'
    target: null, // index of the stack a drag is carrying
    moved: 0, // the largest distance from the start point, in CSS pixels
    consumed: false, // a long press has already decided this gesture
  };
}

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const count = (state) => Object.keys(state.pointers).length;
const slopOf = (state) => (state.pointers[Object.keys(state.pointers)[0]]?.touch ? SLOP.touch : SLOP.mouse);

/**
 * A pointer went down.
 *
 * @param {object} state the gesture state (mutated)
 * @param {{id: number, x: number, y: number, touch?: boolean, hit?: number|null}} event
 * @returns {{kind: string, [key: string]: any}} the intent
 */
export function pointerDown(state, event) {
  const point = { x: event.x, y: event.y, startX: event.x, startY: event.y, touch: Boolean(event.touch) };
  state.pointers[event.id] = point;
  state.moved = 0;
  state.consumed = false;

  if (count(state) >= 2) {
    // A second finger means zooming, and it cancels whatever the first one was doing: a pinch that
    // turns into a drag (or leaves a stack half-moved) is the classic multi-touch wart.
    const wasDragging = state.mode === 'drag' && state.target !== null;
    const target = state.target;
    state.mode = 'pinch';
    state.target = null;
    return wasDragging ? { kind: 'release', index: target } : { kind: 'none' };
  }

  state.mode = 'pending';
  state.target = event.hit === null || event.hit === undefined ? null : event.hit;
  // Selecting on the way down, not on the way up: the ring has to appear under the finger, and
  // "did that register?" is exactly the doubt a touch interface cannot afford.
  return state.target === null ? { kind: 'none' } : { kind: 'select', index: state.target };
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
  if (state.mode === 'spent') return { kind: 'none' };

  const fromStart = distance(point, { x: point.startX, y: point.startY });
  state.moved = Math.max(state.moved, fromStart);
  if (state.moved <= slopOf(state)) return { kind: 'none' }; // still a tap, whatever else happens

  if (state.mode === 'pending') {
    state.mode = state.target === null ? 'orbit' : 'drag';
    // The first turn of the table starts from *here*, so entering a drag does not jump the view by
    // the slop distance.
    return state.mode === 'drag' ? { kind: 'drag-start', index: state.target } : { kind: 'none' };
  }
  if (state.mode === 'drag') {
    return { kind: 'drag', index: state.target, dx: point.x - previous.x, dy: point.y - previous.y };
  }
  if (state.mode === 'orbit') {
    return { kind: 'orbit', dx: point.x - previous.x, dy: point.y - previous.y };
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
  const wasPending = state.mode === 'pending';
  const target = state.target;
  reset(state);

  if (tapped && wasPending) return { kind: 'place', index: target };
  return { kind: 'none' };
}

/**
 * The system took the pointer away (a notification, an edge gesture).
 *
 * @returns {{kind: string, [key: string]: any}}
 */
export function pointerCancel(state, event = {}) {
  const target = state.mode === 'drag' ? state.target : null;
  if (event.id !== undefined) delete state.pointers[event.id];
  if (count(state) > 0) {
    state.mode = 'pinch';
    state.target = null;
    return { kind: 'none' };
  }
  reset(state);
  return target === null ? { kind: 'cancel' } : { kind: 'release', index: target };
}

/**
 * The long-press timer fired. The caller owns the timer; this decides whether it means anything.
 *
 * @returns {{kind: string, [key: string]: any}}
 */
export function longPress(state) {
  if (state.mode !== 'pending' || state.target === null || state.consumed) return { kind: 'none' };
  state.consumed = true;
  return { kind: 'remove', index: state.target };
}

/** Whether the gesture is still a tap so far -- the caller's long-press timer reads this to
 *  decide if the finger has wandered off. */
export function stillTapping(state) {
  return state.moved <= slopOf(state);
}

/** How far apart the two fingers are, for a pinch. */
function pinch(state) {
  const ids = Object.keys(state.pointers);
  if (ids.length < 2) return { kind: 'none' };
  const [a, b] = ids.map((id) => state.pointers[id]);
  return { kind: 'pinch', distance: distance(a, b) };
}

function reset(state) {
  state.mode = 'idle';
  state.target = null;
  state.moved = 0;
  state.consumed = false;
}
