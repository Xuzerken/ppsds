/**
 * The first-visit tour: a tiny, pure state machine and the five things it asks you to do.
 *
 * The page has a guide already, but a guide is a thing you open *after* you have decided the page
 * is worth learning. A player's first thirty seconds happen before that decision, so the first
 * round gets a coach mark instead: non-blocking, closeable, and re-openable from the guide.
 *
 * The steps are gates, not pages. `tutorialReady(state, context)` says whether the player has done
 * what the current step asks -- placed a stack, covered the support, changed a height, dealt,
 * moved on -- and `tutorialNext` refuses to advance until it is true. The shell enables the Next
 * button from the same predicate, so the tour never asks the player to read something and take it
 * on faith: a step is complete when the table says it is.
 *
 * Nothing here is persisted. The page's promise is that a visit leaves nothing behind, so "first
 * visit" means "this visit": every boot starts the tour again, and Skip lasts until the tab closes.
 *
 * The module is pure -- no DOM, no timers, no storage -- so the order of the steps, their gates and
 * the way back are testable in node. The shell owns the element, the corner it sits in, and the
 * clock.
 */

export const TOUR_STEPS = [
  {
    id: 'prediction',
    text: 'Turn on prediction. It draws where the drift alone would put the puck, and the noise '
      + 'support around it: the gold region is everywhere the law says the puck can land.',
    covers: ['prediction', 'support'],
    needs: 'prediction',
    target: 'prediction',
  },
  {
    id: 'place',
    text: 'Tap the felt to put a stack where you think the puck lands. '
      + 'Drag a stack to move it, drag the felt to turn the table, and scroll or pinch to zoom.',
    covers: ['tap', 'place', 'drag-stack', 'orbit', 'wheel', 'pinch'],
    needs: 'stack',
    target: 'felt',
  },
  {
    id: 'cover',
    text: 'Keep adding stacks until every landing is inside a payout square -- the badge at the '
      + 'lower left stops saying "Not safe" and the table is safe.',
    covers: ['cover', 'safe'],
    needs: 'safe',
    target: 'felt',
  },
  {
    id: 'height',
    text: 'Tap a stack to open its toolbar, then drag the slider or use − / +. '
      + 'The height is its weight; the percentage is its share of the whole bet.',
    covers: ['select-stack', 'height', 'toolbar'],
    needs: 'height',
    target: 'felt',
  },
  {
    id: 'deal',
    text: 'Press Deal. The ball lands, and every stack within ε of it pays its share of the bet.',
    covers: ['deal'],
    needs: 'deal',
    target: 'deal',
  },
  {
    id: 'next',
    text: 'Press Next round. Keep my layout is on, so this hand comes back next round, '
      + 're-anchored on the new prediction.',
    covers: ['result', 'next-round', 'keep-layout'],
    needs: 'next-round',
    target: 'deal',
  },
];

/**
 * What each `needs` key asks of the table.
 *
 * The shell supplies the context from the game's own state, so a step is complete because the
 * table is in that state -- not because an event was seen. That distinction is what keeps "put a
 * stack down" from staying done after the player clears the table.
 */
const NEEDS = {
  prediction: (context) => Boolean(context.prediction),
  stack: (context) => (context.stacks || 0) > 0,
  safe: (context) => Boolean(context.safe),
  height: (context) => Boolean(context.heightChanged),
  deal: (context) => (context.deals || 0) > 0,
  'next-round': (context) => (context.nextRounds || 0) > 0,
};

const copy = (state) => ({
  enabled: Boolean(state.enabled),
  index: Number(state.index) || 0,
  done: Boolean(state.done),
});

/** A fresh tour. `enabled: false` means the player turned teaching off before it began. */
export function createTutorial({ enabled = true } = {}) {
  return { enabled: Boolean(enabled), index: 0, done: !enabled };
}

/** The step to show, or `null` when the tour is off or finished. */
export function tutorialStep(state) {
  if (!state || !state.enabled || state.done) return null;
  return TOUR_STEPS[state.index] || null;
}

/** Whether the current step's action has been done. `Next` is disabled until it has. */
export function tutorialReady(state, context = {}) {
  const step = tutorialStep(state);
  return step !== null && Boolean(NEEDS[step.needs](context));
}

/** Advance one step -- but only once the current step's own action is done. */
export function tutorialNext(state, context = {}) {
  const next = copy(state);
  if (!next.enabled || next.done) return next;
  if (!tutorialReady(state, context)) return next;
  next.index += 1;
  next.done = next.index >= TOUR_STEPS.length;
  return next;
}

/**
 * Go back one step. The table is untouched, so a step already done stays done: going back is for
 * re-reading, not for undoing the round.
 */
export function tutorialBack(state) {
  const next = copy(state);
  if (!next.enabled || next.done) return next;
  next.index = Math.max(0, next.index - 1);
  return next;
}

/** Turn teaching on (restarting from the first step) or off (ending it for this visit). */
export function tutorialSetEnabled(state, enabled) {
  if (enabled) return { enabled: true, index: 0, done: false };
  return { enabled: false, index: 0, done: true };
}

/** "Show me the tour again", from the guide. */
export function tutorialReopen(state) {
  return tutorialSetEnabled(state, true);
}

/**
 * Skip: off for this visit, not a preference that survives a reload.
 *
 * The shell treats "teaching is off" -- by Skip or by the switch -- as "do not show the static
 * Place your stacks hint either": the player has said, in the clearest available way, that they
 * do not want to be talked at. Turning teaching back on brings both back.
 */
export function tutorialSkip(state) {
  return tutorialSetEnabled(state, false);
}

/**
 * Where the bubble sits, in viewport pixels: the stage's lower-left corner, just above the dock.
 *
 * A corner is the one place a step is never talking about -- the felt, the hand and the toolbar are
 * all elsewhere -- so the card can stay put instead of following its subject. Pure, so the corner
 * is checkable without a browser; the shell owns the elements and the rects.
 */
export function tourSpot(stageRect, dockRect, ownRect, margin = 12) {
  const own = ownRect || { width: 0, height: 0 };
  return {
    left: stageRect.left + margin,
    // Clamped to the stage so a very short table still shows the top of the card rather than
    // pushing it off the felt; on any table with room, the dock decides.
    top: Math.max(stageRect.top + margin, dockRect.top - margin - own.height),
  };
}
