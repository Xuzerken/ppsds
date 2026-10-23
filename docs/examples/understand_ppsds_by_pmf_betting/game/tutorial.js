/**
 * The first-visit tour: a tiny, pure state machine and the six things it asks you to do.
 *
 * The page has a guide already, but a guide is a thing you open *after* you have decided the page
 * is worth learning. A player's first thirty seconds happen before that decision, so the first
 * round gets a coach mark instead: non-blocking, closeable, and re-openable from the guide.
 *
 * The steps are gates, not pages. `tutorialReady(state, context)` says whether the player has done
 * what the current step asks -- placed a stack, covered the support, changed a height, dealt,
 * moved on -- and `tutorialNext` refuses to advance until it is true. The shell watches the same
 * predicate on its own clock and advances when it turns true, so a step is complete when the table
 * says it is, and the card carries no button for going on. There is no going back for the same
 * reason: the gate that was just met would push the tour forward again a beat later.
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
    text: 'Turn on prediction — the gold region is everywhere the law lets the puck land.',
    covers: ['prediction', 'support'],
    needs: 'prediction',
    target: 'prediction',
  },
  {
    id: 'place',
    text: 'Tap the felt to place a stack. Drag the felt to turn it; two fingers move and zoom.',
    covers: ['tap', 'place', 'orbit', 'pinch', 'pan'],
    needs: 'stack',
    target: 'felt',
  },
  {
    id: 'cover',
    text: 'Cover every landing; the badge stops warning when the table is safe.',
    covers: ['cover', 'safe'],
    needs: 'safe',
    target: 'felt',
  },
  {
    id: 'height',
    text: 'Tap a stack to set its height — the height is its weight.',
    covers: ['select-stack', 'height', 'toolbar'],
    needs: 'height',
    target: 'felt',
  },
  {
    id: 'deal',
    text: 'Press Deal. Stacks within ε of the ball pay their share.',
    covers: ['deal'],
    needs: 'deal',
    target: 'deal',
  },
  {
    id: 'next',
    text: 'Press Next round — your hand comes back on the new prediction.',
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
