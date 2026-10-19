/**
 * The game: state, the round flow, and the wiring to the page.
 *
 * Two halves, and the split is deliberate:
 *
 * * a **pure core** -- a plain state object plus functions that move it along (place a stack, deal,
 *   settle, next round, build a view). No DOM, no canvas, no `window`. `tests/test_game_page.py`
 *   imports this half in node and plays whole runs against the exported assets, which is the only
 *   way to test a round flow without a browser;
 * * a **shell** (`boot`) -- event listeners, the renderers, the animation clock, the HUD. It is
 *   the only part that touches the document, and it decides nothing that matters.
 *
 * A stack is an atom with a **continuous weight**, stored as an offset from the predicted point.
 * Not a count of stacks: the score is the *share* of the hand that landed within `eps`, so only the
 * ratios between stacks are observable, and the page never asks the player to think about an
 * absolute size. What is drawn is the height, and what is written on the stack is the share it
 * works out to. Storing offsets (rather than absolute places) is what makes a hand a rule: the same
 * shape means the same thing every round and on every run, so "Copy this round" and "Export this
 * run" can be scored against thousands of trajectories by `run.py`.
 */

import { coveringLayout, gaussianMissProbability, uncoveredReport } from './coverage.js';
import { formatSelfCheck, mean, roundStats, selfCheck } from './scoring.js';
import { createFlatScene, drawFanPreview, drawScoreChart, scoreIndexAt } from './scene2d.js';
import { feltOf } from './theme.js';
import { WEIGHT_CEILING, WEIGHT_MIN, WEIGHT_STEP, discFootprint, discRadius, discPlan,
  fanStarts, frameBox, grabReach, latticeStep, nearestIndex, nominalNext, nominalPath, noiseProse,
  panelWidth, safeBox, sameHand, shares, sliderFromWeight, smoothstep, snapToLattice, stackHeight,
  supportBox, viewSpan, weightFromSlider } from './world.js';
import { LONGPRESS_MS, createGestureState, longPress, pointerCancel, pointerDown,
  pointerMove, pointerUp, stillTapping } from './gestures.js';

/** How far ahead the aim assist draws the drift's path. */
const PATH_STEPS = 5;

/** How many rounds a run lasts, and the default. The replays are exported long enough for the
 *  longest of them (`EXPORT_STEPS` on the Python side), so this is a choice, not a limit. */
export const RUN_LENGTHS = [10, 20, 30, 50];
export const DEFAULT_RUN_LENGTH = 20;

/** The two ways the table can be drawn, in the order the picker offers them. It is a setting and not
 *  a fallback: a reader on a machine without a GPU, or one who wants the table legible at a glance,
 *  should be able to ask for the flat one. */
export const SURFACES = [['3d', '3-D table'], ['flat', 'Flat table']];

/** The weight "Cover the table" gives every stack. Equal weights are the whole point of that
 *  layout -- it is the one that cannot miss -- so they are all the same number. */
const COVER_WEIGHT = 1;

/** The weight a stack the player just put down gets. One, so a fresh stack weighs the same as a
 *  covered one and the shares say what the shape says. */
const NEW_STACK_WEIGHT = 1;

/** What one press of −/＋ multiplies a stack's height by. Multiplicative because the slider's range
 *  is a ratio -- a fixed step would crawl at the top and leap at the bottom. */
export const NUDGE_FACTOR = 1.25;

/** The throw and the settlement, in milliseconds, and the change between two rounds.
 *
 *  The order matters: a deal during a change ends the change, so a round is never settled while the
 *  hand is still arriving, and both are well inside auto-play's own beat. */
export const THROW_MS = 820;
export const SETTLE_MS = 900;
/** The three beats of a new round: the hand leaves, the camera moves over an empty table, the hand
 *  arrives. Hiding the discontinuity is the whole trick -- a hand that slides while the camera
 *  swings is two things moving at once, and the eye believes neither. */
export const HAND_OUT_MS = 220;
export const CAMERA_MOVE_MS = 380;
export const HAND_IN_MS = 220;
export const TRANSITION_MS = HAND_OUT_MS + CAMERA_MOVE_MS + HAND_IN_MS;
/** How long a stack takes to acknowledge a touch: a short scale that says "I have it". */
export const PRESS_MS = 240;

const clone = (stacks) => stacks.map((stack) => ({ ...stack }));
const to = (value, places) => Number(value.toFixed(places));

/**
 * A game over the exported world. Pure: it takes the assets and returns functions over them.
 *
 * @param {{presets: object, trajectories: object, selfcheck: object[]}} assets
 */
export function createGame(assets) {
  const state = {
    presetKey: Object.keys(assets.presets)[0],
    eps: 0,
    runIndex: 0,
    step: 0,
    stacks: [], // {x, y, w}: offsets from the predicted point, w a continuous weight
    phase: 'bet', // 'bet' | 'dealing' | 'settled' | 'over'
    selected: null,
    hover: null, // the stack under the cursor, for the one renderer that has a cursor
    pressedAt: -Infinity, // when the hand was last touched, for the tap's acknowledgement
    aimAssist: true,
    showHistory: true,
    keepLayout: false,
    autoPlay: false,
    snapping: true,
    runLength: DEFAULT_RUN_LENGTH,
    endReason: null, // 'played-out' | 'ruined' once the run is over
    rounds: [], // settled rounds of this run
    undo: [],
    // The glide that opens a round: `null` once it is over. See `nextRound`.
    transition: null, // {startedAt, from: [x, y], out: [{x, y, w}]}
    dealStarted: 0,
    settledAt: 0,
    landed: null,
    result: null,
    lastLayout: null, // the previous round's hand, for "same as last round"
    // The quick layout the table is currently holding, if it is still holding it untouched:
    // {name, before, produced}. See `applyPreset` -- it is what makes a second press undo.
    presetApplied: null,
  };

  const preset = () => assets.presets[state.presetKey];
  const trajectory = () => assets.trajectories[state.presetKey];
  /** Rounds in a run: what the player asked for, capped by what the replay holds. */
  const stepCount = () => Math.min(state.runLength, trajectory().noise[state.runIndex].length);
  const isSecret = () => Boolean(preset().hidden_noise);
  const here = () => trajectory().states[state.runIndex][state.step];
  const noise = () => trajectory().noise[state.runIndex][state.step];
  const predicted = () => nominalNext(preset(), here());

  /**
   * The reference span every piece of geometry is measured against.
   *
   * It is `viewSpan` -- the frame a table with nothing on it needs -- and deliberately *not* the
   * wider frame the view model ends up with, which depends on where the stacks are. A stack's
   * radius, the lattice it lands on and the reach that grabs it have to be the same numbers before
   * and after it is placed, or the rule would chase its own tail.
   */
  const geometrySpan = () => viewSpan(here(), predicted(), state.eps);

  const stateAt = (step) => trajectory().states[state.runIndex][step];
  const noiseAt = (step) => trajectory().noise[state.runIndex][step];
  const predictedAt = (step) => nominalNext(preset(), stateAt(step));
  /** The landing in the state space: the predicted point plus the noise that actually happened. */
  const landingAt = (step) => {
    const base = predictedAt(step);
    const draw = noiseAt(step);
    return [base[0] + draw[0], base[1] + draw[1]];
  };

  /** Where a stack sits on the table: its stored offset, drawn back at this round's prediction. */
  const restingAt = (stack, base = predicted()) => [base[0] + stack.x, base[1] + stack.y];
  const resting = (base) => state.stacks.map((stack) => restingAt(stack, base));

  /** The lattice a tap lands on, and how far a press may be from a stack and still hold it. */
  const lattice = () => latticeStep(state.eps, geometrySpan(), state.snapping);
  const footprint = () => discFootprint(state.eps, geometrySpan());

  /** Drop a selection or a hover that no longer points at a stack (every mutating path calls it). */
  function clampCursors() {
    if (state.selected !== null && state.selected >= state.stacks.length) state.selected = null;
    if (state.hover !== null && state.hover >= state.stacks.length) state.hover = null;
  }

  /** A hand edit: the glide stops being about a round that has already happened. */
  function endGlide() {
    state.transition = null;
  }

  function snapshot() {
    state.undo.push({ stacks: clone(state.stacks), step: state.step, rounds: state.rounds.length });
    if (state.undo.length > 50) state.undo.shift();
    // Every edit that is not the quick layout itself goes through here, so this one line is the
    // whole of "you changed it, so the button no longer means 'take it back'".
    state.presetApplied = null;
  }

  /**
   * Open one undo step for the gesture that is about to start.
   *
   * Called once per gesture, not once per event: dragging a stack emits a move on every pointer
   * frame, and a dragged stack that cannot be undone (or that needs fifty undos to undo) is the
   * kind of thing a player notices immediately.
   */
  function markEdit() {
    snapshot();
  }

  /** Mark the hand as just touched: the shell turns this into a short scale on the stack. */
  function touch(now = 0) {
    state.pressedAt = now;
    endGlide();
  }

  function setStacks(stacks, { keepUndo = false } = {}) {
    if (!keepUndo) snapshot();
    state.stacks = clone(stacks).filter((stack) => stack.w > 0);
    state.result = null;
    clampCursors();
  }

  // -- the round flow ----------------------------------------------------------------------

  function deal(now = 0) {
    if (state.phase !== 'bet' || state.stacks.length === 0) return false;
    // A deal ends whatever the change was doing: the player has asked to play, so the hand comes
    // back at once rather than arriving while the ball is in the air.
    endGlide();
    state.phase = 'dealing';
    state.dealStarted = now;
    state.landed = landingAt(state.step);
    return true;
  }

  /** Called by the shell when the throw lands: settle the round and bank the numbers. */
  function settle(now = 0) {
    if (state.phase !== 'dealing') return null;
    const draw = noise();
    const stats = roundStats(state.stacks, draw, state.eps);
    state.result = {
      ...stats,
      perStack: state.stacks.map((stack) => (
        Math.max(Math.abs(stack.x - draw[0]), Math.abs(stack.y - draw[1])) <= state.eps
      )),
    };
    state.rounds.push({
      step: state.step,
      stacks: clone(state.stacks),
      roi: stats.roi,
      utility: stats.utility,
    });
    state.lastLayout = clone(state.stacks);
    state.settledAt = now;

    // A run ends here, rather than silently starting another one. It ends for one of two
    // reasons, and both have been promised by the page's own text since the first version:
    //   * the rounds ran out -- that is simply the end;
    //   * the round paid nothing. The score is then -inf, which is not a number you can average
    //     and not a stake you can keep betting, so the run is over rather than dented.
    if (stats.roi === 0) {
      state.phase = 'over';
      state.endReason = 'ruined';
    } else if (state.step + 1 >= stepCount()) {
      state.phase = 'over';
      state.endReason = 'played-out';
    } else {
      state.phase = 'settled';
      state.endReason = null;
    }
    return state.result;
  }

  /**
   * The next round of *this* run. At the end of a run there is no next round: `restart()` is a
   * separate, explicit action, because a run that quietly restarts itself is not a run.
   *
   * The hand is re-anchored here, and that is the one thing on this page that used to jump:
   * offsets are measured from the predicted point, so a kept hand moves by however far the
   * prediction moved -- up to a third of the view, every round. Nothing can make that move look
   * like a move, so the page does not try: the hand leaves, the camera moves over the empty
   * table, and the hand arrives already in its new place (see `glideAt`). The `out` handed over
   * is the hand that was on the table, whether or not it is coming back -- it is what the first
   * beat fades away.
   */
  function nextRound(now = 0) {
    if (state.phase !== 'settled') return false;
    const from = predicted();
    state.step += 1;
    const kept = state.keepLayout && state.lastLayout ? state.lastLayout : [];
    return beginRound(kept, { from, startedAt: now, out: clone(state.stacks) });
  }

  function restart(now = 0) {
    const from = predicted();
    const kept = state.keepLayout && state.lastLayout ? state.lastLayout : [];
    const out = clone(state.stacks);
    state.rounds = [];
    state.undo = [];
    state.step = 0;
    state.endReason = null;
    return beginRound(kept, { from, startedAt: now, out });
  }

  function beginRound(stacks, { from = null, startedAt = 0, out = [] } = {}) {
    state.phase = 'bet';
    state.stacks = clone(stacks);
    state.landed = null;
    state.result = null;
    state.selected = null;
    state.presetApplied = null;
    // A round only changes with a beat when the world stayed put: changing table, radius or run is
    // a new game and reads as one -- `selectTable` and friends call this bare.
    state.transition = from === null ? null : { startedAt, from, out };
    clampCursors();
    return true;
  }

  const atEnd = () => state.phase === 'over';

  // -- the hand ----------------------------------------------------------------------------

  /** Where a tap lands: on the lattice this table and this snap setting call for. */
  function snapPoint(point) {
    return snapToLattice(point, predicted(), lattice());
  }

  /**
   * The stack whose square a point falls in, or `null`.
   *
   * The test is the stack's own footprint, so "inside" means "inside the space this stack owns"
   * and two stacks that both pass it are two stacks that overlap.
   */
  function stackAt(point) {
    return nearestIndex(point, resting(), footprint());
  }

  /**
   * Put a stack down where the player tapped.
   *
   * A tap resolves to a cell, and a cell belongs to one stack: if the resolved point lands in one
   * that is taken, this is a tap *on that stack*, and it reports which -- the shell selects it. It
   * does not merge, and it does not nudge: a stack's height only ever changes from the toolbar,
   * where the player can see what they are doing.
   *
   * @returns {{placed: boolean, index: number|null}} `index` is the stack that owns the cell when
   *   the placement was refused, so the caller can say what the tap meant instead.
   */
  /**
   * Hold a stack, or let it go.
   *
   * A second tap on the stack that is already held puts it down again -- which is the only way to
   * close the toolbar without reaching for the Move/Remove buttons you were trying to get rid of.
   */
  function toggleSelection(index) {
    if (state.phase !== 'bet' || index === null) return false;
    state.selected = state.selected === index ? null : index;
    return true;
  }

  function placeStack(point, now = 0) {
    if (state.phase !== 'bet') return { placed: false, index: null };
    const base = predicted();
    const snapped = snapPoint(point);
    const taken = stackAt(snapped);
    if (taken !== null) return { placed: false, index: taken };
    snapshot(); // one placement, one undo -- this used to record nothing unless the table was bare
    state.stacks.push({
      x: to(snapped[0] - base[0], 9),
      y: to(snapped[1] - base[1], 9),
      w: NEW_STACK_WEIGHT,
    });
    state.result = null;
    // A new stack is not the held one: its toolbar would cover the thing the player just put down.
    state.selected = null;
    touch(now);
    return { placed: true, index: state.stacks.length - 1 };
  }

  /** Carry a stack to a new cell. A cell that belongs to somebody else is not a place to land. */
  function moveStack(index, point) {
    if (state.phase !== 'bet' || !state.stacks[index]) return false;
    const base = predicted();
    const snapped = snapPoint(point);
    const x = to(snapped[0] - base[0], 9);
    const y = to(snapped[1] - base[1], 9);
    const taken = nearestIndex([base[0] + x, base[1] + y], resting(), footprint());
    if (taken !== null && taken !== index) return false;
    state.stacks[index].x = x;
    state.stacks[index].y = y;
    touch();
    return true;
  }

  /**
   * Set one stack's height, and let every share on the table follow.
   *
   * The shares are not stored anywhere: they are the weights over their total, computed where they
   * are shown. So raising one stack lowers everybody else's share on screen, exactly, with no
   * bookkeeping to get out of step with the geometry.
   *
   * No undo step of its own: a slider emits a value per pixel, and one drag is one change. The
   * caller opens the step (`markEdit`) when the gesture starts.
   */
  function setStackWeight(index, weight) {
    const stack = state.stacks[index];
    if (!stack || state.phase !== 'bet') return false;
    const next = Math.min(WEIGHT_CEILING, Math.max(WEIGHT_MIN, weight));
    if (next === stack.w) return false;
    stack.w = next;
    state.result = null;
    return true;
  }

  /** One press of a −/＋ button: a step, not a drag, so it records its own undo. */
  function nudgeWeight(index, factor) {
    const stack = state.stacks[index];
    if (!stack || state.phase !== 'bet') return false;
    snapshot();
    stack.w = Math.min(WEIGHT_CEILING, Math.max(WEIGHT_MIN, stack.w * factor));
    state.result = null;
    touch();
    return true;
  }

  function removeStack(index) {
    if (!state.stacks[index] || state.phase !== 'bet') return false;
    snapshot();
    state.stacks.splice(index, 1);
    state.selected = null;
    state.hover = null;
    state.result = null;
    touch();
    return true;
  }

  function undo() {
    const last = state.undo.pop();
    if (!last) return false;
    state.stacks = clone(last.stacks);
    state.rounds = state.rounds.slice(0, last.rounds);
    state.selected = null;
    state.hover = null;
    state.result = null;
    state.presetApplied = null;
    endGlide();
    return true;
  }

  // -- the quick layouts -------------------------------------------------------------------

  /**
   * The safe layout: stacks that tile the box the noise cannot leave (or a chosen window).
   *
   * The tiler hands back *shares*, and every stack here gets the same weight, so the hand is the
   * tiling itself. Equal weights are the whole point -- it is the one layout that cannot miss --
   * and they are exact numbers rather than a fixed pile of stacks rounded to whole ones.
   */
  function coverTable() {
    // Only between rounds. Placing a hand while the previous one is still settling would look like
    // it worked -- the stacks change on screen -- and then be cleared by the next round, because a
    // run only carries a layout over when the player asked it to.
    if (state.phase !== 'bet' || state.eps <= 0 || isSecret()) return false;
    const { atoms } = coveringLayout(safeBox(preset(), state.eps), state.eps);
    setStacks(atoms.map((atom) => ({ x: atom[0], y: atom[1], w: COVER_WEIGHT })));
    return true;
  }

  /**
   * The experiment's baseline: a 4x4 grid spanning `[-2 eps, 2 eps]`, one stack at each point.
   *
   * Spacing `4 eps / 3` rather than `2 eps` on purpose -- a grid at exactly `2 eps` divides a box
   * exactly, which would make the payout constant and the spread across rounds zero. These are the
   * experiment's `default_bets` numbers, so a hand laid out here is the baseline the script scores.
   */
  function evenGrid() {
    if (state.phase !== 'bet' || state.eps <= 0) return false;
    const axis = [0, 1, 2, 3].map((i) => -2 * state.eps + (4 * state.eps * i) / 3);
    const stacks = [];
    for (const x of axis) for (const y of axis) stacks.push({ x, y, w: COVER_WEIGHT });
    setStacks(stacks);
    return true;
  }

  /** Repeat the last hand, re-anchored at this round's predicted point. */
  function sameAsLastRound() {
    if (state.phase !== 'bet' || !state.lastLayout) return false;
    setStacks(state.lastLayout);
    return true;
  }

  /**
   * A quick layout, as a switch rather than a one-way door.
   *
   * A player who lays out the safe hand and then wants their own hand back should not have to
   * reach for Undo and hope: pressing the same button again gives it back. The rule is narrow on
   * purpose -- the second press takes it back only while the table is still holding *exactly* what
   * this layout put there. Touch anything in between and the button means "lay it out again",
   * because that is what a hand that has been edited since is asking for.
   *
   * The layouts themselves stay one-way (`coverTable` and friends): auto-play leans on
   * `sameAsLastRound` to fill an empty table, and a layout that undid itself there would fight it.
   */
  function applyPreset(name, apply) {
    const applied = state.presetApplied;
    if (applied !== null && applied.name === name && sameHand(state.stacks, applied.produced)) {
      setStacks(applied.before);
      state.presetApplied = null;
      return true;
    }
    const before = clone(state.stacks);
    if (!apply()) return false;
    // After the layout, not before: `setStacks` opens an undo step, and an undo step is an edit.
    state.presetApplied = { name, before, produced: clone(state.stacks) };
    return true;
  }

  /** Whether `name` is the layout the table is holding, untouched -- what the button shows. */
  function presetIsOn(name) {
    const applied = state.presetApplied;
    return applied !== null && applied.name === name && sameHand(state.stacks, applied.produced);
  }

  function clearStacks() {
    if (state.stacks.length === 0) return false;
    setStacks([]);
    return true;
  }

  // -- what the page shows -----------------------------------------------------------------

  /** The safety of the current hand: a guarantee when one is possible, a probability when not. */
  function safety() {
    // No verdict on a table that keeps its law, and that comes before anything else: the verdict
    // is computed from the support, so "safe" or "not safe" would answer the very question the
    // table is asking. It is also the first thing to say about such a table, hand or no hand.
    if (isSecret()) return { kind: 'unpublished' };
    if (state.stacks.length === 0) return { kind: 'empty' };
    if (state.eps <= 0) return { kind: 'no-neighbourhood' };
    const offsets = state.stacks.map((stack) => [stack.x, stack.y]);
    const weights = state.stacks.map((stack) => stack.w);
    const box = supportBox(preset());
    if (box !== null) {
      const report = uncoveredReport(offsets, weights, box.low, box.high, state.eps);
      return {
        kind: 'bounded',
        safe: !report.possibleNegInf,
        uncovered: report.uncoveredAreaFraction,
        worstCaseRoi: report.worstCaseRoi,
      };
    }
    const miss = gaussianMissProbability(
      offsets, weights, preset().noise.mean, preset().noise.cov, state.eps);
    return { kind: 'gaussian', catchProbability: 1 - miss };
  }

  /**
   * The scoreboard: this round, the run so far, and the par line the run is chasing.
   *
   * No stake. It used to carry one -- the payout shares multiplied together, from a notional ×1 --
   * and it was the same information as the score written twice: `log(stake)` *is* the number of
   * rounds times the average score, and because a payout share is always below one it decays to
   * nothing by the end of any run. The two means below say what it was trying to say, and say it
   * in units the player can check: `meanRoi` is what a round pays on average, and `compoundRoi` is
   * the rate the same numbers would grow at if they were rolled over.
   */
  function scores() {
    const rois = state.rounds.map((item) => item.roi);
    const utilities = state.rounds.map((item) => item.utility);
    const meanUtility = utilities.length === 0 ? null : mean(utilities);
    const par = preset().upper_bounds[String(state.eps)];
    return {
      rounds: state.rounds.length,
      lastRoi: state.result ? state.result.roi : null,
      lastUtility: state.result ? state.result.utility : null,
      meanRoi: rois.length === 0 ? null : mean(rois),
      meanUtility,
      compoundRoi: meanUtility === null ? null : Math.exp(meanUtility),
      par: par === undefined ? null : par,
      gap: meanUtility === null || par === undefined ? null : meanUtility - par,
    };
  }

  /**
   * What the run amounted to.
   *
   * The numbers are the ones the scoreboard has been showing all along, gathered so they can be
   * read at once; the per-hand rows are the run's own rounds grouped by the hand that was played
   * (the safe layout is the same hand every round, so that is usually one row). They carry the
   * *achieved* average on this replay, not the expected score over many -- one trajectory cannot
   * give the latter, which is what `run.py --bets` is for.
   */
  function runSummary() {
    const rois = state.rounds.map((item) => item.roi);
    const utilities = state.rounds.map((item) => item.utility);
    const meanUtility = utilities.length === 0 ? null : mean(utilities);
    const par = preset().upper_bounds[String(state.eps)];
    const best = state.rounds.reduce(
      (top, item, index) => (top === null || item.utility > top.utility ? { ...item, index } : top),
      null);
    const worst = state.rounds.reduce(
      (low, item, index) => (low === null || item.utility < low.utility ? { ...item, index } : low),
      null);

    const hands = [];
    const seen = new Map();
    state.rounds.forEach((item) => {
      const key = JSON.stringify([
        item.stacks.map((stack) => [stack.x, stack.y]),
        item.stacks.map((stack) => stack.w),
      ]);
      if (seen.has(key)) {
        const entry = hands[seen.get(key)];
        entry.rounds += 1;
        entry.total += Number.isFinite(item.utility) ? item.utility : -Infinity;
      } else {
        seen.set(key, hands.length);
        hands.push({
          nAtoms: item.stacks.length,
          rounds: 1,
          total: item.utility,
        });
      }
    });
    hands.forEach((entry) => { entry.meanScore = entry.total / entry.rounds; });

    return {
      rounds: state.rounds.length,
      planned: stepCount(),
      reason: state.endReason,
      ruined: rois.some((roi) => roi === 0),
      meanRoi: rois.length === 0 ? null : mean(rois),
      meanUtility,
      compoundRoi: meanUtility === null ? null : Math.exp(meanUtility),
      par: par === undefined ? null : par,
      gap: meanUtility === null || par === undefined ? null : meanUtility - par,
      best: best && { round: best.index + 1, utility: best.utility, roi: best.roi },
      worst: worst && { round: worst.index + 1, utility: worst.utility, roi: worst.roi },
      hands,
    };
  }

  /**
   * Where the hand is between two rounds.
   *
   * Three beats, and the middle one is the point: the hand fades out where it stood, the camera
   * moves across a table with nothing on it to compare against, and the hand fades in at its new
   * anchor. `handAlpha` is the third beat and `out` is the first; the middle beat is simply the
   * stretch where both are empty, which is also when the camera's own easing has the stage to
   * itself. Nothing here needs a timer of its own -- it is all a function of `now`, which is what
   * lets a test sample the three beats instead of watching them.
   */
  function glideAt(now, base) {
    const glide = state.transition;
    if (glide === null) {
      return { active: false, alpha: 1, from: base, out: [] };
    }
    const elapsed = now - glide.startedAt;
    const arriving = smoothstep((elapsed - HAND_OUT_MS - CAMERA_MOVE_MS) / HAND_IN_MS);
    return {
      active: elapsed < TRANSITION_MS,
      alpha: arriving,
      from: glide.from,
      // The hand this round did not keep is the same list as the one it did: either way it is what
      // was on the table, and either way it is what leaves in the first beat.
      out: elapsed < HAND_OUT_MS ? glide.out.map((stack) => ({
        at: [glide.from[0] + stack.x, glide.from[1] + stack.y],
        weight: stack.w,
        alpha: 1 - smoothstep(elapsed / HAND_OUT_MS),
      })) : [],
    };
  }

  /** The render view model: geometry and flags, in state-space coordinates. */
  function view(now = 0) {
    const base = predicted();
    const total = state.stacks.reduce((sum, stack) => sum + stack.w, 0);
    const radius = discRadius(state.eps, geometrySpan());
    // How tall the tallest tower stands, which is the one thing about the hand that can push the
    // camera back: a height is a weight, so a tall stack is the player's own doing (see `frameBox`).
    const tower = state.stacks.reduce(
      (tallest, stack) => Math.max(tallest, stackHeight(stack.w, radius)), 0,
    );
    // `landings` joins the frame: the trail behind the puck is the system's own motion, and a
    // frame that only held the hand would let it drift off screen (see `frameBox`).
    const landings = state.rounds.map((item) => landingAt(item.step));
    const { centre, span } = frameBox(here(), base, resting(base), state.eps, tower, landings);
    const glide = glideAt(now, base);
    const settleT = state.phase === 'settled'
      ? Math.min(1, (now - state.settledAt) / SETTLE_MS) : 0;
    const throwT = state.phase === 'bet' ? 0 : Math.min(1, (now - state.dealStarted) / THROW_MS);
    const pressT = Math.min(1, (now - state.pressedAt) / PRESS_MS);
    return {
      presetKey: state.presetKey,
      span,
      centre,
      radius,
      eps: state.eps,
      // No `puck` to draw beside the ball: the state *is* where the ball is (see `ballAt`), and two
      // spheres for one object was the thing to fix rather than a thing to keep in step.
      predicted: state.aimAssist ? base : null,
      nominalPath: state.aimAssist ? nominalPath(preset(), here(), PATH_STEPS) : null,
      // How visible the hand is: 0 while the camera moves, 1 once it has arrived.
      handAlpha: glide.alpha,
      entering: glide.active,
      stacks: state.stacks.map((stack, index) => ({
        at: [base[0] + stack.x, base[1] + stack.y],
        share: total > 0 ? stack.w / total : 0,
        weight: stack.w,
        // The tower both renderers draw: one plan per stack per frame, so the 3-D table and the
        // flat one cannot disagree about how tall a height is.
        plan: discPlan(stack.w, radius),
        height: stackHeight(stack.w, radius),
        state: state.result && state.result.perStack
          ? (state.result.perStack[index] ? 'hit' : 'miss') : 'bet',
        selected: state.selected === index,
        hover: state.phase === 'bet' && state.hover === index,
        pressT: Math.sin(Math.PI * Math.min(1, Math.max(0, pressT))) || 0,
      })),
      leaving: glide.out,
      ball: ballAt(here(), state.landed, throwT, span),
      throwT,
      settleT,
      history: {
        show: state.showHistory,
        // Where the past draws are re-centred: this round's prediction, or nowhere at all. The
        // decision is here rather than in the renderers because "the assist is off, so nothing
        // about it is drawn" is one fact, and a renderer that decided it for itself is how the 3-D
        // table ended up plotting a noise cloud at the origin with the prediction switched off.
        anchor: state.aimAssist ? base : null,
        paths: historyPaths(),
        // The same points the frame was fitted to, so what is drawn as history and what the camera
        // made room for cannot disagree.
        landings,
        draws: state.rounds.map((item) => [noiseAt(item.step)[0], noiseAt(item.step)[1]]),
      },
      phase: state.phase,
    };
  }

  /** The wake walked so far this run. Nothing ahead of the puck: that would be the answer. */
  function historyPaths() {
    const states = trajectory().states[state.runIndex];
    return [states.slice(0, state.step + 1).map((point) => [point[0], point[1]])];
  }

  // -- the two ways out --------------------------------------------------------------------

  function bets(item) {
    return {
      preset: state.presetKey,
      eps: state.eps,
      trajectory: state.runIndex,
      step: item.step,
      atoms: item.stacks.map((stack) => [stack.x, stack.y]),
      weights: item.stacks.map((stack) => stack.w),
    };
  }

  const lastRound = () => (state.rounds.length === 0 ? null : state.rounds[state.rounds.length - 1]);

  /** The whole run: every round's hand, with the world it was played against. */
  function runLog() {
    return {
      preset: state.presetKey,
      eps: state.eps,
      trajectory: state.runIndex,
      n_step: stepCount(),
      rounds: state.rounds.map((item) => ({
        step: item.step,
        atoms: item.stacks.map((stack) => [stack.x, stack.y]),
        weights: item.stacks.map((stack) => stack.w),
      })),
    };
  }

  function selectTable(key) {
    if (!assets.presets[key]) return false;
    const table = assets.presets[key];
    state.presetKey = key;
    state.eps = table.epsilons[Math.min(1, table.epsilons.length - 1)];
    state.runIndex = 0;
    state.rounds = [];
    state.undo = [];
    state.lastLayout = null;
    state.step = 0;
    return beginRound([]);
  }

  function setRunLength(rounds) {
    if (!RUN_LENGTHS.includes(rounds)) return false;
    state.runLength = rounds;
    return restart();
  }

  function setEps(eps) {
    state.eps = eps;
    state.rounds = [];
    state.undo = [];
    state.lastLayout = null;
    state.step = 0;
    return beginRound([]);
  }

  function selectRun(index) {
    state.runIndex = index;
    state.rounds = [];
    state.undo = [];
    state.lastLayout = null;
    state.step = 0;
    return beginRound([]);
  }

  function toggleSnapping() {
    state.snapping = !state.snapping;
    return state.snapping;
  }

  return {
    state, assets, preset, here, predicted, landingAt, stepCount,
    placeStack, moveStack, setStackWeight, nudgeWeight, removeStack, undo, markEdit,
    snapPoint, stackAt, lattice, footprint, touch, toggleSelection,
    coverTable, evenGrid, sameAsLastRound, applyPreset, presetIsOn, clearStacks,
    deal, settle, nextRound, restart, atEnd,
    safety, scores, view, bets, runLog, lastRound, runSummary,
    selectTable, setEps, selectRun, setRunLength, toggleSnapping,
  };
}

/**
 * Where the ball is: resting where it landed, or in the air on its way there.
 *
 * Returned as a state-space position plus a height above the felt, because the two renderers lift
 * it differently (the 3-D one along its own up axis, the flat one by an offset). The fall is
 * `(1 - t)^2`, so the ball arrives with no speed left -- a drop, not a crash.
 *
 * Between rounds there is no throw to draw, and the ball is where it came to rest -- which is also
 * where the system now is, because a landing *is* the next state (`x' = A x + w`). So the ball is
 * the one solid thing on the table at every moment, and there is no separate marker for the state:
 * the state is where the ball is. It used to be two spheres, a white one for the state and a brown
 * one for the ball, sitting a wobble apart on a table where they are the same object.
 */
export function ballAt(from, landing, t, span) {
  if (landing === null) return { at: [from[0], from[1]], height: 0, landed: true };
  // The ball travels from where it *is* to where it lands. It used to arrive from off in the
  // distance at a fixed offset, which read as a prop being flown in -- and in auto-play, where
  // rounds run back to back, as the same prop being flown in over and over.
  const ease = t * t * (3 - 2 * t);
  return {
    at: [from[0] + (landing[0] - from[0]) * ease, from[1] + (landing[1] - from[1]) * ease],
    height: 0.4 * span * Math.sin(Math.PI * Math.min(1, t * 1.1)),
    landed: t >= 1,
  };
}

/**
 * Where the stack toolbar goes: above the stack, flipped below if that leaves the stage, and
 * clamped inside it horizontally. Pure, because a popover that walks off the edge on a phone is
 * only found by playing on a phone unless the rule is a function someone can test.
 *
 * @param {{x: number, y: number}} anchor the stack, in stage pixels
 * @param {{width: number, height: number}} own the toolbar's size
 * @param {{width: number, height: number}} stage the stage's size
 * @param {number} bottomReserve pixels at the bottom the HUD owns
 */
export function placeToolbar(anchor, own, stage, bottomReserve = 96) {
  const gap = 26;
  const fits = (y) => y >= 8 && y + own.height <= stage.height - bottomReserve;
  let y = anchor.y - gap - own.height;
  if (!fits(y)) {
    y = anchor.y + gap;
    if (!fits(y)) y = Math.min(Math.max(8, y), Math.max(8, stage.height - bottomReserve - own.height));
  }
  const x = Math.min(Math.max(own.width / 2 + 8, anchor.x), stage.width - own.width / 2 - 8);
  return { x: x - own.width / 2, y };
}

// ---------------------------------------------------------------------------------------------
// The shell: everything below touches the document, and decides nothing.
// ---------------------------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

export async function boot() {
  const load = (name) => fetch(`assets/${name}`).then((response) => response.json());
  const [presets, trajectories, selfcheck] = await Promise.all(
    ['presets.json', 'trajectories.json', 'selfcheck.json'].map(load));

  const game = createGame({ presets, trajectories, selfcheck });
  const stage = $('stage');
  const surfaces = await createSurfaces(stage);
  // Which table is being drawn. Reassigned by the switch in Game settings (see `showSurface`), which
  // is why every use of it reads the variable rather than holding on to a method.
  let scene = surfaces.scene3d === null ? surfaces.flat : surfaces.scene3d;
  let autoPlayTimer = null;
  let hoverIndex = null;
  let moveMode = null;      // the stack the "Move" button is relocating
  let restartArmed = 0;     // when the restart button first said "Sure?"
  let coachUntil = 0;       // a transient line under the badge

  // -- the HUD ------------------------------------------------------------------------------

  const show = (value, places = 4) => {
    if (value === null || value === undefined) return '—';
    return Number.isFinite(value) ? value.toFixed(places) : '−∞';
  };
  const pct = (value) => (value === null ? '—' : `${(value * 100).toFixed(1)}%`);

  function coach(text, ms = 4000) {
    if (text === null) { coachUntil = 0; return; }
    $('announcement').textContent = text;
    $('announcement').className = 'announce';
    coachUntil = performance.now() + ms;
  }

  function refreshCard() {
    const preset = game.preset();
    document.body.dataset.felt = feltOf(preset.key).name;
    $('tableTitle').textContent = preset.title;
    $('tableBlurb').textContent = preset.blurb;
    $('tableMotion').textContent = preset.motion;
    $('tableNoise').textContent = noiseProse(preset);
    $('tableMatrix').textContent = preset.drift_matrix
      .map((row) => `[ ${row.map((value) => Number(value.toFixed(3))).join('  ')} ]`)
      .join('\n');
    $('digestTable').textContent = preset.noise.kind === 'unpublished'
      ? `${preset.title} · law unpublished` : `${preset.title}`;
  }

  function refreshControls() {
    const { state } = game;
    const preset = game.preset();

    const epsPicker = $('eps');
    epsPicker.innerHTML = '';
    epsPicker.append(new Option('0 — no neighbourhood', '0'));
    for (const value of preset.epsilons) epsPicker.append(new Option(String(value), String(value)));
    epsPicker.value = String(state.eps);

    const runPicker = $('run');
    runPicker.innerHTML = '';
    const runs = trajectories[preset.key].states.length;
    for (let i = 0; i < runs; i += 1) {
      runPicker.append(new Option(`Run ${i + 1} of ${runs}`, String(i)));
    }
    runPicker.value = String(state.runIndex);

    $('roundsPerRun').value = String(state.runLength);
    $('aimAssist').checked = state.aimAssist;
    $('showHistory').checked = state.showHistory;
    $('keepLayout').checked = state.keepLayout;
    $('autoPlay').checked = state.autoPlay;
    $('snap').checked = state.snapping;

    const label = state.phase === 'over' ? 'Play again'
      : (state.phase === 'settled' ? 'Next round' : 'Deal');
    const spent = state.phase === 'over';
    const idle = state.phase === 'bet' && state.stacks.length === 0;
    // The stage's button is the one that matters: the panel does not hold a second copy, because a
    // second copy is a second thing to keep in step -- and the whole point of the stage carrying
    // Deal is that the player never has to open the panel to play on.
    $('stageDeal').textContent = label;
    $('stageDeal').disabled = state.phase === 'dealing' || idle;

    const placing = state.phase === 'bet';
    $('undo').disabled = state.undo.length === 0 || spent;
    $('sameAsLast').disabled = state.lastLayout === null || !placing;
    $('stageRestart').disabled = spent;
    // Which quick layout the table is holding, if it is still holding it: the button says so, so
    // that "press it again to take it back" is a thing you can see rather than a thing you know.
    [['coverTable', 'cover'], ['evenGrid', 'grid'], ['sameAsLast', 'last']].forEach(([id, name]) => {
      const on = game.presetIsOn(name);
      $(id).classList.toggle('on', on);
      $(id).setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    if (performance.now() - restartArmed > 4000) {
      restartArmed = 0;
      $('stageRestart').textContent = 'Restart';
    }
    // The two generated layouts are placed *around the prediction*, so offering them while the
    // prediction is hidden would be handing over the answer the player chose not to see -- and on a
    // table that keeps its law, the layout would be built from the very support it is keeping.
    const sealed = preset.hidden_noise;
    $('coverTable').disabled = !placing || state.eps <= 0 || !state.aimAssist || sealed;
    $('evenGrid').disabled = !placing || state.eps <= 0 || !state.aimAssist;
    $('clear').disabled = !placing || state.stacks.length === 0;
    $('emptyBet').hidden = state.stacks.length > 0 || state.phase !== 'bet';

    $('digestRun').textContent = `${state.runLength} rounds · ε ${state.eps} · Run ${state.runIndex + 1}`;
    $('digestMoves').textContent = state.stacks.length === 0
      ? 'nothing on the table' : `${state.stacks.length} stacks`;
    // Why "Cover the table" is off, when it is: it is built around the prediction, and on a table
    // that keeps its law it would be built out of the very thing that is hidden. The reason lives
    // on the button rather than in a paragraph somewhere else.
    $('coverTableHelp').dataset.help = sealed
      ? 'This table keeps its landing law to itself, and this layout is built from that law -- so '
        + 'the button is off here.'
      : !state.aimAssist
        ? 'This layout is placed around the prediction, so it is off while the prediction is hidden.'
        : state.eps <= 0
          ? 'This layout tiles the payout squares, so it needs a payout radius. Pick one.'
          : 'Lays equal stacks so their payout squares tile the whole support: no landing can pay '
            + 'nothing. Safe, and not always best -- a table with a fat band pays more if you aim '
            + 'at it.';
    $('digestExport').textContent = state.rounds.length === 0
      ? 'nothing to take yet' : `${state.rounds.length} rounds so far`;
  }

  function refreshScores() {
    const scores = game.scores();
    const summary = game.runSummary();
    const over = game.state.phase === 'over';

    $('lastRoi').textContent = pct(scores.lastRoi);
    $('lastUtility').textContent = show(scores.lastUtility);
    $('meanRoi').textContent = pct(scores.meanRoi);
    $('meanUtility').textContent = show(scores.meanUtility);
    $('par').textContent = show(scores.par);
    $('gap').textContent = show(scores.gap);
    $('compoundRoi').textContent = pct(scores.compoundRoi);
    $('rounds').textContent = `${scores.rounds} of ${game.stepCount()} rounds settled`;
    $('digestScore').textContent = scores.rounds === 0
      ? 'no rounds yet'
      : `${scores.rounds} rounds · average ${show(scores.meanUtility)}`;

    // The end of a run gets its own block rather than replacing the scoreboard: the run's numbers
    // are worth reading next to the round-by-round ones that produced them.
    $('summary').hidden = !over;
    if (over) {
      $('sumReason').textContent = summary.reason === 'ruined'
        ? `Run over — a round paid nothing. ${summary.rounds} of ${summary.planned} rounds played.`
        : `Run over — all ${summary.rounds} rounds played.`;
      $('sumMean').textContent = show(summary.meanUtility);
      $('sumPar').textContent = show(summary.par);
      $('sumBest').textContent = summary.best
        ? `#${summary.best.round} · ${show(summary.best.utility)}` : '—';
      $('sumWorst').textContent = summary.worst
        ? `#${summary.worst.round} · ${show(summary.worst.utility)}` : '—';
      const rows = summary.hands.map((hand) => `<tr><td class="k">${hand.nAtoms} `
        + `stack${hand.nAtoms === 1 ? '' : 's'} × ${hand.rounds} round${hand.rounds === 1 ? '' : 's'}</td>`
        + `<td class="v">${show(hand.meanScore)}</td></tr>`).join('');
      $('sumHands').innerHTML = rows;
    }

    const stacks = game.state.stacks;
    const weights = stacks.map((stack) => stack.w);
    const parts = shares(weights);
    $('stackSummary').textContent = stacks.length === 0 ? 'nothing on the table'
      : `${stacks.length} stack${stacks.length === 1 ? '' : 's'}`;
    if (stacks.length > 0) {
      $('stackList').innerHTML = stacks.map((stack, index) => (
        `<tr><td class="k">${index + 1}. (${stack.x.toFixed(2)}, ${stack.y.toFixed(2)})</td>`
        + `<td class="v">${(parts[index] * 100).toFixed(1)}%</td></tr>`)).join('');
    } else {
      $('stackList').innerHTML = '';
    }

    const selected = game.state.selected;
    $('stackTools').hidden = selected === null || game.state.phase !== 'bet';
    if (selected !== null && stacks[selected]) {
      $('stackShare').textContent = `${(parts[selected] * 100).toFixed(1)}%`;
      $('stackWeight').value = String(sliderFromWeight(stacks[selected].w));
    }

    // The badge is a label, not a paragraph: one line, and the sentence that explains it lives
    // behind the `?` in its corner. A verdict that has to be read twice is a verdict nobody reads.
    const badge = $('badge');
    const badgeHelp = $('badgeHelp');
    const safety = game.safety();
    if (safety.kind === 'unpublished') {
      badge.textContent = 'Law unpublished';
      badge.className = 'badge maybe';
      badgeHelp.dataset.help = 'This table does not publish its landing law -- and no verdict, '
        + 'because the verdict would be built from it. Read the landings and work it out.';
    } else if (safety.kind === 'empty') {
      badge.textContent = game.state.eps > 0 ? 'No verdict yet' : 'Every hand scores −∞';
      badge.className = 'badge';
      badgeHelp.dataset.help = game.state.eps > 0
        ? 'Put stacks down and this says what they guarantee. “Cover the table” lays equal stacks '
          + 'so that no landing can pay nothing.'
        : 'With ε = 0 no hand can guarantee anything: a landing has to be exactly on a stack, so '
          + 'every round scores −∞. Try a neighbourhood.';
    } else if (safety.kind === 'no-neighbourhood') {
      badge.textContent = 'Every hand scores −∞';
      badge.className = 'badge warn';
      badgeHelp.dataset.help = 'With ε = 0 no hand can guarantee anything: a landing has to be '
        + 'exactly on a stack. Pick a payout radius and the payout becomes a share rather than a '
        + 'lottery.';
    } else if (safety.kind === 'bounded' && safety.safe) {
      badge.textContent = `Safe · worst case ${(safety.worstCaseRoi * 100).toFixed(1)}%`;
      badge.className = 'badge ok';
      badgeHelp.dataset.help = 'No landing pays nothing: your payout squares tile the whole '
        + `support, so the worst the ball can do is pay ${(safety.worstCaseRoi * 100).toFixed(1)}% `
        + 'of the bet. Safe is not the same as best -- a table with a fat band pays more if you '
        + 'aim at it, and that is a round that can pay nothing.';
    } else if (safety.kind === 'bounded') {
      badge.textContent = `Not safe · ${(safety.uncovered * 100).toFixed(1)}% uncovered`;
      badge.className = 'badge warn';
      badgeHelp.dataset.help = `${(safety.uncovered * 100).toFixed(1)}% of the support is outside `
        + 'every payout square, so a landing there pays nothing and ends the run. Aiming pays '
        + 'more; covering cannot miss.';
    } else {
      badge.textContent = `Catches ${(safety.catchProbability * 100).toFixed(1)}%`;
      badge.className = 'badge maybe';
      badgeHelp.dataset.help = 'This noise can land anywhere, so no hand can guarantee anything. '
        + `The number is the share of landings these stacks would catch, computed from the noise's `
        + 'law -- a probability, not a worst case.';
    }

    // Two charts, one function: the stage's miniature is the same picture with the reading
    // furniture left off, and the hover crosshair belongs to the one in the panel.
    drawScoreChart($('stageChart'), game.state.rounds, scores.par, null, true);
    drawScoreChart($('chart'), game.state.rounds, scores.par, hoverIndex);
  }

  function refresh() {
    refreshCard();
    refreshControls();
    refreshScores();
  }

  // -- what sits on top of the table --------------------------------------------------------

  const labelEls = new Map();

  /**
   * One share label per stack, sitting on top of its tower.
   *
   * Every stack gets one -- not just the tall ones. The height is the weight and the label is what
   * that weight works out to, so a number that appeared only past some threshold would make the
   * small stacks look like they were not being counted. It is a percentage because the percentage
   * is the only part of a weight that means anything: the size of a weight on its own is a free
   * parameter, and the page never asks the player to think about it.
   */
  function positionOverlays(view) {
    for (const [index, el] of labelEls) {
      if (index >= view.stacks.length) { el.remove(); labelEls.delete(index); }
    }
    view.stacks.forEach((stack, index) => {
      let el = labelEls.get(index);
      if (!el) {
        el = document.createElement('span');
        el.className = 'stackLabel';
        $('stackLabels').appendChild(el);
        labelEls.set(index, el);
      }
      const percent = stack.share * 100;
      el.textContent = `${percent.toFixed(percent < 10 ? 1 : 0)}%`;
      el.classList.toggle('held', stack.selected);
      // The labels belong to the hand, so they leave and arrive with it.
      el.style.opacity = String(view.handAlpha);
      const at = scene.projectAt(stack.at, stack.height);
      el.style.transform = `translate(-50%, -100%) translate(${at.x}px, ${at.y - 7}px)`;
    });

    const tools = $('stackTools');
    if (!tools.hidden && view.stacks[view.stacks.length - 1] !== undefined) {
      const index = game.state.selected;
      if (index !== null && view.stacks[index]) {
        const stack = view.stacks[index];
        const at = scene.projectAt(stack.at, stack.height);
        const own = { width: tools.offsetWidth || 300, height: tools.offsetHeight || 56 };
        const size = { width: stage.clientWidth, height: stage.clientHeight };
        const spot = placeToolbar(at, own, size);
        tools.style.transform = `translate(${spot.x}px, ${spot.y}px)`;
      }
    }
  }

  // -- the loop ------------------------------------------------------------------------------

  function frame(now) {
    const view = game.view(now);
    if (game.state.phase === 'dealing' && view.throwT >= 1) {
      announce(game.settle(now));
      refresh();
    }
    scene.render(view);
    positionOverlays(view);
    if (coachUntil && now > coachUntil) { coachUntil = 0; $('announcement').textContent = ''; }
    requestAnimationFrame(frame);
  }

  function announce(result) {
    if (!result || coachUntil) return;
    const line = $('announcement');
    if (result.hit > 0) {
      line.textContent = `${(result.roi * 100).toFixed(1)}% of the hand paid.`;
      line.className = 'announce ok';
    } else if (result.total > 0) {
      line.textContent = 'Nothing landed. Every stack missed, and the run ends here.';
      line.className = 'announce warn';
    }
  }

  // -- the stage: gestures in, game calls out -----------------------------------------------

  function pointerPoint(event) {
    const rect = stage.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  /** The stack under a point in state space, or null. Fingers get a wider reach than a cursor. */
  function stackUnder(point, touch = false) {
    const view = game.view(performance.now());
    return nearestIndex(point, view.stacks.map((stack) => stack.at), grabReach(view.eps, view.span, touch));
  }

  /**
   * What a click *here* would do: the stack it would hold, or the stack the tap would resolve into.
   *
   * Both halves are needed to answer the question the cursor is asking, because a tap resolves to a
   * cell first and only then means a stack -- so a pointer that is a little off a stack, but whose
   * tap would still snap onto it, is a pointer that is pointing at that stack.
   */
  function wouldHit(point, touch = false) {
    const under = stackUnder(point, touch);
    if (under !== null) return under;
    return game.stackAt(game.snapPoint(point));
  }

  function setMoveMode(index) {
    moveMode = index;
    $('stackMove').textContent = index === null ? 'Move' : 'Cancel move';
    if (index === null) coach(null);
    else coach('Tap where the stack should go.');
  }

  function wireStage() {
    const gesture = createGestureState();
    let timer = null;
    let pinchBase = null;
    let dragFrom = null;
    let lastPoint = { x: 0, y: 0 };

    const worldAt = (point) => scene.pick(point.x, point.y);
    const hitAt = (point, touch) => {
      const world = worldAt(point);
      return world === null ? null : stackUnder(world, touch);
    };

    function act(action, point) {
      switch (action.kind) {
        case 'select':
          // Selecting works on a table that keeps its law too: what is hidden is the *noise*, and
          // holding a stack is not information about it. A second tap on the held stack lets it go.
          game.toggleSelection(action.index);
          refresh();
          break;
        case 'drag-start':
          dragFrom = game.state.stacks[action.index]
            ? { x: game.state.stacks[action.index].x, y: game.state.stacks[action.index].y } : null;
          game.markEdit();
          break;
        case 'drag': {
          const world = worldAt(point);
          if (world !== null) { game.moveStack(action.index, world); refreshScores(); }
          break;
        }
        case 'release':
          if (dragFrom && game.state.stacks[action.index]) {
            game.state.stacks[action.index] = { ...game.state.stacks[action.index], ...dragFrom };
            refreshScores();
          }
          dragFrom = null;
          break;
        case 'orbit':
          scene.orbit(action.dx, action.dy);
          break;
        case 'pinch':
          if (pinchBase === null) pinchBase = action.distance;
          else if (action.distance > 0) scene.zoomBy(pinchBase / action.distance);
          break;
        case 'place': {
          // A tap that went down on a stack has already done its whole job: it selected it on the
          // way down. Placing here as well is what used to leave a second stack in the same cell.
          if (action.index !== null) break;
          const world = worldAt(point);
          if (world === null) break;
          if (moveMode !== null) {
            game.markEdit();
            game.moveStack(moveMode, world);
            setMoveMode(null);
          } else if (game.state.phase === 'bet') {
            if (game.state.autoPlay) setAutoPlay(false);
            // A cell belongs to one stack. If this tap resolved into a taken one, the tap is about
            // that stack -- so hold it (or let it go, if it was already held), which is also the
            // only way a tap never lands on top of something.
            const outcome = game.placeStack(world, performance.now());
            if (!outcome.placed && outcome.index !== null) game.toggleSelection(outcome.index);
          }
          refresh();
          break;
        }
        default:
          break;
      }
    }

    /**
     * What the cursor is over, and the cursor itself.
     *
     * A mouse gets to ask "what happens if I click here?" before clicking, and on a table with a
     * hand on it the answer is worth showing: "add a stack" and "pick this stack up" are the two
     * things a click can mean, and which one you are about to get is the whole interaction. A
     * finger has no such question -- it presses, and the ring and the toolbar appear under it --
     * so this is mouse-only, and touch is answered by the selection instead.
     */
    function trackHover(event, point) {
      if (event.pointerType !== 'mouse') return;
      const world = worldAt(point);
      const index = world === null ? null : wouldHit(world);
      if (index === game.state.hover) return;
      game.state.hover = index;
      stage.style.cursor = index === null ? 'crosshair' : 'pointer';
      refreshScores();
    }

    stage.addEventListener('pointerdown', (event) => {
      // Only a gesture that starts on the felt belongs to the table. The overlays inside #stage --
      // the help pill, the deal button, a stack's toolbar, the labels -- are ordinary buttons, and
      // capturing the pointer here would retarget every later event to the stage and swallow their
      // clicks: the page would look broken in a way no logic test can see.
      if (event.target?.tagName !== 'CANVAS') return;
      stage.setPointerCapture(event.pointerId);
      lastPoint = pointerPoint(event);
      const action = pointerDown(gesture, {
        id: event.pointerId, x: lastPoint.x, y: lastPoint.y,
        touch: event.pointerType !== 'mouse', hit: hitAt(lastPoint, event.pointerType !== 'mouse'),
      });
      clearTimeout(timer);
      if (gesture.mode === 'pinch') pinchBase = null;
      else if (gesture.mode === 'pending' && gesture.target !== null) {
        timer = setTimeout(() => {
          const fired = longPress(gesture);
          if (fired.kind === 'remove' && game.state.phase === 'bet') {
            game.removeStack(fired.index);
            refresh();
          }
        }, LONGPRESS_MS);
      }
      act(action, lastPoint);
    });

    stage.addEventListener('pointermove', (event) => {
      const point = pointerPoint(event);
      lastPoint = point;
      const action = pointerMove(gesture, { id: event.pointerId, x: point.x, y: point.y });
      if (!stillTapping(gesture)) clearTimeout(timer);
      act(action, point);
      trackHover(event, point);
    });

    // The cursor left the table: nothing is under it any more, and a stale highlight would claim
    // otherwise until the next move.
    stage.addEventListener('pointerleave', () => {
      if (game.state.hover === null) return;
      game.state.hover = null;
      stage.style.cursor = 'crosshair';
      refreshScores();
    });

    stage.addEventListener('pointerup', (event) => {
      const point = pointerPoint(event);
      clearTimeout(timer);
      pinchBase = null;
      act(pointerUp(gesture, { id: event.pointerId, x: point.x, y: point.y }), point);
      stage.releasePointerCapture?.(event.pointerId);
    });

    // A touch the system takes away (a notification, an edge gesture) must not leave a stack
    // halfway across the table.
    stage.addEventListener('pointercancel', (event) => {
      clearTimeout(timer);
      pinchBase = null;
      act(pointerCancel(gesture, { id: event.pointerId }), pointerPoint(event));
    });

    // Wheel is zoom, always. Changing a count is a deliberate press on the toolbar (or a key),
    // because that was the whole complaint about mis-touches.
    stage.addEventListener('wheel', (event) => {
      event.preventDefault();
      scene.zoom(event.deltaY);
    }, { passive: false });
  }

  // -- the panels ---------------------------------------------------------------------------

  function openGuide(open) {
    $('guide').hidden = !open;
    document.body.classList.toggle('guide-open', open);
    if (open) $('guideClose').focus?.();
  }

  function primaryAction() {
    const { state } = game;
    const now = performance.now();
    if (state.phase === 'over') game.restart(now);
    else if (state.phase === 'settled') game.nextRound(now);
    else game.deal(now);
    setMoveMode(null);
    refresh();
  }

  function onRestart() {
    const now = performance.now();
    if (now - restartArmed > 4000) {
      restartArmed = now;
      $('stageRestart').textContent = 'Sure? Drops the run';
      return;
    }
    restartArmed = 0;
    $('stageRestart').textContent = 'Restart';
    setMoveMode(null);
    game.restart(now);
    refresh();
  }

  function setAutoPlay(on) {
    game.state.autoPlay = on;
    if (autoPlayTimer) clearInterval(autoPlayTimer);
    autoPlayTimer = null;
    if (!on) return;
    if (game.state.stacks.length === 0) game.sameAsLastRound();
    autoPlayTimer = setInterval(() => {
      if (game.atEnd()) {                 // the run is over: stop, and leave the summary up
        setAutoPlay(false);
        $('autoPlay').checked = false;
        return;
      }
      const now = performance.now();
      if (game.state.phase === 'bet') {
        if (game.deal(now)) refresh();
        else {
          game.sameAsLastRound();
          if (game.state.stacks.length > 0 && game.deal(now)) refresh();
        }
      } else if (game.state.phase === 'settled') {
        game.nextRound(now);
        if (game.state.stacks.length === 0) game.sameAsLastRound();
        refresh();
      }
    }, 1500);
  }

  function copyRound() {
    const item = game.lastRound();
    if (!item) {
      $('exportStatus').textContent = 'Play a round first — there is nothing to copy yet.';
      return;
    }
    const text = JSON.stringify(game.bets(item));
    navigator.clipboard?.writeText(text).then(
      () => { $('exportStatus').textContent = 'Copied. Score it with: python experiments/pmf_betting/run.py --bets my_bets.json'; },
      () => { $('exportStatus').textContent = text; });
  }

  function exportRun() {
    const log = game.runLog();
    if (log.rounds.length === 0) {
      $('exportStatus').textContent = 'Play a round first — there is nothing to export yet.';
      return;
    }
    const name = `ppsds-bets-${log.preset}-run${log.trajectory + 1}.json`;
    const blob = new Blob([JSON.stringify(log, null, 1)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    URL.revokeObjectURL(link.href);
    $('exportStatus').textContent = `Saved ${name} to your downloads. Nothing was sent anywhere.`;
  }

  /**
   * The panel's edge, as something you can drag.
   *
   * The column holds three cards, and how much room they deserve depends on what you are doing --
   * reading the numbers wants width, playing wants the table -- so the edge between them is a real
   * edge. The cards reflow to whatever width it is left at: that is the grid's `auto-fit`, not
   * anything here.
   *
   * The width goes into a custom property rather than onto the element, because the narrow-screen
   * rules have to go on winning. On a phone the panel is a sheet under the table, and an inline
   * width would out-rank the media query that says so.
   *
   * Nothing is remembered between visits: the panel's width lives in this session like everything
   * else on the page, and the page's promise is that nothing is kept.
   */
  function wireSplitter() {
    const split = $('split');
    const side = $('side');
    let grabbed = null;
    const set = (width) => document.body.style.setProperty('--side-w', `${width}px`);
    const room = () => $('bench').clientWidth;
    split.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      grabbed = { x: event.clientX, width: side.getBoundingClientRect().width };
      split.setPointerCapture?.(event.pointerId);
      split.classList.add('dragging');
    });
    split.addEventListener('pointermove', (event) => {
      if (grabbed === null) return;
      set(panelWidth(grabbed.width + (grabbed.x - event.clientX), room()));
    });
    ['pointerup', 'pointercancel'].forEach((type) => split.addEventListener(type, () => {
      grabbed = null;
      split.classList.remove('dragging');
    }));
    // The keys that point at the edge move it: left widens the panel, right gives the table room.
    split.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const step = event.key === 'ArrowLeft' ? 24 : -24;
      set(panelWidth(side.getBoundingClientRect().width + step, room()));
    });
  }

  /**
   * The height control: one slider, two buttons, and one undo step per gesture.
   *
   * The slider emits a value per pixel, so the step is opened when the gesture starts and not
   * before -- a per-value `snapshot()` would fill the undo stack with a drag's worth of identical
   * states, and the first Ctrl+Z after a drag would then do nothing at all. The keyboard opens its
   * own step on the first press and closes it on `change`, so an arrow key is one undo.
   */
  function wireWeightControls() {
    const slider = $('stackWeight');
    // The range is the model's own coordinate, not a weight: `0..1` is a *position* along a
    // logarithmic scale, and `weightFromSlider` is what turns it into the number the core uses.
    // A slider whose top were a weight would be a ceiling, and a ceiling is what this replaced.
    slider.min = '0';
    slider.max = '1';
    slider.step = String(WEIGHT_STEP);
    let sliding = false;
    const openStep = () => {
      if (sliding) return;
      sliding = true;
      game.markEdit();
    };
    slider.addEventListener('pointerdown', openStep);
    slider.addEventListener('input', () => {
      if (game.state.selected === null) return;
      openStep();
      game.setStackWeight(game.state.selected, weightFromSlider(parseFloat(slider.value)));
      refreshScores();
    });
    ['pointerup', 'pointercancel', 'change', 'blur'].forEach((type) => {
      slider.addEventListener(type, () => { sliding = false; });
    });

    // Press and hold repeats, because the steps are ratios: getting from 2 down to 0.1 is a dozen
    // presses otherwise, and holding a button to watch a tower shrink is how a height wants to be
    // controlled. `setTimeout` then `setInterval` so the first repeat waits for a deliberate press.
    const repeat = (factor) => {
      let delay = null;
      let ticker = null;
      const step = () => { if (game.state.selected !== null) { game.nudgeWeight(game.state.selected, factor); refresh(); } };
      const stop = () => { clearTimeout(delay); clearInterval(ticker); delay = null; ticker = null; };
      return {
        start() {
          if (delay !== null) return;
          step();
          delay = setTimeout(() => { ticker = setInterval(step, 70); }, 380);
        },
        stop,
      };
    };
    const down = repeat(1 / NUDGE_FACTOR);
    const up = repeat(NUDGE_FACTOR);
    [['stackDown', down], ['stackUp', up]].forEach(([id, control]) => {
      const button = $(id);
      button.addEventListener('pointerdown', (event) => { event.preventDefault(); control.start(); });
      ['pointerup', 'pointerleave', 'pointercancel'].forEach((type) => {
        button.addEventListener(type, () => control.stop());
      });
      button.addEventListener('click', (event) => event.preventDefault());
    });
  }

  /**
   * Show the 3-D table or the flat one.
   *
   * The two are the same game -- the same view model, the same numbers, the same settlement -- so
   * this changes nothing about the round, only what it is drawn with. Asking for the 3-D table on a
   * machine that cannot run it gets the flat one, and the switch says so rather than pretending: the
   * option is disabled when the page finds out, and the value is written back here.
   */
  function showSurface(wanted) {
    const named = wanted === '3d' && surfaces.scene3d !== null ? '3d' : 'flat';
    scene = named === '3d' ? surfaces.scene3d : surfaces.flat;
    // The flat canvas is in the markup; the 3-D one is made by the library, inside the stage. The
    // lookup is guarded for the shell's own test shim, which stubs the stage as a plain box.
    $('flat').hidden = named === '3d';
    const gl = stage.querySelector?.('canvas.gl');
    if (gl) gl.hidden = named !== '3d';
    stage.dataset.mode = named;
    $('surface').value = named;
    scene.resize();
  }

  /**
   * The `?` in the corner of a control, and the box it opens.
   *
   * Everything on the table that is not simply "carry on" has one. A page whose rules are a
   * paragraph somewhere else is a page you have to leave in order to read, and the panel's hints
   * were exactly that; this puts each explanation next to the thing it explains.
   *
   * Listening in the capture phase, and stopping the event: the badge sits *inside* the button it
   * belongs to, so without this a tap on the `?` would also do whatever the button does.
   */
  function wireHelp() {
    const pop = $('helpPop');
    let open = null;
    const close = () => {
      if (open === null) return;
      pop.hidden = true;
      open.classList.remove('asking');
      open = null;
    };
    document.addEventListener('click', (event) => {
      const badge = event.target?.closest?.('.q');
      if (badge === null || badge === undefined) {
        if (open !== null && !event.target?.closest?.('#helpPop')) close();
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (open === badge) { close(); return; }
      close();
      const text = badge.dataset.help || '';
      if (text === '') return;
      pop.textContent = text;
      pop.hidden = false;
      const at = badge.getBoundingClientRect();
      const own = pop.getBoundingClientRect();
      const left = Math.min(Math.max(8, at.left + at.width / 2 - own.width / 2),
        Math.max(8, window.innerWidth - own.width - 8));
      const below = at.bottom + 8;
      const top = below + own.height < window.innerHeight - 8
        ? below : Math.max(8, at.top - own.height - 8);
      pop.style.left = `${left}px`;
      pop.style.top = `${top}px`;
      open = badge;
      badge.classList.add('asking');
    }, true);
    window.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });
  }

  function wireControls() {
    $('preset').addEventListener('change', (event) => { game.selectTable(event.target.value); refresh(); });
    $('eps').addEventListener('change', (event) => { game.setEps(parseFloat(event.target.value)); refresh(); });
    $('roundsPerRun').addEventListener('change', (event) => {
      game.setRunLength(parseInt(event.target.value, 10));
      refresh();
    });
    $('run').addEventListener('change', (event) => { game.selectRun(parseInt(event.target.value, 10)); refresh(); });
    $('stageDeal').addEventListener('click', primaryAction);
    $('stageRestart').addEventListener('click', onRestart);
    $('sumRestart').addEventListener('click', () => { game.restart(performance.now()); refresh(); });
    $('coverTable').addEventListener('click', () => { game.applyPreset('cover', game.coverTable); refresh(); });
    $('evenGrid').addEventListener('click', () => { game.applyPreset('grid', game.evenGrid); refresh(); });
    $('sameAsLast').addEventListener('click', () => { game.applyPreset('last', game.sameAsLastRound); refresh(); });
    $('clear').addEventListener('click', () => { game.clearStacks(); refresh(); });
    $('undo').addEventListener('click', () => { game.undo(); refresh(); });
    $('aimAssist').addEventListener('change', (event) => { game.state.aimAssist = event.target.checked; refresh(); });
    $('showHistory').addEventListener('change', (event) => { game.state.showHistory = event.target.checked; refreshScores(); });
    $('keepLayout').addEventListener('change', (event) => { game.state.keepLayout = event.target.checked; });
    $('snap').addEventListener('change', () => { game.toggleSnapping(); refreshScores(); });
    $('autoPlay').addEventListener('change', (event) => setAutoPlay(event.target.checked));
    $('copyRound').addEventListener('click', copyRound);
    $('exportRun').addEventListener('click', exportRun);
    $('sumExport').addEventListener('click', exportRun);
    $('stackRemove').addEventListener('click', () => { game.removeStack(game.state.selected); refresh(); });
    $('stackMove').addEventListener('click', () => {
      setMoveMode(moveMode === null ? game.state.selected : null);
    });
    // The whole column folds away, and the stage takes the window. An attribute rather than a
    // class, because the page's own test shim has a `dataset` and only a stub `classList`: this way
    // the state is both the thing that drives the CSS and a thing a test can read back.
    $('sideToggle').addEventListener('click', () => {
      const hidden = document.body.dataset.side === 'hidden';
      document.body.dataset.side = hidden ? 'shown' : 'hidden';
      $('sideToggle').textContent = hidden ? 'Hide panel' : 'Show panel';
      $('sideToggle').setAttribute('aria-expanded', hidden ? 'true' : 'false');
      scene.resize();
    });
    ['help', 'helpOpen', 'emptyBetHelp'].forEach((id) => $(id).addEventListener('click', () => openGuide(true)));
    $('guideClose').addEventListener('click', () => openGuide(false));
    $('guide').addEventListener('click', (event) => { if (event.target === $('guide')) openGuide(false); });
    $('fullscreen').addEventListener('click', toggleFullscreen);
    document.addEventListener('fullscreenchange', () => {
      $('fullscreen').textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
      scene.resize();
    });
    wireWeightControls();
    wireSplitter();
    wireHelp();

    const chart = $('chart');
    const tip = $('chartTip');
    chart.addEventListener('pointermove', (event) => {
      const rect = chart.getBoundingClientRect();
      const index = scoreIndexAt(chart, event.clientX - rect.left, game.state.rounds.length);
      if (index === null) return;
      hoverIndex = index;
      const item = game.state.rounds[index];
      tip.hidden = false;
      tip.style.left = `${event.clientX - rect.left}px`;
      tip.textContent = `round ${index + 1} · ${Number.isFinite(item.utility) ? item.utility.toFixed(4) : '−∞'} · ${(item.roi * 100).toFixed(1)}% paid`;
      refreshScores();
    });
    chart.addEventListener('pointerleave', () => {
      hoverIndex = null;
      tip.hidden = true;
      refreshScores();
    });

    window.addEventListener('keydown', (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
      const key = event.key.toLowerCase();
      if (event.key === 'Escape') {
        if (!$('guide').hidden) openGuide(false);
        else if (moveMode !== null) setMoveMode(null);
        else if (game.state.selected !== null) { game.state.selected = null; refresh(); }
        return;
      }
      if (event.key.startsWith('Arrow') && game.state.selected !== null) {
        event.preventDefault();
        const factor = event.key === 'ArrowUp' ? NUDGE_FACTOR
          : event.key === 'ArrowDown' ? 1 / NUDGE_FACTOR : 1;
        if (factor !== 1) { game.nudgeWeight(game.state.selected, factor); refresh(); }
        return;
      }
      if (key === 'z' && (event.ctrlKey || event.metaKey)) { game.undo(); refresh(); return; }
      if (key === 'd') primaryAction();
      else if (key === 'f') toggleFullscreen();
      else if (key === 'g') { game.toggleSnapping(); $('snap').checked = game.state.snapping; refreshScores(); }
      else if (key === 'h') { game.state.showHistory = !game.state.showHistory; $('showHistory').checked = game.state.showHistory; refreshScores(); }
      else if (key === 'a') { game.state.aimAssist = !game.state.aimAssist; refresh(); }
      else if (key === 'm' && game.state.selected !== null) setMoveMode(moveMode === null ? game.state.selected : null);
      else if (key === 'c') copyRound();
      else if (key === 'u') { game.undo(); refresh(); }
      else if (key === 'p') $('sideToggle').click();
      else if (key === '?') openGuide(true);
      else if (event.key === 'Delete' && game.state.selected !== null) { game.removeStack(game.state.selected); refresh(); }
    });
  }

  /** Open a stack's toolbar, or close it: the "Move" button is the one mode in the page. */
  let fullscreenBlocked = false;

  function toggleFullscreen() {
    if (fullscreenBlocked) {
      window.open?.('./index.html', '_blank', 'noopener');
      return;
    }
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
      return;
    }
    const request = document.documentElement?.requestFullscreen?.();
    if (!request || typeof request.catch !== 'function') {
      fullscreenBlocked = true;
      $('fullscreen').textContent = 'Open the full page';
      return;
    }
    request.catch(() => {
      fullscreenBlocked = true;
      $('fullscreen').textContent = 'Open the full page';
      $('fullscreen').title = 'This frame does not allow full screen; the standalone page does';
    });
  }

  // -- the table card's preview -------------------------------------------------------------

  function animatePreview() {
    const canvas = $('preview');
    if (canvas.clientWidth > 0) {
      drawFanPreview(canvas, game.preset(), (performance.now() % 7000) / 7000, fanStarts(), nominalPath);
    }
    requestAnimationFrame(animatePreview);
  }

  // -- boot ---------------------------------------------------------------------------------

  const report = selfCheck(selfcheck, presets);
  const line = $('selfcheck');
  line.textContent = formatSelfCheck(report);
  // `wide` stays: this assignment is the only thing that ever writes this element's class list, and
  // dropping it here silently un-spanned the line from the two-column grid it sits in.
  line.className = report.ok ? 'selfcheck ok wide' : 'selfcheck warn wide';
  if (!report.ok) console.warn('ppsds game self-check failed', report.problems);

  const picker = $('preset');
  for (const [key, preset] of Object.entries(presets)) picker.append(new Option(preset.title, key));
  picker.value = game.state.presetKey;
  const lengthPicker = $('roundsPerRun');
  for (const rounds of RUN_LENGTHS) lengthPicker.append(new Option(`${rounds} rounds`, String(rounds)));
  const surfacePicker = $('surface');
  // The 3-D option is kept, because the page may have to disable it: a machine that cannot run the
  // 3-D table should not be offered one. The option stays in the list -- a missing one reads as an
  // answer about the page, and this is a fact about where you are reading it.
  const offered3d = new Option(SURFACES[0][1], SURFACES[0][0]);
  surfacePicker.append(offered3d);
  for (const [key, label] of SURFACES.slice(1)) surfacePicker.append(new Option(label, key));
  if (surfaces.scene3d === null) offered3d.disabled = true;
  surfacePicker.addEventListener('change', (event) => showSurface(event.target.value));
  showSurface(surfaces.scene3d === null ? 'flat' : '3d');
  game.selectTable(game.state.presetKey);

  // A phone gets the panels folded: the first screen is the table and the hand.
  if (window.innerWidth <= 820) {
    ['panelTable', 'panelData'].forEach((id) => $(id).removeAttribute('open'));
  }

  wireControls();
  wireStage();
  window.addEventListener('resize', () => scene.resize());
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => scene.resize()).observe(stage);
  }
  refresh();
  animatePreview();
  requestAnimationFrame(frame);
  return game;
}

/**
 * The two tables the page can draw, and which one to start on.
 *
 * The 3-D table is what the page wants to be; the flat one is the same game with the depth left off
 * -- the same view model, the same numbers, the same settlement -- and it needs neither a GPU nor
 * the library from the CDN. So the flat one is always built, and the 3-D one is built when it can
 * be. Which is *shown* is a setting (`showSurface`), and the fallback is just its starting value:
 * a machine that cannot run the 3-D table should not be offered one.
 */
async function createSurfaces(stage) {
  const flat = createFlatScene($('flat'));
  let scene3d = null;
  try {
    const { createTableScene } = await import('./scene.js');
    scene3d = await createTableScene(stage);
  } catch (error) {
    console.warn('the 3-D table is unavailable, showing the flat one instead:', error);
    const notice = $('renderNotice');
    notice.textContent = '3-D is unavailable here — this is the flat table.';
    notice.hidden = false;
  }
  return { flat, scene3d };
}
