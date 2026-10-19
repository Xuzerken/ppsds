"""Run the game page's JavaScript against the exported assets, in node.

The page carries a **second implementation** of several library numbers: the scoring rule, the
coverage arithmetic, the layout that cannot miss, the table's own dynamics. ppsds computes them in
Python; the browser recomputes them so the page works with no server. That is deliberate, and it is
also the obvious place for the two to drift apart -- so the page carries self-check vectors, and
this test is what makes them mean something: it loads the page's own modules in a JavaScript
engine, against the committed assets, and requires every one of them to reproduce.

What is executed here, and what is not
--------------------------------------
Everything the page *decides* is a module with no DOM in it -- `scoring.js`, `coverage.js`,
`world.js`, and the pure core of `game.js` -- so it runs here in full: a whole 30-round run is
played, the safe layout is checked to sit exactly on the par line, the aim assist is checked to
change no number, and the run log is checked to have the shape `run.py` reads.

`scene.js` (the 3-D table) is only **syntax-checked**. Its behaviour needs a GPU and a document,
which this suite does not have, and pretending otherwise would be worse than saying so: the honest
statement is that the rendering is verified by opening the page.

Skipped when no JavaScript runtime is present: CI installs Python, and making the whole suite
depend on node would be a heavier requirement than this check is worth. A skip is visible, so the
absence is not silent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _paths import public_root

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The page, in whichever layout this tree uses -- see :mod:`_paths`.
GAME = ((public_root() or REPO_ROOT / 'public') / 'docs'
        / 'examples' / 'understand_ppsds_by_pmf_betting' / 'game')
ASSETS = GAME / 'assets'
EXAMPLE_DIR = REPO_ROOT / 'experiments' / 'pmf_betting'

# The page and the script are two halves of one funnel, so the test that runs the page has to be
# able to run the script's own reading and replaying of what the page wrote out.
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import presets  # noqa: E402  (after the path insert above)
import run as pmf_run  # noqa: E402

NODE = shutil.which('node')

pytestmark = pytest.mark.skipif(
    NODE is None or public_root() is None,
    reason='this test needs a node runtime and the published page; neither is guaranteed here '
           '(CI installs no node, and an sdist has no public content)',
)

#: The page's modules, by the job each does. The pure ones are imported and executed; the renderer
#: is only parsed.
PURE_MODULES = ('scoring.js', 'coverage.js', 'world.js', 'theme.js', 'scene2d.js', 'gestures.js', 'game.js')
RENDERER = 'scene.js'

HARNESS = r'''
// Loaded from a copy of the page's modules, so the relative imports resolve exactly as they do in
// the browser. `package.json` beside them says `type: module`; without it node would read the
// `.js` files as CommonJS and refuse the `import`/`export` they are written in.
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const [moduleDir, assetsDir] = process.argv.slice(2);
const read = (name) => JSON.parse(fs.readFileSync(`${assetsDir}/${name}`, 'utf8'));
const load = (name) => import(pathToFileURL(`${moduleDir}/${name}`).href);

const scoring = await load('scoring.js');
const game = await load('game.js');
const world = await load('world.js');
const gestures = await load('gestures.js');

const presets = read('presets.json');
const trajectories = read('trajectories.json');
const selfcheck = read('selfcheck.json');

const result = {};

// 1. Every exported case, recomputed here.
const report = scoring.selfCheck(selfcheck, presets);
result.selfcheck = scoring.formatSelfCheck(report);
result.selfcheckOk = report.ok;
result.selfcheckProblems = report.problems;

// 2. A whole run on the table whose safe layout tiles its noise exactly, so every round of it must
//    come out at par -- not near it. That is the strongest statement the page can make about its
//    own numbers, and it exercises the round flow, the settlement and the bankroll at once.
const play = game.createGame({ presets, trajectories, selfcheck });
play.selectTable('shear');
const eps = 0.5;
play.setEps(eps);
const covered = play.coverTable();
const safety = play.safety();
result.coverTable = { covered, stacks: play.state.stacks.length, safety };

// 2. The quick layouts: "even grid" must be the experiment's own baseline layout, which is already
//    in the assets as an unnormalised scoring case (16 atoms spanning +-2 eps, at eps = 0.5).
//    Checked before the run, because placing a hand is only allowed between rounds.
//    Compared at 1e-6 because the exported atoms are rounded to six decimals -- the tolerance is
//    set by that rounding, not by the arithmetic.
const baseline = selfcheck.find((item) => item.bet_name && item.bet_name.startsWith('weights of 2.0'));
play.evenGrid();
const gridAtoms = play.state.stacks.map((stack) => [stack.x, stack.y]);
result.evenGridMatchesBaseline = gridAtoms.length === baseline.atoms.length
  && gridAtoms.every((atom, i) => Math.abs(atom[0] - baseline.atoms[i][0]) < 1e-6
    && Math.abs(atom[1] - baseline.atoms[i][1]) < 1e-6);
play.clearStacks();   // the run below should start from an empty table, or it has two hands in it


let rounds = 0;
let worstGap = 0;
let firstRound = null;
let stalled = null;
let lastProgress = -1;
let still = 0;
// A run ends now, so the loop ends with it -- and it must make progress getting there. A loop that
// keeps calling deal() on a state that cannot advance would spin until the guard, which hides the
// reason; this stops and says what the state was.
while (play.state.phase !== 'over' && rounds < 200) {
  still = rounds === lastProgress ? still + 1 : 0;
  lastProgress = rounds;
  if (still > 4) {
    stalled = { phase: play.state.phase, stacks: play.state.stacks.length, eps: play.state.eps,
                runLength: play.state.runLength, step: play.state.step, reason: play.state.endReason };
    break;
  }
  if (play.state.phase === 'settled') {
    play.nextRound();
    play.coverTable();
  }
  // Self-sufficient: a round needs a hand on the table, and this loop should not depend on some
  // earlier section having left one there.
  if (play.state.phase === 'bet' && play.state.stacks.length === 0) play.coverTable();
  play.deal(0);
  const settled = play.settle(game.THROW_MS);
  if (settled) {
    if (firstRound === null) firstRound = settled;
    worstGap = Math.max(worstGap, Math.abs(settled.utility - play.scores().par));
    rounds += 1;
  }
}
result.stalled = stalled;
result.run = {
  rounds,
  worstGapToPar: worstGap,
  firstRound,
  expected: play.stepCount(),
  phase: play.state.phase,
  reason: play.state.endReason,
  nextRoundRefused: play.nextRound() === false,
};
const summary = play.runSummary();
result.summary = {
  rounds: summary.rounds,
  planned: summary.planned,
  reason: summary.reason,
  ruined: summary.ruined,
  compoundRoi: summary.compoundRoi,
  geometricMeanRoi: Math.exp(summary.meanUtility),
  hands: summary.hands.length,
  distinctHandNames: summary.hands.map((h) => `${h.nAtoms} stacks x${h.rounds}`),
  bestRound: summary.best && summary.best.round,
  worstRound: summary.worst && summary.worst.round,
  gapToPar: summary.gap,
};

// 3. The aim assist is a drawing switch: with it off the prediction is not drawn, and not one
//    number -- stacks, bet payload, scores -- moves. If it ever stops being true, the page has
//    handed a player who turned the assist off a different game.
// Self-sufficient: whatever the sections above did to the run, this one needs a settled round.
if (play.lastRound() === null) {
  if (play.state.phase === 'over') play.restart();
  play.coverTable();
  play.deal(0);
  play.settle(game.THROW_MS);
}
const betsWith = JSON.stringify(play.bets(play.lastRound()));
const viewWith = play.view(game.THROW_MS);
play.state.aimAssist = false;
const viewWithout = play.view(game.THROW_MS);
result.aimAssist = {
  hidden: viewWithout.predicted === null && viewWithout.nominalPath === null,
  shownBefore: viewWith.predicted !== null && viewWith.nominalPath !== null,
  chipsUnchanged: JSON.stringify(viewWithout.stacks) === JSON.stringify(viewWith.stacks),
  betsUnchanged: JSON.stringify(play.bets(play.lastRound())) === betsWith,
  // Nothing about the assist is left half-drawn: the anchor the past draws are re-centred on is
  // gone too, so the 3-D table cannot fall back to plotting them at the origin.
  ghostAnchor: viewWithout.history.anchor,
  ghostAnchorWhenShown: JSON.stringify(viewWith.history.anchor),
  predictedWhenShown: JSON.stringify(viewWith.predicted),
};
play.state.aimAssist = true;

// 5. The two ways out: the payload `run.py --bets` reads, and the run log it can replay.
const log = play.runLog();
result.runLog = {
  keys: Object.keys(log).sort(),
  roundKeys: log.rounds.length > 0 ? Object.keys(log.rounds[0]).sort() : [],
  rounds: log.rounds.length,
  nStep: log.n_step,
  expected: play.stepCount(),
  atomsArePairs: log.rounds.every((item) => item.atoms.every((atom) => atom.length === 2)),
  weightsMatchAtoms: log.rounds.every((item) => item.weights.length === item.atoms.length),
};
const single = play.bets(play.lastRound());
result.betsKeys = Object.keys(single).sort();

// 5b. That same way out, replayed. `run.py --bets` is meant to reproduce the numbers the player
//     watched, so this records them beside the hands that produced them and the test below holds
//     the script to them. The run above plays the safe layout every round, where every round pays
//     the same -- exactly the shape that a wrong coordinate space or a mis-transcribed hand would
//     survive. So this run alternates two layouts, and a third of its rounds carry a weighted
//     tower: a pmf pays the *share* that landed, which a hand of equal weights cannot tell apart
//     from the raw sum that happens to agree with it.
play.setRunLength(20);
const settledRounds = [];
while (play.state.phase !== 'over' && settledRounds.length < play.stepCount()) {
  if (play.state.phase === 'settled') play.nextRound();
  play.clearStacks();
  const round = settledRounds.length;
  if (round % 3 === 1) {
    play.evenGrid();
  } else {
    play.coverTable();
    if (round % 3 === 2) play.setStackWeight(0, 6);   // six against three ones
  }
  const step = play.state.step;
  play.deal(0);
  const settled = play.settle(game.THROW_MS);
  if (!settled) break;
  settledRounds.push({ step, roi: settled.roi, utility: settled.utility });
}
result.replay = {
  log: play.runLog(),
  rounds: settledRounds,
  // The draws the page settled against, so the script's regenerated ones can be held to them.
  draws: play.view(0).history.draws,
};

// 3b. A shorter run: the length is a choice, and the page must play exactly that many rounds.
play.setRunLength(10);
play.coverTable();
let shortRounds = 0;
while (play.state.phase !== 'over' && shortRounds < 50) {
  if (play.state.phase === 'settled') { play.nextRound(); play.coverTable(); }
  play.deal(0);
  if (play.settle(game.THROW_MS)) shortRounds += 1;
}
result.shortRun = { rounds: shortRounds, declared: play.stepCount(), planned: play.runSummary().planned };

// 3c. A table that keeps its law: no verdict, no layout built from the thing that is hidden.
play.selectTable('house_rules');
play.coverTable();
result.secret = {
  hidden: play.preset().hidden_noise,
  publishedLaw: play.preset().noise.kind,
  publishedSupport: play.preset().support.kind,
  par: play.scores().par,
  safety: play.safety().kind,
  coverRefused: play.state.stacks.length === 0,
};
play.selectTable('shear');

// 6. A hand that misses entirely: the cliff, which the page has to be able to show -- and which
//    now ends the run rather than being averaged into the next round.
play.restart();
play.clearStacks();
play.placeStack([40, 40]);
play.deal(0);
const missed = play.settle(game.THROW_MS);
result.miss = {
  roi: missed.roi,
  utility: missed.utility,
  phase: play.state.phase,
  reason: play.state.endReason,
  summaryRuined: play.runSummary().ruined,
};

// 7. A dragged stack is undoable: one undo step per gesture, not one per pointer frame.
play.restart(0);
play.clearStacks();
play.placeStack([0.0, 0.0]);
const before = { ...play.state.stacks[0] };
play.markEdit();
play.moveStack(0, [4.0, -3.0]);
const dragged = { ...play.state.stacks[0] };
play.undo();
result.dragUndo = {
  moved: dragged.x !== before.x || dragged.y !== before.y,
  restored: play.state.stacks[0] && Math.abs(play.state.stacks[0].x - before.x) < 1e-9
    && Math.abs(play.state.stacks[0].y - before.y) < 1e-9,
};

// 8. One cell, one stack. The bug this replaces was a tap on a stack putting a second one in the
//    same place; the rule that fixes it is the one every mutation has to respect, so it is checked
//    after every mutation rather than once at the end.
play.restart(0);
play.clearStacks();
const spots = [];
for (let i = -2; i <= 2; i += 1) for (let j = -2; j <= 2; j += 1) spots.push([i * 1.5, j * 1.5]);
let doubled = 0;
let refused = 0;
spots.forEach((spot) => {
  if (!play.placeStack(spot).placed) return;
  if (play.placeStack(spot).placed) doubled += 1; else refused += 1;
});
const resting = play.state.stacks.map((stack) => [stack.x, stack.y]);
let closest = Infinity;
for (let a = 0; a < resting.length; a += 1) {
  for (let b = a + 1; b < resting.length; b += 1) {
    closest = Math.min(closest, Math.max(Math.abs(resting[a][0] - resting[b][0]),
                                         Math.abs(resting[a][1] - resting[b][1])));
  }
}
// A drag cannot land on a neighbour either: it sticks at the last cell that is its own.
const base = play.predicted();
const neighbour = resting.length > 1
  ? [base[0] + resting[1][0], base[1] + resting[1][1]] : [base[0], base[1]];
const dragOnto = play.moveStack(0, neighbour);
result.overlap = {
  stacks: play.state.stacks.length,
  doubled,
  refused,
  closest,
  footprint: play.footprint(),
  dragOntoNeighbour: dragOnto,
  movedOntoNeighbour: Math.abs(play.state.stacks[0].x - resting[0][0]) > 1e-12
    || Math.abs(play.state.stacks[0].y - resting[0][1]) > 1e-12,
};

// 9. The lattice. The snap toggle switches between the payout grid and a finer one, and the finer
//    one is still coarse enough that two discs cannot touch -- which is why turning the snap off
//    can never create the overlap it looks like it might.
result.lattice = [0.1, 0.25, 0.5, 0.75].map((each) => {
  const span = 3.0;
  const on = world.latticeStep(each, span, true);
  const off = world.latticeStep(each, span, false);
  const radius = world.discRadius(each, span);
  return {
    eps: each, on, off, radius,
    neverTouching: off >= 2 * radius - 1e-9 && on >= off - 1e-9,
  };
});

// 10. The frame. With nothing on the table it is `viewSpan` exactly -- the view does not change
//     until there is something to show -- and with a hand on it, the box holds every stack, the
//     puck and the prediction.
play.restart(0);
play.clearStacks();
const empty = play.view(0);
play.coverTable();
const full = play.view(0);
result.frame = {
  spanEmpty: empty.span,
  spanFloor: world.viewSpan(play.here(), play.predicted(), play.state.eps),
  centreEmpty: empty.centre,
  midpoint: [0.5 * (play.here()[0] + play.predicted()[0]),
             0.5 * (play.here()[1] + play.predicted()[1])],
  radiusShared: full.radius === world.discRadius(play.state.eps, empty.span),
  covers: full.stacks.every((stack) => Math.abs(stack.at[0] - full.centre[0]) <= full.span
    && Math.abs(stack.at[1] - full.centre[1]) <= full.span),
  neverFurtherThanDouble: full.span <= 2 * empty.span + 1e-9,
};

// 11. The change between two rounds: three beats. The hand leaves where it stood, the camera moves
//     across a table with nothing on it to compare against, and the hand arrives already in its new
//     place -- so nothing is ever sliding while the camera swings. All three are a function of
//     `now`, which is what lets them be sampled here instead of watched.
play.restart(0);
play.clearStacks();
// The safe layout, so the round cannot miss: a round that pays nothing ends the run, and this
// section is about the change, not about the cliff.
play.coverTable();
play.deal(0);
play.settle(game.THROW_MS);
play.state.keepLayout = true;
const atBefore = play.view(game.THROW_MS).stacks[0].at;
play.nextRound(game.THROW_MS);
const near = (point, x, y) => Math.abs(point[0] - x) < 1e-7 && Math.abs(point[1] - y) < 1e-7;
const leavingBeat = play.view(game.THROW_MS);
const cameraBeat = play.view(game.THROW_MS + game.HAND_OUT_MS + game.CAMERA_MOVE_MS / 2);
const arrivingBeat = play.view(game.THROW_MS + game.HAND_OUT_MS + game.CAMERA_MOVE_MS
  + game.HAND_IN_MS / 2);
const arrived = play.view(game.THROW_MS + game.TRANSITION_MS);
result.change = {
  leaving: {
    hand: leavingBeat.handAlpha,
    leaving: leavingBeat.leaving.length,
    leavingWhereItWas: near(leavingBeat.leaving[0].at, atBefore[0], atBefore[1]),
    leavingOpaque: leavingBeat.leaving[0].alpha,
    active: leavingBeat.entering,
  },
  camera: { hand: cameraBeat.handAlpha, leaving: cameraBeat.leaving.length },
  arriving: { hand: arrivingBeat.handAlpha, leaving: arrivingBeat.leaving.length },
  arrived: { hand: arrived.handAlpha, leaving: arrived.leaving.length, active: arrived.entering },
};

// 11b. The layout is dropped unless the player keeps it -- and the dropping is animated too, so it
//      is watched rather than inferred. The hand is gone from the state at once, which is what
//      makes the deal button refuse until there is a new one.
play.coverTable();
play.deal(game.TRANSITION_MS + game.THROW_MS);
play.settle(game.TRANSITION_MS + 2 * game.THROW_MS);
play.state.keepLayout = false;
const droppedAt = game.TRANSITION_MS + 2 * game.THROW_MS;
play.nextRound(droppedAt);
result.discard = {
  goneAtOnce: play.state.stacks.length === 0,
  leaves: play.view(droppedAt).leaving.length,
  opaqueAtStart: play.view(droppedAt).leaving.every((stack) => Math.abs(stack.alpha - 1) < 1e-9),
  handStaysEmpty: play.view(droppedAt + game.TRANSITION_MS).handAlpha,
  goneByTheEnd: play.view(droppedAt + game.TRANSITION_MS).leaving.length,
  dealRefused: play.deal(droppedAt + game.TRANSITION_MS) === false,
};

// 12. Arrowheads: enough of them to read a direction, few enough not to become a scribble, and the
//     recent end of the wake is the end that keeps them.
const straight = Array.from({ length: 60 }, (_, i) => [i * 0.5, 0]);
const withArrows = world.arrowJoints(straight, 3.0);
result.arrows = {
  limit: world.ARROW_LIMIT,
  capped: withArrows.length,
  direction: world.arrowJoints([[0, 0], [1, 0]], 3.0)[0].dir,
  shortDropped: world.arrowJoints([[0, 0], [0.01, 0]], 3.0).length,
  keepsTheRecent: withArrows[withArrows.length - 1].at,
  noneForAPoint: world.arrowJoints([[0, 0]], 3.0).length,
};

// 13. Hover is a view-model fact, not a renderer's private state: the renderer is handed the flag
//     and decides nothing, and a stale index cannot survive the stack it pointed at.
play.restart(0);
play.clearStacks();
play.placeStack([0, 0]);
play.placeStack([2, 0]);
play.state.hover = 1;
const hovering = play.view(0);
const phase = play.state.phase;
play.state.phase = 'settled';
const settledHover = play.view(0).stacks.filter((stack) => stack.hover).length;
play.state.phase = phase;
play.state.hover = 99;
play.clearStacks();
result.hover = {
  marked: hovering.stacks.filter((stack) => stack.hover).length,
  onlyTheOne: hovering.stacks[1].hover === true && hovering.stacks[0].hover === false,
  hiddenMidRound: settledHover,
  clampedAway: play.state.hover,
};

// 14. The height control. Continuous, with no ceiling to hit: only the ratio between stacks
//     matters, so the slider's own coordinate is a logarithm and its top is a ratio rather than a
//     number the page picked. What a tower draws is that weight against the hand's tallest stack,
//     which is the only thing a height can honestly show once there is nothing to stop at.
play.restart(0);
play.clearStacks();
play.placeStack([0, 0]);
play.placeStack([2, 0]);
play.setStackWeight(0, 1.5);
const shares = play.view(0).stacks.map((stack) => stack.share);
play.setStackWeight(0, 99);
const keptHigh = play.state.stacks[0].w;
play.setStackWeight(0, -5);
const tooLow = play.state.stacks[0].w;
play.setStackWeight(0, 1);
play.nudgeWeight(0, game.NUDGE_FACTOR);
const nudged = play.state.stacks[0].w;
play.setStackWeight(0, 3);
play.setStackWeight(1, 1);
const towers = play.view(0).stacks.map((stack) => stack.height);
// A height is a weight, so raising one stack is the only thing that moves it: no hand-wide
// normaliser, no other tower shifting under the slider.
play.setStackWeight(0, 6);
const raisedTowers = play.view(0).stacks.map((stack) => stack.height);
const fractional = scoring.roundStats(
  [{ x: 0, y: 0, w: 0.5 }, { x: 1, y: 0, w: 1.5 }], [0, 0], 0.5);
result.weights = {
  min: world.WEIGHT_MIN,
  ceiling: world.WEIGHT_CEILING,
  sharesSum: shares.reduce((sum, value) => sum + value, 0),
  shareFollowsHeight: shares[0] > shares[1],
  keptHigh,
  clampedLow: tooLow,
  nudged,
  sliderRoundTrip: Math.abs(world.weightFromSlider(world.sliderFromWeight(7.5)) - 7.5) < 1e-9,
  towerIsProportional: towers[0] > towers[1] && Math.abs(towers[0] / towers[1] - 3) < 1e-9,
  otherUnmoved: raisedTowers[1] === towers[1],
  towerGrew: raisedTowers[0] > towers[0],
  fractionalRoi: fractional.roi,
  fractionalIsLogOfShare: Math.abs(fractional.utility - Math.log(fractional.roi)) < 1e-12,
};

// 15. What a tap landed on, reported back. Leaving the hit out of the report is exactly how a tap
//     on a stack used to place a second stack underneath it.
const onStack = gestures.createGestureState();
gestures.pointerDown(onStack, { id: 1, x: 10, y: 10, hit: 3 });
const tappedStack = gestures.pointerUp(onStack, { id: 1, x: 10, y: 10 });
const onFelt = gestures.createGestureState();
gestures.pointerDown(onFelt, { id: 1, x: 10, y: 10, hit: null });
const tappedFelt = gestures.pointerUp(onFelt, { id: 1, x: 10, y: 10 });
result.gesture = {
  onStack: tappedStack.kind === 'place' && tappedStack.index === 3,
  onFelt: tappedFelt.kind === 'place' && tappedFelt.index === null,
};

// 16. The panel's edge. How wide the column is and how wide it may get are facts about the page,
//     not about a drag handler, so they are decided here and the shell only writes the number out.
result.panel = {
  oneCard: world.CARD_MIN,
  stageMin: world.STAGE_MIN,
  tooNarrow: world.panelWidth(10, 1600),
  exact: world.panelWidth(420, 1600),
  tooWide: world.panelWidth(9999, 1600),
  noRoomForBoth: world.panelWidth(9999, 700),
};

// 17. Two hands are the same hand, or they are not: this is the question a quick layout asks before
//     it takes itself back, and it is answered from the stacks rather than from a flag that a
//     direct call into the store could leave stale.
result.sameHand = {
  copy: world.sameHand([{ x: 0, y: 0, w: 1 }], [{ x: 0, y: 0, w: 1 }]),
  edited: world.sameHand([{ x: 0, y: 0, w: 1 }], [{ x: 0, y: 0, w: 2 }]),
  moved: world.sameHand([{ x: 0, y: 0, w: 1 }], [{ x: 0, y: 0.5, w: 1 }]),
  shorter: world.sameHand([{ x: 0, y: 0, w: 1 }], []),
  bothEmpty: world.sameHand([], []),
};

// 18. A quick layout is a switch, not a one-way door: press it again while the table is still
//     holding exactly what it laid out and you get your hand back. Edit anything in between and the
//     button means "lay it out again" -- a hand that has been touched since is a different hand.
play.restart(0);
play.clearStacks();
play.placeStack([0.5, 0.5]);
play.setStackWeight(0, 0.7);
const myWeight = play.state.stacks[0].w;
play.applyPreset('cover', play.coverTable);
const coveredStacks = play.state.stacks.length;
const onWhileHeld = play.presetIsOn('cover');
play.applyPreset('cover', play.coverTable);
const mineBack = play.state.stacks.length === 1 && play.state.stacks[0].w === myWeight;
const offAfterTakingItBack = play.presetIsOn('cover');
play.applyPreset('cover', play.coverTable);
play.placeStack([3, 3]);
const onAfterAnEdit = play.presetIsOn('cover');
play.applyPreset('cover', play.coverTable);
result.preset = {
  coveredStacks,
  mineBack,
  onWhileHeld,
  offAfterTakingItBack,
  onAfterAnEdit,
  laidOutAgain: play.state.stacks.length === coveredStacks,
};
// 19. The layouts themselves stay one-way. Auto-play leans on `sameAsLastRound` to fill an empty
//     table, and a layout that undid itself there would fight it -- so the switch lives in the
//     shell's `applyPreset` and the core call stays a plain "lay it out".
const hadALastHand = play.state.lastLayout !== null;
play.sameAsLastRound();
result.oneWay = {
  hadALastHand,
  switchedOnByTheCore: play.presetIsOn('last'),
};

// 20. The camera makes room for a tower. A height is a weight and a weight has no ceiling, so the
//     frame is the only thing between a tall stack and the top of the picture -- and it has to give
//     the room back when the stack comes down, or the table would shrink and stay shrunk.
const flatFrame = world.frameBox([0, 0], [0, 0], [], 0.5, 0);
const tallFrame = world.frameBox([0, 0], [0, 0], [], 0.5, 12);
play.restart(0);
play.clearStacks();
play.placeStack([0, 0]);
const lowSpan = play.view(0).span;
play.setStackWeight(0, 30);
const raised = play.view(0);
play.setStackWeight(0, 1);
result.towerFrame = {
  room: world.TOWER_ROOM,
  floor: flatFrame.span,
  withTower: tallFrame.span,
  centreUnmoved: tallFrame.centre[0] === flatFrame.centre[0]
    && tallFrame.centre[1] === flatFrame.centre[1],
  grew: raised.span > lowSpan,
  holdsTower: raised.span >= raised.stacks[0].height,
  cameBack: Math.abs(play.view(0).span - lowSpan) < 1e-9,
};

// 21. A landing keeps the ball that made it. The ghosts fade with age -- the puck stands on the
//     newest one, because a landing *is* the next state -- and they never fade to nothing: a table
//     where the older landings have gone is a table where the puck got where it is by itself.
result.landings = {
  fresh: world.LANDING_ALPHA.fresh,
  dim: world.LANDING_ALPHA.oldest,
  only: world.landingFade(0, 1),
  newest: world.landingFade(4, 5),
  middle: world.landingFade(2, 5),
  oldest: world.landingFade(0, 5),
};

// 22. The frame makes room for the trail the same way it makes room for a tower -- by widening,
//     not by moving. The puck never moves on its own, so the landings behind it are the only place
//     the system's motion is visible; but the frame still cannot follow them indefinitely, or a run
//     that wanders would walk the puck itself off the screen.
const noTrail = world.frameBox([0, 0], [0, 0], [], 0.5, 0, []);
const someTrail = world.frameBox([0, 0], [0, 0], [], 0.5, 0, [[3, 0]]);
const farTrail = world.frameBox([0, 0], [0, 0], [], 0.5, 0, [[500, 500]]);
result.trailFrame = {
  floor: noTrail.span,
  emptyIsUnchanged: noTrail.span === world.frameBox([0, 0], [0, 0], [], 0.5, 0).span
    && noTrail.centre[0] === 0 && noTrail.centre[1] === 0,
  grew: someTrail.span > noTrail.span,
  centredOnTheHand: someTrail.centre[0] === 0 && someTrail.centre[1] === 0,
  holds: Math.abs(3 - someTrail.centre[0]) <= someTrail.span,
  capped: farTrail.span <= 2 * noTrail.span + 1e-9,
  stillCentred: farTrail.centre[0] === 0 && farTrail.centre[1] === 0,
  puckStillFramed: Math.abs(0 - farTrail.centre[0]) <= farTrail.span,
};

console.log('RESULT ' + JSON.stringify(result));
if (!result.selfcheckOk) console.log(result.selfcheck);
'''


@pytest.fixture(scope='module')
def module_dir(tmp_path_factory):
    """A copy of the page's modules, with a `package.json` marking them as the modules they are.

    Copied rather than checked in place for one reason: `node --check` decides how to parse a `.js`
    file from the nearest `package.json`, and there is none beside the page. Node 22 and later
    detect the module type from the syntax anyway, so checking in place usually works -- "usually"
    is not a guarantee a test should rest on.
    """
    workdir = tmp_path_factory.mktemp('game')
    (workdir / 'package.json').write_text('{"type": "module"}\n', encoding='utf-8')
    for name in PURE_MODULES + (RENDERER,):
        shutil.copy(GAME / name, workdir / name)
    return workdir


@pytest.fixture(scope='module')
def node_run(module_dir):
    """The page's modules, run once in node and shared by the assertions below."""
    harness = module_dir / 'harness.mjs'
    harness.write_text(HARNESS, encoding='utf-8')
    proc = subprocess.run([NODE, str(harness), str(module_dir), str(ASSETS)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f'the page modules failed to run\n--- stdout ---\n{proc.stdout}\n'
        f'--- stderr ---\n{proc.stderr}')
    marker = [line for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
    assert marker, f'the harness printed no result\n{proc.stdout}\n{proc.stderr}'
    return json.loads(marker[0][len('RESULT '):])


#: Boots the page itself, in a shimmed document, against a no-op canvas.
#:
#: This is the closest thing to opening the page that a test runner can do: it builds the real
#: element graph the shell reaches for by id, lets it load the real assets, runs the animation
#: frames, and plays rounds -- so every drawing call and every HUD write happens at least once. A
#: misspelled id, a canvas method that does not exist, a chart drawn before its data arrives: all
#: of those throw here. What it does **not** check is what any of it looks like, or the WebGL path,
#: which needs a GPU. The 3-D module is expected to be unavailable here, so this doubles as a test
#: of the fallback: the page must notice and keep playing.
BOOT_HARNESS = r'''
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const [moduleDir, assetsDir, pagePath] = process.argv.slice(2);
const problems = [];
const gradient = { addColorStop() {} };
const context = new Proxy({}, {
  get: (target, key) => (key in target ? target[key] : () => gradient),
  set: (target, key, value) => { target[key] = value; return true; },
});

const elements = new Map();
// The ids the page declares, and whether each starts hidden -- read off the markup rather than
// assumed. A stub that starts everything visible would paper over a panel that only ever gets
// shown because the script unhid it, and the explanation panel is toggled from its own initial
// state, so assuming would make the fold look broken when it is not.
const markup = fs.readFileSync(pagePath, 'utf8');
const hiddenAtStart = new Map();
for (const tag of markup.matchAll(/<[a-z]+([^>]*)>/g)) {
  const id = /\bid="([^"]+)"/.exec(tag[1]);
  // `$` in the lookahead matters: the capture stops at the tag's `>`, so an attribute at the end
  // of a tag has nothing after it to match, and a `hidden` there would go unnoticed.
  if (id) hiddenAtStart.set(id[1], /\shidden(?=[\s/>]|$)/.test(tag[1]));
}

function element(id) {
  const listeners = new Map();
  return {
    id, textContent: '', innerHTML: '', className: '', value: '', checked: false,
    hidden: hiddenAtStart.get(id) === true, disabled: false,
    style: {}, dataset: {}, width: 620, height: 620,
    clientWidth: 600, clientHeight: 400, tagName: 'DIV',
    // Faithful to the one thing the shell does with it: a select is built by appending its options,
    // and the page has to be able to reach back into it (to disable the 3-D table's option on a
    // machine that cannot run it). A plain array stands in for the HTMLCollection.
    children: [],
    append(child) { this.children.push(child); },
    appendChild() {}, focus() {}, remove() {},
    setPointerCapture() {}, releasePointerCapture() {}, click() {},
    removeAttribute() {}, setAttribute() {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    offsetWidth: 300, offsetHeight: 56,
    addEventListener(type, handler) { listeners.set(type, handler); },
    // The one thing a shim has to add beyond the browser's surface: a way to press a button.
    // Driving the page through its own listeners is the point -- the wiring is what is on trial.
    press(type = 'click', event = {}) { const handler = listeners.get(type); if (handler) handler(event); },
    getContext: () => context,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 600, height: 400 }),
  };
}
const get = (id) => {
  if (!elements.has(id)) elements.set(id, element(id));
  return elements.get(id);
};

globalThis.document = {
  body: { dataset: {}, classList: { add() {}, remove() {}, toggle() {} } },
  documentElement: { requestFullscreen: () => Promise.resolve() },
  fullscreenElement: null,
  getElementById: get,
  createElement: (tag) => element(tag),
  addEventListener() {},
};
globalThis.ResizeObserver = class ResizeObserver {
  constructor(callback) { this.callback = callback; }
  observe() {}
  disconnect() {}
};
globalThis.Option = function Option(text, value) { return { text, value }; };
// `navigator` is a getter-only global in modern node, so it has to be redefined rather than
// assigned -- an assignment silently throws and takes the whole harness down with it.
Object.defineProperty(globalThis, 'navigator', {
  value: { clipboard: { writeText: async () => {} } }, configurable: true,
});
globalThis.Blob = class Blob { constructor(parts) { this.parts = parts; } };
globalThis.URL.createObjectURL = () => 'blob:stub';
globalThis.URL.revokeObjectURL = () => {};
globalThis.fetch = async (path) => ({
  json: async () => JSON.parse(fs.readFileSync(`${assetsDir}/${path.replace('assets/', '')}`, 'utf8')),
});

let clock = 0;
const frames = [];
globalThis.performance = { now: () => clock };
globalThis.requestAnimationFrame = (callback) => { frames.push(callback); return frames.length; };
globalThis.window = {
  devicePixelRatio: 1,
  innerWidth: 1440,
  innerHeight: 900,
  addEventListener() {},
  open() {},
};

const warnings = [];
const realWarn = console.warn;
console.warn = (...args) => { warnings.push(args.join(' ')); realWarn(...args); };

const { boot } = await import(pathToFileURL(`${moduleDir}/game.js`).href);
const game = await boot();

function runFrames(count, stepMs = 40) {
  for (let i = 0; i < count; i += 1) {
    clock += stepMs;
    const pending = frames.splice(0, frames.length);
    pending.forEach((callback) => callback(clock));
  }
}

const result = {};
runFrames(3);
result.selfcheck = get('selfcheck').textContent;
result.selfcheckClass = get('selfcheck').className;
result.renderNotice = get('renderNotice').textContent;
result.renderNoticeHidden = get('renderNotice').hidden;
// Which table is on screen, and whether the switch tells the truth about it. The 3-D module cannot
// load in node, so this run is the fallback -- which is the case the switch has to be honest about.
result.surface = {
  mode: get('stage').dataset.mode,
  value: get('surface').value,
  offers: get('surface').children.map((option) => ({
    text: option.text, value: option.value, disabled: option.disabled === true,
  })),
  flatShown: get('flat').hidden === false,
};
result.tableTitle = get('tableTitle').textContent;
result.tableMotion = get('tableMotion').textContent;
result.tableNoise = get('tableNoise').textContent;
result.matrixLines = get('tableMatrix').textContent.split('\n').length;
result.badgeOnEmpty = get('badge').textContent;
result.dealLabel = get('stageDeal').textContent;
result.dealDisabled = get('stageDeal').disabled;

// A round, played through the page's own buttons -- and through the *stage's* buttons only: the
// point of the dock is that a player never has to open the panel to play on.
get('coverTable').press('click');
runFrames(2);
result.dealAfterBet = get('stageDeal').disabled === false;
result.badgeOnCover = get('badge').textContent;
result.badgeHelp = get('badgeHelp').dataset.help;
result.coverHelp = get('coverTableHelp').dataset.help;
result.badgeClass = get('badge').className;
result.safeStacks = game.state.stacks.length;

get('stageDeal').press('click');
runFrames(30, 40);           // long enough for the throw and the settlement to finish
result.phase = game.state.phase;
result.dealLabelSettled = get('stageDeal').textContent;
result.lastRoi = get('lastRoi').textContent;
result.meanUtility = get('meanUtility').textContent;
result.par = get('par').textContent;
result.announcement = get('announcement').textContent;

get('stageDeal').press('click');  // "Next round"
get('coverTable').press('click');
runFrames(3);
result.rounds = game.state.rounds.length;
result.step = game.state.step;

// The panel folds away and the table takes the window. Read back off the body, which is what the
// stylesheet reads too -- no second copy of the state to fall out of step.
result.sideAtStart = document.body.dataset.side || null;
get('sideToggle').press('click');
result.sideHidden = document.body.dataset.side;
result.sideLabel = get('sideToggle').textContent;
get('sideToggle').press('click');
result.sideBack = document.body.dataset.side;

// A tap on the felt puts a stack down; the same tap again picks it *up* instead of putting a
// second one underneath it. This is the bug the whole placement rule exists for, so it is driven
// through the real listeners rather than through the core.
const tapAt = (x, y, pointerId = 11) => {
  const event = { target: { tagName: 'CANVAS' }, clientX: x, clientY: y,
                  pointerId, pointerType: 'mouse' };
  get('stage').press('pointerdown', event);
  get('stage').press('pointerup', event);
};
get('clear').press('click');
runFrames(40);               // let the view settle, so the stage centre means one place
tapAt(300, 200);
runFrames(40);
result.tapOnce = game.state.stacks.length;
tapAt(300, 200);
runFrames(2);
result.tapTwice = { stacks: game.state.stacks.length, selected: game.state.selected };
tapAt(300, 200);             // and a third tap lets it go again
runFrames(2);
result.tapThrice = { stacks: game.state.stacks.length, selected: game.state.selected };
result.toolbarAfterThird = get('stackTools').hidden;

// The cursor asks what a click would hit, before it clicks.
get('stage').press('pointermove', { target: { tagName: 'CANVAS' }, clientX: 300, clientY: 200,
                                    pointerId: 12, pointerType: 'mouse' });
result.hover = { index: game.state.hover, cursor: get('stage').style.cursor };
get('stage').press('pointerleave', {});
result.hoverAfterLeave = game.state.hover;

// The two ways out, through the buttons, in a document that has no downloads and no clipboard.
get('copyRound').press('click');
get('exportRun').press('click');
result.exportStatus = get('exportStatus').textContent;

// The guide: openable from the table, closable again, and it does not need the run to be idle.
get('helpOpen').press('click');
result.guideOpened = get('guide').hidden === false;
get('guideClose').press('click');
result.guideClosed = get('guide').hidden === true;

console.warn = realWarn;
result.warnings = warnings;

// The shim answers for *any* id, which is exactly what a typo needs to stay invisible: the page
// would quietly update an element that does not exist and the reader would see a HUD that never
// changes. What the script asked for is collected here; whether it asked for everything the page
// declares is checked against the script's source (see the test below), because "did this path run
// today" is not the question worth asking.
const declared = new Set(
  [...fs.readFileSync(pagePath, 'utf8').matchAll(/id="([^"]+)"/g)].map((match) => match[1]),
);
const requested = [...elements.keys()];
result.missingIds = requested.filter((id) => !declared.has(id));
result.unusedIds = [...declared].filter((id) => !elements.has(id));

console.log('RESULT ' + JSON.stringify(result));
'''


@pytest.fixture(scope='module')
def boot_run(module_dir):
    """The page, booted in a shimmed document. Shared by the assertions below."""
    harness = module_dir / 'boot.mjs'
    harness.write_text(BOOT_HARNESS, encoding='utf-8')
    proc = subprocess.run([NODE, str(harness), str(module_dir), str(ASSETS), str(GAME / 'index.html')],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f'boot() threw\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    marker = [line for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
    assert marker, f'the boot harness printed no result\n{proc.stdout}\n{proc.stderr}'
    return json.loads(marker[0][len('RESULT '):])


#: Ids that exist for the stylesheet alone -- a wrapper element the script has no business
#: reaching for. Everything else in the markup must be something the script actually uses, or one
#: of the two sides has a typo.
#: Ids that are markup only -- a layout wrapper, a summary element, a dialog's label.
LAYOUT_ONLY_IDS = {'bench', 'explain', 'guideTitle', 'guideClose'}


def test_every_element_the_page_reaches_for_exists(boot_run):
    """A typo in an id is silent by construction -- the script updates an element that is not
    there, and the reader sees a HUD that never changes. Both directions are checked."""
    assert boot_run['missingIds'] == [], (
        'the script asks for elements the page does not declare: '
        f'{boot_run["missingIds"]}')


def test_the_page_declares_no_id_the_script_never_mentions():
    """The other direction, checked against the script's source rather than against one run.

    A running check would only see the ids whose branch happened to execute (the narrow-screen
    defaults, the end-of-run block, a selected stack), so it would fail on paths the harness does
    not walk and pass on a typo it never reached. Reading the source asks the question that
    matters: is this id wired to anything at all? Ids that are purely markup -- a layout wrapper,
    a dialog's label -- say so in LAYOUT_ONLY_IDS.
    """
    declared = set(re.findall(r'id="([^"]+)"', (GAME / 'index.html').read_text(encoding='utf-8')))
    source = '\n'.join(
        (GAME / name).read_text(encoding='utf-8') for name in PURE_MODULES + (RENDERER,))
    unwired = sorted(i for i in declared - LAYOUT_ONLY_IDS if f"'{i}'" not in source)
    assert unwired == [], (
        f'index.html declares ids no script mentions: {unwired}. Wire them, drop them, or add '
        'them to LAYOUT_ONLY_IDS with a reason.')


def test_the_page_boots_and_fills_its_panels(boot_run):
    """Boot is where a typo'd id or a missing element shows up, and nowhere else."""
    assert boot_run['selfcheck'].startswith('self-check ok')
    assert boot_run['selfcheckClass'].split() == ['selfcheck', 'ok', 'wide'], (
        'the self-check line spans both columns of the side column; the class that says so is '
        'written by the same assignment that reports the result, which is how it went missing')
    assert boot_run['tableTitle'] == 'Long Table'
    assert 'rail' in boot_run['tableMotion']
    assert boot_run['tableNoise']
    assert boot_run['matrixLines'] == 2, 'the drift matrix is two rows'


def test_the_page_falls_back_to_the_flat_table_when_the_3d_module_cannot_load(boot_run):
    """The CDN is a runtime dependency, so this path is not hypothetical -- and it has to be
    *silent* about it in the sense that matters: the game keeps playing.

    It also has to be honest in the other sense: the switch in Game settings offers both tables, and
    says by disabling the option that this machine cannot draw the one it is not showing.
    """
    assert 'flat table' in boot_run['renderNotice']
    assert boot_run['renderNoticeHidden'] is False
    assert boot_run['phase'] == 'settled', 'the round played through the fallback'
    assert boot_run['dealLabelSettled'] == 'Next round'

    surface = boot_run['surface']
    assert surface['mode'] == 'flat', 'the stage is not marked with what is on it'
    assert surface['value'] == 'flat', 'the switch does not say what is on screen'
    assert surface['flatShown'] is True
    assert [option['value'] for option in surface['offers']] == ['3d', 'flat'], (
        'the control has to offer both, or the flat table reads as the only game there is')
    assert surface['offers'][0]['disabled'] is True, (
        'this machine cannot draw the 3-D table; the option should say so by being unpickable')


def test_the_hud_reports_the_round_it_just_settled(boot_run):
    """The badge, the scoreboard and the announcement all follow the hand that was played.

    The badge is one line now, and the sentence that explains it lives behind its `?` -- so the two
    are checked separately: the label has to be short, and the explanation has to still be there.
    """
    assert len(boot_run['badgeOnCover']) <= 30, 'the badge grew back into a paragraph'
    assert 'no landing pays nothing' in boot_run['badgeHelp'].lower()
    # "Cover the table" is off while the prediction is hidden, and the reason lives on the button.
    assert 'tile' in boot_run['coverHelp'], boot_run['coverHelp']
    assert boot_run['badgeClass'] == 'badge ok'
    assert boot_run['safeStacks'] == 4, 'the safe layout on the Long Table is four stacks'
    assert boot_run['dealAfterBet'] is True
    assert boot_run['lastRoi'] == '25.0%', 'the safe layout on the Long Table pays a quarter'
    assert boot_run['par'] == '-1.3863'
    assert boot_run['meanUtility'] == '-1.3863', 'and it lands exactly on par'
    assert 'paid' in boot_run['announcement']
    assert boot_run['rounds'] == 1, 'one round settled, and the next one started'
    assert boot_run['step'] == 1


def test_the_two_ways_out_both_answer(boot_run):
    """Copy and Export are the funnel's bottom half; a button that silently does nothing would
    leave a reader with a hand they cannot score."""
    status = boot_run['exportStatus']
    assert 'downloads' in status or 'my_bets.json' in status or status.startswith('{'), (
        f'the export buttons said nothing useful: {status!r}')


def test_the_guide_opens_and_closes(boot_run):
    """The page leads with the game and keeps the instructions one tap away, at any time -- the
    complaint was that the only guidance vanished with the first stack."""
    assert boot_run['guideOpened'] is True
    assert boot_run['guideClosed'] is True


def test_booting_warns_about_nothing_unexpected(boot_run):
    """The one expected warning is the missing 3-D module. Anything else is a bug being ignored."""
    unexpected = [line for line in boot_run['warnings']
                  if '3-D table is unavailable' not in line and 'self-check' not in line.lower()]
    assert unexpected == [], f'the page warned about something it should not have: {unexpected}'


def test_a_tap_on_a_stack_does_not_place_a_second_one(boot_run):
    """The bug, driven through the page's own listeners rather than through the core.

    Two taps on the same spot are one stack and a selection: the second tap is read as "this one",
    which is what the guide has always claimed a tap on a stack does.
    """
    assert boot_run['tapOnce'] == 1, 'the first tap should have placed a stack'
    assert boot_run['tapTwice']['stacks'] == 1, 'the second tap placed a stack under the first'
    assert boot_run['tapTwice']['selected'] == 0, 'the second tap should have held the stack'
    assert boot_run['tapThrice']['stacks'] == 1
    assert boot_run['tapThrice']['selected'] is None, 'a third tap should have let it go'
    assert boot_run['toolbarAfterThird'] is True, 'and taken the toolbar with it'


def test_the_cursor_shows_what_a_click_would_do(boot_run):
    """On a table with a hand on it, the difference between "add a stack" and "pick this one up" is
    the whole interaction, so the cursor says which one is about to happen."""
    assert boot_run['hover']['index'] == 0
    assert boot_run['hover']['cursor'] == 'pointer'
    assert boot_run['hoverAfterLeave'] is None, 'the highlight outlived the pointer'


def test_the_panel_folds_away(boot_run):
    """The column is a control panel, and a game you cannot stop looking at controls is not
    immersive. The state lives on the body as an attribute, which is what the stylesheet reads."""
    assert boot_run['sideAtStart'] is None
    assert boot_run['sideHidden'] == 'hidden'
    assert boot_run['sideLabel'] == 'Show panel'
    assert boot_run['sideBack'] == 'shown'


def test_the_three_d_renderer_keeps_its_lines_from_being_culled():
    """A second canary, for the same reason as the first.

    The wake and the drift path are `THREE.Line`s whose geometry is refilled every frame, and
    `setFromPoints` does not invalidate a bounding sphere that has already been computed: it stays
    a zero-radius sphere at the origin, so on the drifting tables the line gets culled exactly
    when there is most of it to see. Two one-line fixes, and neither of them is visible to
    anything this suite can run -- the screenshot tool is what actually checks the picture.
    """
    source = (GAME / RENDERER).read_text(encoding='utf-8')
    code = '\n'.join(
        line for line in source.splitlines()
        if not line.strip().startswith(('*', '//', '/*'))
    )
    assert 'frustumCulled = false' in code, f'{RENDERER} lets the wake be culled away'
    assert '.computeBoundingSphere()' in code, (
        f'{RENDERER} refills a line geometry without recomputing its bounding sphere')


def test_the_three_d_renderer_issues_a_draw_call():
    """A canary, and an honest one.

    This suite cannot run ``scene.js`` -- it needs a GPU -- so the one thing that must be true of
    any renderer is checked by reading its source: that it draws at all. A scene that is built,
    positioned and never drawn looks **exactly** like a page that failed to load (a transparent
    canvas over a dark background), and that is not hypothetical: it shipped, and only a browser
    could have caught it. This test is what is available instead.

    Comment lines are stripped first, so a mention of the call in prose does not satisfy it.
    """
    source = (GAME / RENDERER).read_text(encoding='utf-8')
    code = '\n'.join(
        line for line in source.splitlines()
        if not line.strip().startswith(('*', '//', '/*'))
    )
    assert 'renderer.render(scene, camera)' in code, (
        f'{RENDERER} never issues a draw call; the table would render as an empty canvas')


def test_every_control_that_is_not_carry_on_explains_itself():
    """The rules live next to the things they are rules about.

    A page whose rules are a paragraph in a side panel is a page you have to leave in order to read,
    and that is what the panel's hints were. So every control on the table that is not simply
    "carry on" carries a `?` in its corner and the sentence behind it -- and the two that do only
    carry on (Deal, Restart) do not, because a `?` on the obvious is noise.

    Checked in the markup rather than in the browser: the popover needs a real document, and what
    can go wrong here is a control that was added without its explanation.
    """
    html = (GAME / 'index.html').read_text(encoding='utf-8')

    helped = re.findall(r'<button[^>]*class="helped"[^>]*>(.*?)</button>', html, re.S)
    assert len(helped) == 8, 'the hand buttons, the two edit buttons and the way out'
    for inner in helped:
        assert 'class="q"' in inner, f'a helped button has no ? badge: {inner[:40]}'
        badge = re.search(r'<i class="q"([^>]*)>', inner).group(1)
        if 'id=' in badge:
            # One badge is filled in by the script, because why "Cover the table" is off depends on
            # the round (the prediction can be hidden, the table can keep its law). What it says is
            # asserted from the running page instead -- `coverHelp` in the tests above.
            continue
        text = re.search(r'data-help="([^"]*)"', inner)
        assert text and len(text.group(1)) > 40, f'the ? badge has nothing to say: {inner[:40]}'

    options = re.findall(r'<label class="opt">(.*?)</label>', html, re.S)
    assert len(options) == 5, 'the five round options are pills with a ? each'
    for inner in options:
        assert 'class="q"' in inner and 'data-help="' in inner

    for basic in ('stageDeal', 'stageRestart'):
        button = re.search(r'<button[^>]*id="%s"[^>]*>(.*?)</button>' % basic, html, re.S)
        assert button and 'class="q"' not in button.group(1), f'{basic} does not need a ? badge'

    assert '<div id="helpPop"' in html, 'there is nowhere to show the explanation'
    # The score is the one number on the table that is not self-evident, so it is named and explained.
    settle = re.search(r'<div class="settle">(.*?)</div>', html, re.S).group(1)
    assert 'paid' in settle and 'score' in settle and 'data-help' in settle


def test_the_panel_is_three_folded_rows_and_the_options_are_on_the_table():
    """The column is settings, data and the way out -- one folded row each, so that opening it is a
    choice rather than the price of playing. Everything a round needs is on the table."""
    html = (GAME / 'index.html').read_text(encoding='utf-8')
    panels = re.findall(r'<details class="panel"([^>]*)>', html)
    assert len(panels) == 3, f'expected three folded rows, found {len(panels)}'
    assert 'id="panelTable"' in panels[0]
    assert 'id="panelData"' in panels[1]
    assert 'panelMoves' not in html, 'the Advanced panel has been emptied and should be gone'
    assert html.index('class="options"') < html.index('class="side"'), (
        'the round options belong on the table, not in the column')
    assert 'assistHint' not in html, 'that hint now lives on the button it is about'


def test_the_options_are_grouped_by_what_they_do():
    """Five switches in a row is a wall you have to read to find the one you want. They are three
    groups, one per row, each named on its left -- what you see, how a stack lands, and what runs
    itself. Checked in the markup: the grouping is the labels and the rows, and what can go wrong
    here is a switch quietly moved to a group that does not describe it, which no run can see.
    """
    html = (GAME / 'index.html').read_text(encoding='utf-8')
    rows = re.findall(r'<div class="optRow">(.*?)</div>', html, re.S)
    assert len(rows) == 3, f'the switches are three groups, found {len(rows)}'
    names = [re.search(r'<span class="rowLabel">([^<]*)</span>', row).group(1) for row in rows]
    assert names == ['see', 'place', 'auto'], f'the groups are named {names}'
    inside = [[i for i in ('aimAssist', 'showHistory', 'snap', 'keepLayout', 'autoPlay')
               if f'id="{i}"' in row] for row in rows]
    assert inside == [['aimAssist', 'showHistory'], ['snap'], ['keepLayout', 'autoPlay']], (
        f'the switches are grouped as {inside}')
    # The bottom of the table is grouped the same way, with the same kind of label.
    dock = html[html.index('class="dockRight"'):]
    assert [re.search(r'<span class="rowLabel">([^<]*)</span>', row).group(1)
            for row in re.findall(r'<div class="btnRow">(.*?)</div>', dock, re.S)] == [
                'hands', 'edit', 'run']


def test_the_three_d_renderer_cannot_lose_the_prediction_marker():
    """The gold marker is the reference the whole screen is read against, and it used to vanish:
    its ring stood on edge (half of it under the felt) and turned its back on the camera twice a
    turn, and any stack placed on the spot hid it. All three are drawing calls no test can run --
    they need a GPU -- so the two that matter are pinned here as canaries, and the third (the flat
    ring) is what a screenshot is for.
    """
    source = (GAME / RENDERER).read_text(encoding='utf-8')
    code = '\n'.join(
        line for line in source.splitlines()
        if not line.strip().startswith(('*', '//', '/*'))
    )
    material = re.search(r'const predictionMaterial = new THREE\.MeshBasicMaterial\(\{(.*?)\}\);',
                         code, re.S)
    assert material, 'the prediction material is gone or was renamed'
    assert 'depthTest: false' in material.group(1), (
        'the marker is depth-tested again: a stack on the spot will hide it')
    assert 'side: THREE.DoubleSide' in material.group(1), (
        'the marker is single-sided, and it spins: half a turn and it disappears')
    assert 'markerRing.rotation.x' in code, (
        'the ring is standing on edge again, which buries half of it in the felt')


def test_the_page_modules_are_syntactically_valid(module_dir):
    """A typo in a renderer is otherwise only found by opening a browser -- by a reader, not by CI."""
    for name in PURE_MODULES + (RENDERER,):
        proc = subprocess.run([NODE, '--check', str(module_dir / name)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f'{name} does not parse:\n{proc.stderr}'


def test_the_page_reproduces_every_exported_case(node_run):
    """The two implementations must agree on every case, or the page is scoring its own game."""
    assert node_run['selfcheckOk'], node_run['selfcheckProblems']
    assert 'self-check ok' in node_run['selfcheck']
    assert node_run['selfcheck'].startswith('self-check ok (')


def test_the_safe_layout_pays_exactly_par_on_a_table_it_tiles(node_run):
    """The Long Table's noise is exactly two boxes wide at eps = 0.5, so the layout that covers it
    must pay the same share every single round -- and that share is the par line itself."""
    cover = node_run['coverTable']
    assert cover['covered'] is True
    assert cover['stacks'] == 4
    assert cover['safety']['safe'] is True
    assert cover['safety']['worstCaseRoi'] == pytest.approx(0.25)

    run = node_run['run']
    assert run['rounds'] == run['expected'], 'a run plays every step it says it has'
    assert run['worstGapToPar'] < 1e-9, (
        f'a round came out {run["worstGapToPar"]} away from par, so the tiling is not exact')
    assert run['firstRound']['roi'] == pytest.approx(0.25)


def test_the_aim_assist_changes_no_number(node_run):
    """Hiding the prediction is a drawing decision, and nothing else."""
    assist = node_run['aimAssist']
    assert assist['shownBefore'] and assist['hidden']
    assert assist['chipsUnchanged'], 'hiding the prediction moved the stacks'
    assert assist['betsUnchanged'], 'hiding the prediction changed the exported hand'


def test_a_run_ends_and_says_why(node_run):
    """The complaint that started this: auto-play used to finish a run and quietly start another.

    A run now stops, and the reason it stopped is recorded, because "the rounds ran out" and "a
    round paid nothing" are different endings and the page has to be able to say which.
    """
    run = node_run['run']
    assert run['phase'] == 'over'
    assert run['reason'] == 'played-out'
    assert run['nextRoundRefused'] is True, 'a finished run must not hand out another round'

    summary = node_run['summary']
    assert summary['rounds'] == run['rounds'] == summary['planned']
    assert summary['reason'] == 'played-out'
    assert summary['ruined'] is False
    # `compoundRoi` is `exp(mean score)`: the geometric mean payout, which is the rate that
    # compounds. It is what replaced the stake, and it agrees with the mean of the logs by
    # definition -- the assertion is here so that a change to either side has to face the other.
    assert summary['compoundRoi'] == pytest.approx(summary['geometricMeanRoi'])
    assert summary['hands'] == 1, 'the safe layout is one hand, played every round'
    # The hand is the safe layout: four equal tiles, played in all 20 rounds.
    assert '4 stacks x20' in summary['distinctHandNames'][0]
    assert 1 <= summary['bestRound'] <= summary['rounds']
    assert 1 <= summary['worstRound'] <= summary['rounds']
    assert summary['gapToPar'] == pytest.approx(0.0, abs=1e-9), 'the safe layout sits on par'


def test_a_run_can_be_however_long_the_player_asked_for(node_run):
    """The length is a choice now, and it is capped by the replay rather than by a constant."""
    short = node_run['shortRun']
    assert short['declared'] == 10
    assert short['rounds'] == 10
    assert short['planned'] == 10


def test_a_table_that_keeps_its_law_gives_no_verdict_and_no_free_layout(node_run):
    """Withholding has to be consistent: no published law, no support, no par, no safety verdict
    -- and the "cover the table" button cannot build its layout from what is hidden."""
    secret = node_run['secret']
    assert secret['hidden'] is True
    assert secret['publishedLaw'] == 'unpublished'
    assert secret['publishedSupport'] == 'unpublished'
    assert secret['par'] is None, 'par is a function of the hidden law'
    assert secret['safety'] == 'unpublished'
    assert secret['coverRefused'] is True


def test_a_round_that_pays_nothing_ends_the_run(node_run):
    """The page has said this since its first version; now it is true."""
    miss = node_run['miss']
    assert miss['phase'] == 'over'
    assert miss['reason'] == 'ruined'
    assert miss['summaryRuined'] is True
    assert miss['roi'] == 0


def test_a_dragged_stack_can_be_undone(node_run):
    """One gesture, one undo step -- a drag that needs fifty undos is a drag nobody can fix."""
    drag = node_run['dragUndo']
    assert drag['moved'] is True
    assert drag['restored'] is True


def test_two_stacks_never_share_a_cell(node_run):
    """The bug the placement rule exists for.

    A tap on a stack used to select it on the way down *and* place a second stack on the way up,
    in the same cell -- which is what "the stacks overlap" was. Checking it once at the end of a
    scripted hand would be checking a coincidence, so the rule is checked after every mutation.
    """
    overlap = node_run['overlap']
    assert overlap['stacks'] > 4, 'the scripted hand should have placed a real spread'
    assert overlap['doubled'] == 0, 'a second tap on the same cell placed another stack'
    assert overlap['refused'] > 0, 'and the second tap should have been read as a tap on the stack'
    assert overlap['closest'] >= overlap['footprint'] - 1e-9, (
        f'two stacks ended up {overlap["closest"]} apart -- closer than their own diameter')
    assert overlap['dragOntoNeighbour'] is False, 'a drag landed on top of its neighbour'
    assert overlap['movedOntoNeighbour'] is False, 'the refused drag moved the stack anyway'


def test_the_finer_lattice_can_never_produce_an_overlap(node_run):
    """Turning the snap off gives a grid three times finer, and that grid is still coarse enough
    that two discs cannot touch. This is what makes the toggle safe to flip: it can only ever
    change where a stack lands, never whether two of them are on top of each other."""
    for case in node_run['lattice']:
        assert case['neverTouching'], (
            f'at eps={case["eps"]} a disc of radius {case["radius"]} would not fit the '
            f'{case["off"]} lattice')
        assert case['off'] <= case['on'] + 1e-9, 'the fine grid must not be coarser than the payout grid'


def test_the_view_frames_the_hand_without_creeping(node_run):
    """The camera follows the hand now, which is two new ways to get it wrong: it must not change
    on an empty table, and it must not let a stack parked at the edge of the screen ratchet the
    view outwards a little further every round."""
    frame = node_run['frame']
    assert frame['spanEmpty'] == pytest.approx(frame['spanFloor']), (
        'an empty table must frame exactly as it did before there was anything to frame')
    assert frame['centreEmpty'] == pytest.approx(frame['midpoint'])
    assert frame['radiusShared'] is True, 'the discs and the lattice must measure the same epsilon'
    assert frame['covers'] is True, 'the frame does not hold every stack'
    assert frame['neverFurtherThanDouble'] is True, 'the view is zooming out with the hand'


def test_the_view_makes_room_for_the_trail(node_run):
    """The one place the system's own motion is visible is the landings behind the puck, so they
    have to stay in the frame -- without the frame following them away from the puck."""
    trail = node_run['trailFrame']
    assert trail['emptyIsUnchanged'] is True, (
        'a run with no rounds behind it must frame exactly as it did before the trail was a '
        'consideration at all')
    assert trail['grew'] is True, 'a landing off to one side did not widen the frame'
    assert trail['centredOnTheHand'] is True, 'the trail moved the frame instead of widening it'
    assert trail['holds'] is True, 'the frame does not hold a nearby landing'
    assert trail['capped'] is True, (
        'a distant landing ratcheted the frame past the same ceiling a stack is held to')
    assert trail['stillCentred'] is True and trail['puckStillFramed'] is True, (
        'a trail longer than the frame can show dragged the camera off the puck')


def test_a_new_round_has_three_beats(node_run):
    """The hand leaves, the camera moves over an empty table, the hand arrives.

    The middle beat is the point: a hand that slides while the camera swings is two things moving
    at once, and the eye believes neither. So the hand is invisible for the whole of the camera's
    move, and the stack it drew from is the one it left behind.
    """
    change = node_run['change']
    assert change['leaving']['hand'] == 0, 'the hand was still visible as the camera started'
    assert change['leaving']['active'] is True
    assert change['leaving']['leaving'] >= 1, 'the hand that was on the table did not fade'
    assert change['leaving']['leavingWhereItWas'] is True, 'it faded somewhere else'
    assert change['leaving']['leavingOpaque'] == pytest.approx(1)
    assert change['camera']['hand'] == 0, 'the hand was on screen while the camera moved'
    assert change['camera']['leaving'] == 0, 'the old hand was still there during the move'
    assert 0 < change['arriving']['hand'] < 1, 'the hand popped in rather than arriving'
    assert change['arrived']['hand'] == 1
    assert change['arrived']['leaving'] == 0
    assert change['arrived']['active'] is False


def test_a_layout_that_is_not_kept_fades_where_it_stood(node_run):
    """The default is that a round clears the table, and that is now watched rather than inferred:
    the hand is gone from the state at once -- so Deal refuses until there is a new one -- while
    the picture of it fades over the same window."""
    discard = node_run['discard']
    assert discard['goneAtOnce'] is True
    assert discard['leaves'] >= 1, 'the cleared hand should be drawn on its way out'
    assert discard['opaqueAtStart'] is True
    assert discard['goneByTheEnd'] == 0
    assert discard['handStaysEmpty'] == 1, 'there was nothing to fade in, and nothing was drawn'
    assert discard['dealRefused'] is True, 'a round cannot start with nothing on the table'


def test_the_wake_gets_arrowheads_but_not_fifty_of_them(node_run):
    """Past trajectories need joints, and a fifty-round run must not sprout fifty arrowheads: the
    recent end of the wake is the end that is being read."""
    arrows = node_run['arrows']
    assert arrows['capped'] == arrows['limit']
    assert arrows['direction'] == [1, 0], 'the arrowhead points the wrong way'
    assert arrows['shortDropped'] == 0, 'an arrowhead on a two-pixel step is a blob'
    assert arrows['noneForAPoint'] == 0
    assert arrows['keepsTheRecent'] == pytest.approx([29.5, 0]), 'the wrong end of the wake kept them'


def test_the_cursor_is_answered_by_the_view_model(node_run):
    """Which stack a click would hold is a fact about the game, not a renderer's private state --
    the renderer is handed the flag along with the geometry."""
    hover = node_run['hover']
    assert hover['marked'] == 1
    assert hover['onlyTheOne'] is True
    assert hover['hiddenMidRound'] == 0, 'nothing is clickable while a round is settling'
    assert hover['clampedAway'] is None, 'a hover outlived the stack it pointed at'


def test_the_height_is_continuous_and_the_shares_follow_it(node_run):
    """A weight is a height, its size means nothing, and every share on the table is that weight
    over the total -- computed where it is shown, so nothing can store a share and fall out of
    step with the geometry.

    There is no top to reach. A ceiling on a weight was a ceiling on nothing -- only the ratio
    between stacks is observable -- and hitting it is what made the height control feel broken. So
    the slider's coordinate is the *logarithm* of the weight: dragging right keeps multiplying, and
    the one clamp left is the bottom, which is what keeps a slider from deleting what it slides.
    """
    weights = node_run['weights']
    assert weights['sharesSum'] == pytest.approx(1.0)
    assert weights['shareFollowsHeight'] is True, 'a taller stack must be a bigger share'
    assert weights['keptHigh'] == pytest.approx(99), 'a weight was capped on the way in'
    assert weights['clampedLow'] == pytest.approx(weights['min'])
    assert weights['nudged'] == pytest.approx(1.25)
    assert weights['sliderRoundTrip'] is True, 'the slider does not read back what it wrote'
    # A tower is its own weight and nothing else: raising one stack never moves another, which is
    # what a hand-wide normaliser could not promise.
    assert weights['towerIsProportional'] is True
    assert weights['otherUnmoved'] is True, 'raising one stack moved another'
    assert weights['towerGrew'] is True
    # Fractional weights are the point of the whole change: the score is `log` of the share, so it
    # has to be `log` of the share for weights that are not whole numbers either.
    assert weights['fractionalRoi'] == pytest.approx(0.25)
    assert weights['fractionalIsLogOfShare'] is True


def test_a_landing_keeps_the_ball_that_made_it(node_run):
    """The ball comes to rest where the puck will be -- `x' = A x + w` is what a landing is -- so
    the trail of landings is the trail of the thing the puck follows. Each one is drawn as a ghost
    of the ball, at the ball's size, fading with age and never quite gone."""
    fade = node_run['landings']
    assert fade['only'] == fade['fresh'], 'the only landing on the table was drawn as an old one'
    assert fade['newest'] == fade['fresh'], 'the landing the puck stands on is not the clearest'
    assert fade['oldest'] == pytest.approx(fade['dim'])
    assert fade['dim'] < fade['middle'] < fade['fresh'], 'the ghosts do not fade with age'


def test_the_camera_makes_room_for_a_tower(node_run):
    """A weight is a height and a weight has no ceiling, so a stack the player makes tall has to
    stay in the picture somehow. The frame is what backs off -- and it has to come back in when the
    stack comes down, or the table would shrink once and stay shrunk."""
    frame = node_run['towerFrame']
    assert frame['withTower'] > frame['floor'], 'a tall stack did not push the camera back'
    assert frame['withTower'] >= 12 * frame['room'], 'the frame does not cover the tower'
    assert frame['centreUnmoved'] is True, 'making room moved the table'
    assert frame['grew'] is True, 'raising a stack past the frame changed no frame'
    assert frame['holdsTower'] is True, 'the tower is taller than the view that is meant to hold it'
    assert frame['cameBack'] is True, 'the frame never came back in'


def test_the_panel_width_is_decided_where_it_can_be_checked(node_run):
    """A drag is three lines of arithmetic and one line of DOM. The arithmetic is here, because a
    panel that can be dragged shut is a bug no logic test can see any other way."""
    panel = node_run['panel']
    assert panel['tooNarrow'] == panel['oneCard'], 'the column can be dragged below one card'
    assert panel['exact'] == 420, 'a width with room around it should be left alone'
    assert panel['tooWide'] == 1600 - panel['stageMin'], (
        'the panel was allowed to take the table\'s floor')
    assert panel['noRoomForBoth'] == panel['oneCard'], (
        'on a window with room for neither, the card floor has to win over the stage floor')


def test_two_hands_are_the_same_hand_or_they_are_not(node_run):
    """What a quick layout asks itself before it takes itself back. Asked of the stacks, not of a
    flag: a flag set by the shell is a flag a call into the store can leave lying."""
    same = node_run['sameHand']
    assert same['copy'] is True
    assert same['edited'] is False, 'a changed weight is a changed hand'
    assert same['moved'] is False, 'a moved stack is a changed hand'
    assert same['shorter'] is False
    assert same['bothEmpty'] is True


def test_a_quick_layout_presses_twice_to_undo_itself(node_run):
    """Lay out the safe hand, then want your own hand back: the same button gives it back. The rule
    is narrow on purpose -- it only takes it back while the table is still holding exactly what this
    layout put there, because a hand that has been edited since is asking for a layout, not an
    undo."""
    preset = node_run['preset']
    assert preset['coveredStacks'] == 4, 'the safe layout on this table is four tiles'
    assert preset['onWhileHeld'] is True, 'the button does not say it can be pressed again'
    assert preset['mineBack'] is True, 'the second press did not give the hand back'
    assert preset['offAfterTakingItBack'] is False, 'the button still claims to be holding it'
    assert preset['onAfterAnEdit'] is False, 'an edit left the layout claiming to be untouched'
    assert preset['laidOutAgain'] is True, 'the third press laid nothing out'

    one_way = node_run['oneWay']
    assert one_way['hadALastHand'] is True, 'the case was vacuous: there was no hand to repeat'
    assert one_way['switchedOnByTheCore'] is False, (
        'a core layout turned the switch on; auto-play would then undo its own filler')


def test_a_tap_reports_what_it_landed_on(node_run):
    """The fix, at the level it was made: a tap says which stack it hit, and the shell can then
    leave the hand alone. A tap on bare felt reports `null` and means "place one here"."""
    gesture = node_run['gesture']
    assert gesture['onStack'] is True
    assert gesture['onFelt'] is True


def test_the_even_grid_is_the_experiments_baseline_layout(node_run):
    """The quick layout the page offers and the fallback bet the script scores must be one layout,
    or "try this in the experiment" would hand the reader a different hand."""
    assert node_run['evenGridMatchesBaseline'] is True


def test_a_hand_that_misses_scores_minus_infinity(node_run):
    """The cliff the neighbourhood exists to avoid, still reachable on purpose."""
    miss = node_run['miss']
    assert miss['roi'] == 0
    assert miss['utility'] is None or miss['utility'] == float('-inf')


def test_the_exported_shapes_are_what_run_py_reads(node_run):
    """The funnel's bottom half: what the page writes out has to be what the script takes in."""
    assert node_run['betsKeys'] == ['atoms', 'eps', 'preset', 'step', 'trajectory', 'weights']
    log = node_run['runLog']
    assert log['keys'] == ['eps', 'n_step', 'preset', 'rounds', 'trajectory']
    assert log['roundKeys'] == ['atoms', 'step', 'weights']
    assert log['rounds'] == log['expected']
    assert log['nStep'] == log['expected']
    assert log['atomsArePairs'] and log['weightsMatchAtoms']


def test_a_run_log_replays_to_the_numbers_the_page_settled(node_run, tmp_path):
    """The funnel's bottom half in *numbers*, not only in shape.

    The page tells the player a run can be taken away and scored properly; this is that claim
    executed. The log the page writes is handed to ``run.py --bets``, replayed on the trajectory it
    names, and every round has to come back with the payout and the score the page settled it with
    -- computed by the library, on noise the script regenerated from a seed rather than the draws
    the page settled against.

    The tolerance is set by the assets, not by the arithmetic: ``trajectories.json`` carries the
    draws rounded to six decimals, so the two sides start ~5e-7 apart. What this test is here to
    catch -- a hand landing in the wrong coordinate space, a payout computed as a sum where the
    library computes a share -- is off by far more than that.
    """
    replay = node_run['replay']
    run_file = tmp_path / 'my_run.json'
    run_file.write_text(json.dumps(replay['log']), encoding='utf-8')

    run = pmf_run.read_run(run_file)
    preset = presets.get(run['preset'])
    _, noise, rows = pmf_run.replay(preset, run)

    assert len(rows) > 1, 'one round would compare a single number'
    assert len(rows) == len(replay['rounds'])
    for row, seen, draw in zip(rows, replay['rounds'], replay['draws']):
        assert row['step'] == seen['step']
        # The landing the page drew is the one the script regenerates.
        assert noise[row['step']] == pytest.approx(draw, abs=1e-6)
        assert row['roi'] == pytest.approx(seen['roi'], abs=1e-5)
        assert row['utility'] == pytest.approx(seen['utility'], abs=1e-5)

    # And the comparison is worth something: the hands differ from round to round, and at least one
    # weights its stacks unequally, so "share" and "sum" are not the same number anywhere in it.
    assert len({round(row['roi'], 9) for row in rows}) > 1, 'every round paid the same'
    assert any(len({round(w, 9) for w in item['weights']}) > 1 for item in run['rounds']), \
        'no hand weights its stacks unequally, so a sum would pass for a share'
