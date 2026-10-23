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
PURE_MODULES = ('scoring.js', 'coverage.js', 'world.js', 'theme.js', 'scene2d.js', 'gestures.js',
                'tutorial.js', 'game.js')
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
const tutorial = await load('tutorial.js');

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
play.state.aimAssist = true; // the page starts with it off; this section is about the switch
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

// 3b. A run of a length the player picked: the length is a choice, and the page must play exactly
//     that many rounds. 30 on purpose -- a length equal to the default would prove nothing.
play.setRunLength(30);
play.coverTable();
let chosenRounds = 0;
while (play.state.phase !== 'over' && chosenRounds < 50) {
  if (play.state.phase === 'settled') { play.nextRound(); play.coverTable(); }
  play.deal(0);
  if (play.settle(game.THROW_MS)) chosenRounds += 1;
}
result.chosenRun = { rounds: chosenRounds, declared: play.stepCount(), planned: play.runSummary().planned };

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

// 7. A drag stays where it is dropped (there is no undo any more), and a tap on bare felt while
//    anything is selected only closes the selection instead of placing another stack under the
//    hand that was reaching for the edge of the screen.
play.restart(0);
play.clearStacks();
play.placeStack([0.0, 0.0]);
const before = { ...play.state.stacks[0] };
play.beginEdit();
play.moveStack(0, [4.0, -3.0]);
const dragged = { ...play.state.stacks[0] };
result.drag = {
  moved: dragged.x !== before.x || dragged.y !== before.y,
  persists: play.state.stacks[0] && Math.abs(play.state.stacks[0].x - dragged.x) < 1e-9
    && Math.abs(play.state.stacks[0].y - dragged.y) < 1e-9,
};
play.toggleSelection(0);
const selectedBefore = play.state.selection.length;
const dismissed = play.placeStack([3.0, 3.0]);
result.dismiss = {
  selectedBefore,
  placed: dismissed.placed,
  dismissed: dismissed.dismissed === true,
  stacksAfterFirst: play.state.stacks.length,
  selectionAfterFirst: play.state.selection.length,
  secondPlacement: play.placeStack([3.0, 3.0]).placed,
  stacksAfterSecond: play.state.stacks.length,
};

// 7b. A selection is a group: one weight for all, one step for all, one move for all -- and a move
//     that would put any member on an occupied cell moves nothing rather than half the group.
play.restart(0);
play.clearStacks();
play.placeStack([-2.0, 0.0]);
play.placeStack([0.0, 0.0]);
play.placeStack([2.0, 0.0]);
play.placeStack([4.0, 0.0]);
const batchInitial = play.state.stacks.map((stack) => [stack.x, stack.y]);
play.selectIndices([0, 1, 2]);
const batchSelected = play.state.selection.slice();
play.setSelectionWeight(2);
const batchWeights = play.state.stacks.map((stack) => stack.w);
play.nudgeSelection(2);
const batchNudged = play.state.stacks.map((stack) => stack.w);
const batchBase = play.predicted();
const anchorX = play.state.stacks[1].x;
const anchorY = play.state.stacks[1].y;
const batchMoved = play.moveStack(1, [batchBase[0] + anchorX + 1, batchBase[1] + anchorY]);
const batchOffsets = play.state.stacks.map((stack) => [stack.x, stack.y]);
const blockedMove = play.moveStack(
  1, [batchBase[0] + batchOffsets[1][0] + 1, batchBase[1] + batchOffsets[1][1]]);
const afterBlocked = play.state.stacks.map((stack) => [stack.x, stack.y]);
const removed = play.removeSelection();
result.batch = {
  selected: batchSelected,
  initial: batchInitial,
  weights: batchWeights,
  nudged: batchNudged,
  moved: batchMoved,
  offsets: batchOffsets,
  // The fourth stack was not selected and did not move; the three moved by exactly one cell.
  movedTogether: batchOffsets.every((at, index) => (index === 3
    ? at[0] === batchInitial[3][0] && at[1] === batchInitial[3][1]
    : Math.abs(at[0] - (batchInitial[index][0] + 1)) < 1e-9
      && Math.abs(at[1] - batchInitial[index][1]) < 1e-9)),
  blocked: blockedMove,
  unchangedAfterBlock: JSON.stringify(afterBlocked) === JSON.stringify(batchOffsets),
  removed,
  remaining: play.state.stacks.length,
  remainingX: play.state.stacks[0] && play.state.stacks[0].x,
  fourthX: batchInitial[3][0],
  selectionAfterRemove: play.state.selection.length,
};

// 7c. The noise support, as the renderers receive it. One switch now: support is drawn with the
//     prediction and hidden with it.
result.support = {
  shear: world.supportShape(presets.shear),
  stacked: world.supportShape(presets.stacked_deck),
  rotation: world.supportShape(presets.rotation),
  house: world.supportShape(presets.house_rules),
};
play.state.aimAssist = false;
result.support.off = play.view(0).support;
play.state.aimAssist = true;
result.support.on = play.view(0).support;
play.state.aimAssist = false;
result.support.noAssist = play.view(0).support;
play.state.aimAssist = true;
play.selectTable('house_rules');
result.support.houseView = play.view(0).support;
play.selectTable('shear');
play.state.aimAssist = false;

// 7d. Auto-continue is a decision, not a timer: the shell drives it from the frame clock.
play.restart(0);
play.clearStacks();
play.coverTable();
play.deal(0);
play.settle(game.THROW_MS);
const settledAt = play.state.settledAt;
const settleDue = (offset) => play.shouldAutoContinue(settledAt + offset);
play.state.autoContinue = true;
const dueBefore = settleDue(game.AUTO_CONTINUE_MS - 1);
const dueAt = settleDue(game.AUTO_CONTINUE_MS);
play.state.autoPlay = true;
const suppressedByAutoPlay = settleDue(game.AUTO_CONTINUE_MS + 1);
play.state.autoPlay = false;
play.state.autoContinue = false;
result.autoContinue = {
  off: play.shouldAutoContinue(settledAt + game.AUTO_CONTINUE_MS),
  dueBefore,
  dueAt,
  suppressedByAutoPlay,
};

// 7e. The tour is a pure, gated state machine: a step cannot be left until the table is in the
//     state it asks for. There is no Back: the card advances itself, so a step that was undone
//     would be pushed forward again a beat later.
const freshTour = tutorial.createTutorial();
const ctx = [
  { prediction: true },
  { prediction: true, stacks: 1 },
  { prediction: true, stacks: 1, safe: true },
  { prediction: true, stacks: 1, safe: true, heightChanged: true },
  { prediction: true, stacks: 1, safe: true, heightChanged: true, deals: 1 },
  { prediction: true, stacks: 1, safe: true, heightChanged: true, deals: 1, nextRounds: 1 },
];
const tourWalk = [];
const readyAt = [];
const blockedWithoutItsAction = [];
let walk = freshTour;
for (let i = 0; i < ctx.length; i += 1) {
  tourWalk.push(tutorial.tutorialStep(walk).id);
  readyAt.push(tutorial.tutorialReady(walk, ctx[i]));
  // Every step refuses to advance on the previous step's context: the gate is its own action.
  const earlier = i === 0 ? {} : ctx[i - 1];
  blockedWithoutItsAction.push(tutorial.tutorialNext(walk, earlier).index === walk.index);
  walk = tutorial.tutorialNext(walk, ctx[i]);
}
const gatedAtStart = tutorial.tutorialNext(freshTour, {}).index;
const skippedTour = tutorial.tutorialSkip(walk);
const reopenedTour = tutorial.tutorialReopen(skippedTour);
const disabledTour = tutorial.tutorialSetEnabled(reopenedTour, false);
// The bubble is pinned, not anchor-following: its corner is a pure function of the stage, the
// dock and its own size, so it can be held here even though the page has no layout in node.
const pinnedSpot = tutorial.tourSpot(
  { left: 16, top: 105, width: 738, height: 594 },
  { left: 29, top: 588, width: 712, height: 98 },
  { width: 300, height: 120 }, 12);
const crowdedSpot = tutorial.tourSpot(
  { left: 0, top: 0, width: 400, height: 200 },
  { left: 12, top: 100, width: 376, height: 88 },
  { width: 280, height: 150 }, 12);
result.tutorial = {
  fresh: tutorial.tutorialStep(freshTour).id,
  walk: tourWalk,
  readyAt,
  blockedWithoutItsAction,
  gatedAtStart,
  finished: tutorial.tutorialStep(walk),
  skipped: tutorial.tutorialStep(skippedTour),
  reopened: tutorial.tutorialStep(reopenedTour).id,
  disabled: tutorial.tutorialStep(disabledTour),
  covers: [...new Set(tutorial.TOUR_STEPS.flatMap((step) => step.covers))].sort(),
  pinned: pinnedSpot,
  pinnedLeftGap: pinnedSpot.left - 16,
  pinnedDockGap: 588 - (pinnedSpot.top + 120),
  crowdedTop: crowdedSpot.top,
};

// 7f. The tallest reachable tower still fits the frame the empty table asks for -- the cap is a
//     claim about the picture, so it is checked against the picture's own rule.
result.heightBudget = Object.entries(presets).flatMap(([key, preset]) => preset.epsilons.map((eps) => {
  const span = world.viewSpan(preset.x0, world.nominalNext(preset, preset.x0), eps);
  const tower = world.TOWER_ROOM * world.stackHeight(world.WEIGHT_CEILING, world.discRadius(eps, span));
  return { key, eps, span, tower, fits: tower <= span + 1e-9 };
}));

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

// 10. The frame. The **span** is the run's (`runSpan`), so it does not move when the hand or the
//     drift does; the **centre** is the moment's, which is what pans. With nothing on the table the
//     frame is the run's span exactly, and with a hand on it the box holds every stack, the puck and
//     the prediction.
play.restart(0);
play.clearStacks();
// Read after the round change has finished: during the slide the camera is deliberately still
// showing the frame it had, which is the previous section's hand, not this one.
const empty = play.view(10000);
play.coverTable();
const full = play.view(10000);
const geometrySpan = world.viewSpan(play.here(), play.predicted(), play.state.eps);
result.frame = {
  spanEmpty: empty.span,
  spanRun: world.runSpan(presets.shear,
    trajectories.shear.states[play.state.runIndex].slice(0, play.state.runLength + 1),
    play.state.eps),
  centreEmpty: empty.centre,
  midpoint: [0.5 * (play.here()[0] + play.predicted()[0]),
             0.5 * (play.here()[1] + play.predicted()[1])],
  // The discs and the lattice are sized from the *geometry* span, which deliberately does not
  // follow the camera: a stack's radius must not change when the view does.
  radiusShared: full.radius === world.discRadius(play.state.eps, geometrySpan),
  covers: full.stacks.every((stack) => Math.abs(stack.at[0] - full.centre[0]) <= full.span
    && Math.abs(stack.at[1] - full.centre[1]) <= full.span),
  // A hand can widen the frame up to the old ceiling, but it is the *same* frame otherwise.
  neverFurtherThanDouble: full.span <= 2 * empty.span + 1e-9,
  handDoesNotZoom: full.span === empty.span,
};

// 11. The change between two rounds: **the hand moves first, the camera second** -- never both at
//     once. `now` decides which beat is running, so the order can be sampled here instead of
//     watched. The two things that matter: during the slide the camera holds the frame it had, and
//     the hand is on screen the whole time (it is the thing that is moving).
play.restart(0);
play.clearStacks();
// The safe layout, so the round cannot miss: a round that pays nothing ends the run, and this
// section is about the change, not about the cliff.
play.coverTable();
play.deal(0);
play.settle(game.THROW_MS);
play.state.keepLayout = true;
const atBefore = play.view(game.THROW_MS).stacks[0].at;
const centreBefore = play.view(game.THROW_MS).centre;
play.nextRound(game.THROW_MS);
const near = (point, x, y) => Math.abs(point[0] - x) < 1e-7 && Math.abs(point[1] - y) < 1e-7;
// Three samples: the first frame of the slide, the middle of it, and the camera beat after it.
const startBeat = play.view(game.THROW_MS);
const slideBeat = play.view(game.THROW_MS + game.HAND_SLIDE_MS / 2);
const cameraBeat = play.view(game.THROW_MS + game.HAND_SLIDE_MS + game.CAMERA_MOVE_MS / 2);
const arrived = play.view(game.THROW_MS + game.TRANSITION_MS);
const atAfter = arrived.stacks[0].at;
result.change = {
  start: {
    // The hand is where it stood when the round turned over, and it is visible.
    handAtOldAnchor: near(startBeat.stacks[0].at, atBefore[0], atBefore[1]),
    cameraHeld: near(startBeat.centre, centreBefore[0], centreBefore[1]),
    active: startBeat.entering,
    visible: startBeat.stacks.length,
  },
  slide: {
    // Halfway: the hand is between the two anchors, and the camera has still not moved.
    handBetween: !near(slideBeat.stacks[0].at, atBefore[0], atBefore[1])
      && !near(slideBeat.stacks[0].at, atAfter[0], atAfter[1]),
    cameraHeld: near(slideBeat.centre, centreBefore[0], centreBefore[1]),
  },
  camera: {
    // The hand has arrived and is standing still; now the camera moves.
    handArrived: near(cameraBeat.stacks[0].at, atAfter[0], atAfter[1]),
    cameraMoved: !near(cameraBeat.centre, centreBefore[0], centreBefore[1]),
    centreAtTarget: near(cameraBeat.centre, arrived.centre[0], arrived.centre[1]),
  },
  arrived: { active: arrived.entering, span: arrived.span },
  // Whatever the camera did, it did not resize the hand: a round is a move, not a zoom.
  spanUnchanged: startBeat.span === play.view(game.THROW_MS + game.TRANSITION_MS).span
    && slideBeat.span === cameraBeat.span,
  oldHandGone: startBeat.leaving.length === 0,
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
  // A hand that is not coming back has to leave where it stood: it is the only `leaving` list the
  // page still draws, because a kept hand now travels instead of fading.
  leaves: play.view(droppedAt).leaving.length,
  opaqueAtStart: play.view(droppedAt).leaving.every((stack) => Math.abs(stack.alpha - 1) < 1e-9),
  handStaysEmpty: play.view(droppedAt + game.TRANSITION_MS).stacks.length === 0,
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
// 15b. A pinch reports the spread *and* the pair's midpoint, and the zoom it hands back is a step
//      between two samples rather than a ratio from where the pinch began. The second half is the
//      regression: multiplying by the running ratio made a two-step 10% spread zoom by 21%, which
//      is what "far too fast to aim" was.
const pinchState = gestures.createGestureState();
gestures.pointerDown(pinchState, { id: 1, x: 0, y: 0, hit: null, touch: true });
gestures.pointerDown(pinchState, { id: 2, x: 100, y: 0, hit: null, touch: true });
const firstPinch = gestures.pointerMove(pinchState, { id: 2, x: 110, y: 0 });
const secondPinch = gestures.pointerMove(pinchState, { id: 2, x: 121, y: 0 });
// Two fingers that slide together without changing their spread: distance constant, midpoint moves.
const slideState = gestures.createGestureState();
gestures.pointerDown(slideState, { id: 1, x: 0, y: 0, hit: null, touch: true });
gestures.pointerDown(slideState, { id: 2, x: 100, y: 0, hit: null, touch: true });
gestures.pointerMove(slideState, { id: 1, x: 40, y: 30 });
const slid = gestures.pointerMove(slideState, { id: 2, x: 140, y: 30 });
result.pinch = {
  kind: firstPinch.kind,
  midpointIsMean: Math.abs(firstPinch.x - 55) < 1e-12 && Math.abs(firstPinch.y - 0) < 1e-12,
  distance: Math.abs(firstPinch.distance - 110) < 1e-12,
  steps: Math.abs(gestures.pinchZoom(firstPinch.distance, secondPinch.distance)
    - (110 / 121) ** gestures.PINCH_GAIN) < 1e-12,
  composes: Math.abs(gestures.pinchZoom(100, 110) * gestures.pinchZoom(110, 121)
    - gestures.pinchZoom(100, 121)) < 1e-12,
  // Spreading the fingers zooms in, so the factor is below 1 -- but not as far below as the raw
  // ratio: doubling the spread must move the camera less than doubling it.
  damped: gestures.pinchZoom(100, 200) > 0.5 && gestures.pinchZoom(100, 200) < 1,
  dampedAmount: gestures.pinchZoom(100, 200),
  undoPair: Math.abs(gestures.pinchZoom(100, 110) * gestures.pinchZoom(110, 100) - 1) < 1e-12,
  // Two fingers that end where they started, 40 px to the right and 30 px down, with their spread
  // unchanged: the midpoint is what a pan has to follow, not the first finger.
  slideMid: [slid.x, slid.y],
  slideDistance: Math.abs(slid.distance - 100) < 1e-12,
};
// A wheel's magnitude matters, and a huge delta saturates instead of flying.
result.wheel = {
  tiny: gestures.wheelZoom(2),
  notch: gestures.wheelZoom(120),
  clamped: gestures.wheelZoom(5000),
  sameClamp: Math.abs(gestures.wheelZoom(5000) - gestures.wheelZoom(60)) < 1e-12,
  out: gestures.wheelZoom(10) > 1,
  back: gestures.wheelZoom(-10) < 1,
  halves: Math.abs(gestures.wheelZoom(10) * gestures.wheelZoom(-10) - 1) < 1e-12,
};

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
const flatFrame = world.frameBox([0, 0], [0, 0], [], 0.5, 3.0, 0);
const tallFrame = world.frameBox([0, 0], [0, 0], [], 0.5, 3.0, 12);
play.restart(0);
play.clearStacks();
play.placeStack([0, 0]);
const lowSpan = play.view(0).span;
play.setStackWeight(0, 30);
const raised = play.view(0);
const capped = Math.abs(play.state.stacks[0].w - world.WEIGHT_CEILING) < 1e-9;
play.setStackWeight(0, 1);
result.towerFrame = {
  room: world.TOWER_ROOM,
  floor: flatFrame.span,
  withTower: tallFrame.span,
  centreUnmoved: tallFrame.centre[0] === flatFrame.centre[0]
    && tallFrame.centre[1] === flatFrame.centre[1],
  grewDirect: tallFrame.span > flatFrame.span,
  capped,
  fitsWithoutMoving: raised.span <= lowSpan + 1e-9,
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

// 22. The span is decided for the whole run, not for the moment.
//     It used to grow with the drift (3.03 -> 8.36 over six rounds on the Long Table), and every
//     growth was the player's towers getting smaller. `runSpan` scans the run once; `frameBox` takes
//     that span and returns only a centre and, if the hand or a tower demands it, a larger one.
const runStates = trajectories.shear.states[0].slice(0, 21);
const fixedSpan = world.runSpan(presets.shear, runStates, 0.5);
const narrowBox = world.frameBox(runStates[1], world.nominalNext(presets.shear, runStates[1]),
  [], 0.5, fixedSpan);
result.runFrame = {
  span: fixedSpan,
  // The widest moment of the run is what the frame is sized for, so it never needs to grow.
  coversEveryStep: runStates.every((step) => world.viewSpan(step, world.nominalNext(presets.shear, step), 0.5)
    <= fixedSpan + 1e-9),
  steady: narrowBox.span === fixedSpan,
  // A frame fitted to the moment would have been smaller here -- which is exactly the shrinking.
  wouldHaveBeenSmaller: world.viewSpan(runStates[1], world.nominalNext(presets.shear, runStates[1]), 0.5)
    < fixedSpan,
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

const documentListeners = new Map();
globalThis.document = {
  body: { dataset: {}, classList: { add() {}, remove() {}, toggle() {} } },
  documentElement: { requestFullscreen: () => Promise.resolve() },
  fullscreenElement: null,
  getElementById: get,
  createElement: (tag) => element(tag),
  // The page listens on the document for the click that puts a stack down, and for the `?`
  // popover. Both are real paths, so the shim keeps them and lets a test dispatch a click.
  addEventListener(type, handler) {
    if (!documentListeners.has(type)) documentListeners.set(type, []);
    documentListeners.get(type).push(handler);
  },
};
const pressDocument = (type, event = {}) => {
  for (const handler of documentListeners.get(type) || []) handler(event);
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
result.dealAct = get('stageDeal').dataset.act;

// The tour starts on every visit, and Skip is session-only; the guide's button brings it back.
// A step is a gate in the shell as well as in the reducer, and the shell *is* the button: it
// watches the gate on its own clock and advances after a beat. There is nothing to press. The
// first step is the prediction switch itself, so it is driven here through the same change event
// a player produces.
result.tour = {
  visibleAtBoot: get('tour').hidden === false,
  firstText: get('tourText').textContent,
  step: get('tourStep').textContent,
  // Teaching has no switch of its own any more: the guide's button is the control, and it says
  // which way it will go.
  guideLabel: get('guideTour').textContent,
  readyAtBoot: get('tour').dataset.ready,
};
get('aimAssist').checked = true;
get('aimAssist').press('change', { target: get('aimAssist') });
result.tour.readyAfterPrediction = get('tour').dataset.ready;
// Done, but not yet moved on: the card holds for a beat so its green edge can be read.
runFrames(1);
result.tour.readyAfterFrame = get('tour').dataset.ready;
result.tour.stepBeforeBeat = get('tourStep').textContent;
runFrames(30, 40);                           // past `TOUR_ADVANCE_MS`
result.tour.stepAfterBeat = get('tourStep').textContent;
get('aimAssist').checked = false;            // back to the page's default for the checks below
get('aimAssist').press('change', { target: get('aimAssist') });
get('tourSkip').press('click');
result.tour.hintAfterSkip = get('emptyBet').hidden;
result.tour.hiddenAfterSkip = get('tour').hidden;
result.tour.guideAfterSkip = get('guideTour').textContent;
get('guideTour').press('click');       // the guide's button is the switch: it turns teaching back on
runFrames(1);
result.tour.visibleAfterReopen = get('tour').hidden === false;
result.tour.guideAfterReopen = get('guideTour').textContent;
get('guideTour').press('click');       // and pressing it again turns teaching off for the visit
runFrames(1);
result.tour.hiddenAfterGuideOff = get('tour').hidden;
result.tour.guideAfterGuideOff = get('guideTour').textContent;
// Off by the guide's button -- not just Skip -- also silences the static hint.
result.tour.hintAfterTeachingOff = get('emptyBet').hidden;

// The support is one switch with the prediction: drawn while it is on, gone when it is off.
result.supportToggle = { offAtBoot: game.view(performance.now()).support };
get('aimAssist').checked = true;
get('aimAssist').press('change', { target: get('aimAssist') });
result.supportToggle.on = game.view(performance.now()).support;
get('aimAssist').checked = false;
get('aimAssist').press('change', { target: get('aimAssist') });
result.supportToggle.offAgain = game.view(performance.now()).support;
// The preset layouts are built around the prediction, so the round below needs it back on --
// exactly what the page does when the player turns it on for a round.
get('aimAssist').checked = true;
get('aimAssist').press('change', { target: get('aimAssist') });

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
result.dealActSettled = get('stageDeal').dataset.act;
result.stagePhase = get('stage').dataset.phase;

get('stageDeal').press('click');  // "Next round"
// `keepLayout` starts on, so the hand comes back by itself; pressing Cover on top of it would
// toggle the held layout back off. Fill only a table that is actually empty.
if (game.state.stacks.length === 0) get('coverTable').press('click');
runFrames(3);
result.rounds = game.state.rounds.length;
result.step = game.state.step;
result.keepLayoutAtStart = get('keepLayout').checked;

// The play button beside Deal is the old "auto all rounds", now a two-state control: press to
// deal on its own, press again to stop. Its state is its icon, so the state has to be readable.
result.autoAtStart = get('stageAuto').dataset.state;
get('stageAuto').press('click');
result.autoPlaying = get('stageAuto').dataset.state;
get('stageAuto').press('click');   // always stop it: a live interval would keep node awake
result.autoPaused = get('stageAuto').dataset.state;

// The panel folds away and the table takes the window. Read back off the body, which is what the
// stylesheet reads too -- no second copy of the state to fall out of step.
result.sideAtStart = document.body.dataset.side || null;
get('sideToggle').press('click');
result.sideHidden = document.body.dataset.side;
// The button's word lives in its own span: the button also carries an icon, and the label has to
// change without taking the icon with it.
result.sideLabel = get('sideLabel').textContent;
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
get('clear').press('click');   // first press arms...
result.clearArmed = get('clearLabel').textContent;
get('clear').press('click');   // ...second press clears
runFrames(40);               // let the view settle, so the stage centre means one place
tapAt(300, 200);
runFrames(40);
result.tapOnce = game.state.stacks.length;
tapAt(300, 200);
runFrames(2);
result.tapTwice = { stacks: game.state.stacks.length, selection: game.state.selection };
tapAt(300, 200);             // and a third tap lets it go again
runFrames(2);
result.tapThrice = { stacks: game.state.stacks.length, selection: game.state.selection };
result.toolbarAfterThird = get('stackTools').hidden;

// With something selected, a tap on bare felt is the player putting the toolbar down -- not a
// request for another stack under the hand.
tapAt(300, 200);
runFrames(2);
result.dismissTap = { before: game.state.selection.slice(), stacks: game.state.stacks.length };
tapAt(150, 300);
runFrames(2);
result.dismissTap.afterStacks = game.state.stacks.length;
result.dismissTap.afterSelection = game.state.selection.slice();
tapAt(150, 300);
runFrames(2);
result.dismissTap.placedNext = game.state.stacks.length === 2;

// A click on a control (not the toolbar) both does the control's job and lets go of the stack.
tapAt(150, 300);            // select it (it is already there, so this is a select)
runFrames(2);
pressDocument('click', { target: { tagName: 'BUTTON', id: 'snap' } });
result.dismissFromControl = { selection: game.state.selection.slice() };

// The cursor asks what a click would hit, before it clicks. Put one stack back on the centre so
// the question has a stable answer (the two stacks above moved the frame's centre).
get('clear').press('click');
get('clear').press('click');
runFrames(40);
tapAt(300, 200);
runFrames(40);
get('stage').press('pointermove', { target: { tagName: 'CANVAS' }, clientX: 300, clientY: 200,
                                    pointerId: 12, pointerType: 'mouse' });
result.hover = { index: game.state.hover, cursor: get('stage').style.cursor };
get('stage').press('pointerleave', {});
result.hoverAfterLeave = game.state.hover;
// Pointer capture retargets the click after a canvas gesture to the stage itself; that click is
// still the table, and must not clear a selection the gesture just made.
tapAt(300, 200);
runFrames(2);
pressDocument('click', { target: get('stage') });
result.stageClickKeepsSelection = game.state.selection.slice();

// The two ways out, through the buttons, in a document that has no downloads and no clipboard.
get('copyRound').press('click');
get('exportRun').press('click');
result.exportStatus = get('exportStatus').textContent;

// The guide: openable from the table, closable again, and it does not need the run to be idle.
get('help').press('click');
result.guideOpened = get('guide').hidden === false;
get('guideClose').press('click');
result.guideClosed = get('guide').hidden === true;

// Auto-continue, through the page's own clock: settle a safe round, then let the frame loop run
// past the delay. The hand comes back by itself (keep-layout is on), and the next round is a new
// bet, not a deal.
game.restart(performance.now());
game.coverTable();
get('autoContinue').checked = true;
get('autoContinue').press('change', { target: get('autoContinue') });
get('stageDeal').press('click');
runFrames(30, 40);
const autoSettled = { step: game.state.step, phase: game.state.phase };
runFrames(40, 40);
result.autoContinued = {
  settled: autoSettled,
  step: game.state.step,
  phase: game.state.phase,
};
get('autoContinue').checked = false;
get('autoContinue').press('change', { target: get('autoContinue') });

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
LAYOUT_ONLY_IDS = {'bench', 'guideTitle', 'guideClose'}


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


def test_the_game_saves_nothing_between_visits():
    """The tour, the switches, the table and the panel width are session-only.

    The page says so in its own text, and `localStorage` would be the quiet way to break that
    promise: the tutorial's "first visit" flag is the obvious temptation. A reader's browser
    storage belongs to the sites they trust; this page is not collecting any.
    """
    sources = [(GAME / name).read_text(encoding='utf-8') for name in PURE_MODULES + (RENDERER,)]
    sources.append((GAME / 'index.html').read_text(encoding='utf-8'))
    for source in sources:
        assert 'localStorage' not in source, 'the game page writes browser storage'
        assert 'sessionStorage' not in source, 'the game page writes session storage'


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
    assert boot_run['tapTwice']['selection'] == [0], 'the second tap should have held the stack'
    assert boot_run['tapThrice']['stacks'] == 1
    assert boot_run['tapThrice']['selection'] == [], 'a third tap should have let it go'
    assert boot_run['toolbarAfterThird'] is True, 'and taken the toolbar with it'
    dismiss = boot_run['dismissTap']
    assert dismiss['before'] == [0]
    assert dismiss['afterStacks'] == dismiss['stacks'], (
        'a tap on bare felt placed a stack while it was closing the toolbar')
    assert dismiss['afterSelection'] == []
    assert dismiss['placedNext'] is True, 'the next tap should place again'
    assert boot_run['dismissFromControl']['selection'] == [], (
        'a click on a control left the toolbar up')
    assert boot_run['stageClickKeepsSelection'] == [0], (
        'the click that follows a canvas gesture cleared the selection it just made')


def test_the_cursor_shows_what_a_click_would_do(boot_run):
    """On a table with a hand on it, the difference between "add a stack" and "pick this one up" is
    the whole interaction, so the cursor says which one is about to happen."""
    assert boot_run['hover']['index'] == 0
    assert boot_run['hover']['cursor'] == 'pointer'
    assert boot_run['hoverAfterLeave'] is None, 'the highlight outlived the pointer'


def test_the_panel_folds_away(boot_run):
    """The column is a control panel, and a game you cannot stop looking at controls is not
    immersive. The state lives on the body as an attribute, which is what the stylesheet reads.

    The word lives in its own span because the button also carries an icon; rewriting its
    `textContent` is what would silently take the icon away.
    """
    assert boot_run['sideAtStart'] is None
    assert boot_run['sideHidden'] == 'hidden'
    assert boot_run['sideLabel'] == 'Show panel'
    assert boot_run['sideBack'] == 'shown'


def test_clear_takes_two_presses(boot_run):
    """There is no undo any more, so the one control that empties the table asks first."""
    assert 'Sure?' in boot_run['clearArmed'], 'the first press cleared without asking'


def test_keep_my_layout_starts_on(boot_run):
    """A hand is meant to be a rule you can score twice; the switch starts where the game does."""
    assert boot_run['keepLayoutAtStart'] is True


def test_the_play_button_beside_deal_is_the_auto_switch(boot_run):
    """Auto deal is an action, not a setting, so it sits with Deal and says its state twice over:
    a play triangle while it is paused, two bars while it is dealing."""
    assert boot_run['autoAtStart'] == 'paused'
    assert boot_run['autoPlaying'] == 'playing', 'the button did not start dealing on its own'
    assert boot_run['autoPaused'] == 'paused', 'the same button did not pause it'


def test_the_deal_button_says_what_it_will_do(boot_run):
    """The icon follows the phase, and the settled button is the one that asks for a press."""
    assert boot_run['dealAct'] == 'deal'
    assert boot_run['dealActSettled'] == 'next'
    assert boot_run['stagePhase'] == 'settled'


def test_the_tour_starts_on_and_skip_is_session_only(boot_run):
    """Every visit starts the tour; Skip lasts until the tab closes; the guide can reopen it.

    The card has no Next: the shell watches the same gate the reducer does and moves on by itself
    after a beat. So the two things to pin are "it is not ready at boot" and "it advanced without
    anyone pressing anything".
    """
    tour = boot_run['tour']
    assert tour['visibleAtBoot'] is True
    assert tour['step'] == '1 / 6'
    assert tour['guideLabel'] == 'Turn teaching off', 'the guide does not say what its button does'
    assert tour['readyAtBoot'] == 'no', 'the first step was done before the player did it'
    assert tour['readyAfterFrame'] == 'yes', 'the card did not notice the step was done'
    # The beat: the step number is still the old one immediately after the action, and the next
    # step is on screen once the clock has passed.
    assert tour['stepBeforeBeat'] == '1 / 6', 'the card changed before its green edge could be read'
    assert tour['stepAfterBeat'] == '2 / 6', 'the card did not advance on its own'
    assert tour['hiddenAfterSkip'] is True
    # One control, two labels: the guide's button is the whole teaching switch.
    assert tour['guideAfterSkip'] == 'Show the tour', 'the guide cannot bring the tour back'
    # A player who pressed Skip asked not to be talked at; the static "Place your stacks" hint
    # stays away for the visit instead of reappearing in the tour's place.
    assert tour['hintAfterSkip'] is True, 'the empty-table hint came back after Skip'
    assert tour['visibleAfterReopen'] is True
    assert tour['guideAfterReopen'] == 'Turn teaching off'
    assert tour['hiddenAfterGuideOff'] is True, 'the guide button did not turn teaching off'
    assert tour['guideAfterGuideOff'] == 'Show the tour'
    assert tour['hintAfterTeachingOff'] is True, (
        'the empty-table hint came back when teaching was turned off')


def test_the_support_is_drawn_with_the_prediction(boot_run):
    """One switch, two drawings: the support rises and falls with the prediction it is drawn around."""
    support = boot_run['supportToggle']
    assert support['offAtBoot'] is None
    assert support['on'] is not None and support['on']['kind'] == 'box'
    assert support['offAgain'] is None


def test_auto_continue_advances_a_settled_round(boot_run):
    """The clock lives in the frame loop, so the page can be asked whether it moved on."""
    auto = boot_run['autoContinued']
    assert auto['settled']['phase'] == 'settled'
    assert auto['step'] == auto['settled']['step'] + 1
    assert auto['phase'] == 'bet'


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
    "carry on" carries a `?` in its corner and the sentence behind it -- and the one that only
    carries on (Deal) does not, because a `?` on the obvious is noise. Restart is not on the table
    any more; it lives in the panel with a hint that says what it drops.

    Checked in the markup rather than in the browser: the popover needs a real document, and what
    can go wrong here is a control that was added without its explanation.
    """
    html = (GAME / 'index.html').read_text(encoding='utf-8')

    # A helped button may carry other classes too (`helped quiet` is a rare control that still has
    # to explain itself), so the class list is matched as a list rather than as one exact string.
    helped = re.findall(r'(<button[^>]*class="[^"]*\bhelped\b[^"]*"[^>]*>)(.*?)</button>', html, re.S)
    names = set()
    for opening, inner in helped:
        found = re.search(r'id="([^"]+)"', opening)
        assert found, 'a helped button has no id to name it'
        names.add(found.group(1))
        assert 'class="q"' in inner, f'a helped button has no ? badge: {inner[:40]}'
        badge = re.search(r'<i class="q"([^>]*)>', inner).group(1)
        if 'id=' in badge:
            # One badge is filled in by the script, because why "Cover the table" is off depends on
            # the round (the prediction can be hidden, the table can keep its law). What it says is
            # asserted from the running page instead -- `coverHelp` in the tests above.
            continue
        text = re.search(r'data-help="([^"]*)"', inner)
        assert text and len(text.group(1)) > 40, f'the ? badge has nothing to say: {inner[:40]}'
    assert names == {'coverTable', 'evenGrid', 'sameAsLast', 'selectMode', 'clear',
                     'sumExport', 'copyRound', 'exportRun', 'stageAuto', 'resetView'}, (
        f'the helped set is {names}')

    options = re.findall(r'<label class="opt"[^>]*>(.*?)</label>', html, re.S)
    assert len(options) == 2, 'the two things a round shows are pills with a ? each'
    for inner in options:
        assert 'class="q"' in inner and 'data-help="' in inner
    # Settings, not round switches; they explain themselves where they live.
    for key in ('snap', 'keepLayout', 'autoContinue'):
        row = re.search(rf'<label class="teachOpt">(.*?id="{key}".*?)</label>', html, re.S)
        assert row and 'class="q"' in row.group(1) and 'data-help="' in row.group(1), (
            f'{key} does not explain itself')

    button = re.search(r'<button[^>]*id="stageDeal"[^>]*>(.*?)</button>', html, re.S)
    assert button and 'class="q"' not in button.group(1), 'stageDeal does not need a ? badge'

    assert '<div id="helpPop"' in html, 'there is nowhere to show the explanation'
    # The score is the one number on the table that is not self-evident, so it is named and explained.
    settle = re.search(r'<div class="settle">(.*?)</div>', html, re.S).group(1)
    assert 'paid' in settle and 'score' in settle and 'data-help' in settle


def test_the_panel_is_five_folded_rows_in_the_order_you_meet_them():
    """The column is five folded rows and each one answers a single question.

    It used to be three, and the settings row had seventeen controls in it -- a 1281 px card whose
    height alone was the argument for splitting it. The split is by *when you would touch it*: the
    hand every round, the table once, the run once; then the data, then the way out. Only the hand
    starts open (it is the only one a round makes you come back to), and every row carries its own
    state in the summary line, so a closed row can be read.
    """
    html = (GAME / 'index.html').read_text(encoding='utf-8')
    panels = re.findall(r'<details class="panel"([^>]*)>', html)
    assert len(panels) == 5, f'expected five folded rows, found {len(panels)}'
    for panel, key in zip(panels, ('panelHand', 'panelTable', 'panelRun', 'panelData')):
        assert f'id="{key}"' in panel, f'{key} is not in the order it is listed in: {panels}'
    # Exactly one row starts open, and it is the hand.
    assert sum('open' in panel for panel in panels) == 1, f'opened rows: {panels}'
    assert 'open' in panels[0], 'the hand is not the row that starts open'
    assert 'panelMoves' not in html, 'the Advanced panel has been emptied and should be gone'
    assert html.index('class="options"') < html.index('class="side"'), (
        'the round options belong on the table, not in the column')
    assert 'assistHint' not in html, 'that hint now lives on the button it is about'
    # Each settings row says its state where it is folded.
    for key in ('digestHand', 'digestTable', 'digestRun'):
        assert f'id="{key}"' in html, f'{key} is missing: that row cannot be read while closed'


def test_the_options_are_grouped_by_what_they_do():
    """The table keeps the switches that describe what you are looking at, and nothing else.

    A first visit's list of six controls was a wall, and the wall was mostly settings: what carries
    between rounds and what runs itself belong in the panel with the other settings, while "keep
    dealing" belongs next to Deal -- one press, one icon, its state visible. What stays on the felt
    is `see`, because it changes what the felt shows.
    """
    html = (GAME / 'index.html').read_text(encoding='utf-8')
    rows = re.findall(r'<div class="optRow">(.*?)</div>', html, re.S)
    assert len(rows) == 1, f'the table switches are one group, found {len(rows)}'
    names = [re.search(r'<span class="rowLabel">([^<]*)</span>', row).group(1) for row in rows]
    assert names == ['see'], f'the groups are named {names}'
    inside = [[i for i in ('aimAssist', 'showHistory', 'keepLayout', 'autoPlay', 'autoContinue')
               if f'id="{i}"' in row] for row in rows]
    assert inside == [['aimAssist', 'showHistory']], f'the switches are grouped as {inside}'
    assert 'id="autoPlay"' not in html, (
        'auto-play is a button beside Deal now; a leftover checkbox is a second source of truth')
    # The bottom of the table is the two edit/run groups; the preset layouts are in the panel.
    dock = html[html.index('class="dockRight"'):]
    assert [re.search(r'<span class="rowLabel">([^<]*)</span>', row).group(1)
            for row in re.findall(r'<div class="btnRow">(.*?)</div>', dock, re.S)] == [
                'edit', 'run']
    # Auto deal is a button beside Deal, not a checkbox anywhere: it is an action, and the icon is
    # the state.
    run_row = re.findall(r'<div class="btnRow">(.*?)</div>', dock, re.S)[-1]
    assert 'id="stageDeal"' in run_row and 'id="stageAuto"' in run_row
    # Each row holds what its name promises, and nothing that belongs to another.
    def row(key):
        """One folded row's markup, up to the next row that starts."""
        start = html.index(f'id="{key}"')
        nxt = html.find('<details class="panel"', start + 1)
        return html[start:nxt if nxt != -1 else len(html)]
    hand = row('panelHand')
    for key in ('coverTable', 'evenGrid', 'sameAsLast'):
        assert f'id="{key}"' in hand, f'{key} is not with the hand'
    table = row('panelTable')
    for key in ('preset', 'surface', 'snap', 'resetView'):
        assert f'id="{key}"' in table, f'{key} is not with the table'
    run_panel = row('panelRun')
    for key in ('roundsPerRun', 'eps', 'run', 'keepLayout', 'autoContinue', 'restartRun'):
        assert f'id="{key}"' in run_panel, f'{key} is not with the run'
    assert 'id="keepLayout"' not in table and 'id="snap"' not in run_panel, (
        'a setting is filed under the wrong row')
    actions = re.search(r'<div class="actions">(.*?)</div>', html, re.S).group(1)
    assert 'id="sideToggle"' in actions and 'id="helpIcon"' in actions
    assert 'id="helpOpen"' not in html, 'the floating ? came back next to the always-present link'
    # Teaching has no switch of its own: the guide's button is the control, one press either way.
    assert 'id="teaching"' not in html, 'the teaching checkbox came back next to the guide button'
    guide = html[html.index('id="guideTitle"'):]
    assert 'id="guideTour"' in guide


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
    # The hand is the safe layout: four equal tiles, played in every round of the run. The count
    # comes from the run rather than from a literal: the default length is a product decision and
    # has already changed once (20 -> 10), which a hardcoded "x20" turns into a broken test rather
    # than a passing one that says what it means.
    assert f"4 stacks x{summary['rounds']}" in summary['distinctHandNames'][0]
    assert 1 <= summary['bestRound'] <= summary['rounds']
    assert 1 <= summary['worstRound'] <= summary['rounds']
    assert summary['gapToPar'] == pytest.approx(0.0, abs=1e-9), 'the safe layout sits on par'


def test_a_run_can_be_however_long_the_player_asked_for(node_run):
    """The length is a choice now, and it is capped by the replay rather than by a constant."""
    chosen = node_run['chosenRun']
    assert chosen['declared'] == 30
    assert chosen['rounds'] == 30
    assert chosen['planned'] == 30


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


def test_a_dragged_stack_stays_where_you_drop_it(node_run):
    """Undo is gone: a drag is a move, and the stack stays where the hand left it."""
    drag = node_run['drag']
    assert drag['moved'] is True
    assert drag['persists'] is True, 'the stack moved back on its own; there is no undo any more'


def test_a_tap_on_bare_felt_only_closes_a_selection(node_run):
    """The tap that closes the toolbar must not also put a stack under the hand.

    This is the difference between "I am done with this stack" and "I want another one here" --
    and the player has already said which by selecting something first.
    """
    dismiss = node_run['dismiss']
    assert dismiss['selectedBefore'] == 1
    assert dismiss['placed'] is False
    assert dismiss['dismissed'] is True
    assert dismiss['stacksAfterFirst'] == 1, 'the tap placed a stack while closing the toolbar'
    assert dismiss['selectionAfterFirst'] == 0
    assert dismiss['secondPlacement'] is True, 'the next tap should place again'
    assert dismiss['stacksAfterSecond'] == 2


def test_a_selection_is_a_group(node_run):
    """One weight, one step, one move: and a move that would land on somebody else moves nothing."""
    batch = node_run['batch']
    assert batch['selected'] == [0, 1, 2]
    assert batch['weights'] == [2, 2, 2, 1], 'the unselected stack changed weight'
    assert batch['nudged'] == [4, 4, 4, 1]
    assert batch['moved'] is True
    assert batch['movedTogether'] is True, f'the group did not keep its shape: {batch}'
    assert batch['blocked'] is False, 'a move into an occupied cell was allowed'
    assert batch['unchangedAfterBlock'] is True, 'a refused group move moved half the group'
    assert batch['removed'] is True
    assert batch['remaining'] == 1
    assert batch['remainingX'] == pytest.approx(batch['fourthX'])
    assert batch['selectionAfterRemove'] == 0


def test_the_noise_support_is_what_each_law_publishes(node_run):
    """The overlay is a picture of the exported law, and a table that keeps its law has no picture.

    The Gaussian has no support at all, so the page draws a chosen 3σ window and says so; drawing
    it with the prediction hidden would hand back the point the prediction switch hides.
    """
    support = node_run['support']
    assert support['shear']['kind'] == 'box'
    assert support['shear']['low'] == [-1, -1] and support['shear']['high'] == [1, 1]
    assert support['stacked']['kind'] == 'bands'
    assert [band['weight'] for band in support['stacked']['bands']] == [0.1, 0.8, 0.1]
    assert support['rotation']['kind'] == 'ellipse'
    assert support['rotation']['rx'] == pytest.approx(1.5)
    assert support['rotation']['ry'] == pytest.approx(1.5)
    assert support['house'] is None
    assert support['houseView'] is None, 'a table that keeps its law drew a support window'
    assert support['off'] is None
    assert support['on']['kind'] == 'box'
    assert support['noAssist'] is None, 'the support disclosed the prediction with the assist off'


def test_auto_continue_is_a_decision_not_a_timer(node_run):
    """The shell drives the clock; this only says whether the clock has arrived."""
    auto = node_run['autoContinue']
    assert auto['off'] is False
    assert auto['dueBefore'] is False
    assert auto['dueAt'] is True
    assert auto['suppressedByAutoPlay'] is False, 'auto-play and auto-continue double-stepped'


def test_the_tour_starts_on_is_gated_and_can_be_reopened(node_run):
    """Session-only teaching: start on, every step waits for its own action, skip, re-open."""
    tour = node_run['tutorial']
    assert tour['fresh'] == 'prediction'
    assert tour['walk'] == ['prediction', 'place', 'cover', 'height', 'deal', 'next']
    # Each step is ready only when its own action is on the table; the step before it is not
    # enough, which is what makes the step a verdict rather than a click.
    assert tour['readyAt'] == [True] * 6
    assert tour['blockedWithoutItsAction'] == [True] * 6
    assert tour['gatedAtStart'] == 0, 'the tour moved on before prediction was on'
    assert tour['finished'] is None
    assert tour['skipped'] is None
    assert tour['reopened'] == 'prediction'
    assert tour['disabled'] is None
    # Pinned to the table's lower-left corner, above the dock: it is a corner card now, not a
    # coach mark that follows the thing each step describes.
    assert tour['pinned'] == {'left': 28, 'top': 456}
    assert tour['pinnedLeftGap'] == 12 and tour['pinnedDockGap'] == 12
    assert tour['crowdedTop'] == 12, 'a short table pushed the card off the felt'
    covered = set(tour['covers'])
    # Gestures are still the tour's business: moving the table out from under the card is one of
    # the things it has to teach, or the card is in the way of its own instructions.
    assert {'tap', 'place', 'orbit', 'pinch', 'pan', 'support', 'cover', 'safe',
            'select-stack', 'height', 'toolbar', 'deal', 'result', 'next-round',
            'keep-layout'} <= covered


def test_a_pinch_is_a_sequence_of_steps_not_one_ratio(node_run):
    """Two fingers zoom and slide: the zoom is incremental, damped, and commutes between samples."""
    pinch = node_run['pinch']
    assert pinch['kind'] == 'pinch'
    assert pinch['midpointIsMean'], 'a pinch must report where the pair is, not where one finger is'
    assert pinch['distance'], 'the pinch spread is the distance between the two fingers'
    assert pinch['steps'], 'the zoom handed back is the ratio between the last two samples'
    # The regression: 100→110 then 110→121 has to equal 100→121 exactly. The old shell multiplied
    # by `start / now` on every event, so the two-step spread composed to 1.21 instead of 1.21**0.8.
    assert pinch['composes'], 'pinch steps do not compose -- the zoom is compounding again'
    assert pinch['damped'], 'a doubled finger spread must not double the zoom'
    assert pinch['undoPair'], 'spreading and closing the same amount must cancel exactly'
    assert pinch['slideMid'] == [90, 30], 'the pair moved 40 right and 30 down'
    assert pinch['slideDistance'], 'moving both fingers together must not change their spread'


def test_a_wheel_delta_is_a_magnitude(node_run):
    """A trackpad's 2 px stream and a mouse's 120 px notch are not the same step."""
    wheel = node_run['wheel']
    assert wheel['out'] and wheel['back']
    assert wheel['tiny'] < wheel['notch'], 'a 2 px delta zoomed as hard as a whole notch'
    assert wheel['clamped'] < 1.2, 'a flick reported thousands of pixels and flew'
    assert wheel['sameClamp'], 'the delta is not clamped'
    assert wheel['halves'], 'wheel in and wheel out by the same amount must cancel'


def test_the_tallest_height_fits_the_frame(node_run):
    """The cap is a promise about the picture, checked against the frame's own rule."""
    budget = node_run['heightBudget']
    assert len(budget) == 15
    failed = [item for item in budget if not item['fits']]
    assert failed == [], f'the tallest tower overflows the frame: {failed}'


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
    """The span is the run's, so the camera can only pan -- and a pan does not resize the hand.

    This is the rule that stops a run from slowly zooming out. A frame fitted to the moment grew
    with the drift (3.03 -> 8.36 over six rounds on the Long Table) and every growth shrank the
    towers the player had built, for a reason that had nothing to do with them.
    """
    frame = node_run['frame']
    assert frame['spanEmpty'] == pytest.approx(frame['spanRun']), (
        'an empty table must frame at the span the whole run was sized for')
    assert frame['handDoesNotZoom'] is True, (
        'placing a hand changed the zoom: the span must be the run\'s, not the moment\'s')
    assert frame['centreEmpty'] == pytest.approx(frame['midpoint'])
    assert frame['radiusShared'] is True, 'the discs and the lattice must measure the same epsilon'
    assert frame['covers'] is True, 'the frame does not hold every stack'
    assert frame['neverFurtherThanDouble'] is True, 'the view is zooming out with the hand'


def test_the_frame_is_sized_for_the_run_not_for_the_moment(node_run):
    """Every step of the run fits inside one span, so the span never has to grow mid-run.

    The failure this replaces: `past` (the trail) widened the frame each round, and the trail's
    reach grows without bound on any table that drifts, so the frame followed it outwards until it
    hit the old `2 * floor` ceiling. The towers rode that zoom all the way down.
    """
    run = node_run['runFrame']
    assert run['coversEveryStep'] is True, 'the run frame does not hold every step of the run'
    assert run['steady'] is True, 'the frame grew when the drift opened the gap'
    assert run['wouldHaveBeenSmaller'] is True, (
        'the test is not exercising a run whose gap actually varies')




def test_a_new_round_moves_the_hand_then_the_camera(node_run):
    """Two beats, in that order: the hand travels, and only then does the camera follow.

    It used to be the other way round -- the hand faded out, the camera moved over an empty table,
    and the hand reappeared already in its new place -- so the one thing that had actually changed
    was the one thing you could not see move. The samples below are the proof of the order: during
    the slide the hand is halfway between the two anchors *and the camera has not moved*, and by
    the camera beat the hand has arrived and stands still.
    """
    change = node_run['change']
    assert change['start']['active'] is True
    assert change['start']['visible'] >= 1, 'the hand was hidden before the round even started'
    assert change['start']['handAtOldAnchor'] is True, 'the hand started somewhere else'
    assert change['start']['cameraHeld'] is True
    assert change['slide']['handBetween'] is True, 'the hand jumped instead of travelling'
    assert change['slide']['cameraHeld'] is True, (
        'the camera moved while the hand was still travelling')
    assert change['camera']['handArrived'] is True, 'the hand had not arrived when the camera moved'
    assert change['camera']['cameraMoved'] is True, 'the camera never followed the hand'
    assert change['camera']['centreAtTarget'] is True, 'the camera settled somewhere else'
    assert change['arrived']['active'] is False
    # A round is a move, not a zoom: nothing in the transition may change the span.
    assert change['spanUnchanged'] is True, 'the round changed the zoom'
    assert change['oldHandGone'] is True, 'a kept hand faded out instead of travelling'


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


def test_the_height_has_a_reachable_ceiling_and_the_shares_follow_it(node_run):
    """A weight is a height, its size means nothing, and every share on the table is that weight
    over the total -- computed where it is shown, so nothing can store a share and fall out of
    step with the geometry.

    The ceiling is a **comfort** decision, not a claim about weights: only the ratio between stacks
    is observable, but a tower 300 times a fresh stack turns the table into a grid, and a
    mis-touch on the way to closing the toolbar used to be able to do it. The slider stays
    logarithmic; it just stops at a height a person can read.
    """
    weights = node_run['weights']
    assert weights['sharesSum'] == pytest.approx(1.0)
    assert weights['shareFollowsHeight'] is True, 'a taller stack must be a bigger share'
    assert weights['ceiling'] == pytest.approx(8)
    assert weights['keptHigh'] == pytest.approx(weights['ceiling']), 'the cap is not reachable'
    assert weights['clampedLow'] == pytest.approx(weights['min'])
    assert weights['min'] == pytest.approx(0.1)
    assert weights['nudged'] == pytest.approx(1.12)
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
    """A height the player makes has to stay in the picture.

    With a reachable cap the camera no longer *has* to move for the tallest tower -- the cap is
    chosen so it fits the frame the empty table already asks for -- but `frameBox` must still honor
    a tower that does exceed the frame (the direct 12-unit case), never move the centre to do it,
    and come back in when the tower goes down.
    """
    frame = node_run['towerFrame']
    assert frame['withTower'] > frame['floor'], 'a tall stack did not push the camera back'
    assert frame['withTower'] >= 12 * frame['room'], 'the frame does not cover the tower'
    assert frame['centreUnmoved'] is True, 'making room moved the table'
    assert frame['grewDirect'] is True, 'the direct tower case changed no frame'
    assert frame['capped'] is True, 'the height control has no reachable ceiling'
    assert frame['fitsWithoutMoving'] is True, 'the cap is taller than the frame it promised to fit'
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
