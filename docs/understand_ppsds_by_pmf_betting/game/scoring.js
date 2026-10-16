/**
 * The scoring rule, in JavaScript -- the second implementation, and the one the self-check
 * exists to keep honest.
 *
 * ppsds scores a hand in Python; the page has to score it in the browser, instantly and with no
 * server. Two implementations of one rule drift apart eventually, so the exported vectors are
 * recomputed here on every page load and a disagreement is said out loud rather than discovered
 * later as "the game and the script disagree".
 *
 * What a hand pays
 * ----------------
 * The stacks are a **pmf** over where the next state will be, so what settles is the *share* of
 * the bet that landed within `eps` of the landing:
 *
 *     hit     = sum of the weights within eps of the landing   (infinity-norm, <= eps)
 *     roi     = hit / total                                    (the payout share)
 *     utility = log(roi)                                       (the epsilon-logarithm score)
 *
 * The share, not the sum: the library normalises a pmf on construction, so `log(hit / total)` is
 * what `EpsilonLogarithmScore` returns and `log(hit)` is a different number that happens to
 * agree when the weights sum to one. The utility is therefore exactly `log(roi)`, which is why
 * the page can show a payout percentage and a log score without either being decoration.
 *
 * The average of the utilities is a growth rate -- `exp` of it is the geometric mean payout, the
 * rate a stake rolled over every round would compound at. The page shows that number next to the
 * *arithmetic* mean payout, because the two differ and only one of them compounds. It does not
 * show the stake itself any more: as a running total it is `exp(rounds x average score)`, the same
 * information written twice, and because a payout share is always below one it only ever decays to
 * nothing, which is a strange thing to put a number on.
 */

import { coveringLayout, gaussianMissProbability, uncoveredReport } from './coverage.js';
import { nominalPath } from './world.js';

/**
 * How far this file's answers may sit from ppsds' before it complains.
 *
 * The exported vectors are rounded to six decimals, so the recorded number and a recomputation
 * from the recorded inputs already differ by up to ~5e-7 -- these tolerances are set by that
 * rounding, not by the arithmetic. They are still four orders of magnitude tighter than any
 * real disagreement: a missed normalisation or an off-by-one tiling is off by ≫ 1e-3.
 */
export const SCORE_TOLERANCE = 1e-5;
export const GEOMETRY_TOLERANCE = 1e-6;

/** The total weight of the stacks within `eps` of the landing, in the infinity norm. */
export function hitWeight(stacks, landing, eps) {
  let hit = 0;
  for (const stack of stacks) {
    if (Math.max(Math.abs(stack.x - landing[0]), Math.abs(stack.y - landing[1])) <= eps) {
      hit += stack.w;
    }
  }
  return hit;
}

/** The weight on the table -- every stack's, added up. */
export function totalWeight(stacks) {
  let total = 0;
  for (const stack of stacks) total += stack.w;
  return total;
}

/**
 * What one round paid: the share that landed, and its logarithm.
 *
 * A hand that misses entirely pays nothing and scores `-inf`. That is a legitimate settlement --
 * "nothing landed" -- and it is the whole reason a neighbourhood is used at all.
 */
export function roundStats(stacks, landing, eps) {
  const total = totalWeight(stacks);
  const hit = total > 0 ? hitWeight(stacks, landing, eps) : 0;
  const roi = total > 0 ? hit / total : 0;
  return { hit, total, roi, utility: roi > 0 ? Math.log(roi) : -Infinity };
}

/** The arithmetic mean -- of utilities, this is the quantity the par line bounds. */
export function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

const close = (mine, theirs, tolerance) =>
  mine === theirs
  || (Number.isFinite(mine) && Number.isFinite(theirs) && Math.abs(mine - theirs) <= tolerance);

/**
 * Recompute every exported case and report the ones this file gets wrong.
 *
 * @param {object[]} cases the `selfcheck.json` payload
 * @param {object} presets the `presets.json` payload (the drift cases need the matrices)
 * @returns {{ok: boolean, checked: number, problems: string[]}}
 */
export function selfCheck(cases, presets) {
  const problems = [];
  let checked = 0;
  for (const item of cases) {
    const label = `${item.preset} eps=${item.eps}`;
    const before = problems.length;

    if (item.kind === 'score') {
      const stacks = item.atoms.map((atom, i) => ({ x: atom[0], y: atom[1], w: item.weights[i] }));
      const mine = item.noise.map((landing) => roundStats(stacks, landing, item.eps).utility);
      const wrong = mine.findIndex((value, k) => !close(value, item.per_step_scores[k], SCORE_TOLERANCE));
      if (wrong >= 0) {
        problems.push(`${label} (${item.bet_name}): round ${wrong} scored ${mine[wrong]} `
          + `but ppsds says ${item.per_step_scores[wrong]}`);
      } else {
        const running = mean(mine);
        if (!close(running, item.mean_score, SCORE_TOLERANCE)) {
          problems.push(`${label}: mean ${running} vs ppsds ${item.mean_score}`);
        }
      }
    } else if (item.kind === 'coverage') {
      if (item.layout === 'covering') {
        // The page builds this hand itself, so the case asks it to build it *and* measure it.
        const made = coveringLayout({ low: item.low, high: item.high }, item.eps);
        const same = made.atoms.length === item.atoms.length
          && made.atoms.every((atom, i) => close(atom[0], item.atoms[i][0], GEOMETRY_TOLERANCE)
            && close(atom[1], item.atoms[i][1], GEOMETRY_TOLERANCE));
        if (!same) {
          problems.push(`${label}: the covering layout has ${made.atoms.length} stacks, `
            + `ppsds has ${item.atoms.length}`);
        }
      }
      if (item.uncovered_area_fraction !== undefined) {
        const report = uncoveredReport(item.atoms, item.weights, item.low, item.high, item.eps);
        if (!close(report.uncoveredAreaFraction, item.uncovered_area_fraction, GEOMETRY_TOLERANCE)) {
          problems.push(`${label}: uncovered ${report.uncoveredAreaFraction} `
            + `vs ppsds ${item.uncovered_area_fraction}`);
        }
        if (!close(report.worstCaseRoi, item.worst_case_roi, GEOMETRY_TOLERANCE)) {
          problems.push(`${label}: worst-case payout ${report.worstCaseRoi} `
            + `vs ppsds ${item.worst_case_roi}`);
        }
        if (report.possibleNegInf !== item.possible_neg_inf) {
          problems.push(`${label}: says ${report.possibleNegInf ? 'can' : 'cannot'} miss, `
            + `ppsds says ${item.possible_neg_inf ? 'can' : 'cannot'}`);
        }
      } else {
        const miss = gaussianMissProbability(item.atoms, item.weights, item.mean, item.cov, item.eps);
        if (!close(miss, item.miss_probability, GEOMETRY_TOLERANCE)) {
          problems.push(`${label}: uncatchable mass ${miss} vs ppsds ${item.miss_probability}`);
        }
      }
    } else if (item.kind === 'drift') {
      const path = nominalPath(presets[item.preset], item.start, item.steps);
      if (path.length !== item.path.length) {
        problems.push(`${label}: the drift path has ${path.length} points, ppsds has ${item.path.length}`);
      } else {
        const wrong = path.findIndex((point, i) => !close(point[0], item.path[i][0], GEOMETRY_TOLERANCE)
          || !close(point[1], item.path[i][1], GEOMETRY_TOLERANCE));
        if (wrong >= 0) {
          problems.push(`${label}: the drift path is off at step ${wrong} `
            + `(${path[wrong]} vs ${item.path[wrong]})`);
        }
      }
    } else {
      problems.push(`unknown self-check case kind ${item.kind}`);
    }

    if (problems.length === before) checked += 1;
  }
  return { ok: problems.length === 0, checked, problems };
}

/** The one line the page shows and the test asserts, so both say the same thing. */
export function formatSelfCheck(result) {
  return result.ok
    ? `self-check ok (${result.checked} cases)`
    : `SELF-CHECK FAILED: ${result.problems.join('; ')}`;
}
