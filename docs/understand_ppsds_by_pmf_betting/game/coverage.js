/**
 * What a hand of stacks covers -- the JavaScript half of `examples/understand_ppsds_by_pmf_betting/experiment/bet_coverage.py`.
 *
 * A landing pays nothing when it falls outside every stack's neighbourhood, so "can this hand
 * ever lose?" is a geometry question: does the union of the stacks' boxes cover the noise's
 * support? This file answers it the same way the Python side does -- **exactly**, by cutting the
 * plane at every box edge, so that within one cell every stack pays or does not pay as a whole.
 * No sampling, because a sampled answer cannot distinguish "safe" from "safe so far".
 *
 * The page needs this in the browser: the safety badge updates on every drag, with no server.
 * The exported self-check vectors tie this implementation to the Python one, and the Python one
 * is the authority (see `tests/test_game_page.py`).
 *
 * Mirrored from `bet_coverage.py`: keep the two in step, and let the self-check prove it.
 */

/** How far out the catch probability is evaluated for unbounded noise, in standard deviations.
 *  Must match `bet_coverage.MISS_WINDOW_SIGMA`; beyond this window the tail is below 1e-8. */
export const MISS_WINDOW_SIGMA = 6.0;

/**
 * A layout of stacks whose neighbourhoods **tile** the box, so nothing there pays nothing.
 *
 * Tiling is what makes it safe: with the boxes exactly `2 eps` wide and their centres no further
 * than `2 eps` apart, neighbouring boxes touch, so every point of the box is in at least one.
 * The stacks sit one per tile centre with equal weights, so the layout is a pmf -- and when the
 * box's half-width is a whole multiple of `eps` the tiles fit exactly, every landing pays the
 * same share, and more stacks stop buying anything.
 *
 * @param {{low: number[], high: number[]}} box the box to cover
 * @param {number} eps the payout radius; with `eps = 0` no layout can cover anything
 */
export function coveringLayout(box, eps) {
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error(`a covering layout needs a positive radius, got ${eps}`);
  }
  const axes = [0, 1].map((axis) => {
    const centre = 0.5 * (box.low[axis] + box.high[axis]);
    const half = 0.5 * (box.high[axis] - box.low[axis]);
    // `ceil(half / eps)` tiles at a spacing of at most `2 eps`, which is what makes the boxes
    // touch. One tile is enough when the box is no wider than a box -- and then the centre is
    // the box's own centre, not the first point of a `linspace` that would sit off to one side.
    const count = Math.max(1, Math.ceil(half / eps));
    if (count === 1) return [centre];
    const reach = half - eps;
    const centres = [];
    for (let i = 0; i < count; i += 1) centres.push(centre - reach + (2 * reach * i) / (count - 1));
    return centres;
  });
  const atoms = [];
  for (const x of axes[0]) for (const y of axes[1]) atoms.push([x, y]);
  const weight = 1 / atoms.length;
  return { atoms, weights: atoms.map(() => weight) };
}

/**
 * The grid of cells on which the payout is constant: the box's own edges plus every stack's.
 *
 * Clipping before de-duplicating matters: an edge outside the box would add a strip that is not
 * part of the support, and those must not enter the areas or the worst case.
 */
export function compressedGrid(atoms, low, high, eps) {
  const edges = (lo, hi, coordinate) => {
    const values = [lo, hi];
    for (const value of coordinate) {
      values.push(Math.min(Math.max(value - eps, lo), hi));
      values.push(Math.min(Math.max(value + eps, lo), hi));
    }
    return [...new Set(values)].sort((a, b) => a - b);
  };
  return {
    xs: edges(low[0], high[0], atoms.map((a) => a[0])),
    ys: edges(low[1], high[1], atoms.map((a) => a[1])),
  };
}

/** `mass[i][j]`: the total weight of the stacks covering cell `(i, j)`. */
export function coveredMass(atoms, weights, xs, ys, eps) {
  const middleX = xs.slice(0, -1).map((x, i) => 0.5 * (x + xs[i + 1]));
  const middleY = ys.slice(0, -1).map((y, j) => 0.5 * (y + ys[j + 1]));
  return middleX.map((x) =>
    middleY.map((y) => {
      let mass = 0;
      for (let k = 0; k < atoms.length; k += 1) {
        // A cell's middle decides for the whole cell: no stack edge crosses a cell interior.
        if (Math.abs(x - atoms[k][0]) <= eps && Math.abs(y - atoms[k][1]) <= eps) {
          mass += weights[k];
        }
      }
      return mass;
    }));
}

/**
 * What a hand can and cannot do over the box `[low, high]`.
 *
 * Returns `uncoveredAreaFraction` (the share of the box no stack reaches), `worstCaseRoi` (the
 * smallest payout share over the box -- zero as soon as any part is uncovered) and
 * `possibleNegInf` (whether some landing pays nothing).
 */
export function uncoveredReport(atoms, weights, low, high, eps) {
  const total = weights.reduce((sum, w) => sum + w, 0);
  const { xs, ys } = compressedGrid(atoms, low, high, eps);
  const mass = coveredMass(atoms, weights, xs, ys, eps);
  const boxArea = (high[0] - low[0]) * (high[1] - low[1]);

  let coveredArea = 0;
  let worst = Infinity;
  let possibleNegInf = false;
  for (let i = 0; i < mass.length; i += 1) {
    for (let j = 0; j < mass[i].length; j += 1) {
      const share = mass[i][j] / total;
      if (mass[i][j] > 0) {
        coveredArea += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]);
      } else {
        possibleNegInf = true;
      }
      worst = Math.min(worst, share);
    }
  }
  return {
    uncoveredAreaFraction: 1 - coveredArea / boxArea,
    worstCaseRoi: worst,
    possibleNegInf,
  };
}

/**
 * The probability a landing pays nothing, when the noise is a Gaussian.
 *
 * A Gaussian can land anywhere, so no hand can guarantee a payout and the worst case is always
 * zero. The honest number is this one: the mass the stacks fail to catch, over the whole plane
 * (the tail beyond the `MISS_WINDOW_SIGMA` window is added rather than dropped).
 *
 * Assumes independent axes -- the exporter refuses to emit a case for correlated noise, so a
 * preset like that fails at build time rather than quietly printing a wrong probability here.
 */
export function gaussianMissProbability(atoms, weights, mean, cov, eps) {
  const sigma = [Math.sqrt(cov[0][0]), Math.sqrt(cov[1][1])];
  const low = [mean[0] - MISS_WINDOW_SIGMA * sigma[0], mean[1] - MISS_WINDOW_SIGMA * sigma[1]];
  const high = [mean[0] + MISS_WINDOW_SIGMA * sigma[0], mean[1] + MISS_WINDOW_SIGMA * sigma[1]];
  const { xs, ys } = compressedGrid(atoms, low, high, eps);
  const mass = coveredMass(atoms, weights, xs, ys, eps);

  let miss = 1 - boxProbability(low, high, mean, sigma);
  for (let i = 0; i < mass.length; i += 1) {
    for (let j = 0; j < mass[i].length; j += 1) {
      if (mass[i][j] <= 0) {
        miss += boxProbability([xs[i], ys[j]], [xs[i + 1], ys[j + 1]], mean, sigma);
      }
    }
  }
  return miss;
}

/** `P(low <= X <= high)` for independent axes. */
function boxProbability(low, high, mean, sigma) {
  const upper = low.map((_, axis) => normalCdf((high[axis] - mean[axis]) / sigma[axis]));
  const lower = low.map((lo, axis) => normalCdf((lo - mean[axis]) / sigma[axis]));
  return upper.reduce((product, value, axis) => product * (value - lower[axis]), 1);
}

/** The standard normal CDF. */
export function normalCdf(z) {
  return 0.5 * erfc(-z / Math.SQRT2);
}

/**
 * The complementary error function, to near machine precision.
 *
 * There is no `Math.erfc`, and the usual short approximations are good to ~1e-7 -- not enough to
 * compare against `scipy.special.ndtr` on the far side of a 6-sigma window. So: a Taylor series
 * where it converges fast (|x| <= 1) and the standard continued fraction
 * `erfc(x) = exp(-x^2) / sqrt(pi) / (x + 1/2 / (x + 1 / (x + 3/2 / (x + ...))))` beyond it,
 * evaluated by backward recurrence. Negative arguments come from the identity
 * `erfc(-x) = 2 - erfc(x)` -- without it the series would be asked to converge at |x| = 6,
 * which it does not.
 *
 * The one place precision is lost is `normalCdf` far in the *upper* tail, where the answer is
 * `1 - tiny` and the subtraction spends the tiny part's digits. The error there is ~1e-10
 * absolute, which is far below the tolerance the self-check compares at.
 */
function erfc(x) {
  if (x < 0) return 2 - erfc(-x);
  if (x >= 1) {
    let fraction = 0;
    for (let n = 60; n >= 1; n -= 1) fraction = n / 2 / (x + fraction);
    return Math.exp(-x * x) / Math.sqrt(Math.PI) / (x + fraction);
  }
  let term = x;
  let sum = x;
  const square = x * x;
  for (let n = 1; n < 80; n += 1) {
    term *= -square / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
  }
  return 1 - (2 / Math.sqrt(Math.PI)) * sum;
}
