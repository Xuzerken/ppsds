/**
 * One palette for both renderers, the HUD and the charts, so nothing means two things.
 *
 * The colours are the page's presentation, not the library's: nothing here is exported, and a
 * table's felt is a mood rather than a property of the world (which is why the presets carry no
 * colour). Kept in one file because four places draw with it -- the 3-D scene, the flat fallback,
 * the table card's preview and the scoreboard charts.
 *
 * Two families, with different jobs:
 *
 * * **marks on the felt** (stacks, the puck, the prediction, the landing). Their colours say what a
 *   thing *is*, and the hit/miss pair is the one place where colour alone would be a trap: green
 *   against red is the classic red-green confusion, so a settled round never relies on it -- the
 *   stacks also rise or sink, the payout is written next to them, and the announcement line says
 *   it in words.
 * * **chart series** (`CHART`). Validated as a two-slot palette against this page's panel surface:
 *   lightness band, chroma floor, protan/deutan separation, and >= 3:1 contrast all pass
 *   (worst adjacent CVD delta-E 15.9, normal-vision 26.5). Re-run the validator before changing
 *   these two -- they are the only pair in the page that has to survive colour blindness on its
 *   own, because a line has no second encoding.
 */

export const PALETTE = {
  ink: '#e8ecf3',
  dim: '#9aa5b8',
  pale: 'rgba(232, 236, 243, 0.55)',
  puck: '#f6f9ff',
  predicted: '#f5c451',
  stack: '#63b3ed',
  hit: '#4ade80',
  miss: '#f87171',
  landed: '#e8985a',
  eps: 'rgba(125, 211, 252, 0.14)',
  epsEdge: 'rgba(125, 211, 252, 0.42)',
  safe: '#4ade80',
  unsafe: '#f87171',
  maybe: '#f5c451',
  grid: 'rgba(232, 236, 243, 0.07)',
  panel: '#161a22',
};

/** The two chart colours, validated together against `PALETTE.panel`. */
export const CHART = {
  series: '#3987e5',
  reference: '#d55181',
  ink: '#9aa5b8',
};

/** Each table's felt: the Long Table is baize, the Roulette wheel is red, the Whirlpool is deep. */
export const FELT = {
  shear: { name: 'green', top: '#17563a', edge: '#0a2f1f' },
  rotation: { name: 'red', top: '#7a1f22', edge: '#430f12' },
  damped: { name: 'blue', top: '#1c3d7a', edge: '#0e2149' },
};

export function feltOf(key) {
  return FELT[key] || FELT.shear;
}

/** The colour a stack stack takes once the round has settled. */
export function settleColour(state) {
  if (state === 'hit') return PALETTE.hit;
  if (state === 'miss') return PALETTE.miss;
  return PALETTE.stack;
}
