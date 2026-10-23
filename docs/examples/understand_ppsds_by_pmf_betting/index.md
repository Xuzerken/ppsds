# Understand ppsds by playing bet game

The fastest way to see what this package is about is to play it. The page below is the whole idea
with the abstraction removed: a puck moves around a table, you put stacks where you think it will
land next, the dealer drops a ball, and every stack within `epsilon` of the landing pays its share.

<style>
  /* The frame is the only interactive thing on this page, and its height is a reading decision as
     much as a playing one. On a phone a fixed 900 px frame is the *entire* screen: the only way
     left to scroll the article is to start the drag on the frame, and the game reads a drag as a
     turn of the camera -- so the reader trying to get to the text below ends up rotating the table.
     Small screens therefore get a share of the viewport instead of a fixed height, which leaves
     the article enough room to be scrolled by, and the frame is one press of Full screen away from
     the whole window. A mouse does not have that problem, so the desktop height stays fixed. */
  .game-frame {
    display: block; width: 100%; height: 900px;
    border: 1px solid #ccc; border-radius: 8px;
  }
  @media (max-width: 700px) { .game-frame { height: 66dvh; } }
  /* Sideways on a phone: the frame is what is short, not the reading column. */
  @media (max-height: 520px) { .game-frame { height: 74dvh; } }
</style>

<iframe class="game-frame" src="game/index.html" title="Place your bets" allow="fullscreen"
        loading="lazy"></iframe>

If the frame does not load, open
[the game directly](https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/game/index.html) — it is a static page with no
server behind it. The **Full screen** button (or `F`) gives it the whole window, and **Hide
panel** (or `P`) folds the control column away, leaving only the table. That costs you nothing but
the settings: everything a round needs — the hand buttons, the switches, the score, the running
average — is on the table itself. The panel is five folded rows — **Your hand**, **The table**,
**This run**, **Data** and **Take your bets with you** — and each one says its own state on the
summarised line, so a closed row can still be read. It is there when you want it: the edge between it and
the table is a real edge, and dragging it is how the panel goes from one card wide to two and three
as the cards flow into the room you give them. On a phone — either way up — the header shrinks to a
row of icons and the panel becomes a sheet that slides up from the bottom when asked for, so the
table gets the whole screen instead of two thirds of it. The one thing the page
needs from the network is the 3-D library, and without it it shows a flat table and keeps playing —
that flat table is also a choice, under **How it is drawn**, for a reader on a machine without a GPU
or on one where the table reads better without the depth.

The first visit starts a short **guided tour**: a card in the table's lower-left corner and a
bouncing arrow that points at the one thing each step is about. **The six steps are the skeleton of
this page, and the sections below follow them in order**: turn on prediction, place a stack, cover
the noise support, change a height, deal, next round. The card has no **Next**: a step ends when you
have actually done what it asks — the table, not a button, says when a step is done — and the card's
edge turns green for a moment before the next step takes its place. **Skip** closes the tour for
this visit — and takes the static "Place your stacks" hint with it — the **teaching** switch in Game
settings brings it back, and so does **Show the tour** at the bottom of How to play. Nothing
about the tour is stored — every visit starts it again — which is part of the same promise as the
rest of the page: no server, no files, nothing kept between visits.

## The three things on the screen

Three things are on the screen, and they are the three things this package is made of:

| on screen | in the package |
|---|---|
| the moving puck | the **SDS** — a drift plus a known noise law |
| the stacks you place | the **probabilistic predictor** — a pmf over where the puck will be |
| how they pay | the **scoring rule** — the `epsilon`-logarithm score |

The panel is split by *when you would touch it*, not by kind: **Your hand** (the preset predictors,
the one row that starts open, because it is the one a round brings you back to), **The table** (the
drift in plain words, an animated picture of what it does to a puck, the noise law, the matrix, and
how it is drawn), **This run** (how long, the payout radius, which saved run, what carries between
rounds), then **Data** and the way to take your bets with you. Buttons are sized by how often they
are pressed: **Deal** is the only filled one, the three hand buttons are the largest in the panel,
and **Restart run** and **Recentre the table** are quiet because they are rare (and, in Restart's
case, destructive). Every control
that is not simply "carry on" carries a **?** in its corner. Tapping it opens the sentence that
explains it — the same sentences a manual would have to repeat, kept next to the things they are
about instead of in a paragraph somewhere else.

## 1. Turn on prediction

The tour starts here because the prediction is the thing the rest of the round is about. The
switch is in the table's top-right corner and starts **off**; turning it on draws two things that
belong together:

- where the drift alone would put the puck, and the drift's next few steps;
- the **noise support** around it, in the marker's gold at a low opacity: the region where the
  landing law says the puck can land.

The table's own movement is **given**: pick a table and you are saying its drift is known to you.
That is the setting the package works in — a known nominal model, an unknown disturbance — and it
is why the page draws the drift instead of hiding it. Turning the switch off changes no number,
only what you can see while you choose. The preset layouts are built around the prediction, so they
are disabled while it is hidden.

The support is drawn for the law the table publishes: an exact box for uniform noise, weighted
bands for a block law, or a 3σ window for a Gaussian (a window, not a guarantee). A table that keeps
its law draws nothing. **`last rounds`** is the other see switch and also starts off: it draws the
wake the puck has walked this run, with a ghost of the ball at every place it has landed.

??? note "The five tables, and the one that keeps its law"
    The tables differ in their **noise** — what the dealer's ball does — and the point of having
    more than one is that they are not equally easy to bet well on:

    | table | what makes it interesting |
    |---|---|
    | **Long Table** | uniform noise on a square; the system from the predictability paper |
    | **Roulette** | the landing wobbles with a Gaussian, so no hand can be guaranteed anything |
    | **Whirlpool** | a contraction: each round is a smaller step than the last, so the action crowds into a shrinking region |
    | **Stacked Deck** | three bands, and 80% of landings fall in the middle one. Covering the noise support is still *safe* — and it is no longer the best you can do: aiming at the fat band pays more, and risks a round that pays nothing |
    | **House Rules** | the same idea with four unequal corners, and the table **does not publish its landing law** |

    On **House Rules** the page withholds everything derived from that law: no description of it,
    no safety verdict, no par line, and "Cover the noise support" is off (the layout would be built
    from the very support you are not being told). What you have instead is the payout, the score,
    and the landings you have already seen — which is the point of the table. It is a test of
    reading, not a secret in any cryptographic sense: the replays the page plays are the recorded
    draws, and anyone who opens the network tab can have them.

## 2. Place a stack

Tap the felt to put a stack where you think the puck lands. Drag a stack to move it, drag the felt
to turn the table, scroll or pinch to zoom — and with two fingers on the felt, slide the pair to
move the table itself: the tour card, a tall tower or a corner of the table you want to see can all
get out of the way (`Shift`+drag does the same on a computer, and `0` puts the view back). A tap on
a stack already there picks it up instead
of stacking a second one on the same spot; tap it again to put it down. With **snap to grid** on,
every stack lands on a payout square, so your squares tile the table and a landing is worth exactly
one stack; off, a grid three times finer lets two stacks stand a third of a cell apart.

Your stacks are a **pmf**: each one sits on a spot and carries a **weight**, drawn as its height and
written on it as a percentage. Only the percentages are ever compared — a weight's own size is a
free parameter, so the page never asks you to choose one — and the number written on each stack is
its share of the whole bet.

The **Your hand** row builds or recalls a whole layout, so you do not
have to place stacks one at a time: **Even spread** is the baseline the experiment below falls back
to, at a spacing that deliberately does not divide a box evenly (see the note in
`experiments/pmf_betting/presets.py` for why); **Repeat last hand** puts the previous round's
layout back, re-anchored on the new prediction; and **Cover the noise support** tiles the whole
support — that is the next step's story. They are switches rather than one-way doors: press the one
you just pressed, while the table is still holding exactly what it laid out, and you get the hand
you had before it. Touch anything in between — move a stack, change a height — and the button means
"lay it out again", because a hand that has been edited since is a different hand.

??? note "Select several stacks at once"
    **Select** (or Shift-click without it) gathers several stacks, and the slider, `−`/`+` and
    Remove then act on all of them; dragging any selected stack moves the group and refuses if a
    destination is taken. While Select is on, tap a stack to add or remove it, or drag on the empty
    felt to draw a box; a tap on empty felt clears the selection, and placing needs Select off.

??? note "A pmf is a legitimate prediction"
    The stacks are a **pmf**: a set of atoms with weights, whose support need not contain the true
    distribution's support. Under the ordinary logarithm score that prediction is `-inf` almost
    surely, and it would be reasonable to conclude that pmfs are useless here. Under a
    neighbourhood score they are perfectly usable, and they buy something a Gaussian cannot:

    - the predictive distribution can be **multimodal**, or a shape you drew by hand;
    - it can be **sparse**, which is what you want when you have a few strong beliefs rather than a
      broad one;
    - it can be **honest about a bounded support**, without a Gaussian's tails.

    ```python
    from ppsds.numerics.atom_pmf import AtomPmf
    from ppsds.probabilistic_predictor.pmf.atom import DriftShiftedAtomPmfPredictor
    from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

    bet = AtomPmf(atoms, weights)                       # offsets from the nominal next state
    predictor = DriftShiftedAtomPmfPredictor(bet, drift)

    predictor.update(x_observe, u_observe)
    predictor.predict()                                 # an AtomPmf
    EpsilonLogarithmScore(0.5)(predictor.predict(), x_next)
    ```

    > A pmf has no density, so the ordinary `LogarithmScore` returns `-inf` for it, and the `O(eps)`
    > approximation from the [the predictability paper](../../research/probabilistic-predictability.md)
    > page does not exist for it. That is not a gap in the library: it is the reason the
    > neighbourhood score exists.

## 3. Cover the noise support

A stack's payout square is its `epsilon`-neighbourhood: with `epsilon = 0.5` (the default) it is a
square one unit wide, and a landing inside it counts that stack's weight. **Cover the noise
support** places equal stacks so that their squares tile the noise's whole support. No landing can
then fall outside every stack, so no round can pay nothing. The badge above the table says what
that layout guarantees: `Safe · worst case 25.0%`.

That is the safe hand, and it is one end of a trade. On a table whose noise is **not uniform**, the
safe hand is safe and no longer best: it pays the same share wherever the ball lands, while par —
the best average any hand could reach — is higher, because a table with a fat band is a table with
less entropy. The gap between them is the money you leave on the table by not aiming, and the
scoreboard shows it. Trying to close it means betting on the fat band, which is the moment a round
can pay nothing at all. Both hands are one click away; the choice is the game. The badge's other
verdict is `Not safe · …% uncovered`: some landing is outside every square, and that landing pays
nothing and ends the run.

When the noise is **unbounded** — the Roulette table's Gaussian wobble — no layout can promise
anything, and the badge changes its claim rather than its colour: it reports the share of landings
the current stacks catch, and says it is a probability, not a guarantee.

## 4. Change a height

Tap a stack to open its toolbar, then drag the slider or use `−`/`+` (12% a press, or hold to
repeat). The height is its **weight**, and the percentage on the stack is its share of the whole
bet — the number the scoring rule uses.

A tower is its own weight: a stack twice as heavy is drawn twice as tall, and moving one stack never
moves another. Only the ratios between stacks are ever compared, so what a weight means is its
share. The slider is logarithmic, because a ratio wants a ratio scale, but it stops at eight times
a fresh stack: that is a comfort cap, not a claim about weights. It keeps the tallest tower readable
on a table and keeps a mis-touch from turning the felt into a very tall grid; the shares, which are
the numbers that matter, are unaffected. With a selection, the slider and `−`/`+` act on the whole
group.

## 5. Deal

Press Deal. The ball lands, and every stack within `epsilon` of it pays its share. What settles is
the **share** of the bet that landed:

```
paid    = the weight within eps of the landing, over all the weight on the table
score   = log(paid)
```

The page shows both, because they answer different questions. `paid` is the one to feel — "a
quarter of my bet came back". The score is the one that accumulates: the *average of the log
scores* is a rate, and `exp` of it is the growth a round — what the same numbers would compound at
if they were rolled over. The page shows that next to the plain average payout, because the two
differ and only one of them compounds.

## 6. Next round

Between rounds the table has two beats, in this order: **the hand travels to its new place, and
then the camera follows it.** Never at the same time — two things moving at once is two things the
eye believes neither of — but which goes second matters. The camera used to go first, over an empty
table, with the hand reappearing already in its new place, so the one thing that had actually
changed was the one thing you could not see move. After a round settles the page stops, so you can
read it and change the hand; **Next round** starts the next one.

The camera only ever **pans** during a run, it does not zoom: the frame is sized once for the whole
run, wide enough for the widest gap the puck and its prediction will ever open. A frame fitted to
the moment grew every round as the drift opened up, and every growth shrank the towers you had
built — on the Long Table the frame went from 3.0 to 8.4 over six rounds, so a hand that was worth
reading became a quarter of its size for a reason that had nothing to do with you. The cost is that
the first round is framed a little wider than it strictly needs; the wake behind the puck is drawn
as far as that frame holds and clipped past its edge.

**keep my layout** starts on, because a hand is meant to be a rule you can score twice. **auto next
round**, in the panel with it, moves on after a pause without touching your hand. **Auto deal** is
the play button beside Deal: press it and the page deals, settles and deals again on your last hand
until the run ends; the triangle becomes two bars, and pressing it again pauses. A run is a fixed
number of rounds — ten to fifty, ten by default. It ends when the rounds
run out, or the moment a round pays nothing: `log(0)` is `-inf`, and the run stops there rather
than being averaged into the next round. When it ends the page stops with it and shows what the run
amounted to: the average score, the best and worst rounds, and each hand you played with what it
paid. Auto deal stops too, and its button goes back to a triangle.

??? note "Take your bets with you"
    Two buttons take your play back with you, and they are different things:

    | button | what it gives you | what it is for |
    |---|---|---|
    | **Copy this round** | one hand: `{preset, eps, trajectory, step, atoms, weights}` | scoring *the hand* — the same atoms and weights mean the same thing on every round and every trajectory, so it can be scored against thousands of draws |
    | **Export this run** | every round you played, with the run it was played on | replaying *the run* — the script walks the same trajectory and reproduces the numbers you watched |

    Both buttons live in the page's own panel, under **Take your bets with you**, and the run-end
    summary carries the export as well: the end of a run is when you most likely want the file.
    Nothing is saved anywhere. The page has no server, writes no files, and keeps nothing between
    visits: what you want to keep, you export, and it lands in your own downloads folder.

## From the game to an experiment

The page scores one trajectory, one step at a time. Your browser is not the right tool for a
claim, so the package ships the same world as a script:

```bash
python experiments/pmf_betting/quickstart.py
python experiments/pmf_betting/run.py --preset shear --eps 0.5 --n_traj 10000
python experiments/pmf_betting/run.py --covering --preset shear --eps 0.5   # the safe hand
```

The systems are defined **once**, in `experiments/pmf_betting/presets.py`, and the page's data is
exported from it — so the game and the experiment are the same world rather than two demos that
happen to look alike. The files from **Copy this round** and **Export this run** are what the same
script replays:

```bash
python experiments/pmf_betting/run.py --bets my_bets.json    # one hand, over many trajectories
python experiments/pmf_betting/run.py --bets my_run.json     # one run, replayed round by round
```

## Why the page is checked automatically

**How to play** is a button in the page, not a paragraph at the start: it opens a guide with the
gestures (tap to place, tap a stack to select it — and tap it again to let it go — set its height on
the slider, drag a stack to move it, drag the felt to turn the table, two fingers to move and zoom,
scroll to zoom), the
**Select** mode that turns taps and a box into a group, the three things you will do most, what the
safe hand and par mean, the noise-support overlay, how a run ends (with **auto next round** told
apart from the **play button** beside Deal), and where your numbers go. The first visit also gets the
six-step tour, and the button at the bottom of the guide is the switch for it: it starts the tour
again, or turns it off for this visit.

The page recomputes the scoring rule — and the coverage arithmetic, and the safe layout, and the
table's own dynamics — in JavaScript. It has to, to answer your click instantly and without a
server. Two implementations of one rule will drift apart eventually, so the page carries a list of
`(system, radius, stacks) -> score` vectors computed by `ppsds`, recomputes them on load, and says
so in the corner if it disagrees.

The same vectors are asserted in the test suite, which also plays a whole round of the page's
logic headlessly: the two implementations must agree on every case, the safe layout must come out
exactly on the par line on a table whose noise it tiles, and hiding the prediction must not move a
single number.

## Next

- **The same thing, measured properly**: [the experiment behind this page](../../experiments/pmf_betting.md) — your
  exported hand, thousands of trajectories, and the ceiling to compare against
- **What a pmf-as-prediction buys you in a library**: [Research](../../research/index.md)
- **Put a predictor on a system of your own**: [Getting Started](../../tutorials/getting-started.md)
- **How the four examples connect**: [the map](../../tutorials/map.md)
