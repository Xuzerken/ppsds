/**
 * The 3-D table.
 *
 * Three.js, pinned and loaded from a CDN. That is a deliberate trade: the page stays free of
 * vendored third-party code (the repository has none, and the licence/attribution question would
 * arrive with it), and in exchange the 3-D view needs a network. Everything that matters -- the
 * scoring, the coverage, the dynamics, the self-check -- is in modules that do not import this
 * file, so a CDN that does not answer costs the page its depth, not its meaning: `game.js` falls
 * back to the flat table and says so.
 *
 * This module is *rendering only*. It receives a view model of geometry and flags and turns it
 * into pixels; every decision (what is safe, what paid, where the drift goes) was made upstream.
 * That is also why the test suite only syntax-checks it: behaviour here needs a GPU.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
import { feltOf } from './theme.js';
import { arrowJoints, landingFade } from './world.js';
import { wheelZoom } from './gestures.js';

/** Divisions in the felt grid. Together with the view span this fixes the grid's cell size, which
 *  is what lets the grid be snapped to whole cells instead of sliding under the puck. */
const GRID_DIVISIONS = 12;
/** An arrowhead's size, as a fraction of the view span. An open "→" reads smaller than the solid
 *  wedge it replaced, so it is drawn a little larger to carry the same weight. */
const ARROW_SIZE = 0.05;
/** The ball, as a fraction of the view span. It is the only solid thing on the felt, so its size is
 *  the table's one fixed point of scale -- and its ghosts are drawn at the same size: a ghost of
 *  the ball that is not the ball's size is a ghost of something else. */
const BALL_SIZE = 0.018;
/** The barbs of that "→": how long each stroke is, how thick, and how far each swings back from
 *  the direction of travel. Fractions of the arrowhead itself. */
const BARB_LENGTH = 0.62;
const BARB_WIDTH = 0.13;
const BARB_SPREAD = 0.62;

export async function createTableScene(container) {
  const canvas = document.createElement('canvas');
  canvas.className = 'gl';
  container.appendChild(canvas);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (error) {
    throw new Error(`WebGL is unavailable: ${error.message}`);
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
  const target = new THREE.Vector3(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x0a0d12, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.7);
  key.position.set(-2.4, 5.2, 2.0);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x9fc4ff, 0.5);
  rim.position.set(3.5, 2.5, -3.0);
  scene.add(rim);

  const felt = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshStandardMaterial({ color: 0x17563a, roughness: 0.96, metalness: 0.02 }),
  );
  felt.rotation.x = -Math.PI / 2;
  scene.add(felt);

  // The rim of the table, in a darker shade of whatever the felt is. Nearly black reads as a wall
  // standing behind the action rather than as the table's own edge.
  const edge = new THREE.Mesh(
    new THREE.RingGeometry(0.9, 1.0, 64),
    new THREE.MeshStandardMaterial({ color: 0x0d3324, roughness: 0.8, metalness: 0.15 }),
  );
  edge.rotation.x = -Math.PI / 2;
  edge.position.y = 0.002;
  scene.add(edge);

  const grid = new THREE.GridHelper(1, 12, 0x9fb6d6, 0x7f93ad);
  grid.material.transparent = true;
  grid.material.opacity = 0.16;
  scene.add(grid);

  // -- pools ---------------------------------------------------------------------------------
  // The view model rebuilds itself every frame; allocating a mesh per frame would be the one
  // place this page could stutter, so every repeated object is created once and re-used.

  const discGeometry = new THREE.CylinderGeometry(1, 1, 1, 28);
  const slabGeometry = new THREE.PlaneGeometry(1, 1);
  const outlineGeometry = new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1));
  const dotGeometry = new THREE.CircleGeometry(1, 24);
  const sphereGeometry = new THREE.SphereGeometry(1, 22, 16);
  const ringGeometry = new THREE.RingGeometry(0.86, 1, 48);

  const discMaterials = {
    bet: new THREE.MeshStandardMaterial({ color: 0x63b3ed, roughness: 0.35, metalness: 0.25 }),
    hit: new THREE.MeshStandardMaterial({ color: 0x4ade80, roughness: 0.3, metalness: 0.25, emissive: 0x0d3a1c }),
    miss: new THREE.MeshStandardMaterial({ color: 0xf87171, roughness: 0.55, metalness: 0.15 }),
    rim: new THREE.MeshStandardMaterial({ color: 0x0d1117, roughness: 0.6, metalness: 0.2 }),
  };
  // The hand that a new round did not carry over. It keeps the bet colour and fades, so what the
  // player sees is their layout being cleared rather than having been cleared.
  const leaveMaterial = new THREE.MeshStandardMaterial({
    color: 0x63b3ed, roughness: 0.45, metalness: 0.2, transparent: true, opacity: 0.45,
  });
  const epsMaterial = new THREE.MeshBasicMaterial({
    color: 0x7dd3fc, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
  });
  const epsEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0x7dd3fc, transparent: true, opacity: 0.7, depthWrite: false,
  });
  const predictionMaterial = new THREE.MeshBasicMaterial({
    color: 0xf5c451, side: THREE.DoubleSide, transparent: true, opacity: 0.95,
    // Drawn over everything the table puts down. The prediction is the thing the whole screen is
    // read against, and a mark you lose behind your own stacks is a mark you cannot bet by -- the
    // cost is that it also paints over the ball as it lands, which is the lesser loss.
    depthTest: false, depthWrite: false,
  });
  // One flat colour, no emissive: a ball that changes brightness as it travels reads as a ball
  // that is changing colour, and the ball carries no information to change *about*. It is the only
  // solid thing on the felt -- between throws it rests where it landed, and the system's state has
  // caught up to it (a landing *is* the next state), so there is no second sphere for the state.
  const ballMaterial = new THREE.MeshStandardMaterial({
    color: 0xe8985a, roughness: 0.45, metalness: 0.3,
  });
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: 0xe8ecf3, transparent: true, opacity: 0.3, depthWrite: false,
  });
  // One material per ghost, not one between them: each fades with its own age (`landingFade`).
  const landingMaterial = () => new THREE.MeshBasicMaterial({
    color: 0xe8985a, transparent: true, opacity: 0.42, depthWrite: false,
  });
  const historyMaterial = new THREE.LineBasicMaterial({
    color: 0xe8ecf3, transparent: true, opacity: 0.4, depthWrite: false,
  });
  const aimMaterial = new THREE.LineBasicMaterial({
    color: 0xf5c451, transparent: true, opacity: 0.55, depthWrite: false,
  });
  // The support is drawn under the hand and above the felt, in the prediction marker's gold at a
  // low opacity: it is the same reading as the marker, at the scale of the law. Per-slab materials
  // because a block law has one opacity per band, and the alpha is the weight -- that is the
  // information the bands carry. The Gaussian's window is a thin ring; WebGL has no dashed line
  // width worth having.
  const supportSlabMaterial = () => new THREE.MeshBasicMaterial({
    color: 0xf5c451, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide,
  });
  const supportEdgeMaterial = () => new THREE.LineBasicMaterial({
    color: 0xf5c451, transparent: true, opacity: 0.5, depthWrite: false,
    // The boundary is drawn over the hand: a support whose edge disappears under a stack answers
    // "where does the law reach?" with "not here", which is the one thing it must not say.
    depthTest: false,
  });

  /** A pool of identical objects: created once, then shown or hidden as the frame needs them. */
  const pool = (make, parent) => {
    const items = [];
    return (count) => {
      while (items.length < count) {
        const object = make();
        parent.add(object);
        items.push(object);
      }
      items.forEach((object, i) => { object.visible = i < count; });
      return items;
    };
  };

  const slabs = pool(() => new THREE.Mesh(slabGeometry, epsMaterial), scene);
  const outlines = pool(() => new THREE.LineSegments(outlineGeometry, epsEdgeMaterial), scene);
  const dots = pool(() => new THREE.Mesh(dotGeometry, ghostMaterial), scene);
  // A landing keeps the ball that made it: the ghosts are spheres the size of the ball, so the
  // trail of landings is legible as the trail of the thing the puck follows. It was a small dot,
  // which is neither the ball's size nor its shape, and read as a mark of some other kind.
  const landings = pool(() => new THREE.Mesh(sphereGeometry, landingMaterial()), scene);
  // The noise's support: one slab and one outline per box/band, and one thin ring for a Gaussian's
  // chosen window. Drawn before the ε squares so the hand always reads on top of the law.
  const supportSlabs = pool(() => new THREE.Mesh(slabGeometry, supportSlabMaterial()), scene);
  const supportEdges = pool(() => {
    const object = new THREE.LineSegments(outlineGeometry, supportEdgeMaterial());
    object.renderOrder = 11; // above the discs, below the prediction marker (12)
    return object;
  }, scene);
  const supportEllipse = new THREE.Mesh(
    new THREE.RingGeometry(0.968, 1, 96),
    new THREE.MeshBasicMaterial({ color: 0xf5c451, transparent: true, opacity: 0.55,
      depthWrite: false, depthTest: false, side: THREE.DoubleSide }),
  );
  supportEllipse.rotation.x = -Math.PI / 2;
  supportEllipse.renderOrder = 11;
  supportEllipse.visible = false;
  scene.add(supportEllipse);
  const discs = [];
  const leftDiscs = [];
  const ball = new THREE.Mesh(sphereGeometry, ballMaterial);
  // One ring per selected stack, from a pool: a group is selected as often as a single stack, and
  // a single mesh would silently mark only one of them.
  const selectionMaterial = new THREE.MeshBasicMaterial({
    color: 0xf6f9ff, transparent: true, opacity: 0.85,
    depthWrite: false, side: THREE.DoubleSide,
  });
  const selectionGeometry = new THREE.RingGeometry(0.82, 1, 40);
  const selectionRings = pool(() => {
    const ring = new THREE.Mesh(selectionGeometry, selectionMaterial);
    ring.rotation.x = -Math.PI / 2;
    return ring;
  }, scene);
  // The stack the cursor is over, at a smaller radius than the selection ring: one ring means "a
  // click lands here", two mean "you are holding this one".
  const hoverRing = new THREE.Mesh(
    new THREE.RingGeometry(0.84, 1, 40),
    new THREE.MeshBasicMaterial({ color: 0xe8ecf3, transparent: true, opacity: 0.3,
      depthWrite: false, side: THREE.DoubleSide }),
  );
  hoverRing.rotation.x = -Math.PI / 2;
  scene.add(hoverRing);
  const marker = new THREE.Group();
  const markerRing = new THREE.Mesh(ringGeometry, predictionMaterial);
  // Flat on the felt, like the selection and hover rings -- a ring standing on edge is half buried
  // in the baize and, because it spins, turns its back on the camera twice a turn.
  markerRing.rotation.x = -Math.PI / 2;
  const markerBarX = new THREE.Mesh(new THREE.BoxGeometry(1, 0.01, 0.012), predictionMaterial);
  const markerBarZ = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.01, 1), predictionMaterial);
  // A pin standing in the middle of the reticle. A flat ring seen from a shallow angle is an
  // ellipse a few pixels tall; the pin is what is still there when the table is nearly edge-on.
  const markerPin = new THREE.Mesh(
    new THREE.BoxGeometry(0.07, 0.6, 0.07).translate(0, 0.3, 0), predictionMaterial,
  );
  marker.add(markerRing, markerBarX, markerBarZ, markerPin);
  // Per mesh, not per group: three.js sorts the transparent pass by each object's own order, and a
  // group's is not passed down to its children.
  marker.traverse((child) => { child.renderOrder = 12; });
  scene.add(marker, ball);

  const line = (material) => {
    const object = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      material,
    );
    // The geometry is refilled in place every frame, and `setFromPoints` does not invalidate a
    // bounding sphere that has already been computed -- so the sphere stays the zero-radius one at
    // the origin and the line is culled the moment the table drifts away from it, which is exactly
    // when the wake matters most. Culling off, and the sphere recomputed in `drawLine`.
    object.frustumCulled = false;
    scene.add(object);
    return object;
  };
  const walked = line(historyMaterial); // where the puck has been
  const aim = line(aimMaterial); // where the drift says it is going
  // A "→" head: two strokes meeting at the tip, lying in the table's own plane so the heading is
  // one yaw. Not a filled wedge -- a triangle and its mirror read the same at a glance, and showing
  // which way the puck is going is the whole of what this mark is for. Boxes rather than a
  // `THREE.Line`, because WebGL ignores `linewidth`: a hairline head is thinner than the wake it
  // sits on.
  const barb = (side) => {
    const geometry = new THREE.BoxGeometry(BARB_LENGTH, 0.012, BARB_WIDTH);
    geometry.translate(-BARB_LENGTH / 2, 0, 0); // the tip lands on the joint
    geometry.rotateY(side * BARB_SPREAD);
    return geometry;
  };
  const barbGeometry = [barb(1), barb(-1)];
  const arrowHead = (material) => {
    const group = new THREE.Group();
    barbGeometry.forEach((geometry) => group.add(new THREE.Mesh(geometry, material)));
    return group;
  };
  const wakeArrows = pool(() => arrowHead(historyMaterial), scene);
  const aimArrows = pool(() => arrowHead(aimMaterial), scene);

  /** The pool a tower's discs come from, so the cleared hand and the live hand never share one. */
  function discPool(items) {
    return (count) => {
      while (items.length < count) {
        const disc = new THREE.Mesh(discGeometry, discMaterials.bet);
        items.push(disc);
        scene.add(disc);
      }
      items.forEach((disc, i) => { disc.visible = i < count; });
      return items;
    };
  }
  const stackDiscs = discPool(discs);
  const leavingDiscs = discPool(leftDiscs);

  // -- camera --------------------------------------------------------------------------------

  const HOME = { azimuth: -0.7, elevation: 0.85 };
  let azimuth = HOME.azimuth;
  let elevation = HOME.elevation;
  let zoomFactor = 1;
  /**
   * How far the player has dragged the table away from where the round put it, in world units on
   * the ground plane. It is added to the view's centre rather than replacing it, so a round change
   * still frames the hand -- the same way the tour card sitting on the table is the player's
   * problem and not the camera's.
   */
  let panX = 0;
  let panZ = 0;
  /** The world size of one CSS pixel at the current camera distance, for `panBy`. */
  let lastDistance = null;
  /** Where the camera is actually looking, which lags `view.centre` by design: the felt and the
   *  grid hang off this rather than off the target, so a round change does not teleport the baize
   *  out from under a hand that is still gliding across it. */
  const smoothed = new THREE.Vector3();
  /** The span the camera has eased to. It lags the view's, which is what turns "the frame changed"
   *  into "the camera moved". The span is decided per *run* (`world.runSpan`), so in practice it
   *  starts where it is going to stay and this easing only matters when the hand changes size. */
  let smoothedSpan = null;

  function placeCamera(view) {
    smoothedSpan = smoothedSpan === null
      ? view.span : smoothedSpan + (view.span - smoothedSpan) * 0.18;
    const distance = smoothedSpan * 2.9 * zoomFactor;
    lastDistance = distance;
    // The pan is clamped to the table's own size: a gesture that can put the baize out of reach
    // entirely is a gesture with no way back, and "there is nothing on screen" is not a state the
    // page can explain its way out of.
    const limit = 1.4 * smoothedSpan;
    panX = Math.max(-limit, Math.min(limit, panX));
    panZ = Math.max(-limit, Math.min(limit, panZ));
    const want = new THREE.Vector3(view.centre[0] + panX, 0, -view.centre[1] + panZ);
    target.lerp(want, 0.18);
    smoothed.copy(target);
    const horizontal = Math.cos(elevation) * distance;
    camera.position.set(
      target.x + Math.sin(azimuth) * horizontal,
      target.y + Math.sin(elevation) * distance,
      target.z + Math.cos(azimuth) * horizontal,
    );
    camera.lookAt(target);
    camera.updateMatrixWorld();
  }

  // -- drawing -------------------------------------------------------------------------------

  /**
   * The noise's support, in the noise frame (i.e. around the view's anchor).
   *
   * A block law's opacity is the band's weight, because that is the one thing a block law has to
   * say. A Gaussian has no support at all, so its ellipse is a chosen 3σ window and the caption
   * outside the canvas says so. The renderer does not decide whether the support is shown or which
   * table may show it -- `view.support` is already `null` in those cases.
   */
  function drawSupport(support) {
    if (!support) {
      supportSlabs(0);
      supportEdges(0);
      supportEllipse.visible = false;
      return;
    }
    const anchor = support.anchor;
    if (support.kind === 'ellipse') {
      supportSlabs(0);
      supportEdges(0);
      supportEllipse.visible = true;
      supportEllipse.position.set(
        anchor[0] + support.centre[0], 0.004, -(anchor[1] + support.centre[1]));
      supportEllipse.scale.set(support.rx, support.ry, 1);
      return;
    }
    supportEllipse.visible = false;
    const boxes = support.kind === 'box'
      ? [{ low: support.low, high: support.high, weight: 1 }]
      : support.bands;
    const slabs = supportSlabs(boxes.length);
    const edges = supportEdges(boxes.length);
    boxes.forEach((box, index) => {
      const x = anchor[0] + 0.5 * (box.low[0] + box.high[0]);
      const y = anchor[1] + 0.5 * (box.low[1] + box.high[1]);
      const width = box.high[0] - box.low[0];
      const depth = box.high[1] - box.low[1];
      slabs[index].position.set(x, 0.004, -y);
      slabs[index].scale.set(width, depth, 1);
      slabs[index].rotation.x = -Math.PI / 2;
      slabs[index].material.opacity = 0.06 + 0.14 * box.weight;
      edges[index].position.set(x, 0.005, -y);
      edges[index].scale.set(width, depth, 1);
      edges[index].rotation.x = -Math.PI / 2;
    });
  }

  function render(view) {
    const span = view.span;
    // The felt, the rim and the grid are furniture rather than geometry: they follow the camera's
    // own easing so they cannot lurch ahead of it.
    const stage = smoothedSpan === null ? span : smoothedSpan;
    placeCamera(view);

    const feltColour = feltOf(view.presetKey);
    felt.material.color.set(feltColour.top);
    // The ground travels with the action, exactly as the rim and the grid do. It used to be nailed
    // to the origin while everything else followed the camera target, so on a drifting table the
    // puck -- and its wake -- walked off the baize onto the background: a table that stops being a
    // table. Size is generous because the fix should not depend on the camera being tame.
    felt.scale.set(stage * 8, stage * 8, 1);
    felt.position.set(smoothed.x, 0, smoothed.z);
    edge.position.set(smoothed.x, 0.004, smoothed.z);
    edge.scale.set(stage * 2.5, stage * 2.5, 1);
    edge.material.color.copy(felt.material.color).multiplyScalar(0.55);
    // The grid is snapped to whole cells of its own spacing. Without the snap it slides under the
    // puck whenever the view centre moves -- a grid that crawls reads as the table moving, which
    // is exactly the wrong thing to suggest on a page about where something is going.
    const cell = (stage * 2.6) / GRID_DIVISIONS;
    grid.scale.set(stage * 2.6, 1, stage * 2.6);
    grid.position.set(
      Math.round(smoothed.x / cell) * cell, 0.001, Math.round(smoothed.z / cell) * cell);

    // The hand is always solid now. It used to fade out while the camera moved, because the camera
    // moved first; the other way round, the hand is the thing that travels and there is nothing to
    // hide. The one hand that still fades is the one the round did not keep, and it carries its own
    // alpha per stack (`view.leaving`).
    for (const material of [discMaterials.bet, discMaterials.hit, discMaterials.miss]) {
      material.opacity = 1;
    }
    epsMaterial.opacity = 0.22;
    epsEdgeMaterial.opacity = 0.7;

    drawSupport(view.support);

    // The neighbourhoods, under the stacks: where two squares meet, the stacks above them share a
    // landing. The outline is a square, not a ring -- the payout region is a box in the infinity
    // norm, and drawing a circle around it would teach the wrong shape. The squares are allowed to
    // overlap; the discs are not, which is why the discs are sized from the footprint and these
    // are sized from epsilon.
    const squares = slabs(view.stacks.length);
    const borders = outlines(view.stacks.length);
    if (view.eps > 0) {
      view.stacks.forEach((stack, index) => {
        const square = squares[index];
        square.position.set(stack.at[0], 0.006, -stack.at[1]);
        square.scale.set(2 * view.eps, 2 * view.eps, 1);
        square.rotation.x = -Math.PI / 2;
        const border = borders[index];
        border.position.set(stack.at[0], 0.008, -stack.at[1]);
        border.scale.set(2 * view.eps, 2 * view.eps, 1);
        border.rotation.x = -Math.PI / 2;
      });
    }

    // The towers. A stack's height *is* its weight: `full` whole discs and one squashed to the
    // fraction the weight asks for, which is the whole reason a slider drag grows a pile smoothly
    // instead of jumping a disc at a time. Every disc is the same radius -- the weight is the
    // height and the percentage label says what it works out to, so the width has nothing to say
    // and says nothing.
    const radius = view.radius;
    const wanted = view.stacks.reduce(
      (sum, stack) => sum + stack.plan.full + (stack.plan.partial > 0 ? 1 : 0), 0);
    const stacks = stackDiscs(wanted);
    let cursor = 0;
    const selectedSpots = [];
    let hoverSpot = null;
    view.stacks.forEach((stack) => {
      const material = discMaterials[stack.state];
      // The tap's acknowledgement: the stack the hand just touched swells for a moment.
      const scale = radius * (1 + 0.14 * stack.pressT * (stack.selected ? 1 : 0));
      const lift = stack.state === 'hit' ? 0.06 * span * view.settleT : 0;
      if (stack.selected) selectedSpots.push({ at: stack.at, radius: scale, lift });
      if (stack.hover) hoverSpot = { at: stack.at, radius: scale, lift };
      const { full, partial, thickness, step } = stack.plan;
      for (let i = 0; i < full; i += 1, cursor += 1) {
        const disc = stacks[cursor];
        disc.material = material;
        disc.position.set(stack.at[0], i * step + thickness / 2 + lift, -stack.at[1]);
        disc.scale.set(scale, thickness, scale);
      }
      if (partial > 0) {
        const disc = stacks[cursor];
        cursor += 1;
        const height = thickness * partial;
        disc.material = material;
        disc.position.set(stack.at[0], full * step + height / 2 + lift, -stack.at[1]);
        disc.scale.set(scale, height, scale);
      }
    });

    // The hand this round did not keep, fading where it stood.
    const cleared = view.leaving || [];
    leaveMaterial.opacity = cleared.length === 0 ? 0 : 0.45 * cleared[0].alpha;
    const left = leavingDiscs(cleared.reduce(
      (sum, stack) => sum + 1 + (stack.weight > 1 ? 1 : 0), 0));
    let leftCursor = 0;
    cleared.forEach((stack) => {
      const count = 1 + (stack.weight > 1 ? 1 : 0);
      for (let i = 0; i < count; i += 1, leftCursor += 1) {
        const disc = left[leftCursor];
        disc.material = leaveMaterial;
        disc.position.set(stack.at[0], i * radius * 0.6, -stack.at[1]);
        disc.scale.set(radius, radius * 0.34, radius);
      }
    });

    // The rings that say "these are the stacks the toolbar is about", and the dimmer one that says
    // where a click would land. Selecting on a touch screen gives no other feedback at all, so
    // without them the toolbar appears out of nowhere -- and a group with one ring would look like
    // a single stack.
    selectionMaterial.opacity = 0.85;
    const selectedRings = selectionRings(selectedSpots.length);
    selectedSpots.forEach((spot, index) => {
      selectedRings[index].position.set(spot.at[0], 0.012 + spot.lift, -spot.at[1]);
      selectedRings[index].scale.setScalar(spot.radius * 1.9);
    });
    hoverRing.material.opacity = 0.3;
    hoverRing.visible = hoverSpot !== null;
    if (hoverSpot) {
      hoverRing.position.set(hoverSpot.at[0], 0.011 + hoverSpot.lift, -hoverSpot.at[1]);
      hoverRing.scale.setScalar(hoverSpot.radius * 1.5);
    }

    // The one solid thing on the felt, at every moment: where it landed, or in the air on the way.
    ball.visible = Boolean(view.ball);
    if (view.ball) {
      ball.position.set(view.ball.at[0], span * BALL_SIZE + view.ball.height * 0.55, -view.ball.at[1]);
      ball.scale.setScalar(span * BALL_SIZE);
    }

    marker.visible = Boolean(view.predicted);
    if (view.predicted) {
      marker.position.set(view.predicted[0], 0.01, -view.predicted[1]);
      marker.scale.setScalar(span * 0.05);
      marker.rotation.y += 0.01;
    }

    drawLine(walked, view.history.show ? view.history.paths[0] : null, span, 0.012, wakeArrows);
    drawLine(aim, view.nominalPath, span, 0.014, aimArrows);
    drawHistory(view, span, dots, landings);

    // The one line the whole scene hangs on. Everything above only *positions* objects; without
    // this call three.js draws nothing and the stage is a transparent canvas over a dark
    // background -- which looks exactly like a page that failed to load. Nothing in the test suite
    // can run this (it needs a GPU), so `test_game_page.py` asserts this call exists as a canary.
    renderer.render(scene, camera);
  }

  /** One polyline on the felt with an arrowhead at each joint, or nothing when there is no path. */
  function drawLine(object, points, span, height, arrowPool) {
    if (!points || points.length < 2) {
      object.visible = false;
      arrowPool(0);
      return;
    }
    object.visible = true;
    object.geometry.setFromPoints(
      points.map((point) => new THREE.Vector3(point[0], span * height, -point[1])));
    object.geometry.computeBoundingSphere();
    const joints = arrowJoints(points, span);
    const heads = arrowPool(joints.length);
    joints.forEach((joint, index) => {
      const head = heads[index];
      head.position.set(joint.at[0], span * height, -joint.at[1]);
      head.scale.setScalar(span * ARROW_SIZE);
      // The barbs are built along +x and the world's z is the state's -y, so the heading is one
      // yaw about the vertical: `dir = (dx, dy)` points down `(dx, 0, -dy)`.
      head.rotation.y = Math.atan2(joint.dir[1], joint.dir[0]);
    });
  }

  function drawHistory(view, span, ghostPool, landingPool) {
    // The draws so far, re-centred on this round's prediction: the noise's habits, which is what a
    // player is actually betting on. With the prediction hidden there is no centre to re-centre on
    // -- and the raw offsets are zero-mean, so they would pile up at the origin, which is nowhere
    // near where anything is happening. `anchor` is null exactly then, and the whole hint goes.
    // (The flat table has always done it this way; the two renderers must agree.)
    const anchor = view.history.show ? view.history.anchor : null;
    if (!anchor) {
      ghostPool(0);
    } else {
      const ghosts = ghostPool(view.history.draws.length);
      view.history.draws.forEach((draw, index) => {
        ghosts[index].position.set(anchor[0] + draw[0], span * 0.008, -(anchor[1] + draw[1]));
        ghosts[index].scale.setScalar(span * 0.012);
        ghosts[index].rotation.x = -Math.PI / 2;
      });
    }
    const marks = landingPool(view.history.show ? view.history.landings.length : 0);
    if (!view.history.show) return;
    view.history.landings.forEach((point, index) => {
      const ghost = marks[index];
      // Where the ball came to rest, at the ball's own size and height, so the newest ghost sits
      // under the puck -- which is what a landing is: the state the puck moved to.
      ghost.position.set(point[0], span * BALL_SIZE, -point[1]);
      ghost.scale.setScalar(span * BALL_SIZE);
      ghost.material.opacity = landingFade(index, view.history.landings.length);
    });
  }

  // -- the surface the shell talks to --------------------------------------------------------

  function resize() {
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const raycaster = new THREE.Raycaster();
  const hitPoint = new THREE.Vector3();
  const pointer = new THREE.Vector2();

  return {
    render,
    resize,
    pick(offsetX, offsetY) {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      pointer.set((offsetX / width) * 2 - 1, -(offsetY / height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      // A pointer outside the table still lands somewhere on the felt's plane, which is what a
      // player expects when they drag a stack towards the edge of the screen.
      return raycaster.ray.intersectPlane(plane, hitPoint) === null
        ? null
        : [hitPoint.x, -hitPoint.z];
    },
    zoom(deltaY) {
      this.zoomBy(wheelZoom(deltaY));
    },
    /** A pinch hands in a ratio: below one means closer. */
    zoomBy(factor) {
      zoomFactor = Math.min(2.5, Math.max(0.45, zoomFactor * factor));
    },
    orbit(dx, dy) {
      azimuth -= dx * 0.006;
      elevation = Math.min(1.3, Math.max(0.22, elevation + dy * 0.005));
    },
    /**
     * Slide the table under a moving pair of fingers.
     *
     * The camera's own right and up axes are what "under the fingers" means: a pixel is worth
     * `2·d·tan(fov/2) / height` world units, so the motion scales with the zoom the player chose.
     * Both axes are then flattened onto the ground plane, because the target never leaves y = 0 --
     * panning up and down the screen would otherwise tilt the table instead of moving it.
     */
    panBy(dx, dy) {
      const distance = lastDistance === null
        ? (smoothedSpan === null ? 1 : smoothedSpan) * 2.9 * zoomFactor : lastDistance;
      const height = Math.max(1, container.clientHeight);
      const unit = (2 * distance * Math.tan((camera.fov * Math.PI / 180) / 2)) / height;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      panX += (-dx * right.x + dy * up.x) * unit;
      panZ += (-dx * right.z + dy * up.z) * unit;
    },
    /** Back to the frame the rounds are written for: centred, unzoomed, at the home angle. */
    resetView() {
      panX = 0;
      panZ = 0;
      zoomFactor = 1;
      azimuth = HOME.azimuth;
      elevation = HOME.elevation;
    },
    /** Where a state-space point lands on the screen, in CSS pixels inside the container.
     *  The shell positions the stack toolbar and the share labels with this, so both renderers
     *  have to answer it the same way. */
    project(point) {
      return this.projectAt(point, 0);
    },
    /** The same, for a point `lift` above the felt: a label sits on top of a tower, and a tower is
     *  as tall as its weight. */
    projectAt(point, lift = 0) {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      const vector = new THREE.Vector3(point[0], lift, -point[1]).project(camera);
      return {
        x: (vector.x + 1) / 2 * width,
        y: (1 - vector.y) / 2 * height,
        visible: vector.z < 1,
      };
    },
  };
}
