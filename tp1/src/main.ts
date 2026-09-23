// ---------------------------------------------------------------------------
// Point d'entrée : relie l'état, le réseau de neurones, le rendu et l'interface.
// ---------------------------------------------------------------------------

import { describeNode, drawDiagram, hitTestDiagram } from "./diagram";
import { toNormalized } from "./geometry";
import { MLP, type TrainingSample } from "./mlp";
import { draw, pointRadius } from "./renderer";
import { hiddenSizes, makeDemoPoints, state } from "./state";
import { setupUi } from "./ui";

const canvas = document.getElementById("board") as HTMLCanvasElement;
const ctx = canvas.getContext("2d");
if (!ctx) throw new Error("Contexte 2D indisponible");

const W = canvas.width;
const H = canvas.height;

// Second canvas : le schéma du réseau (les neurones et leurs poids).
const diagram = document.getElementById("network") as HTMLCanvasElement;
const dctx = diagram.getContext("2d");
if (!dctx) throw new Error("Contexte 2D indisponible");

// Taille logique en pixels CSS. La résolution réelle est multipliée par la
// densité de l'écran pour que les textes du schéma restent nets. La taille
// affichée, elle, est laissée au CSS : le schéma rétrécit sur petit écran.
const DW = diagram.width;
const DH = diagram.height;
const dpr = window.devicePixelRatio || 1;
diagram.width = Math.round(DW * dpr);
diagram.height = Math.round(DH * dpr);
dctx.scale(dpr, dpr);

let net = new MLP(hiddenSizes(state));

/** Convertit les points (pixels) en exemples d'entraînement (normalisés). */
function samples(): TrainingSample[] {
  return state.points.map((p) => {
    const n = toNormalized(p.px, p.py, W, H);
    return { x: n.x, y: n.y, label: p.label };
  });
}

const ui = setupUi({
  onToggleClass() {
    state.activeLabel = state.activeLabel === 1 ? 0 : 1;
    ui.refreshActiveClass();
  },
  onDemo() {
    state.points = makeDemoPoints(state.demo, W, H);
    net.reset();
  },
  onReset() {
    state.points = [];
    net.reset();
  },
  onToggleTraining() {
    state.training = !state.training;
    ui.refreshTraining();
  },
  onStep() {
    // Mode pas-à-pas : une seule itération (une passe batch) par clic.
    state.training = false;
    ui.refreshTraining();
    net.trainEpoch(samples(), state.learningRate);
  },
  onNewWeights() {
    // Mêmes points, nouveau tirage des poids : le réseau repart de zéro
    // (utile s'il reste coincé dans une mauvaise solution).
    net.reset();
  },
  onArchitectureChange() {
    // Changer le nombre de couches ou de neurones change l'architecture : on
    // reconstruit un réseau neuf.
    net = new MLP(hiddenSizes(state));
    state.hoveredNode = null;
  },
});

// --- Interaction : ajouter ou supprimer un point -----------------------------
// Les « pointer events » couvrent à la fois la souris et le doigt (écran
// tactile).

/**
 * Au doigt, on vise moins précisément qu'à la souris : on élargit de ce
 * nombre de pixels (à l'écran) la zone où toucher un point le supprime.
 */
const TOUCH_SLOP = 8;

/** Type du dernier pointeur posé sur le plan : "mouse", "touch" ou "pen". */
let lastPointerType = "mouse";

/** Position du pointeur (souris ou doigt) en pixels du canvas. */
function canvasPixel(target: HTMLCanvasElement, event: MouseEvent, width: number, height: number) {
  const rect = target.getBoundingClientRect();
  // Le canvas peut être affiché à une taille différente de sa résolution.
  return {
    px: ((event.clientX - rect.left) / rect.width) * width,
    py: ((event.clientY - rect.top) / rect.height) * height,
  };
}

/**
 * Pixels du canvas par pixel affiché : 1 quand le plan est affiché à sa
 * taille réelle, plus quand il est rétréci (sur un téléphone par exemple).
 */
function displayScale(): number {
  return W / canvas.getBoundingClientRect().width;
}

// On ajoute ou supprime au « click » plutôt qu'au contact : au doigt, un
// glissement fait défiler la page sans poser de point.
canvas.addEventListener("click", (event) => {
  const { px, py } = canvasPixel(canvas, event, W, H);

  const hitIndex = findPointAt(px, py);
  if (hitIndex >= 0) {
    state.points.splice(hitIndex, 1); // clic sur un point existant -> suppression
  } else {
    state.points.push({ px, py, label: state.activeLabel }); // sinon -> ajout
  }
});

// Survol du plan (ou toucher au doigt) : on suit ce point à travers le réseau.
function trackProbe(event: PointerEvent): void {
  const { px, py } = canvasPixel(canvas, event, W, H);
  state.probe = toNormalized(px, py, W, H);
}
canvas.addEventListener("pointermove", trackProbe);
canvas.addEventListener("pointerdown", (event) => {
  lastPointerType = event.pointerType;
  trackProbe(event);
});
canvas.addEventListener("pointerleave", (event) => {
  // Au doigt, le pointeur « quitte » le canvas dès qu'on lève le doigt : on
  // garde alors le dernier point touché.
  if (event.pointerType === "mouse") state.probe = null;
});

// Survol du schéma (ou toucher au doigt) : on repère le neurone visé.
function trackNode(event: PointerEvent): void {
  const { px, py } = canvasPixel(diagram, event, DW, DH);
  state.hoveredNode = hitTestDiagram(DW, DH, net.sizes, px, py);
}
diagram.addEventListener("pointermove", trackNode);
diagram.addEventListener("pointerdown", trackNode);
diagram.addEventListener("pointerleave", (event) => {
  if (event.pointerType === "mouse") state.hoveredNode = null;
});

/** Renvoie l'index d'un point sous le pointeur, ou -1. */
function findPointAt(px: number, py: number): number {
  const scale = displayScale();
  const slop = lastPointerType === "touch" ? TOUCH_SLOP * scale : 0;
  const r = pointRadius(scale) + slop;
  for (let i = state.points.length - 1; i >= 0; i--) {
    const p = state.points[i]!;
    if (Math.hypot(p.px - px, p.py - py) <= r) return i;
  }
  return -1;
}

// --- Boucle d'animation -----------------------------------------------------

function step(): void {
  const data = samples();

  if (state.training && data.length > 0) {
    // Chaque frame applique plusieurs itérations batch (réglé par le slider) :
    // à chaque itération on parcourt tous les points puis on met à jour tous
    // les poids une seule fois. On voit ainsi la courbe se déformer.
    for (let i = 0; i < state.speed; i++) {
      net.trainEpoch(data, state.learningRate);
    }
  }

  ui.updateReadouts(net, net.evaluate(data));

  // Un neurone caché survolé dans le schéma : on surligne sa frontière sur le plan.
  const hovered = state.hoveredNode;
  draw(ctx!, W, H, state.points, net, state.showHiddenLines, hovered, displayScale());

  drawDiagram(dctx!, DW, DH, net, state.probe, hovered);
  ui.updateNeuronInfo(describeNode(net, hovered));

  requestAnimationFrame(step);
}

// Démarre avec le jeu de démonstration en place.
state.points = makeDemoPoints(state.demo, W, H);
requestAnimationFrame(step);
