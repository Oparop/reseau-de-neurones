// ---------------------------------------------------------------------------
// Rendu sur le canvas : régions colorées, frontière courbe et points.
// ---------------------------------------------------------------------------

import type { NodeRef } from "./diagram";
import { isLinear } from "./features";
import { contourSegments, decisionLineEndpoints, sampleField, type ScalarField } from "./geometry";
import type { MLP } from "./mlp";
import { isTestPoint, type Point } from "./state";

/** Rayon des points tels qu'on les voit à l'écran, en pixels CSS. */
const POINT_RADIUS = 7;

/**
 * Écart (en pixels) entre deux mesures du réseau pour tracer la frontière.
 * Plus petit = courbe plus fine, mais plus de calculs à chaque image.
 */
const MIN_GRID_STEP = 4;
const MAX_GRID_STEP = 10;

/**
 * Nombre de multiplications qu'on s'autorise, à chaque image, pour mesurer le
 * réseau sur la grille. Chaque mesure coûte environ une multiplication par
 * paramètre : un grand réseau (4 couches de 32 neurones = plus de 3000
 * paramètres) est trop coûteux à mesurer tous les 4 pixels.
 */
const FIELD_BUDGET = 1e7;

/** Écart de grille adapté à la taille du réseau, pour garder une animation fluide. */
function gridStep(net: MLP, width: number, height: number): number {
  const measures = FIELD_BUDGET / net.parameterCount; // mesures permises par image
  const step = Math.round(Math.sqrt((width * height) / measures)); // surface par mesure -> écart
  return Math.min(MAX_GRID_STEP, Math.max(MIN_GRID_STEP, step));
}

/**
 * Rayon d'un point, en pixels du canvas.
 *
 * @param scale pixels du canvas par pixel affiché : 1 quand le canvas est
 *              affiché à sa taille réelle, plus quand il est rétréci (sur un
 *              téléphone par exemple). Les points gardent ainsi la même
 *              taille à l'écran.
 */
export function pointRadius(scale: number): number {
  return POINT_RADIUS * Math.max(1, scale);
}

/** Petit canvas hors écran : une case de la grille = un pixel. */
let regionCanvas: HTMLCanvasElement | null = null;

/**
 * Redessine entièrement la scène.
 *
 * @param scale     pixels du canvas par pixel affiché (voir `pointRadius`)
 * @param testRatio part des points mis de côté pour le test (voir state.ts)
 */
export function draw(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  points: Point[],
  net: MLP,
  showHiddenLines: boolean,
  highlighted: NodeRef | null,
  scale: number,
  testRatio: number,
): void {
  ctx.clearRect(0, 0, width, height);

  // On mesure la sortie p du réseau une fois pour toutes sur la grille :
  // elle sert à la fois à colorer les régions et à tracer la frontière.
  const step = gridStep(net, width, height);
  const field = sampleField((x, y) => net.forward(x, y), width, height, step);

  drawRegions(ctx, field);
  if (showHiddenLines) drawHiddenLines(ctx, width, height, net, step);
  if (highlighted !== null) drawHighlightedNeuron(ctx, width, height, net, highlighted, step);
  drawDecisionCurve(ctx, field);
  drawPoints(ctx, points, scale, testRatio);
}

/**
 * Colore chaque zone selon la classe prédite (rouge = « noir », bleu =
 * « blanc »). L'intensité traduit la confiance du réseau : la couleur est
 * franche là où p est proche de 0 ou 1, et s'efface près de la frontière
 * (p ≈ 0.5), là où le réseau hésite.
 */
function drawRegions(ctx: CanvasRenderingContext2D, field: ScalarField): void {
  const { cols, rows, step, values } = field;

  regionCanvas ??= document.createElement("canvas");
  regionCanvas.width = cols;
  regionCanvas.height = rows;
  const off = regionCanvas.getContext("2d");
  if (!off) return;

  const img = off.createImageData(cols, rows);
  for (let i = 0; i < values.length; i++) {
    const p = values[i];
    const confidence = Math.abs(p - 0.5) * 2; // 0 sur la frontière, 1 si certain
    const [r, g, b] = p >= 0.5 ? [255, 130, 130] : [120, 160, 255];
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = Math.round(confidence * 0.3 * 255);
  }
  off.putImageData(img, 0, 0);

  // On agrandit la petite image à la taille du canvas ; le lissage du
  // navigateur fond les cases entre elles. Le décalage d'une demi-case centre
  // chaque pixel agrandi sur le point de la grille qu'il représente.
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(regionCanvas, -step / 2, -step / 2, cols * step, rows * step);
}

/**
 * Trace la frontière de chaque neurone de la 1re couche cachée. Toute la
 * suite du réseau est construite en combinant ces frontières « adoucies ».
 */
function drawHiddenLines(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  net: MLP,
  step: number,
): void {
  ctx.save();
  ctx.strokeStyle = "rgba(60, 60, 90, 0.45)";
  ctx.lineWidth = 1.5;
  for (let j = 0; j < net.sizes[1]; j++) strokeFirstLayerBoundary(ctx, width, height, net, j, step, [6, 5]);
  ctx.restore();
}

/**
 * Met en évidence la frontière du neurone caché survolé dans le schéma : là
 * où sa somme pondérée s'annule (quelle que soit l'activation, c'est là que
 * le neurone bascule d'un côté à l'autre).
 * - 1re couche cachée : une droite (en pointillés), ou une courbe si le
 *   réseau reçoit x², y² ou x·y.
 * - couches suivantes : c'est déjà une courbe, qu'on trace comme la frontière
 *   de décision (mesure sur la grille puis marching squares).
 */
function drawHighlightedNeuron(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  net: MLP,
  node: NodeRef,
  step: number,
): void {
  const { layer, index } = node;
  const isHidden = layer >= 1 && layer < net.sizes.length - 1;
  if (!isHidden || index >= net.sizes[layer]) return;

  ctx.save();
  ctx.strokeStyle = "rgba(30, 30, 50, 0.9)";
  ctx.lineWidth = 3;
  if (layer === 1) {
    strokeFirstLayerBoundary(ctx, width, height, net, index, step, [10, 6]);
  } else {
    const sum = (x: number, y: number) => {
      net.forward(x, y);
      return net.sums[layer][index];
    };
    strokeContour(ctx, sampleField(sum, width, height, step), 0);
  }
  ctx.restore();
}

/**
 * Trace la frontière du neurone j de la 1re couche cachée (la couleur et
 * l'épaisseur sont choisies par l'appelant).
 * - Entrées x et y seulement : w_x·x + w_y·y + b = 0 est une droite, tracée
 *   d'un bord à l'autre, en pointillés (`dash`).
 * - Avec x², y² ou x·y : c'est une courbe (ellipse, hyperbole...), tracée
 *   comme la frontière de décision.
 */
function strokeFirstLayerBoundary(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  net: MLP,
  j: number,
  step: number,
  dash: number[],
): void {
  const { w, b } = net.layers[0];
  if (!isLinear(net.features)) {
    ctx.setLineDash([]);
    strokeContour(ctx, sampleField((x, y) => net.firstLayerSum(j, x, y), width, height, step), 0);
    return;
  }

  // Poids de x et de y (0 si l'entrée n'est pas donnée au réseau).
  const ix = net.features.indexOf("x");
  const iy = net.features.indexOf("y");
  const line = decisionLineEndpoints(ix >= 0 ? w[j][ix] : 0, iy >= 0 ? w[j][iy] : 0, b[j], width, height);
  if (!line) return;
  const [a, c] = line;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(c.x, c.y);
  ctx.stroke();
}

/** La frontière de décision : la courbe où p = 0.5. */
function drawDecisionCurve(ctx: CanvasRenderingContext2D, field: ScalarField): void {
  ctx.strokeStyle = "#e5484d";
  ctx.lineWidth = 3;
  strokeContour(ctx, field, 0.5);
}

/** Trace la courbe de niveau  field = level  (le style est choisi par l'appelant). */
function strokeContour(ctx: CanvasRenderingContext2D, field: ScalarField, level: number): void {
  ctx.lineCap = "round"; // bouts arrondis : les petits segments se raccordent sans trou
  ctx.beginPath();
  for (const { a, b } of contourSegments(field, level)) {
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
}

/**
 * Les points : ronds pour l'entraînement, carrés pour le test (le réseau ne
 * les voit jamais pendant l'apprentissage).
 */
function drawPoints(ctx: CanvasRenderingContext2D, points: Point[], scale: number, testRatio: number): void {
  const r = pointRadius(scale);
  for (const p of points) {
    ctx.beginPath();
    if (isTestPoint(p, testRatio)) {
      const half = r * 0.9; // à rayon égal, un carré paraît plus gros qu'un rond
      ctx.rect(p.px - half, p.py - half, half * 2, half * 2);
    } else {
      ctx.arc(p.px, p.py, r, 0, Math.PI * 2);
    }
    ctx.fillStyle = p.label === 1 ? "#101018" : "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2 * Math.max(1, scale);
    ctx.strokeStyle = "#101018";
    ctx.stroke();
  }
}
