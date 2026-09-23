// ---------------------------------------------------------------------------
// Rendu sur le canvas : régions colorées, frontière courbe et points.
// ---------------------------------------------------------------------------

import type { NodeRef } from "./diagram";
import { contourSegments, decisionLineEndpoints, sampleField, type ScalarField } from "./geometry";
import type { MLP } from "./mlp";
import type { Point } from "./state";

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

export function pointRadius(): number {
  return POINT_RADIUS;
}

/** Petit canvas hors écran : une case de la grille = un pixel. */
let regionCanvas: HTMLCanvasElement | null = null;

/** Redessine entièrement la scène. */
export function draw(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  points: Point[],
  net: MLP,
  showHiddenLines: boolean,
  highlighted: NodeRef | null,
): void {
  ctx.clearRect(0, 0, width, height);

  // On mesure la sortie p du réseau une fois pour toutes sur la grille :
  // elle sert à la fois à colorer les régions et à tracer la frontière.
  const step = gridStep(net, width, height);
  const field = sampleField((x, y) => net.forward(x, y), width, height, step);

  drawRegions(ctx, field);
  if (showHiddenLines) drawHiddenLines(ctx, width, height, net);
  if (highlighted !== null) drawHighlightedNeuron(ctx, width, height, net, highlighted, step);
  drawDecisionCurve(ctx, field);
  drawPoints(ctx, points);
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
 * Trace en pointillés la droite de chaque neurone de la 1re couche cachée.
 * Toute la suite du réseau est construite en combinant ces droites
 * « adoucies ».
 */
function drawHiddenLines(ctx: CanvasRenderingContext2D, width: number, height: number, net: MLP): void {
  ctx.save();
  ctx.strokeStyle = "rgba(60, 60, 90, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  for (let j = 0; j < net.sizes[1]; j++) strokeHiddenLine(ctx, width, height, net, j);
  ctx.restore();
}

/**
 * Met en évidence la frontière du neurone caché survolé dans le schéma : là
 * où sa réponse change de signe (h = 0).
 * - 1re couche cachée : c'est une droite, tracée en pointillés.
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
    ctx.setLineDash([10, 6]);
    strokeHiddenLine(ctx, width, height, net, index);
  } else {
    const response = (x: number, y: number) => {
      net.forward(x, y);
      return net.activations[layer][index];
    };
    strokeContour(ctx, sampleField(response, width, height, step), 0);
  }
  ctx.restore();
}

/** Trace la droite du neurone j de la 1re couche cachée (le style est choisi par l'appelant). */
function strokeHiddenLine(ctx: CanvasRenderingContext2D, width: number, height: number, net: MLP, j: number): void {
  const [wx, wy] = net.layers[0].w[j];
  const line = decisionLineEndpoints(wx, wy, net.layers[0].b[j], width, height);
  if (!line) return;
  const [a, c] = line;
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

function drawPoints(ctx: CanvasRenderingContext2D, points: Point[]): void {
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.px, p.py, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#101018" : "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#101018";
    ctx.stroke();
  }
}
