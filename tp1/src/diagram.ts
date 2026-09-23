// ---------------------------------------------------------------------------
// Schéma du réseau : les neurones dessinés comme des boules reliées par
// leurs poids, une colonne par couche.
//
//      entrées        couches cachées          sortie
//
//                    (h1) ──── (h1)
//        (x) ─────── (h2) ──── (h2) ─────────── (p)
//        (y) ─────── ...  ──── ...
//                    (hH) ──── (hH)
//
// - Chaque trait est un POIDS : rouge s'il est positif, bleu s'il est
//   négatif, d'autant plus épais et opaque qu'il est grand en valeur absolue.
// - Chaque boule contient une mini-carte du plan : la réponse du neurone en
//   chaque point (rouge = positive, bleu = négative, blanc = proche de 0).
//   On y voit que chaque neurone de la 1re couche coupe le plan en deux par
//   une droite adoucie, que ceux des couches suivantes dessinent déjà des
//   courbes, et que la sortie assemble le tout en la frontière finale.
// - Quand la souris survole le plan, on suit ce point à travers le réseau
//   (propagation avant) : chaque boule s'allume de la couleur de sa réponse
//   et un petit point marque sa position dans chaque mini-carte.
//
// Les biais ne sont pas dessinés (ce ne sont pas des liaisons entre deux
// neurones) : on les lit en survolant un neurone.
// ---------------------------------------------------------------------------

import type { Vec2 } from "./geometry";
import type { MLP } from "./mlp";

/**
 * Désigne un neurone du schéma. `layer` suit la numérotation de `net.sizes` :
 * 0 = entrées (x puis y), 1 = 1re couche cachée, ..., dernière = sortie.
 */
export interface NodeRef {
  layer: number;
  index: number; // numéro du neurone dans sa couche (0 = le premier)
}

type RGB = [number, number, number];

/** Position et rayon d'une boule, en pixels. */
interface Ball {
  x: number;
  y: number;
  r: number;
}

/** columns[l][j] : boule du neurone j de la couche l (mêmes indices que net.sizes). */
type Layout = Ball[][];

// Même convention partout : rouge = positif (côté « noir » pour la sortie),
// bleu = négatif (côté « blanc »), blanc = zéro.
const COLOR_POS: RGB = [235, 95, 100];
const COLOR_NEG: RGB = [95, 140, 245];
const COLOR_ZERO: RGB = [245, 245, 250];

/** Hauteur réservée en haut pour les titres des couches. */
const TITLE_HEIGHT = 36;

/** Résolution des mini-cartes (en cases par côté), agrandies ensuite. */
const THUMB_RES = 32;

/**
 * Échelle des mini-cartes : le rayon de la boule correspond à √2 en
 * coordonnées normalisées, pour que le carré [-1, 1]² (tout le plan) tienne
 * entièrement dans le disque, coins compris.
 */
const THUMB_SPAN = Math.SQRT2;

/** Au-delà de cette valeur absolue, un poids est dessiné au maximum. */
const STRONG_WEIGHT = 3;

/**
 * Au-delà de ce nombre de neurones par couche cachée, on n'écrit plus le nom
 * de chaque neurone (seulement celui du neurone survolé) : ils se
 * chevaucheraient.
 */
const MAX_LABELLED_UNITS = 16;

/**
 * Place les boules : une colonne par couche, régulièrement espacées de
 * gauche à droite, les neurones cachés répartis en hauteur.
 */
function layout(width: number, height: number, sizes: number[]): Layout {
  const top = TITLE_HEIGHT;
  const bottom = height - 8;
  const midY = (top + bottom) / 2;
  const last = sizes.length - 1;
  const left = width * 0.1;
  const right = width * 0.88;
  const gap = (right - left) / last; // écart horizontal entre deux colonnes

  return sizes.map((n, l) => {
    const x = left + l * gap;
    if (l === 0) return [-1, 1].map((side) => ({ x, y: midY + side * 70, r: 22 }));
    if (l === last) return [{ x, y: midY, r: 28 }];

    // Couches cachées : plus il y a de neurones (ou de couches), plus ils sont
    // serrés et petits.
    const spacing = Math.min(64, (bottom - top) / n);
    const r = Math.min(22, spacing * 0.4, gap * 0.3);
    const firstY = midY - (spacing * (n - 1)) / 2;
    return Array.from({ length: n }, (_, j) => ({ x, y: firstY + j * spacing, r }));
  });
}

/** Renvoie le neurone sous le curseur (coordonnées en pixels), ou null. */
export function hitTestDiagram(
  width: number,
  height: number,
  sizes: number[],
  x: number,
  y: number,
): NodeRef | null {
  const columns = layout(width, height, sizes);
  for (let l = 0; l < columns.length; l++) {
    for (let j = 0; j < columns[l].length; j++) {
      const b = columns[l][j];
      if (Math.hypot(b.x - x, b.y - y) <= b.r + 3) return { layer: l, index: j };
    }
  }
  return null;
}

/**
 * Dessine tout le schéma.
 *
 * @param probe   point du plan survolé (coordonnées normalisées), ou null
 * @param hovered neurone survolé dans le schéma, ou null
 */
export function drawDiagram(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  net: MLP,
  probe: Vec2 | null,
  hovered: NodeRef | null,
): void {
  const sizes = net.sizes;
  const last = sizes.length - 1;
  const columns = layout(width, height, sizes);
  ctx.clearRect(0, 0, width, height);

  drawTitles(ctx, columns);

  // Mini-cartes de tous les neurones, calculées en une seule passe.
  const atlas = paintAtlas(net);

  // Propagation avant du point survolé : la réponse de chaque neurone. On en
  // garde une copie, car `activations` est écrasé au prochain appel.
  let probeActs: Float64Array[] | null = null;
  if (probe) {
    net.forward(probe.x, probe.y);
    probeActs = net.activations.map((a) => Float64Array.from(a));
  }

  // --- Les traits (poids), dessinés sous les boules --------------------------
  // Si un neurone est survolé, on estompe les traits qui ne le touchent pas.
  const isHovered = (layer: number, index: number) =>
    hovered !== null && hovered.layer === layer && hovered.index === index;
  for (let l = 0; l < net.layers.length; l++) {
    const { w } = net.layers[l];
    const from = columns[l];
    const to = columns[l + 1];
    // Entre deux grandes couches, les traits se comptent par centaines : on les
    // affine pour que les poids les plus forts restent lisibles.
    const density = Math.min(1, 6 / Math.sqrt(from.length * to.length));
    for (let j = 0; j < to.length; j++) {
      for (let i = 0; i < from.length; i++) {
        const touched = hovered === null || isHovered(l, i) || isHovered(l + 1, j);
        drawEdge(ctx, from[i], to[j], w[j][i], density, !touched);
      }
    }
  }

  // --- Les boules -----------------------------------------------------------
  let k = 0; // numéro du neurone dans l'atlas, toutes couches confondues
  for (let l = 0; l <= last; l++) {
    for (let j = 0; j < sizes[l]; j++, k++) {
      const value = probeActs ? displayValue(probeActs[l][j], l === last) : null;
      drawBall(ctx, columns[l][j], atlas, k, probe, value, isHovered(l, j));
    }
  }

  // --- Les étiquettes, par-dessus tout le reste --------------------------------
  const inputNames = ["x", "y"];
  for (let i = 0; i < 2; i++) {
    const ball = columns[0][i];
    const text = probeActs ? `${inputNames[i]} = ${signed(probeActs[0][i])}` : inputNames[i];
    drawLabel(ctx, text, ball.x, ball.y + ball.r + 12, "center", 12);
  }

  for (let l = 1; l < last; l++) {
    const n = sizes[l];
    // Une seule couche cachée pas trop fournie : on nomme tous les neurones.
    // Sinon, seulement le neurone survolé.
    const labelAll = last === 2 && n <= MAX_LABELLED_UNITS;
    for (let j = 0; j < n; j++) {
      if (!labelAll && !isHovered(l, j)) continue;
      const ball = columns[l][j];
      const text = probeActs ? signed(probeActs[l][j]) : `h${j + 1}`;
      drawLabel(ctx, text, ball.x + ball.r + 5, ball.y, "left", n > 10 ? 10 : 12);
    }
  }

  const out = columns[last][0];
  if (probeActs) {
    const p = probeActs[last][0];
    drawLabel(ctx, `p = ${p.toFixed(2)}`, out.x, out.y + out.r + 12, "center", 12);
    drawLabel(ctx, p >= 0.5 ? "→ noir" : "→ blanc", out.x, out.y + out.r + 29, "center", 12);
  } else {
    drawLabel(ctx, "p", out.x, out.y + out.r + 12, "center", 12);
  }
}

/**
 * Texte explicatif du neurone survolé : sa formule avec les poids actuels.
 */
export function describeNode(net: MLP, node: NodeRef | null): string {
  const sizes = net.sizes;
  const last = sizes.length - 1;
  if (node === null || node.layer > last || node.index >= sizes[node.layer]) {
    return (
      "Survole (ou touche) un neurone pour lire ses poids. " +
      "Survole (ou touche) le plan pour suivre un point à travers le réseau."
    );
  }

  const { layer, index: j } = node;

  // Entrées.
  if (layer === 0) {
    return j === 0
      ? "Entrée x : position horizontale du point, de −1 (bord gauche) à +1 (bord droit)."
      : "Entrée y : position verticale du point, de −1 (bord haut) à +1 (bord bas).";
  }

  // Poids et biais qui arrivent sur ce neurone.
  const { w, b } = net.layers[layer - 1];

  // Sortie.
  if (layer === last) {
    const n = sizes[last - 1];
    const sum = n === 1 ? "v1·h1" : n === 2 ? "v1·h1 + v2·h2" : `v1·h1 + … + v${n}·h${n}`;
    const from = last === 2 ? "" : ` (h = neurones de la couche cachée ${last - 1})`;
    return (
      `Sortie p = sigmoïde(${sum} + bo)${from}, avec bo = ${signed(b[0])} : ` +
      `probabilité que le point soit noir (p ≥ 0.5 → noir).`
    );
  }

  const k = j + 1;
  const where = last === 2 ? "Neurone caché" : `Couche cachée ${layer}, neurone`;

  // 1re couche cachée : une droite adoucie, formule complète avec x et y.
  if (layer === 1) {
    const [wx, wy] = w[j];
    const first = wx < 0 ? "−" : "";
    const formula =
      `${where} h${k} = tanh(${first}${abs(wx)}·x${op(wy)}${abs(wy)}·y${op(b[j])}${abs(b[j])})`;
    // Une seule couche cachée : le poids vers la sortie tient en une valeur.
    const toOutput = last === 2 ? `, poids vers la sortie v${k} = ${signed(net.layers[1].w[0][j])}` : "";
    return `${formula}${toOutput}. Sa droite est surlignée sur le plan.`;
  }

  // Couches suivantes : trop d'entrées pour écrire la formule en entier.
  const nIn = sizes[layer - 1];
  const weights = Array.from(w[j]);
  return (
    `${where} h${k} = tanh(somme pondérée des ${nIn} neurones de la couche ${layer - 1} + b), ` +
    `avec b = ${signed(b[j])} et des poids de ${signed(Math.min(...weights))} à ${signed(Math.max(...weights))}. ` +
    `Sa frontière (h = 0) est tracée sur le plan : c'est déjà une courbe, car ce neurone combine ` +
    `les ${layer === 2 ? "droites" : "courbes"} de la couche précédente.`
  );
}

// --- Dessin des éléments ------------------------------------------------------

function drawTitles(ctx: CanvasRenderingContext2D, columns: Layout): void {
  const last = columns.length - 1;
  const hiddenLayers = last - 1;
  const firstHidden = columns[1][0].x;
  const lastHidden = columns[last - 1][0].x;

  ctx.save();
  ctx.font = "600 11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#9a9ab0";
  ctx.fillText("ENTRÉES", columns[0][0].x, 12);
  ctx.fillText(hiddenLayers === 1 ? "COUCHE CACHÉE" : "COUCHES CACHÉES", (firstHidden + lastHidden) / 2, 12);
  ctx.fillText("SORTIE", columns[last][0].x, 12);

  // Plusieurs couches cachées : on les numérote.
  if (hiddenLayers > 1) {
    ctx.font = "600 10px system-ui, sans-serif";
    for (let l = 1; l < last; l++) ctx.fillText(String(l), columns[l][0].x, 27);
  }
  ctx.restore();
}

/**
 * Un poids : couleur = signe, épaisseur et opacité = force.
 *
 * @param density 1 pour peu de traits, plus petit quand ils sont nombreux
 */
function drawEdge(
  ctx: CanvasRenderingContext2D,
  a: Ball,
  b: Ball,
  w: number,
  density: number,
  dimmed: boolean,
): void {
  const strength = Math.min(Math.abs(w) / STRONG_WEIGHT, 1);
  const [r, g, bl] = w >= 0 ? COLOR_POS : COLOR_NEG;
  const alpha = (0.15 + 0.75 * strength) * density * (dimmed ? 0.12 : 1);
  ctx.strokeStyle = `rgba(${r}, ${g}, ${bl}, ${alpha})`;
  ctx.lineWidth = (0.5 + 4.5 * strength) * Math.max(density, 0.3);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

/**
 * Une boule : la mini-carte de la réponse du neurone sur le plan, le point
 * survolé, et un contour.
 *
 * @param atlas       mini-cartes de tous les neurones (voir `paintAtlas`)
 * @param k           numéro du neurone dans l'atlas
 * @param value       réponse au point survolé, dans [-1, 1] (colore le contour), ou null
 * @param highlighted neurone survolé dans le schéma : halo blanc
 */
function drawBall(
  ctx: CanvasRenderingContext2D,
  ball: Ball,
  atlas: HTMLCanvasElement | null,
  k: number,
  probe: Vec2 | null,
  value: number | null,
  highlighted: boolean,
): void {
  // Mini-carte, découpée en disque. Le lissage du navigateur fond les cases.
  ctx.save();
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
  ctx.clip();
  if (atlas) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(
      atlas,
      k * THUMB_RES, 0, THUMB_RES, THUMB_RES, // la case du neurone k dans l'atlas
      ball.x - ball.r, ball.y - ball.r, ball.r * 2, ball.r * 2, // agrandie dans la boule
    );
  }
  ctx.restore();

  // Où tombe le point survolé dans cette mini-carte.
  if (probe) {
    ctx.beginPath();
    ctx.arc(
      ball.x + (probe.x / THUMB_SPAN) * ball.r,
      ball.y + (probe.y / THUMB_SPAN) * ball.r,
      Math.max(1.5, ball.r * 0.12),
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = "#101018";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }

  // Contour : il « s'allume » de la couleur de la réponse au point survolé.
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
  if (value !== null) {
    ctx.strokeStyle = colorOf(value);
    ctx.lineWidth = ball.r < 8 ? 2 : 3;
  } else {
    ctx.strokeStyle = "#9a9ab0";
    ctx.lineWidth = ball.r < 8 ? 1 : 1.5;
  }
  ctx.stroke();

  if (highlighted) {
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.r + 4, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

/** Petit texte sur fond sombre, lisible même par-dessus les traits. */
function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  align: "left" | "center",
  size: number,
): void {
  ctx.save();
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width;
  const left = align === "center" ? x - w / 2 : x;
  ctx.fillStyle = "rgba(38, 38, 54, 0.85)";
  ctx.fillRect(left - 3, y - size * 0.7, w + 6, size * 1.4);
  ctx.fillStyle = "#e6e6f0";
  ctx.fillText(text, x, y);
  ctx.restore();
}

// --- Mini-cartes ----------------------------------------------------------------

/**
 * Canvas hors écran qui range les mini-cartes de TOUS les neurones côte à
 * côte : la case k (de THUMB_RES pixels de large) est celle du k-ième neurone,
 * en comptant les entrées, puis chaque couche cachée, puis la sortie.
 */
let atlasCanvas: HTMLCanvasElement | null = null;

/**
 * Calcule les mini-cartes de tous les neurones sur tout le plan (et un peu
 * au-delà). Une seule propagation avant par case suffit : elle donne d'un
 * coup la réponse de chaque neurone du réseau en ce point.
 */
function paintAtlas(net: MLP): HTMLCanvasElement | null {
  const sizes = net.sizes;
  const last = sizes.length - 1;
  const total = sizes.reduce((sum, n) => sum + n, 0);

  atlasCanvas ??= document.createElement("canvas");
  if (atlasCanvas.width !== total * THUMB_RES) {
    atlasCanvas.width = total * THUMB_RES;
    atlasCanvas.height = THUMB_RES;
  }
  const off = atlasCanvas.getContext("2d");
  if (!off) return null;

  const img = off.createImageData(atlasCanvas.width, THUMB_RES);
  for (let r = 0; r < THUMB_RES; r++) {
    for (let c = 0; c < THUMB_RES; c++) {
      // Centre de la case, en coordonnées normalisées (y vers le bas, comme le plan).
      const x = (((c + 0.5) / THUMB_RES) * 2 - 1) * THUMB_SPAN;
      const y = (((r + 0.5) / THUMB_RES) * 2 - 1) * THUMB_SPAN;
      net.forward(x, y);

      let k = 0;
      for (let l = 0; l <= last; l++) {
        const acts = net.activations[l];
        for (let j = 0; j < acts.length; j++, k++) {
          const i = (r * img.width + k * THUMB_RES + c) * 4;
          writeColor(img.data, i, displayValue(acts[j], l === last));
        }
      }
    }
  }
  off.putImageData(img, 0, 0);
  return atlasCanvas;
}

/**
 * Valeur d'un neurone ramenée dans [-1, 1] pour la couleur. La sortie p est
 * entre 0 et 1 : on la recentre, pour que p = 0.5 (la frontière) soit blanc.
 */
function displayValue(a: number, isOutput: boolean): number {
  return isOutput ? (a - 0.5) * 2 : a;
}

// --- Petits utilitaires -----------------------------------------------------------

/**
 * Couleur d'une valeur de [-1, 1] : bleu (−1) -> blanc (0) -> rouge (+1).
 * Écrite directement dans un tableau de pixels RGBA, à partir de l'indice i
 * (évite de créer un objet par pixel : il y en a des milliers par image).
 */
function writeColor(data: Uint8ClampedArray, i: number, v: number): void {
  const t = Math.min(Math.abs(v), 1);
  const end = v >= 0 ? COLOR_POS : COLOR_NEG;
  data[i] = COLOR_ZERO[0] + (end[0] - COLOR_ZERO[0]) * t;
  data[i + 1] = COLOR_ZERO[1] + (end[1] - COLOR_ZERO[1]) * t;
  data[i + 2] = COLOR_ZERO[2] + (end[2] - COLOR_ZERO[2]) * t;
  data[i + 3] = 255;
}

/** Même couleur que `writeColor`, sous forme de texte CSS. */
function colorOf(v: number): string {
  const c = new Uint8ClampedArray(4);
  writeColor(c, 0, v);
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Nombre signé à deux décimales, avec un vrai signe moins : « −0.42 ». */
function signed(v: number): string {
  return `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;
}

function abs(v: number): string {
  return Math.abs(v).toFixed(2);
}

/** Opérateur placé devant un terme : « + » ou « − » selon son signe. */
function op(v: number): string {
  return v < 0 ? " − " : " + ";
}
