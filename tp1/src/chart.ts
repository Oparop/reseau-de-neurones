// ---------------------------------------------------------------------------
// Courbe de perte : l'évolution de la perte au fil des itérations, mesurée
// sur les points d'entraînement et sur les points de test.
//
// Ce qu'on y lit :
// - les deux courbes descendent ensemble : le réseau apprend ET généralise ;
// - la courbe d'entraînement continue de descendre pendant que celle de test
//   remonte : le réseau apprend PAR CŒUR ses points d'entraînement au lieu
//   de saisir la forme générale (surapprentissage) ;
// - une courbe en dents de scie : le learning rate est trop grand ;
// - un long plateau : le réseau est coincé (essayer d'autres poids de départ,
//   plus de neurones, une autre activation...).
// ---------------------------------------------------------------------------

/** Une mesure de la perte, à une itération donnée. */
export interface LossRecord {
  iteration: number;
  train: number; // perte moyenne sur les points d'entraînement (NaN s'il n'y en a pas)
  test: number; // perte moyenne sur les points de test (NaN s'il n'y en a pas)
}

/**
 * Nombre maximal de mesures conservées. Au-delà, on n'en garde qu'une sur deux
 * et on espace les suivantes : la courbe montre toujours tout l'historique,
 * sans grossir sans fin.
 */
const MAX_RECORDS = 400;

/** L'historique des mesures depuis le dernier départ du réseau. */
export class LossHistory {
  records: LossRecord[] = [];

  /** Écart minimal (en itérations) entre deux mesures conservées. */
  private minGap = 1;

  clear(): void {
    this.records = [];
    this.minGap = 1;
  }

  /**
   * Enregistre la mesure courante (appelé à chaque image). La dernière mesure
   * reste « provisoire » tant qu'elle est trop proche de l'avant-dernière :
   * elle est alors remplacée au lieu d'en ajouter une nouvelle.
   */
  add(record: LossRecord): void {
    const n = this.records.length;
    const last = this.records[n - 1];
    const beforeLast = this.records[n - 2];
    const provisional =
      last !== undefined &&
      (record.iteration === last.iteration ||
        (beforeLast !== undefined && last.iteration - beforeLast.iteration < this.minGap));

    if (provisional) this.records[n - 1] = record;
    else this.records.push(record);

    if (this.records.length > MAX_RECORDS) {
      const all = this.records;
      this.records = all.filter((_, i) => i % 2 === 0 || i === all.length - 1);
      this.minGap *= 2;
    }
  }
}

// --- Apparence --------------------------------------------------------------

/**
 * Couleurs des deux séries. Le rouge et le bleu étant déjà pris dans
 * l'application (classes des points, signe des poids), on utilise un vert
 * d'eau et un jaune, vérifiés pour rester distincts sur le fond du panneau,
 * y compris pour les daltoniens.
 */
export const SERIES = [
  { key: "train", name: "entraînement", color: "#199e70" },
  { key: "test", name: "test", color: "#c98500" },
] as const;

const SURFACE = "#262636"; // fond du graphique (celui du panneau)
const GRID = "#34344a"; // lignes de repère, discrètes
const BASELINE = "#4a4a66"; // axe du zéro
const INK_MUTED = "#9a9ab0"; // graduations
const INK_SECONDARY = "#c8c8d8"; // étiquettes des courbes

/** Marges autour de la zone de tracé (place pour les graduations et étiquettes). */
const MARGIN = { left: 40, right: 104, top: 10, bottom: 22 };

/**
 * Dessine la courbe de perte.
 *
 * @param hoverX position horizontale du pointeur (pixels CSS), ou null
 * @returns la mesure pointée (pour l'infobulle), ou null
 */
export function drawLossChart(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  history: LossHistory,
  hoverX: number | null,
): LossRecord | null {
  ctx.clearRect(0, 0, width, height);
  const records = history.records;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  ctx.font = "11px system-ui, sans-serif";
  if (records.length < 2) {
    ctx.fillStyle = INK_MUTED;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("La courbe apparaîtra pendant l'entraînement.", width / 2, height / 2);
    return null;
  }

  // --- Échelles -------------------------------------------------------------
  // L'axe part toujours de l'itération 0, même si la première mesure a été
  // prise un peu après (entraînement déjà lancé).
  const x0 = 0;
  const x1 = Math.max(records[records.length - 1].iteration, 1);
  let maxLoss = 0;
  for (const r of records) {
    if (Number.isFinite(r.train)) maxLoss = Math.max(maxLoss, r.train);
    if (Number.isFinite(r.test)) maxLoss = Math.max(maxLoss, r.test);
  }
  const yTicks = niceTicks(0, Math.max(maxLoss, 0.1), 4);
  const yMax = yTicks[yTicks.length - 1];
  const sx = (it: number) => MARGIN.left + ((it - x0) / (x1 - x0)) * plotW;
  const sy = (v: number) => MARGIN.top + plotH - (v / yMax) * plotH;

  // --- Repères et graduations -------------------------------------------------
  ctx.lineWidth = 1;
  ctx.fillStyle = INK_MUTED;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yDecimals = decimalsFor(yTicks[1] - yTicks[0]);
  for (const t of yTicks) {
    const y = Math.round(sy(t)) + 0.5; // + 0.5 : trait d'un pixel bien net
    ctx.strokeStyle = t === 0 ? BASELINE : GRID;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y);
    ctx.lineTo(MARGIN.left + plotW, y);
    ctx.stroke();
    ctx.fillText(t.toFixed(yDecimals), MARGIN.left - 6, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of niceTicks(x0, x1, 4)) {
    if (t < x0 || t > x1) continue;
    ctx.fillText(t.toLocaleString("fr-FR"), sx(t), MARGIN.top + plotH + 6);
  }

  // --- Les courbes --------------------------------------------------------------
  const ends: { y: number; value: number; name: string; color: string }[] = [];
  for (const series of SERIES) {
    ctx.strokeStyle = series.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    let penDown = false;
    let end: LossRecord | null = null;
    for (const r of records) {
      const v = r[series.key];
      if (!Number.isFinite(v)) {
        penDown = false; // trou dans la série (aucun point de ce type à ce moment-là)
        continue;
      }
      if (penDown) ctx.lineTo(sx(r.iteration), sy(v));
      else ctx.moveTo(sx(r.iteration), sy(v));
      penDown = true;
      end = r;
    }
    ctx.stroke();
    if (end) {
      const value = end[series.key];
      drawDot(ctx, sx(end.iteration), sy(value), series.color);
      ends.push({ y: sy(value), value, name: series.name, color: series.color });
    }
  }

  // Étiquettes au bout des courbes, sauf si elles se chevaucheraient : la
  // légende et l'infobulle suffisent alors.
  const overlap = ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 14;
  if (!overlap) {
    ctx.fillStyle = INK_SECONDARY;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const e of ends) {
      ctx.fillText(`${e.name} ${e.value.toFixed(2)}`, MARGIN.left + plotW + 10, e.y);
    }
  }

  // --- Survol : réticule sur la mesure la plus proche ----------------------------
  if (hoverX === null) return null;
  let hovered = records[0];
  for (const r of records) {
    if (Math.abs(sx(r.iteration) - hoverX) < Math.abs(sx(hovered.iteration) - hoverX)) hovered = r;
  }
  const hx = Math.round(sx(hovered.iteration)) + 0.5;
  ctx.strokeStyle = INK_MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(hx, MARGIN.top);
  ctx.lineTo(hx, MARGIN.top + plotH);
  ctx.stroke();
  for (const series of SERIES) {
    const v = hovered[series.key];
    if (Number.isFinite(v)) drawDot(ctx, hx, sy(v), series.color);
  }
  return hovered;
}

/**
 * Remplit l'infobulle du graphique avec la mesure pointée : la valeur d'abord
 * (c'est ce qu'on cherche), le nom de la courbe ensuite.
 */
export function fillTooltip(tip: HTMLElement, record: LossRecord): void {
  const title = document.createElement("div");
  title.className = "tip-title";
  title.textContent = `Itération ${record.iteration.toLocaleString("fr-FR")}`;
  const rows = SERIES.map((series) => {
    const row = document.createElement("div");
    row.className = "tip-row";
    const key = document.createElement("span");
    key.className = "key";
    key.style.background = series.color;
    const value = document.createElement("strong");
    const v = record[series.key];
    value.textContent = Number.isFinite(v) ? v.toFixed(3) : "—";
    const name = document.createElement("span");
    name.textContent = series.name;
    row.append(key, value, name);
    return row;
  });
  tip.replaceChildren(title, ...rows);
}

/**
 * Ajuste la résolution du canvas à sa taille affichée (et à la densité de
 * l'écran) : le texte reste net et lisible, même sur téléphone.
 * @returns la taille affichée, en pixels CSS
 */
export function fitCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): { width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const bw = Math.round(width * dpr);
  const bh = Math.round(height * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

// --- Petits utilitaires -----------------------------------------------------------

/** Pastille de fin de courbe, entourée d'un liseré couleur du fond. */
function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number, color: string): void {
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = SURFACE;
  ctx.stroke();
}

/** Graduations « rondes » (pas de 1, 2, 2.5 ou 5 × 10^k) couvrant [min, max]. */
function niceTicks(min: number, max: number, count: number): number[] {
  const rough = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= rough) ?? 10 * power;
  const ticks: number[] = [];
  for (let t = Math.floor(min / step) * step; t < max + step - 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12))); // évite les 0.30000000000000004
  }
  return ticks;
}

/** Nombre de décimales nécessaires pour écrire un pas de graduation (0.25 -> 2). */
function decimalsFor(step: number): number {
  let d = 0;
  while (d < 4 && Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) > 1e-9) d++;
  return d;
}
