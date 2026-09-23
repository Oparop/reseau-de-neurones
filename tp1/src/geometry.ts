// ---------------------------------------------------------------------------
// Géométrie : conversions pixels <-> espace normalisé, droites et courbes.
//
// Le réseau travaille dans l'espace « normalisé » [-1, 1] plutôt qu'en pixels :
// les entrées ont un ordre de grandeur raisonnable, donc un même learning rate
// fonctionne quelle que soit la taille du canvas.
// ---------------------------------------------------------------------------

export interface Vec2 {
  x: number;
  y: number;
}

/** Pixel -> coordonnée normalisée dans [-1, 1]. */
export function toNormalized(px: number, py: number, width: number, height: number): Vec2 {
  return {
    x: (px / width) * 2 - 1,
    y: (py / height) * 2 - 1,
  };
}

/** Coordonnée normalisée [-1, 1] -> pixel. */
export function toPixel(nx: number, ny: number, width: number, height: number): Vec2 {
  return {
    x: ((nx + 1) / 2) * width,
    y: ((ny + 1) / 2) * height,
  };
}

/**
 * Calcule les deux extrémités (en pixels) du segment représentant la droite
 * w1*x + w2*y + b = 0  à travers tout le canvas. Sert à afficher la droite
 * propre à chaque neurone caché.
 *
 * On raisonne dans l'espace normalisé [-1, 1] puis on reconvertit en pixels.
 * Selon l'orientation de la droite, on la paramètre soit par x, soit par y,
 * pour éviter les divisions par des poids proches de zéro.
 *
 * @returns les deux extrémités, ou null si la droite est indéfinie (w1=w2=0).
 */
export function decisionLineEndpoints(
  w1: number,
  w2: number,
  b: number,
  width: number,
  height: number,
): [Vec2, Vec2] | null {
  if (w1 === 0 && w2 === 0) return null;

  let a: Vec2;
  let c: Vec2;

  if (Math.abs(w2) >= Math.abs(w1)) {
    // Droite « plutôt horizontale » : on balaie x de -1 à 1 et on résout y.
    const yAt = (x: number) => -(b + w1 * x) / w2;
    a = { x: -1, y: yAt(-1) };
    c = { x: 1, y: yAt(1) };
  } else {
    // Droite « plutôt verticale » : on balaie y de -1 à 1 et on résout x.
    const xAt = (y: number) => -(b + w2 * y) / w1;
    a = { x: xAt(-1), y: -1 };
    c = { x: xAt(1), y: 1 };
  }

  return [toPixel(a.x, a.y, width, height), toPixel(c.x, c.y, width, height)];
}

// ---------------------------------------------------------------------------
// Tracé d'une frontière COURBE.
//
// Pour une droite, on connaît l'équation : deux points suffisent. Pour la
// frontière d'un réseau de neurones, il n'y a pas de formule simple. On
// procède donc comme un cartographe qui trace une ligne de niveau :
//   1. on mesure la sortie du réseau sur une grille régulière de points ;
//   2. dans chaque petite case de la grille, si la frontière p = 0.5 passe
//      entre deux coins (l'un >= 0.5, l'autre < 0.5), on estime où elle coupe
//      le bord, puis on relie ces points d'entrée et de sortie par un petit
//      segment (algorithme des « marching squares »).
// Mis bout à bout, ces petits segments dessinent la courbe.
// ---------------------------------------------------------------------------

/** Valeurs d'une fonction mesurées sur une grille régulière du canvas. */
export interface ScalarField {
  cols: number;
  rows: number;
  step: number; // écart en pixels entre deux points de la grille
  values: Float32Array; // values[r * cols + c] = valeur au pixel (c*step, r*step)
}

export interface Segment {
  a: Vec2;
  b: Vec2;
}

/**
 * Mesure f tous les `step` pixels. f reçoit des coordonnées normalisées, comme
 * les entrées du réseau.
 */
export function sampleField(
  f: (x: number, y: number) => number,
  width: number,
  height: number,
  step: number,
): ScalarField {
  const cols = Math.ceil(width / step) + 1;
  const rows = Math.ceil(height / step) + 1;
  const values = new Float32Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const n = toNormalized(c * step, r * step, width, height);
      values[r * cols + c] = f(n.x, n.y);
    }
  }
  return { cols, rows, step, values };
}

/**
 * Découpe la courbe de niveau  f = level  en petits segments (en pixels).
 *
 * Chaque case a 4 coins (haut-gauche, haut-droite, bas-droite, bas-gauche).
 * On note lesquels sont « au-dessus » du niveau : cela donne 16 cas possibles,
 * qui disent par quels bords la courbe entre et sort de la case.
 */
export function contourSegments(field: ScalarField, level: number): Segment[] {
  const { cols, rows, step, values } = field;
  const segments: Segment[] = [];

  // Fraction du chemin entre a et b où la valeur vaut `level` (interpolation
  // linéaire) : c'est ce qui rend la courbe lisse plutôt qu'en escalier.
  const t = (a: number, b: number) => (level - a) / (b - a);

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const tl = values[r * cols + c];
      const tr = values[r * cols + c + 1];
      const br = values[(r + 1) * cols + c + 1];
      const bl = values[(r + 1) * cols + c];

      // Un bit par coin au-dessus du niveau -> un code entre 0 et 15.
      const code =
        (tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
      if (code === 0 || code === 15) continue; // la courbe ne traverse pas cette case

      // Points où la courbe coupe chacun des 4 bords de la case.
      const x0 = c * step;
      const y0 = r * step;
      const top = () => ({ x: x0 + step * t(tl, tr), y: y0 });
      const right = () => ({ x: x0 + step, y: y0 + step * t(tr, br) });
      const bottom = () => ({ x: x0 + step * t(bl, br), y: y0 + step });
      const left = () => ({ x: x0, y: y0 + step * t(tl, bl) });
      const add = (a: Vec2, b: Vec2) => segments.push({ a, b });

      // Cas ambigus (5 et 10) : deux coins opposés au-dessus, deux en dessous.
      // On tranche avec la valeur au centre de la case.
      const centerAbove = (tl + tr + br + bl) / 4 >= level;

      switch (code) {
        case 1:
        case 14:
          add(left(), bottom());
          break;
        case 2:
        case 13:
          add(bottom(), right());
          break;
        case 3:
        case 12:
          add(left(), right());
          break;
        case 4:
        case 11:
          add(top(), right());
          break;
        case 6:
        case 9:
          add(top(), bottom());
          break;
        case 7:
        case 8:
          add(left(), top());
          break;
        case 5: // haut-droite et bas-gauche au-dessus
          if (centerAbove) {
            add(left(), top());
            add(bottom(), right());
          } else {
            add(top(), right());
            add(left(), bottom());
          }
          break;
        case 10: // haut-gauche et bas-droite au-dessus
          if (centerAbove) {
            add(top(), right());
            add(left(), bottom());
          } else {
            add(left(), top());
            add(bottom(), right());
          }
          break;
      }
    }
  }

  return segments;
}
