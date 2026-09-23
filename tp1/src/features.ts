// ---------------------------------------------------------------------------
// Les entrées du réseau : ce qu'on lui donne à « voir » de chaque point.
//
// Par défaut, le réseau ne reçoit que les coordonnées x et y du point. On
// peut aussi lui donner des entrées CALCULÉES à partir d'elles : x², y², x·y.
// C'est ce qu'on appelle choisir des « caractéristiques » (feature
// engineering) : de bonnes entrées peuvent remplacer des couches entières.
//
// Exemples :
// - Le cercle (noir au centre) n'est pas séparable par une droite dans le
//   plan (x, y). Avec les entrées x² et y², un seul neurone suffit : sa
//   « droite »  a·x² + b·y² + c = 0  est une ellipse dans le plan.
// - Le XOR (damier 2×2) : l'entrée x·y est positive en haut à gauche et en
//   bas à droite, négative ailleurs. Elle résout le problème à elle seule.
// ---------------------------------------------------------------------------

export type FeatureName = "x" | "y" | "x2" | "y2" | "xy";

export interface Feature {
  label: string; // nom affiché, ex. « x² »
  description: string; // explication affichée au survol du neurone d'entrée
  f: (x: number, y: number) => number; // calcul à partir des coordonnées du point
}

export const FEATURES: Record<FeatureName, Feature> = {
  x: {
    label: "x",
    description: "position horizontale du point, de −1 (bord gauche) à +1 (bord droit)",
    f: (x) => x,
  },
  y: {
    label: "y",
    description: "position verticale du point, de −1 (bord haut) à +1 (bord bas)",
    f: (_x, y) => y,
  },
  x2: {
    label: "x²",
    description: "carré de x : nul au milieu, grand près des bords gauche et droit",
    f: (x) => x * x,
  },
  y2: {
    label: "y²",
    description: "carré de y : nul au milieu, grand près des bords haut et bas",
    f: (_x, y) => y * y,
  },
  xy: {
    label: "x·y",
    description: "produit de x et y : positif en haut à gauche et en bas à droite, négatif ailleurs",
    f: (x, y) => x * y,
  },
};

/** Ordre d'affichage des entrées, dans l'interface comme dans le schéma. */
export const FEATURE_ORDER: FeatureName[] = ["x", "y", "x2", "y2", "xy"];

/**
 * Vrai si le réseau ne reçoit que x et/ou y. Les neurones de la 1re couche
 * ont alors une vraie DROITE pour frontière ; avec x², y² ou x·y, cette
 * frontière devient une courbe.
 */
export function isLinear(features: FeatureName[]): boolean {
  return features.every((f) => f === "x" || f === "y");
}
