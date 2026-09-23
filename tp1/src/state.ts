// ---------------------------------------------------------------------------
// État partagé de l'application et données de démonstration.
// ---------------------------------------------------------------------------

import type { NodeRef } from "./diagram";
import type { FeatureName } from "./features";
import { toPixel, type Vec2 } from "./geometry";
import type { ActivationName, NetworkConfig } from "./mlp";

/** Un point posé sur le canvas, en coordonnées pixels. */
export interface Point {
  px: number;
  py: number;
  label: 0 | 1; // 0 = blanc, 1 = noir
  /**
   * Tirage au sort dans [0, 1[, fait une fois pour toutes à la création du
   * point : il sert au TEST (et non à l'entraînement) si roll < part de test.
   * Ainsi, changer la part de test ne rebat pas toutes les cartes : les
   * points déjà mis de côté le restent quand on l'augmente.
   */
  roll: number;
}

/** Jeux de points proposés par le bouton « Démo ». */
export type DemoKind = "circle" | "xor" | "moons" | "blobs";

export interface AppState {
  points: Point[];
  activeLabel: 0 | 1; // classe déposée au prochain clic
  testRatio: number; // part des points mis de côté pour le test (0 à 0.5)
  learningRate: number;
  speed: number; // nombre d'itérations (passes batch) par frame
  training: boolean;
  features: FeatureName[]; // entrées du réseau
  hiddenLayers: number; // nombre de couches cachées
  hiddenUnits: number; // nombre de neurones de chaque couche cachée
  activation: ActivationName; // activation des neurones cachés
  demo: DemoKind; // jeu chargé par le bouton « Démo »
  showHiddenLines: boolean; // afficher la frontière de chaque neurone de la 1re couche cachée
  probe: Vec2 | null; // point du plan survolé par la souris (coordonnées normalisées)
  hoveredNode: NodeRef | null; // neurone survolé dans le schéma du réseau
}

export const state: AppState = {
  points: [],
  activeLabel: 1,
  testRatio: 0.2,
  learningRate: 0.5,
  speed: 2,
  training: false,
  features: ["x", "y"],
  hiddenLayers: 2,
  hiddenUnits: 8,
  activation: "tanh",
  demo: "circle",
  showHiddenLines: false,
  probe: null,
  hoveredNode: null,
};

/** Forme du réseau décrite par les réglages actuels. */
export function networkConfig(s: AppState): NetworkConfig {
  return {
    features: s.features,
    hidden: Array.from({ length: s.hiddenLayers }, () => s.hiddenUnits),
    activation: s.activation,
  };
}

/**
 * Vrai si le point est mis de côté pour le TEST : le réseau ne l'utilise
 * jamais pour apprendre. On mesure dessus s'il sait GÉNÉRALISER, c'est-à-dire
 * bien classer des points qu'il n'a jamais vus.
 */
export function isTestPoint(p: Point, testRatio: number): boolean {
  return p.roll < testRatio;
}

/**
 * Génère un jeu de démonstration. Les trois premiers ne sont PAS séparables
 * par une droite : il faut une frontière courbe. Le dernier (deux amas) l'est,
 * pour comparer avec le perceptron simple.
 */
export function makeDemoPoints(kind: DemoKind, width: number, height: number): Point[] {
  const rnd = (min: number, max: number) => Math.random() * (max - min) + min;
  const points: Point[] = [];

  // Les formes sont décrites en coordonnées normalisées [-1, 1], puis
  // converties en pixels.
  const add = (nx: number, ny: number, label: 0 | 1) => {
    const p = toPixel(nx, ny, width, height);
    points.push({ px: p.x, py: p.y, label, roll: Math.random() });
  };

  switch (kind) {
    case "circle":
      // Un disque noir au centre, entouré d'un anneau blanc.
      for (let i = 0; i < 20; i++) {
        const a = rnd(0, Math.PI * 2);
        const r = rnd(0, 0.35);
        add(r * Math.cos(a), r * Math.sin(a), 1);
      }
      for (let i = 0; i < 28; i++) {
        const a = rnd(0, Math.PI * 2);
        const r = rnd(0.6, 0.9);
        add(r * Math.cos(a), r * Math.sin(a), 0);
      }
      break;

    case "xor":
      // Damier 2×2 : noir en haut-gauche et bas-droite, blanc ailleurs.
      // Le « OU exclusif » est l'exemple historique qu'un perceptron seul ne
      // sait pas résoudre (Minsky & Papert, 1969).
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          for (let i = 0; i < 10; i++) {
            add(sx * rnd(0.2, 0.85), sy * rnd(0.2, 0.85), sx === sy ? 1 : 0);
          }
        }
      }
      break;

    case "moons":
      // Deux croissants imbriqués l'un dans l'autre.
      for (let i = 0; i < 24; i++) {
        const a = rnd(0, Math.PI);
        const noise = () => rnd(-0.06, 0.06);
        add(-0.3 + 0.55 * Math.cos(a) + noise(), 0.15 - 0.55 * Math.sin(a) + noise(), 1);
        add(0.3 - 0.55 * Math.cos(a) + noise(), -0.15 + 0.55 * Math.sin(a) + noise(), 0);
      }
      break;

    case "blobs":
      // Deux amas séparables par une simple droite.
      for (let i = 0; i < 12; i++) add(rnd(-0.8, -0.2), rnd(-0.8, -0.1), 1);
      for (let i = 0; i < 12; i++) add(rnd(0.2, 0.8), rnd(0.1, 0.8), 0);
      break;
  }

  return points;
}
