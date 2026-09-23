// ---------------------------------------------------------------------------
// Le perceptron multicouche (MLP) : des neurones organisés en couches.
//
// Un perceptron seul trace une DROITE : il ne sait séparer que des classes
// « linéairement séparables ». Pour obtenir une frontière COURBE, on ajoute
// des couches de neurones « cachés » entre les entrées et la sortie :
//
//        entrées    couche cachée 1   couche cachée 2   ...    sortie
//
//          x ──┬──►   h1 ──┐   ┌──►   h1 ──┐
//              ├──►   h2 ──┼───┼──►   h2 ──┼──► ... ──►  p   (probabilité
//              │      ...  │   │      ...  │                  que le point
//          y ──┴──►   hH ──┘   └──►   hH ──┘                  soit « noir »)
//
// Chaque neurone fait la même chose : une somme pondérée de TOUTES les
// réponses de la couche précédente, plus un biais, passée dans une fonction
// d'activation :
//
//   neurone caché :      h = tanh(w_1*a_1 + w_2*a_2 + ... + b)      (entre -1 et 1)
//   neurone de sortie :  p = sigmoïde(v_1*h_1 + v_2*h_2 + ... + bo)  (entre 0 et 1)
//
// (a_1, a_2, ... sont les réponses de la couche précédente : x et y pour la
// première couche cachée.)
//
// Ce que chaque couche apporte :
// - Un neurone de la 1re couche cachée ne voit que x et y : c'est un
//   perceptron « adouci ». Il a sa propre DROITE, mais passe progressivement
//   de -1 à 1 quand on la traverse.
// - Un neurone de la 2e couche combine les droites adoucies de la 1re : sa
//   propre frontière est déjà une COURBE (un coin, une bande, une bosse...).
// - Chaque couche suivante combine les courbes de la précédente : on obtient
//   des formes de plus en plus complexes avec relativement peu de neurones.
//   C'est l'idée du « deep learning » (apprentissage profond).
//
// La frontière de décision est la courbe où p = 0.5 :
//   p >= 0.5  ->  classe 1 (noir)
//   p <  0.5  ->  classe 0 (blanc)
//
// Pourquoi tanh et sigmoïde plutôt que la « marche » 0/1 du perceptron ?
// Parce qu'elles sont lisses (dérivables) : on peut calculer, pour chaque
// poids, dans quel sens le bouger pour réduire l'erreur — y compris pour les
// neurones cachés, dont on ne connaît pas la « bonne réponse ». C'est la
// rétropropagation du gradient.
// ---------------------------------------------------------------------------

/** Un exemple d'entraînement : deux entrées normalisées et la classe attendue. */
export interface TrainingSample {
  x: number; // première entrée (coordonnée normalisée dans [-1, 1])
  y: number; // seconde entrée (coordonnée normalisée dans [-1, 1])
  label: 0 | 1; // classe cible attendue
}

/** Bilan du réseau sur un jeu de points. */
export interface Evaluation {
  loss: number; // perte moyenne (entropie croisée) : plus c'est bas, mieux c'est
  errors: number; // nombre de points mal classés
}

/**
 * Une couche de neurones : chaque neurone j a un poids par neurone de la
 * couche précédente, plus un biais.
 */
export interface Layer {
  /** w[j][i] : poids qui relie le neurone i de la couche précédente au neurone j. */
  w: Float64Array[];
  /** b[j] : biais du neurone j. */
  b: Float64Array;
}

/**
 * Amplitude des poids aléatoires de départ, pour un neurone à 2 entrées.
 * Assez grande pour que les droites de la 1re couche partent dans des
 * directions variées, assez petite pour que tanh ne soit pas déjà « saturée »
 * (pente quasi nulle = apprentissage lent).
 *
 * Un neurone qui reçoit beaucoup d'entrées additionne beaucoup de termes : on
 * réduit alors ses poids de départ (en 1/√entrées) pour que la somme garde le
 * même ordre de grandeur. Sans cette précaution, les couches profondes
 * démarreraient saturées et n'apprendraient presque rien.
 */
const INIT_SCALE = 1.5;

/** Sigmoïde : écrase n'importe quel nombre dans ]0, 1[. */
function sigmoid(s: number): number {
  return 1 / (1 + Math.exp(-s));
}

export class MLP {
  /**
   * Taille de chaque couche, entrées et sortie comprises.
   * Exemple : [2, 8, 8, 1] = 2 entrées, deux couches cachées de 8, 1 sortie.
   */
  readonly sizes: number[];

  /**
   * Les couches qui ont des poids : layers[0] est la 1re couche cachée,
   * la dernière est la couche de sortie (un seul neurone).
   * La couche l fait passer de activations[l] à activations[l + 1].
   */
  layers: Layer[] = [];

  /**
   * Réponses de tous les neurones pour le DERNIER point passé à `forward` :
   * activations[0] = [x, y], activations[1] = 1re couche cachée, ...,
   * activations[dernier] = [p]. Réutilisées d'un appel à l'autre (pas de
   * nouvelle allocation) : à copier si on veut les garder.
   *
   * (Float64Array = tableau de nombres à taille fixe : même usage qu'un
   * tableau ordinaire, mais bien plus rapide pour ces calculs répétés.)
   */
  readonly activations: Float64Array[];

  /** Nombre total de mises à jour effectuées depuis le dernier reset. */
  iterations = 0;

  /** @param hiddenSizes nombre de neurones de chaque couche cachée, ex. [8, 8] */
  constructor(hiddenSizes: number[]) {
    this.sizes = [2, ...hiddenSizes, 1];
    this.activations = this.sizes.map((n) => new Float64Array(n));
    this.reset();
  }

  /** Nombre de neurones de chaque couche cachée. */
  get hiddenSizes(): number[] {
    return this.sizes.slice(1, -1);
  }

  /** Nombre de paramètres ajustés par l'apprentissage. */
  get parameterCount(): number {
    let count = 0;
    for (let l = 1; l < this.sizes.length; l++) {
      // Chaque neurone : un poids par neurone de la couche précédente + un biais.
      count += this.sizes[l] * (this.sizes[l - 1] + 1);
    }
    return count;
  }

  /**
   * Repart de poids aléatoires. Ici c'est indispensable (et pas seulement
   * utile) : si tous les neurones d'une couche partaient des mêmes poids, ils
   * recevraient exactement les mêmes corrections et resteraient identiques
   * pour toujours — ils traceraient tous la même frontière.
   */
  reset(): void {
    this.layers = [];
    for (let l = 1; l < this.sizes.length; l++) {
      const nIn = this.sizes[l - 1];
      const nOut = this.sizes[l];
      const scale = INIT_SCALE * Math.sqrt(2 / nIn); // = INIT_SCALE pour 2 entrées
      const rnd = () => (Math.random() * 2 - 1) * scale;
      const isOutput = l === this.sizes.length - 1;
      this.layers.push({
        w: Array.from({ length: nOut }, () => Float64Array.from({ length: nIn }, rnd)),
        // Biais aléatoires pour les couches cachées (les droites de la 1re
        // couche ne passent pas toutes par le centre), nul pour la sortie.
        b: Float64Array.from({ length: nOut }, () => (isOutput ? 0 : rnd())),
      });
    }
    this.iterations = 0;
  }

  /**
   * Propagation avant : fait passer le point (x, y) couche après couche,
   * jusqu'à la probabilité p en sortie. Les réponses de chaque neurone restent
   * disponibles dans `activations` (la rétropropagation en a besoin).
   */
  forward(x: number, y: number): number {
    const acts = this.activations;
    acts[0][0] = x;
    acts[0][1] = y;

    const last = this.layers.length - 1;
    for (let l = 0; l <= last; l++) {
      const { w, b } = this.layers[l];
      const input = acts[l];
      const output = acts[l + 1];
      for (let j = 0; j < output.length; j++) {
        // Somme pondérée des réponses de la couche précédente, plus le biais.
        let s = b[j];
        const wj = w[j];
        for (let i = 0; i < input.length; i++) s += wj[i] * input[i];
        // Couches cachées : tanh. Couche de sortie : sigmoïde (une probabilité).
        output[j] = l === last ? sigmoid(s) : Math.tanh(s);
      }
    }
    return acts[last + 1][0];
  }

  /** Classe prédite (0 ou 1) : de quel côté de la courbe tombe le point. */
  predict(x: number, y: number): 0 | 1 {
    return this.forward(x, y) >= 0.5 ? 1 : 0;
  }

  /**
   * Une ITÉRATION = une passe complète (batch) sur tous les échantillons,
   * suivie d'UNE SEULE mise à jour de tous les poids — comme pour le
   * perceptron simple.
   *
   *   pour chaque point (classe attendue t) :
   *     1. propagation avant : on calcule la réponse a de chaque neurone,
   *        couche après couche, jusqu'à p.
   *     2. erreur de sortie :  e = t − p   (entre -1 et 1)
   *     3. on remonte les couches, de la sortie vers les entrées. Chaque
   *        neurone j a une « erreur » e_j ; il suit la règle du perceptron
   *        avec ses propres entrées a_i (les réponses de la couche d'avant) :
   *                             g_w[j][i] += e_j × a_i
   *                             g_b[j]    += e_j
   *        puis il transmet son erreur en arrière. Le neurone i de la couche
   *        précédente reçoit la part qui lui revient : l'erreur de chaque
   *        neurone qu'il alimente, pondérée par le poids qui les relie, et
   *        multipliée par la pente de tanh là où il se trouve (1 − a_i²) :
   *                             e_i = (Σ_j e_j × w[j][i]) × (1 − a_i²)
   *        C'est la RÉTROPROPAGATION : l'erreur de sortie est répartie, couche
   *        par couche, entre tous les neurones qui y ont contribué.
   *   puis, une seule fois (N = nombre de points) :
   *                             chaque poids += lr × g / N
   *
   * Mathématiquement, c'est une descente de gradient sur l'entropie croisée
   * (voir `evaluate`) : avec une sortie sigmoïde, sa dérivée se simplifie
   * exactement en t − p.
   */
  trainEpoch(samples: TrainingSample[], lr: number): void {
    if (samples.length === 0) return;

    const layers = this.layers;
    const acts = this.activations;

    // Gradients accumulés : mêmes dimensions que les poids, partis de zéro.
    const gW = layers.map((layer) => layer.w.map((row) => new Float64Array(row.length)));
    const gB = layers.map((layer) => new Float64Array(layer.b.length));
    // errors[l][j] : erreur du neurone j de la couche de poids l.
    const errors = layers.map((layer) => new Float64Array(layer.b.length));

    // Phase 1 : on parcourt tous les points et on accumule le gradient.
    for (const s of samples) {
      const p = this.forward(s.x, s.y);
      errors[layers.length - 1][0] = s.label - p;

      // On remonte de la couche de sortie vers la 1re couche cachée.
      for (let l = layers.length - 1; l >= 0; l--) {
        const { w } = layers[l];
        const input = acts[l]; // entrées de cette couche
        const err = errors[l];

        // Règle du perceptron pour chaque neurone de la couche.
        for (let j = 0; j < err.length; j++) {
          gB[l][j] += err[j];
          const gWj = gW[l][j];
          for (let i = 0; i < input.length; i++) gWj[i] += err[j] * input[i];
        }

        // Rétropropagation vers la couche cachée précédente (pas vers les
        // entrées x et y : elles n'ont pas de poids à corriger).
        if (l > 0) {
          const prevErr = errors[l - 1];
          for (let i = 0; i < input.length; i++) {
            let share = 0;
            for (let j = 0; j < err.length; j++) share += err[j] * w[j][i];
            prevErr[i] = share * (1 - input[i] * input[i]);
          }
        }
      }
    }

    // Phase 2 : une seule mise à jour de tous les poids.
    const k = lr / samples.length;
    for (let l = 0; l < layers.length; l++) {
      const { w, b } = layers[l];
      for (let j = 0; j < b.length; j++) {
        b[j] += k * gB[l][j];
        for (let i = 0; i < w[j].length; i++) w[j][i] += k * gW[l][j][i];
      }
    }

    this.iterations++;
  }

  /**
   * Mesure la perte moyenne et compte les points mal classés.
   *
   * La perte (entropie croisée) vaut −ln(p) pour un point noir et −ln(1 − p)
   * pour un point blanc : 0 quand le réseau est sûr de lui et a raison, très
   * grande quand il est sûr de lui et a tort. Contrairement au simple nombre
   * d'erreurs, elle continue de baisser tant que le réseau gagne en confiance.
   */
  evaluate(samples: TrainingSample[]): Evaluation {
    let loss = 0;
    let errors = 0;
    for (const s of samples) {
      const p = this.forward(s.x, s.y);
      const pExpected = s.label === 1 ? p : 1 - p; // proba donnée à la bonne classe
      loss -= Math.log(Math.max(pExpected, 1e-12)); // on évite ln(0) = -∞
      if ((p >= 0.5 ? 1 : 0) !== s.label) errors++;
    }
    return { loss: samples.length > 0 ? loss / samples.length : NaN, errors };
  }
}
