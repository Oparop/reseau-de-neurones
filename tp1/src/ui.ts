// ---------------------------------------------------------------------------
// Câblage de l'interface : récupère les éléments du DOM, branche les
// événements et met à jour les affichages temps réel.
// ---------------------------------------------------------------------------

import { FEATURE_ORDER, type FeatureName } from "./features";
import type { ActivationName, Evaluation, MLP } from "./mlp";
import { state, type DemoKind } from "./state";

/** Callbacks fournis par main.ts pour réagir aux actions utilisateur. */
export interface UiHandlers {
  onToggleClass: () => void;
  onDemo: () => void;
  onReset: () => void;
  onToggleTraining: () => void;
  onStep: () => void;
  onNewWeights: () => void;
  onArchitectureChange: () => void;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Élément #${id} introuvable`);
  return el as T;
}

export interface Ui {
  refreshTraining(): void;
  refreshActiveClass(): void;
  updateReadouts(net: MLP, train: Evaluation, test: Evaluation): void;
  updateNeuronInfo(text: string): void;
}

export function setupUi(handlers: UiHandlers): Ui {
  const toggleClassBtn = byId<HTMLButtonElement>("toggle-class");
  const activeClassLabel = byId<HTMLElement>("active-class");
  const demoSelect = byId<HTMLSelectElement>("demo-kind");
  const demoBtn = byId<HTMLButtonElement>("demo");
  const resetBtn = byId<HTMLButtonElement>("reset");
  const testRatioInput = byId<HTMLInputElement>("test-ratio");
  const testRatioVal = byId<HTMLElement>("test-ratio-val");
  const layersInput = byId<HTMLInputElement>("layers");
  const layersVal = byId<HTMLElement>("layers-val");
  const hiddenInput = byId<HTMLInputElement>("hidden");
  const hiddenVal = byId<HTMLElement>("hidden-val");
  const activationSelect = byId<HTMLSelectElement>("activation");
  const showLinesInput = byId<HTMLInputElement>("show-lines");
  const newWeightsBtn = byId<HTMLButtonElement>("new-weights");
  const trainBtn = byId<HTMLButtonElement>("train");
  const stepBtn = byId<HTMLButtonElement>("step");
  const speedInput = byId<HTMLInputElement>("speed");
  const speedVal = byId<HTMLElement>("speed-val");
  const lrInput = byId<HTMLInputElement>("lr");
  const lrVal = byId<HTMLElement>("lr-val");

  // Une case à cocher par entrée possible (x, y, x², y², x·y).
  const featureBoxes = new Map<FeatureName, HTMLInputElement>();
  for (const box of document.querySelectorAll<HTMLInputElement>("input[data-feature]")) {
    featureBoxes.set(box.dataset.feature as FeatureName, box);
  }

  const archEl = byId<HTMLElement>("arch");
  const paramsEl = byId<HTMLElement>("params");
  const iterEl = byId<HTMLElement>("iter");
  const lossTrainEl = byId<HTMLElement>("loss-train");
  const lossTestEl = byId<HTMLElement>("loss-test");
  const errTrainEl = byId<HTMLElement>("err-train");
  const errTestEl = byId<HTMLElement>("err-test");
  const neuronInfoEl = byId<HTMLElement>("neuron-info");

  toggleClassBtn.addEventListener("click", handlers.onToggleClass);
  demoBtn.addEventListener("click", handlers.onDemo);
  resetBtn.addEventListener("click", handlers.onReset);
  newWeightsBtn.addEventListener("click", handlers.onNewWeights);
  trainBtn.addEventListener("click", handlers.onToggleTraining);
  stepBtn.addEventListener("click", handlers.onStep);

  demoSelect.addEventListener("change", () => {
    // Choisir un jeu le charge directement, sans devoir cliquer sur « Démo ».
    state.demo = demoSelect.value as DemoKind;
    handlers.onDemo();
  });

  testRatioInput.addEventListener("input", () => {
    // Pas besoin de réinitialiser le réseau : les points changent simplement
    // de rôle (voir `isTestPoint`), et l'entraînement continue.
    state.testRatio = Number(testRatioInput.value) / 100;
    testRatioVal.textContent = `${testRatioInput.value} %`;
  });

  for (const box of featureBoxes.values()) {
    box.addEventListener("change", () => {
      const chosen = FEATURE_ORDER.filter((f) => featureBoxes.get(f)?.checked);
      if (chosen.length === 0) {
        box.checked = true; // il faut au moins une entrée
        return;
      }
      state.features = chosen;
      handlers.onArchitectureChange();
    });
  }

  layersInput.addEventListener("input", () => {
    state.hiddenLayers = Number(layersInput.value);
    layersVal.textContent = layersInput.value;
    handlers.onArchitectureChange();
  });

  hiddenInput.addEventListener("input", () => {
    state.hiddenUnits = Number(hiddenInput.value);
    hiddenVal.textContent = hiddenInput.value;
    handlers.onArchitectureChange();
  });

  activationSelect.addEventListener("change", () => {
    state.activation = activationSelect.value as ActivationName;
    handlers.onArchitectureChange();
  });

  showLinesInput.addEventListener("change", () => {
    state.showHiddenLines = showLinesInput.checked;
  });

  speedInput.addEventListener("input", () => {
    state.speed = Number(speedInput.value);
    speedVal.textContent = speedInput.value;
  });

  lrInput.addEventListener("input", () => {
    state.learningRate = Number(lrInput.value);
    lrVal.textContent = state.learningRate.toFixed(2);
  });

  // Initialise les affichages à partir des valeurs par défaut de l'état.
  demoSelect.value = state.demo;
  const testPercent = String(Math.round(state.testRatio * 100));
  testRatioInput.value = testPercent;
  testRatioVal.textContent = `${testPercent} %`;
  for (const [name, box] of featureBoxes) box.checked = state.features.includes(name);
  layersInput.value = String(state.hiddenLayers);
  layersVal.textContent = String(state.hiddenLayers);
  hiddenInput.value = String(state.hiddenUnits);
  hiddenVal.textContent = String(state.hiddenUnits);
  activationSelect.value = state.activation;
  showLinesInput.checked = state.showHiddenLines;
  speedInput.value = String(state.speed);
  speedVal.textContent = String(state.speed);
  lrInput.value = String(state.learningRate);
  lrVal.textContent = state.learningRate.toFixed(2);

  /** Perte à 3 décimales, ou « — » s'il n'y a aucun point à évaluer. */
  const formatLoss = (e: Evaluation) => (Number.isNaN(e.loss) ? "—" : e.loss.toFixed(3));
  /** Erreurs sur le nombre de points, ex. « 2 / 38 ». */
  const formatErrors = (e: Evaluation) => (e.count === 0 ? "—" : `${e.errors} / ${e.count}`);

  const ui: Ui = {
    refreshTraining() {
      trainBtn.textContent = state.training ? "Pause" : "Entraîner";
    },
    refreshActiveClass() {
      activeClassLabel.textContent = state.activeLabel === 1 ? "Noir" : "Blanc";
    },
    updateReadouts(net, train, test) {
      archEl.textContent = net.sizes.join(" → ");
      paramsEl.textContent = String(net.parameterCount);
      iterEl.textContent = String(net.iterations);
      lossTrainEl.textContent = formatLoss(train);
      lossTestEl.textContent = formatLoss(test);
      errTrainEl.textContent = formatErrors(train);
      errTestEl.textContent = formatErrors(test);
    },
    updateNeuronInfo(text) {
      // Appelé à chaque image : on ne touche au DOM que si le texte change.
      if (neuronInfoEl.textContent !== text) neuronInfoEl.textContent = text;
    },
  };

  ui.refreshTraining();
  ui.refreshActiveClass();
  return ui;
}
