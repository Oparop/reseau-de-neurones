Alors, ce que je voudrais pour le répertoire TP1, c'est faire un petit TP qui montre ce qu'est un perceptron. Je voudrais juste qu'on voie un neurone artificiel avec, on va dire, juste deux entrées qui réponde 0 ou 1, donc qui fasse de la classification.

Je voudrais que tu me traces sur un plan à deux dimensions, on va dire, des points blancs et des points noirs. À la limite, tu peux laisser l'utilisateur cliquer pour ajouter des points ou enlever des points. Une fois que les points sont là, il faudrait qu'on puisse pouvoir entraîner un perceptron de manière à ce que on voie la droite que le perceptron trace devenir la droite de la frontière entre les points blancs et les points noirs. 

---

## Spécifications clarifiées

### Objectif pédagogique
Démontrer visuellement un perceptron à 2 entrées (classification binaire, sortie 0/1) et
comment il apprend une frontière de décision linéaire entre deux classes de points.

### Technologie
- **Vite + TypeScript**, rendu via l'API **Canvas 2D**.
- Le `package.json` est placé **à la racine du workspace** ; Vite est configuré avec
  `root` pointant vers `tp1/` (ou équivalent) pour que le TP vive dans `tp1/`.
- Fichiers principaux : `index.html` + `main.ts` (+ modules TS dédiés).

### Interaction utilisateur
- Un **toggle / bouton** sélectionne la **classe active** (noir ou blanc).
- **Clic gauche sur une zone vide** : ajoute un point de la classe active.
- **Clic gauche sur un point existant** : le supprime.

### Entraînement
- **Animation pas-à-pas** : la droite (frontière) se redessine à **chaque itération**
  de l'algorithme du perceptron, pour visualiser la convergence.

### Contrôles d'interface
- Bouton **Reset** (efface tous les points).
- Bouton **Entraîner / Pause** (démarre/suspend l'animation).
- **Slider de vitesse** d'animation.
- **Learning rate réglable**.

### Affichages temps réel
- Poids **w1, w2** et biais **b** courants.
- Nombre d'**itérations** et nombre d'**erreurs** de classification.

### Qualité du code
- Commentaires de code **détaillés** (visée pédagogique).

---

## Évolution — frontière courbe

> Je ne veux plus d'une ligne droite mais d'une courbe. Je veux que ça puisse
> résoudre un problème, mais plus de manière linéaire : avec une courbe.

- Le perceptron simple est remplacé par un **perceptron multicouche** :
  2 entrées → une couche cachée de H neurones (tanh) → 1 sortie (sigmoïde),
  entraîné par **rétropropagation** (descente de gradient batch, entropie croisée).
- La frontière de décision (p = 0.5) est tracée comme une **courbe**
  (marching squares sur une grille), les régions sont teintées selon la
  confiance du réseau.
- Jeux de démonstration **non linéairement séparables** : cercle, XOR, deux
  lunes (et deux amas pour comparer).
- Nouveaux contrôles : nombre de **neurones cachés**, affichage des droites des
  neurones cachés, nouveaux poids aléatoires.
- Nouveaux affichages : architecture, nombre de paramètres, **perte**.

---

## Évolution — réseau profond

> Je veux encore plus de neurones dans mon POC.

- Le réseau accepte **plusieurs couches cachées** : de 1 à 4 couches, de 1 à
  32 neurones par couche (jusqu'à 128 neurones cachés). Par défaut : 2 couches
  de 8 neurones.
- La rétropropagation est généralisée à un nombre quelconque de couches.
- Le schéma du réseau affiche une colonne par couche ; survoler un neurone
  d'une couche profonde trace sa frontière (déjà courbe) sur le plan.
- La grille de mesure du plan s'élargit automatiquement pour les gros réseaux,
  afin de garder une animation fluide.

---

## Évolution — généralisation, courbe de perte, activations, entrées

> Coder les améliorations 1, 2, 3 et 5 : points d'entraînement et de test,
> courbe de perte en direct, choix de l'activation, entrées x², y², x·y.

- **Points de test** : une part réglable des points (20 % par défaut) est mise
  de côté et jamais utilisée pour apprendre. Perte et erreurs sont affichées
  pour l'entraînement et pour le test ; les points de test sont des carrés.
- **Courbe de perte** : sous le plan, perte d'entraînement et de test au fil
  des itérations, avec réticule et infobulle au survol.
- **Activation** des neurones cachés au choix : tanh, ReLU, sigmoïde.
- **Entrées** au choix : x, y, x², y², x·y (au moins une).
