# Réseau de neurones interactif

Un petit laboratoire pour **voir** un réseau de neurones apprendre : on pose
des points noirs et blancs sur un plan, on lance l'entraînement, et on regarde
la frontière de décision se déformer jusqu'à séparer les deux classes.

**▶ Essayer en ligne : <https://oparop.github.io/reseau-de-neurones/>**

- du perceptron (une droite) au réseau **profond** : jusqu'à 4 couches cachées
  de 32 neurones ;
- schéma du réseau en direct : poids, réponse de chaque neurone, propagation
  d'un point à travers les couches ;
- points de test mis de côté et courbe de perte en direct, pour voir si le
  réseau généralise ou apprend par cœur ;
- activation au choix (tanh, ReLU, sigmoïde) et entrées calculées (x², y², x·y) ;
- jeux de démonstration : cercle, XOR, deux lunes, deux amas.

Tout le détail (utilisation, expériences à faire, fonctionnement de la
rétropropagation) est dans [tp1/README.md](tp1/README.md).

## Lancer en local

Le projet utilise [Bun](https://bun.sh) (version figée via `mise`) :

```bash
bun install
bun run dev      # serveur de développement, en général http://localhost:5173/
bun run build    # version statique dans dist/
```

## Publication

Chaque push sur `main` reconstruit le site et le publie sur GitHub Pages
(voir [.github/workflows/pages.yml](.github/workflows/pages.yml)).
