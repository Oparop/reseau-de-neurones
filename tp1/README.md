# TP1 — Du perceptron au réseau de neurones

Un petit laboratoire interactif pour **voir** comment un réseau de neurones
apprend à séparer deux classes de points par une **courbe**, là où un
perceptron seul ne sait tracer qu'une droite.

## À quoi ça sert ?

Le réseau prend **2 entrées** (ici les coordonnées `x` et `y` d'un point) et
répond **0 ou 1** (ici : point *blanc* ou point *noir*).

Un perceptron simple calcule `w1·x + w2·y + b` et regarde le signe : sa
frontière est forcément une **droite**. Il échoue dès que les classes ne sont
pas « linéairement séparables » (un disque noir entouré de points blancs, par
exemple).

On ajoute donc une **couche cachée** de plusieurs neurones entre les entrées et
la sortie :

```text
      x ──┬──► h1 ──┐
          ├──► h2 ──┤
          │    ...  ├──► p   (probabilité que le point soit noir)
      y ──┴──► hH ──┘

neurone caché j :    h_j = tanh(wx_j·x + wy_j·y + bh_j)
neurone de sortie :  p   = sigmoïde(v_1·h_1 + … + v_H·h_H + bo)
```

Chaque neurone caché est un perceptron « adouci » : il a sa propre droite,
mais passe **progressivement** de -1 à 1 quand on la traverse. Le neurone de
sortie additionne ces réponses : en combinant plusieurs droites adoucies, il
peut dessiner une frontière **courbe** de presque n'importe quelle forme.

On peut aller plus loin et **empiler plusieurs couches cachées** (jusqu'à 4,
avec jusqu'à 32 neurones chacune, soit 128 neurones) :

```text
      x ──► couche 1 ──► couche 2 ──► … ──► p
      y ──►  (droites)    (courbes)
```

Chaque neurone d'une couche reçoit **toutes** les réponses de la couche
précédente. Les neurones de la 1re couche tracent des droites adoucies ; ceux
de la 2e combinent ces droites et dessinent déjà des **courbes** ; chaque
couche suivante combine les courbes de la précédente. C'est l'idée de
l'apprentissage **profond** (*deep learning*).

La frontière de décision est la courbe où **p = 0.5**. Le but du TP est de la
voir se déformer, itération après itération, jusqu'à séparer correctement tous
les points.

## Démarrer le logiciel

Le projet utilise **Bun** (déjà figé via `mise`). Depuis la **racine du
workspace** (`tps-laval`) :

```bash
# 1. Installer les dépendances (une seule fois)
bun install

# 2. Lancer le serveur de développement
bun run dev
```

Ouvre ensuite l'adresse affichée dans le terminal, en général
<http://localhost:5173/>.

Pour générer une version statique (dossier `dist/`) :

```bash
bun run build
```

## Comment l'utiliser

Au démarrage, le jeu de démonstration **Cercle** s'affiche : un disque de
points noirs entouré d'un anneau de points blancs. Aucune droite ne peut les
séparer.

### Poser des points

- **Choisir la classe** avec le bouton *Classe active* (bascule *Noir* / *Blanc*).
- **Clic sur une zone vide** : ajoute un point de la classe active.
- **Clic sur un point existant** : le supprime.
- **Jeu de démo** : charge un jeu de points prêt à l'emploi :

| Jeu | Séparable par une droite ? |
|-----|----------------------------|
| Cercle (noir au centre) | non |
| XOR (damier 2×2) | non — l'exemple historique qu'un perceptron seul ne sait pas résoudre |
| Deux lunes | non |
| Deux amas | oui — pour comparer avec le perceptron simple |

- **Part de points de test** (0 à 50 %, 20 % par défaut) : une partie des
  points est mise de côté. Le réseau ne les voit **jamais** pendant
  l'apprentissage ; on mesure dessus s'il sait **généraliser**, c'est-à-dire
  bien classer des points qu'il n'a jamais vus. Sur le plan, les points
  d'entraînement sont des **ronds**, les points de test des **carrés**.

### Régler le réseau

- **Entrées** : ce que le réseau reçoit de chaque point. Par défaut `x` et `y`
  (ses coordonnées) ; on peut ajouter des entrées **calculées** : `x²`, `y²`,
  `x·y`. De bonnes entrées peuvent remplacer des couches entières : avec `x²`
  et `y²`, un seul neurone sépare le cercle ; avec `x·y`, un seul neurone
  résout le XOR.
- **Couches cachées** : nombre de couches cachées (1 à 4, 2 par défaut).
- **Neurones par couche** : taille de chaque couche cachée (1 à 32).
- **Activation des neurones cachés** :

| Activation | Réponse | À retenir |
|------------|---------|-----------|
| tanh | de −1 à +1, en douceur | bon choix par défaut, frontières arrondies |
| ReLU | 0 d'un côté, proportionnelle de l'autre | l'activation des grands réseaux actuels ; frontières en morceaux de droites ; un neurone peut « mourir » (rester éteint partout) |
| sigmoïde | de 0 à 1, en douceur | historique ; apprend nettement plus lentement |

  Changer les entrées, les couches, les neurones ou l'activation reconstruit
  un réseau neuf.
- **Afficher la frontière de chaque neurone de la 1re couche** : trace les
  frontières « élémentaires » que le réseau combine pour former sa courbe —
  des droites en pointillés avec `x` et `y` seuls, des courbes avec `x²`, `y²`
  ou `x·y`.
- **Nouveaux poids aléatoires** : garde les points mais repart d'un nouveau
  tirage de poids (utile si le réseau reste coincé).

### Entraîner

- **Entraîner / Pause** : démarre ou suspend l'apprentissage. La courbe rouge se
  redessine à chaque étape : on la voit se déformer jusqu'à épouser les classes.
- **Pas à pas** : exécute **une seule itération** par clic — idéal pour analyser
  finement chaque mise à jour des poids.

> **Qu'est-ce qu'une itération ?** Ici, une itération = une passe complète sur
> **tous** les points : on accumule les corrections souhaitées, puis on applique
> **une seule** mise à jour de tous les poids (descente de gradient *batch*).

- **Vitesse** : nombre d'itérations appliquées par image (animation plus ou
  moins rapide).
- **Learning rate** : amplitude de chaque correction. Trop grand → ça oscille ;
  trop petit → ça converge lentement. À expérimenter !

### Réinitialiser

- **Reset** : efface tous les points et tire de nouveaux poids.
- **Démo** : recharge le jeu de démonstration sélectionné.

### Lire le canvas

- La **courbe rouge** est la frontière de décision (p = 0.5).
- Les **zones colorées** indiquent la classe prédite (rouge = noir, bleu =
  blanc). Plus la couleur est franche, plus le réseau est sûr de lui ; elle
  s'efface près de la frontière, là où il hésite.

### Lire la courbe de perte

Sous le plan, la **perte** est tracée au fil des itérations, pour les points
d'entraînement (vert d'eau) et pour les points de test (jaune). Survoler la
courbe affiche les valeurs d'une itération.

- Les deux courbes descendent ensemble : le réseau apprend **et** généralise.
- La courbe d'entraînement descend pendant que celle de test **remonte** : le
  réseau apprend ses points par cœur, c'est le **surapprentissage**.
- Une courbe en dents de scie : le *learning rate* est trop grand.
- Un long plateau : le réseau est coincé (nouveaux poids, plus de neurones,
  autre activation…).

### Lire le schéma du réseau

À droite du plan, le réseau est dessiné couche par couche : une boule par
neurone, un trait par poids (rouge = positif, bleu = négatif, plus épais =
plus fort). Dans chaque boule, une mini-carte montre la réponse du neurone
sur tout le plan.

- **Survoler le plan** : on suit le point sous la souris à travers le réseau ;
  chaque neurone s'allume de la couleur de sa réponse.
- **Survoler un neurone** : sa formule s'affiche sous le schéma, et sa
  frontière est surlignée sur le plan — une **droite** pour la 1re couche,
  déjà une **courbe** pour les couches suivantes.

### Lire l'état du modèle

| Champ | Signification |
|-------|---------------|
| Architecture | `2 → 8 → 8 → 1` : 2 entrées, la taille de chaque couche cachée, 1 sortie |
| Paramètres | nombre de poids et de biais ajustés par l'apprentissage |
| Itérations | nombre de mises à jour effectuées |
| Perte | erreur moyenne « continue » (entropie croisée), sur les points d'entraînement et sur les points de test : elle baisse tant que le réseau gagne en confiance, même quand il ne fait plus d'erreur |
| Erreurs | points mal classés sur le nombre de points, pour chaque lot (ex. `1 / 10`) |

L'apprentissage a réussi quand les erreurs d'**entraînement** tombent à 0. Le
réseau **généralise** bien si les erreurs de **test** restent faibles, elles
aussi.

## Petites expériences à proposer

1. Lance l'entraînement sur le **Cercle** : la courbe se referme autour des
   points noirs et `Erreurs` descend jusqu'à 0.
2. Passe à **1 couche** de **1 neurone** et relance : le réseau redevient un
   perceptron, la frontière redevient une **droite**… et le cercle n'est plus
   soluble. Monte progressivement à 2, 3, 4 neurones : à partir de combien la
   courbe parvient-elle à entourer le disque ?
3. Coche **Afficher la droite de chaque neurone de la 1re couche** pendant
   l'entraînement : la courbe rouge est construite à partir de ces droites.
4. Essaie **XOR** et **Deux lunes**. Ajoute ensuite des points à la main pour
   créer ta propre forme.
5. Sur **Deux lunes**, compare **1 couche de 8** et **3 couches de 8** : le
   réseau profond converge en général bien plus vite. Survole les neurones
   des couches 2 et 3 : leurs frontières sont déjà courbes.
6. Avec **4 couches de 32 neurones**, ajoute un point noir isolé au milieu des
   blancs : le réseau finit par creuser une petite « poche » rien que pour
   lui. C'est le **surapprentissage** : il colle aux exemples plutôt que de
   généraliser.
7. Augmente le *learning rate* au maximum : la courbe devient instable.
8. **Surapprentissage mesuré** : sur **Deux lunes**, mets la part de test à
   **30 %**, prends **3 couches** en **ReLU** et entraîne longtemps. Les
   erreurs d'entraînement tombent à 0, mais sur la courbe de perte, celle du
   test finit souvent par **remonter**.
9. **Entrées calculées** : sur le **Cercle**, ne garde que les entrées `x²` et
   `y²`, avec **1 couche de 1 neurone** : il suffit ! Coche l'affichage des
   frontières : la « droite » de ce neurone est une ellipse. Sur le **XOR**,
   essaie l'entrée `x·y` seule.
10. **Activations** : sur le **Cercle**, compare tanh, ReLU et sigmoïde avec
    le même réseau. Regarde la forme de la frontière (arrondie ou en
    polygone) et la vitesse de descente de la perte. En ReLU, repère dans le
    schéma les neurones restés blancs partout : ils sont « morts ».

## Comment ça marche (sous le capot)

À chaque itération, le réseau parcourt **tous** les points, accumule le
gradient (la somme des corrections souhaitées), puis applique **une seule**
mise à jour de tous les poids. Pour les neurones cachés, dont on ne connaît pas
la « bonne réponse », on **rétropropage** l'erreur de sortie, couche par
couche, de la sortie vers les entrées :

```text
pour chaque point (classe attendue t) :
  propagation avant :   les entrées (x, y, et éventuellement x², y², x·y),
                        puis chaque neurone j d'une couche cachée calcule
                        a_j = f(Σ_i w_ji·a_i + b_j)      (f : tanh, ReLU ou sigmoïde)
                        et enfin la sortie p = sigmoïde(Σ_i v_i·h_i + bo)
  erreur de sortie :    e = t − p                        (entre -1 et 1)
  pour chaque couche, de la sortie vers la 1re couche cachée :
    règle du perceptron :  g_w_ji += e_j × a_i
                           g_b_j  += e_j
    rétropropagation :     e_i = (Σ_j e_j × w_ji) × pente de f en a_i
                           (part de l'erreur qui revient au neurone i
                            de la couche précédente ; la pente vaut
                            1 − a_i² pour tanh, 1 ou 0 pour ReLU,
                            a_i × (1 − a_i) pour la sigmoïde)
puis, une seule fois (N = nombre de points d'entraînement) :
                        chaque poids += learning_rate × g / N
```

C'est une descente de gradient sur l'entropie croisée. Pour dessiner la
frontière, qui n'a pas d'équation simple, on mesure `p` sur une grille de
points du canvas et on trace la ligne de niveau `p = 0.5` par l'algorithme des
*marching squares* (comme une courbe de niveau sur une carte).

## Organisation du code

```text
tp1/
├─ index.html          # structure de la page et des contrôles
├─ vite.config.ts      # configuration Vite (root = tp1/)
└─ src/
   ├─ main.ts          # point d'entrée : clics, boucle d'animation
   ├─ mlp.ts           # le réseau : propagation avant et rétropropagation
   ├─ features.ts      # les entrées possibles : x, y, x², y², x·y
   ├─ chart.ts         # la courbe de perte (entraînement et test)
   ├─ geometry.ts      # conversions pixels ↔ [-1, 1], droites, tracé de courbe
   ├─ state.ts         # état partagé + jeux de démonstration
   ├─ renderer.ts      # dessin du plan (régions, courbe, points)
   ├─ diagram.ts       # dessin du schéma du réseau (neurones, poids)
   ├─ ui.ts            # câblage des boutons/sliders et affichages
   └─ style.css        # mise en page
```

Le code est abondamment commenté en français, à visée pédagogique : commence par
[`src/mlp.ts`](./src/mlp.ts) pour comprendre l'algorithme.
