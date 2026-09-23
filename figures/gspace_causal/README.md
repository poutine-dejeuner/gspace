# G-space → J-space : Analyses causales

Quatre expériences pour établir la relation entre l'espace de calcul géométrique
(G-space, via crosscoder) et l'espace de lecture Jacobienne (J-space, via
Jacobian lens).

**Modèle** : Qwen3.5-27B, addition modulaire mod 59, couches 48-62.

---

## Expérience 1 : Ablation causale

![exp1_ablation](gspace_causal/exp1_ablation.png)

**Question** : Si on retire les features G-space les plus importantes, le J-space
rank se dégrade-t-il ?

**Méthode** : On classe les 4096 features du crosscoder par |coefficient| de
régression linéaire vers la réponse (plus le coefficient est grand, plus la
feature est importante pour prédire la réponse). On retire progressivement les
top-N features de l'activation au layer 61, et on mesure le J-space rank
complet (tous les sub-tokens de la réponse) sur l'activation nettoyée.

**Résultat** : Même en retirant les 4096 features, le rank passe seulement de
3.50 à 3.53 (Δ = +0.03). Le top-1 passe de 20.7% à 20.5%.

**Interprétation** : L'ablation d'une seule couche ne suffit pas. Soit
l'information de calcul est redondante à travers les couches (le crosscoder
lit 16 couches et les features sont partagées), soit les features importantes
pour la régression ne sont pas celles qui encodent causalement le calcul.
L'hypothèse de redondance inter-couche est la plus probable : le crosscoder
a été entraîné sur 16 couches simultanément, donc l'information de calcul
est distribuée.

---

## Expérience 2 : Mapping G-space ↔ J-lens

![exp2_mapping](gspace_causal/exp2_mapping.png)

**Question** : Les features G-space sont-elles alignées avec des tokens
spécifiques du J-space ?

**Méthode** : Pour chaque feature crosscoder, on calcule son cosinus avec
chaque direction J-lens des 10 digits (0-9). On obtient une matrice
d'alignement [4096 × 10].

- **Panneau 1** (heatmap) : alignement des top-50 features avec les 10 digits
- **Panneau 2** (barres) : meilleur alignement trouvé pour chaque digit
- **Panneau 3** (scatter) : importance de la feature vs son alignement max
- **Panneau 4** (histogramme) : distribution des alignements, toutes features
  vs top-200

**Résultat** :
- Alignement moyen = 0.024, max absolu = 0.106
- Aucune feature top-200 n'a de cos > 0.15 avec un digit
- Corrélation importance/alignement : r = -0.03

**Interprétation** : G-space et J-space sont quasi orthogonaux même au niveau
des features individuelles. Les features qui prédisent le mieux la réponse ne
sont pas plus alignées avec les J-lens que les features aléatoires. G et J
sont deux représentations complètement différentes de la même information —
G encode le calcul, J encode la projection vers les tokens. Le fait qu'ils
soient décorrélés est cohérent avec l'hypothèse que G est un espace de calcul
interne et J un espace de lecture.

---

## Expérience 3 : Évolution couche par couche

![exp3_evolution](gspace_causal/exp3_evolution.png)

**Question** : G et J émergent-ils simultanément ou avec un délai ?

**Méthode** : Pour chaque couche l ∈ [48, 62], on mesure G-R² et J-space rank.
- **Panneau 1** : normalisation [0,1] des deux métriques — visualise l'émergence
  relative.
- **Panneau 2** : amélioration du J-rank par couche (barres orange) et norme
  de Frobenius du décodeur crosscoder (courbe verte).
- **Panneau 3** : trajectoire (G-R², J-rank) — chaque point est une couche,
  la flèche montre la progression.

**Résultat** :
- **Spearman r = 0.979** (p < 0.0001) entre G-R² et 1/J-rank
- G-R² : 0.37 (L48) → 0.93 (L61)
- J-rank : 7.3 (L48) → 3.5 (L61)
- La norme du décodeur est constante (64.0) à travers toutes les couches
- Aucun délai visible : G et J émergent ensemble

**Interprétation** : G et J évoluent de façon quasi parfaitement corrélée.
Deux lectures possibles :
1. G→J est un transfert instantané (même couche), donc pas de délai mesurable
2. G et J sont deux lectures parallèles d'un même calcul sous-jacent dans le
   residuel — le calcul émerge dans le residuel, et les deux méthodes le lisent
   simultanément.

La décorrélation géométrique (exp 2) + la corrélation fonctionnelle (exp 3)
est le résultat le plus intéressant : G et J portent la même information
calculatoire mais dans des directions orthogonales du residuel.

---

## Expérience 4 : Intervention

![exp4_intervention](gspace_causal/exp4_intervention.png)

**Question** : Si on modifie les features G-space pour encoder une réponse
différente, J-space lit-il la nouvelle réponse ?

**Méthode** : Pour 200 échantillons, on calcule la modification nécessaire
des features G-space (top-50 par |coefficient|) pour que la régression
linéaire prédise une réponse cible au lieu de la vraie. On modifie
l'activation en conséquence (via le décodeur crosscoder), et on mesure
le J-space rank de la vraie réponse vs la cible.

- Barres bleues : condition originale (vraie réponse rank ~3.8, cible ~6.5)
- Barres rouges : après intervention
- Barres grises : contrôle (perturbation aléatoire des features)

**Résultat** : Effet net = 0.00. L'intervention ne change rien au J-space rank.

**Interprétation** : Même conclusion que l'ablation — modifier les features
d'une seule couche ne suffit pas. Le calcul est distribué à travers les
couches. Une intervention multi-couche (modifier les features aux couches
60, 61, 62 simultanément) serait nécessaire, mais nécessite de propager
l'intervention à travers le residuel, ce qui est non trivial.

---

## Conclusion globale

| Expérience | Question | Réponse |
|---|---|---|
| 1. Ablation | G → J causal ? | Non démontrable sur une couche |
| 2. Mapping | G aligné avec J ? | Non, cos ~0.02, décorrélés |
| 3. Évolution | G et J simultanés ? | Oui, Spearman r = 0.98 |
| 4. Intervention | Injection G → lecture J ? | Échec sur une couche |

**G ⊥ J géométriquement mais G ∥ J fonctionnellement.** C'est le résultat
central : les deux espaces sont orthogonaux dans le residuel (cos ~0.02)
mais portent la même information calculatoire (corrélation 0.98). Cela
suggère que le calcul émerge dans le residuel de façon distribuée, et que
G et J en sont deux projections complémentaires — G capture la structure
interne du calcul, J capture sa projection vers les tokens de sortie.
