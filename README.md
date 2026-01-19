# Description
Ce projet présente une étude progressive de la segmentation sémantique par Transformers. Il débute par l’analyse et l’expérimentation de la segmentation multi-classes à l’aide du modèle Segmenter, avant d’aborder un cas d’usage ciblé portant sur la segmentation de la classe personne en environnement intérieur. Cette démarche permet d’évaluer à la fois les capacités générales du modèle et son adaptation à une application spécifique.

## Objectifs du projet
- Comprendre les principes de la segmentation sémantique
- Étudier l’architecture du modèle **Segmenter**
- Expérimenter la segmentation multi-classes basée sur les Transformers
- Adapter l’approche à un cas d’usage ciblé (classe *personne*)
- Comparer une approche spécialisée avec un modèle universel (SAM3)



## Phase 1 : Segmentation sémantique multi-classes
Dans cette première phase, le projet s’est concentré sur l’étude et l’expérimentation de la segmentation sémantique générale.  
Le modèle Segmenter repose sur une architecture entièrement Transformer, composée de :

- **Vision Transformer (ViT)** :  
  L’image est découpée en patches, projetée dans un espace d’embedding et traitée par des couches Transformer afin de capturer des relations globales entre toutes les régions de l’image.

- **Mask Transformer** :  
  Les représentations issues du ViT sont décodées afin de produire une carte de segmentation où chaque pixel est associé à une classe sémantique.

Des tests ont été réalisés à partir d’un checkpoint pré-entraîné, permettant de générer des cartes de segmentation multi-classes cohérentes sur des scènes complexes.



## Phase 2 : Cas d’usage – segmentation de la classe personne
La seconde phase du projet vise un cas d’usage précis : la **segmentation de la classe personne dans des scènes intérieures**, un enjeu important pour la sécurité et la navigation en robotique domestique.

### Jeu de données
- Dataset issu de **ADE20K Indoor**
- Sélection des images contenant des personnes
- Simplification des annotations en deux classes :
  - personne
  - arrière-plan
- Prétraitement des images et des masques (redimensionnement, normalisation)



## Contraintes techniques
Le fine-tuning du modèle Segmenter officiel n’a pas pu être réalisé en raison de :
- dépendances lourdes (mmcv-full, MMSegmentation)
- incompatibilités entre PyTorch, CUDA et les ressources GPU disponibles
- difficultés à obtenir un environnement stable et reproductible

Face à ces contraintes, une **architecture Transformer simplifiée**, inspirée de Segmenter, a été implémentée afin de spécialiser le modèle sur la classe *personne*.



## Améliorations proposées
Afin d’améliorer la qualité des masques de segmentation, plusieurs améliorations ont été étudiées :
- **Attention locale** pour mieux capturer les détails fins et les contours
- **Approche multi-échelle** pour gérer les variations de taille des personnes
- Fusion des caractéristiques avant le décodage



## Comparaison avec SAM3
Une comparaison a été effectuée avec le modèle **SAM3** :
- **Segmenter (approche spécialisée)** :
  - segmentation supervisée
  - résultats plus précis pour la classe cible
  - modèle plus léger et adapté à une intégration embarquée

- **SAM3** :
  - segmentation universelle (zero-shot)
  - génération de nombreux masques non pertinents dans un cas d’usage ciblé
  - coût computationnel plus élevé

Cette comparaison montre qu’un modèle spécialisé est plus adapté lorsqu’une segmentation contrôlée et précise est requise.

## Technologies utilisées
- Python
- PyTorch
- Transformers (SegFormer)
- PIL, NumPy, Matplotlib



