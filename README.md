# Multimedia Gallery — Mini-projet de recherche d’images

Application web de gestion et de recherche d’images développée avec Flask dans le cadre du mini-projet Multimedia Mining and Indexing.

## 1. Fonctionnalités

- Inscription, connexion et gestion de session utilisateur.
- Importation, affichage, téléchargement et suppression d’images.
- Organisation des images par catégories personnelles.
- Transformation des images : recadrage et redimensionnement.
- Extraction et visualisation de descripteurs visuels :
  - **Couleur** : histogrammes HSV, moments statistiques et couleurs dominantes.
  - **Texture** : propriétés GLCM et histogramme LBP.
- Segmentation automatique pour analyser l’objet ou l’arrière-plan.
- API REST pour calculer les descripteurs.
- Recherche d’images similaires dans :
  - le dataset GHIM-10K ;
  - la galerie personnelle de l’utilisateur.
- Filtrage par catégorie et retour de pertinence FRE (Feature Relevance Estimation).

## 2. Organisation du projet

```text
Multimedia_Gallery/
├── api/                 # Ressources de l’API REST
├── database/            # Base SQLite et index de caractéristiques
├── dataset/
│   └── GHIM-10K/        # Dataset organisé en catégories
├── generated/           # Images transformées ou générées
├── models/              # Modèles SQLAlchemy
├── routes/              # Routes Flask (authentification, galerie, recherche...)
├── services/            # Extraction, segmentation, similarité et FRE
├── static/              # CSS et JavaScript
├── templates/           # Pages HTML/Jinja
├── uploads/             # Images importées et images requêtes temporaires
├── app.py
├── config.py
├── extensions.py
└── requirements.txt
```

Les dossiers `venv/` et `__pycache__/` sont des éléments locaux d’exécution et ne sont pas nécessaires dans l’archive de livraison.

## 3. Prérequis

- Python 3.10 ou version ultérieure (le projet a été exécuté avec Python 3.13).
- pip.
- Le dataset GHIM-10K et son index de caractéristiques pré-calculées.

## 4. Installation

Ouvrir un terminal dans le dossier racine du projet.

### Windows — Git Bash

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Préparation des données

Placer le dataset de manière à obtenir cette structure :

```text
dataset/
└── GHIM-10K/
    ├── aircraft/
    ├── buildings/
    ├── butterfly/
    ├── cars/
    ├── dragon fly/
    ├── fireworks/
    ├── flowers/
    ├── hen/
    ├── horses/
    ├── insects/
    ├── motorcycles/
    ├── mountains/
    ├── sea shores/
    ├── ships/
    ├── sunset/
    ├── temples/
    ├── trees/
    ├── valleys/
    ├── walls/
    └── yacht/
```

Le fichier d’index utilisé par la recherche sur le dataset doit être présent ici :

```text
database/ghim10k_global_features.npz
```

Le fichier `database/gallery.db` contient la base SQLite de l’application. Si cette base n’est pas livrée, l’application crée les tables au démarrage ; il faudra alors créer un compte et importer des images.

### Télécharger GHIM-10K

Le dataset n’est pas inclus dans l’archive du projet afin de réduire sa taille. Il peut être téléchargé depuis Kaggle :

**Lien de téléchargement :**https://www.kaggle.com/datasets/mcpython/ghim-10k

Après téléchargement, décompresser les fichiers et organiser les dossiers pour obtenir exactement la structure suivante :

```text
Multimedia_Gallery/
└── dataset/
    └── GHIM-10K/
        ├── aircraft/
        ├── buildings/
        ├── butterfly/
        ├── cars/
        └── ... (les 20 catégories)
```

Le dossier `GHIM-10K` doit donc se trouver directement dans `dataset/`. Vérifier que les images sont bien présentes dans les dossiers de catégories.

**Important :** l’index `database/ghim10k_global_features.npz` fourni avec le projet a été calculé pour ce dataset. Il faut conserver les noms et l’organisation des fichiers/catégories attendus par le projet. Si le dataset téléchargé a une structure différente, réorganiser les dossiers sans renommer les catégories utilisées par l’index.

## 6. Lancement

Avec l’environnement virtuel activé :

```bash
python app.py
```

Ouvrir ensuite l’adresse suivante dans le navigateur :

```text
http://127.0.0.1:5000
```

## 7. API REST

Les endpoints suivants acceptent une image envoyée en multipart/form-data sous le champ `image`. Pour les analyses nécessitant un masque, le champ `mask` peut également être fourni.

| Méthode | Endpoint | Fonction |
|---|---|---|
| POST | `/api/color` | Descripteurs de couleur |
| POST | `/api/texture` | Descripteurs de texture |
| POST | `/api/features` | Ensemble des descripteurs |

## 8. Évaluation de la recherche et de FRE

Le projet comprend des scripts de test et un fichier de résultats expérimentaux :

```text
database/fre_20_categories_results.csv
```

Les résultats doivent être interprétés avec les précautions décrites dans le rapport : les exemples pertinents utilisés dans l’évaluation automatisée sont simulés à partir des catégories du dataset. Cette évaluation ne correspond donc pas à une campagne de retours utilisateurs réels.

## 9. Vérification de l’environnement

Pour vérifier les dépendances :

```bash
pip check
```

Le résultat attendu lorsque l’environnement est cohérent est :

```text
No broken requirements found.
```

## 10. Contenu conseillé pour l’archive de livraison

À inclure :
- le code source (`api/`, `models/`, `routes/`, `services/`, `static/`, `templates/`) ;
- `app.py`, `config.py`, `extensions.py`, `requirements.txt` ;
- l’index `database/ghim10k_global_features.npz` ;
- les résultats expérimentaux utiles ;
- le rapport et ce README.

À exclure :
- `venv/` ;
- les dossiers `__pycache__/` ;
- les fichiers temporaires et caches ;
- les images de test ou fichiers personnels dans `uploads/` et `generated/`, sauf si une démonstration précise les nécessite.

**Attention :** ne pas exclure `dataset/GHIM-10K/` ou l’index `.npz` si l’environnement de démonstration doit permettre la recherche sur GHIM-10K. Si le dataset est trop volumineux pour être remis, préciser dans le rapport ou le README comment le récupérer et où le placer.
