# Exploration de l'extension Budget-Constrained
-> Thompson Sampling avec/sans Contrainte Budgétaire et avec/sans contexte

Ce projet propose un simulateur permettant de comparer différentes variantes de l'algorithme **Thompson Sampling** sous une contrainte de budget et avec ou sans le contexte.

---

## Installation et Setup

1. **Cloner le projet**

   ```bash
        git clone [https://github.com/votre-utilisateur/votre-repo.git](https://github.com/votre-utilisateur/votre-repo.git)
        

2. **Créer l'environnement virtuel**

   ```bash
        python -m venv venv
        # Windows
        .\venv\Scripts\activate
        # macOS/Linux
        source venv/bin/activate

3. **Installer les dépendences**

   ```bash
        pip install -r requirements.txt

## Structure du projet
├── output/                 # Stockage des résultats (un dossier créé par run)
├── resources/              # Datasets utilisés pour les simulations
├── src/
│   ├── algorithms/         # Implémentations des algos (CTS.py, TS.py, TSBudget.py et CTSBudget.py)
│   ├── data_management/    # Chargement et traitement des données (dataloader.py)
│   ├── process/            # Logique d'exécution de la simulation (simulator.py)
│   └── reporting/          # Génération des résultats et métriques (report_generator.py et results_storer.py)
├── main.py                 # Point d'entrée principal du programme
└── requirements.txt        # Liste des dépendances Python

## 🛠️ Réglage des paramètres importants

Les paramètres principaux de la simulation et des algorithmes se configurent dans leurs fichiers respectifs.

### Paramètres globaux (`simulator.py`)

| Paramètre | Description |
| :--- | :--- |
| `self.dataset_name` | Définit le nom du dataset à charger depuis `resources/`. |
| `self.algorithm` | Sélectionne l'algorithme à exécuter (ex: TS, CTS, TSBudget...). |
| `self.horizon` | Définit le nombre total d'itérations pour la simulation. |

### Paramètres des algorithmes

| Algorithme | Fichier | Paramètres spécifiques à ajuster |
| :--- | :--- | :--- |
| **Standard TS** | `TS.py` | `self.threshold` |
| **Contextual TS** | `CTS.py` | `self.v`, `self.threshold` |
| **Budget TS** | `TSBudget.py` | `self.budget`, `self.threshold` |
| **Contextual Budget TS** | `CTSBudget.py` | `self.v`, `self.budget`, `self.threshold` |


---

## Évaluation et métriques

À la fin de chaque exécution, le simulateur génère un rapport de performance basé sur les métriques suivantes :

* **Temps d'exécution (running time)** : Durée totale de la simulation.
* **Précision (accuracy)** 
* **Regrets cumulés (cumulated regrets)** 
* **Prix cumulé (cumulated price)** : Dépense totale engagée au cours de la simulation.
* **Récompenses cumulées (cumulated rewards)** : Nombre total de click obtenu.
* **CPC (Cost Per Click)** : Coût moyen par récompense obtenue.

## Run

   ```bash
        python main.py
        
