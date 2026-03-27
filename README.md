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
- Output : stockages des resultats (un dossier créer par run)
- Resources : dataset
- Src:
        - Algorithms
                - CTS.py (contextual thompson sampling)
                -...
        - data_management
                -dataloader.py
        - process
        - Reporting
- main.py
- requierements.txt

## Réglage des paramètres importants
simulator.py:
 self.dataset_name pour choisir le nom du dataset
 self.algorithm pour chosir l'algorithm
 self.horizon pour chosir le nombre d'itération

TS.py:
self.threshold

CTS.py:
self.v
self.threshold

TSBudget.py:
self.budget
self.threshold

CTSBudget.py:
self.v
self.budget
self.threshold
... 

## Run

   ```bash
        python main.py
        
        
