# CO2 vehicles

## TO DO
python.exe -m pip install --upgrade pip

## Intro 
Une application permettant de consulter des données sur les émissions de GES des véhicules en Europe via :
- une API
- une interface web

## Environnement de développement
### Installation des dépendances
```bash
pip install -r requirements.txt
```

## Déploiement
### Déploiement de l'API
```bash
python api/main.py
```
### Déploiement de l'interface web
```bash
streamlit run web/main.py
```
## Structure
Les dossiers du projet sont organisés comme suit :
- api : fichiers de l'API
- config: fichiers de configuration
- data: fichiers de données divers
- helpers : scripts de fonctionnalités transverses
- models: modèles d'apprentissage automatique
- orm : fichiers de liaison ORM à la base de données
- preparation: scripts de préparation initiale des données
- tmp: fichiers temporaires
- web : fichiers de l'interface web
