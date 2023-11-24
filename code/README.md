# CO2 vehicles

## Introduction
Une application permettant de consulter des données sur les émissions de GES des véhicules électriques en Europe via :
- une API
- une interface web

## Environnement de développement
### Installation des dépendances
```bash
python.exe -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Exécution de l'API et du serveur web
### Exécution de l'API
```bash
cd api
uvicorn main:app --reload
```
### Exécution du serveur web
```bash
cd web
flask run
```

## Déploiement de l'application
### Création de l'image Docker
```bash
docker build --tag co2electric .
```

### Exécution dans un conteneur Docker
```bash
# Le site web sera accessible sur le port 80
# L'API sera accessible sur le port 8000
docker run -p 80:80 -p 8000:8000 co2electric
```

### Contrôle du conteneur
```bash
# Consulter les logs du conteneur
docker logs <ID_ou_nom_du_conteneur>

# Lancer une ligne de commande au sein du conteneur
docker exec -it <ID_ou_nom_du_conteneur> /bin/bash

# Arrêter l'exécution du conteneur
docker stop <ID_du_conteneur>

# Relancer l'exécution du conteneur
docker start <ID_ou_nom_du_conteneur>

# Supprimer le conteneur
docker rm <ID_du_conteneur>

# Si nécessaire, pour retrouver l'ID du conteneur
docker ps -a
```

## Structure
Les dossiers du projet sont organisés comme suit :
- api : fichiers de l'API
- config: fichiers de configuration
- data: fichiers de données divers
- helpers : scripts de fonctionnalités transverses
- models: modèles d'apprentissage automatique
- preparation: scripts de préparation initiale des données
- tmp: fichiers temporaires
- web : fichiers de l'interface web
