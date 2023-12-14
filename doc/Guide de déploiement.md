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

## Déploiement du serveur

### Variété de configurations
Bien entendu, selon le serveur choisi, il n'est pas possible de détailler le déploiement sur chaque environnement.
Nous l'aborderons ainsi dans les grandes lignes.

### Déploiement sur serveur unitaire
Pour déployer un serveur sur une machine "locale" il est nécessaire de paramétrer :
- Le pare-feu, en créant des exceptions pour les ports 80 et 8000, afin d'autoriser le trafic entrant et sortant.
- Le routeur (ou box), en créant une redirection des ports 80 et 8000 du routeur sur les mêmes ports du serveur.

### Déploiement sur le cloud
Pour déployer un serveur cloud il est nécessaire de :
- Créer une base de données distribuée (il est conseillé de créer au moins une replica en parallèle). Nous avons testé et validé PostgreSQL sur serveur Azure Flexible.
- Créer une unité de stockage dans laquelle sera stockée l'image Docker.
- Créer un environnement d'exécution distribué de style Azure Web Apps pour y faire tourner l'application déployée.

### Nom de domaine
Dans les deux cas un nom de domaine rédirigeant vers l'adresse IP de votre serveur sera nécessaire pour permettre un accès simple à l'application pour les utilisateurs.

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
