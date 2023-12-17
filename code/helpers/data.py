import string

categories_dict = {"M1": "Transport de personnes", "N1": "Transport de marchandises"}
countries_dict = {
    "DE": "Allemagne",
    "AT": "Autriche",
    "BE": "Belgique",
    "ES": "Espagne",
    "FR": "France",
    "HU": "Hongrie",
    "IT": "Italie",
    "NL": "Pays-Bas",
    "PL": "Pologne",
    "CZ": "République Tchèque",
    "SE": "Suède",
}
fuel_types_dict = {
    "Diesel/electric": "Hybride diesel",
    "Electric": "Electrique",
    "Petrol/electric": "Hybride essence",
}


def safe_string(unchecked_string, pattern="max"):
    """
    Retourne une chaîne de caractères "nettoyée"
    :param unchecked_string: la chaîne de caractères à nettoyer
    :param pattern: le filtre à appliquer (float, integer ou max)
    :return: la chaîne de caractères nettoyée
    """
    match pattern:
        case "float":
            pattern = string.digits + "."
        case "integer":
            pattern = string.digits
        case _:
            pattern = (
                string.ascii_lowercase + string.ascii_uppercase + string.digits + " .-/"
            )

    return "".join([c for c in unchecked_string if c in pattern])
