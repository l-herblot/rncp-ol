import json
import requests
from helpers.logger import logger


def get_public_ip():
    """
    Récupère l'adresse IP publique de la machine
    :return: une chaîne de caractères contenant l'adresse IP (ou vide si la récupération n'a pas fonctionné)
    """
    try:
        return json.loads(
            requests.get(
                "https://api.ipify.org/?format=json", timeout=10
            ).content.decode("utf8")
        )["ip"]
    except requests.RequestException:
        logger.error("Impossible de récupérer l'adresse IP publique")

    return ""
