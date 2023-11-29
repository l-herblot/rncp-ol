import json
import requests
from helpers.logger import logger


def get_public_ip():
    try:
        return json.loads(
            requests.get(
                "https://api.ipify.org/?format=json", timeout=10
            ).content.decode("utf8")
        )["ip"]
    except requests.RequestException:
        logger.error("Impossible de récupérer l'adresse IP publique")

    return ""
