from dynaconf import Dynaconf

# Récupère le dictionnaire contenant les informations de configuration
settings = Dynaconf(
    envvar_prefix="DYNACONF", settings_files=["settings.toml", ".secrets.toml"]
)
