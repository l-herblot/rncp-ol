import requests

from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query

from config.settings import settings
from helpers.data import fuel_types_dict, safe_string
from helpers.db_pg2 import pg2_cursor

router_data = APIRouter()


@router_data.get("/")
async def retrieve_data(
    caller: str = Query(
        "country", description="Elément du formulaire ayant demandé les données"
    ),
    country: str = Query("FR", description="Code du pays"),
    make: str = Query("", description="Marque du véhicule"),
    cn: str = Query("", description="Modèle du véhicule"),
    ft: str = Query("", description="Carburation du véhicule"),
    year: str = Query("", description="Année eu véhicule"),
):
    try:
        match caller:
            case "country":
                if len(country) > 0:
                    pg2_cursor.execute(
                        f"SELECT DISTINCT make FROM {settings['PG_TABLE_VEHICLES_ELECTRIC']} WHERE country='{safe_string(country)}' ORDER BY make;"
                    )
                    makes = pg2_cursor.fetchall()
                    return {"makes": [make for make in makes]}
            case "make":
                if len(country) > 0 and len(make) > 0:
                    pg2_cursor.execute(
                        f"SELECT DISTINCT cn FROM {settings['PG_TABLE_VEHICLES_ELECTRIC']} WHERE country='{safe_string(country)}' AND make='{safe_string(make)}' ORDER BY cn;"
                    )
                    cns = pg2_cursor.fetchall()
                    return {"cns": [cn for cn in cns]}
            case "cn":
                if len(country) > 0 and len(make) > 0 and len(cn) > 0:
                    pg2_cursor.execute(
                        f"SELECT DISTINCT ft FROM {settings['PG_TABLE_VEHICLES_ELECTRIC']} WHERE country='{safe_string(country)}' AND make='{safe_string(make)}' AND cn='{safe_string(cn)}' ORDER BY ft;"
                    )
                    fts = pg2_cursor.fetchall()
                    return {"fts": [ft for ft in fts]}
            case "ft":
                if len(country) > 0 and len(make) > 0 and len(cn) > 0 and len(ft) > 0:
                    pg2_cursor.execute(
                        f"SELECT DISTINCT year FROM {settings['PG_TABLE_VEHICLES_ELECTRIC']} WHERE country='{safe_string(country)}' AND make='{safe_string(make)}' AND cn='{safe_string(cn)}' AND ft='{safe_string(ft)}' ORDER BY year;"
                    )
                    years = pg2_cursor.fetchall()
                    return {"years": [year for year in years]}
            case "year":
                if (
                    len(country) > 0
                    and len(make) > 0
                    and len(cn) > 0
                    and len(ft) > 0
                    and len(year) > 0
                ):
                    pg2_cursor.execute(
                        f"SELECT category, mro, co2e, length, width, ec, z FROM {settings['PG_TABLE_VEHICLES_ELECTRIC']} WHERE country='{safe_string(country)}' AND make='{safe_string(make)}' AND cn='{safe_string(cn)}' AND ft='{safe_string(ft)}' AND year={safe_string(year,'integer')} LIMIT 1;"
                    )
                    vehicle = pg2_cursor.fetchall()
                    return {"vehicle": [item for item in vehicle]}
            case _:
                return {
                    "error": "L'identifiant de l'appelant n'a pas été fourni ou n'est pas reconnu"
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_data.get("/thumbnail")
async def retrieve_thumbnail_url(
    vehicle: str = Query("", description="Véhicule à rechercher"),
):
    try:
        url = rf'https://www.google.com/search?client=firefox-b-d&sca_esv=584679428&q={vehicle.replace(" ", "+")}&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiNwfWvwdiCAxWfUKQEHa7PAlUQ0pQJegQIDBAB&biw=1920&bih=955&dpr=1'

        page = requests.get(url).text

        parse_result = BeautifulSoup(page, "html.parser")

        for img in parse_result.find_all("img"):
            url = img.get("src")
            if url and url.startswith("https://"):
                return {"thumbnail_url": url}

        return {"error": "Aucune miniature n'a pu être récupérée pour " + vehicle}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
