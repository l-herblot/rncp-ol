from fastapi import APIRouter, HTTPException, Query

from models.lstm_electric import get_forecast

router_forecast_electric = APIRouter()


@router_forecast_electric.get("/")
async def retrieve_forecast(
    year_from: int = Query(2025, description="Année de départ des projections souhaitées pour les émissions de GES du mix électrique"),
    year_to: int = Query(2025, description="Année de fin des projections souhaitées pour les émissions de GES du mix électrique"),
):
    try:
        return get_forecast(year_from, year_to)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
