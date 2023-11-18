#!/usr/bin/env python3
from fastapi import FastAPI

from valorization.food_scribe.food_scribe import router_food_scribe
from valorization.rungis_wizz.rungis_wizz import router_rungis_wizz

app = FastAPI(
    title="CO2 vehicles",
    description="CO2 vehicles vous informe sur les émissions des véhicules européens",
    version="0.0.1",
    terms_of_service="https://github.com/l-herblot/rncp-ol/blob/main/conditions_utilisation.md",
    contact={
        "name": "Assistance",
        "url": "https://github.com/l-herblot/rncp-ol/",
        "email": "assistance@co2vehicles.fr",
    },
)

app.include_router(router_rungis_wizz, prefix="/rungis", tags=["Rungis Wizz"])
app.include_router(router_food_scribe, prefix="/scribe", tags=["Food Scribe"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
