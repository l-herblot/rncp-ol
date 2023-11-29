import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.data import router_data
from api.forecast_electric import router_forecast_electric

from helpers.network import get_public_ip

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

origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:5000",
    f"http://{get_public_ip()}",
    f"http://{get_public_ip()}:80",
    f"http://{get_public_ip()}:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router_forecast_electric, prefix="/forecast", tags=["Forecast electric"]
)
app.include_router(router_data, prefix="/data", tags=["Data retrieval"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
