from fastapi import FastAPI

from api.forecast_electric import router_forecast_electric

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

app.include_router(router_forecast_electric, prefix="/forecast", tags=["Forecast electric"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
