"""HTTP application entry point."""

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


def create_app() -> FastAPI:
    app = FastAPI(title="ResolveLane API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


app = create_app()
