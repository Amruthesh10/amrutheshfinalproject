"""API routes for the recommendation system."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.get("/")
def root() -> dict:
    return {"message": "Recommendation backend is running"}
