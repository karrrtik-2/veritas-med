"""Production API layer for the Medical AI System."""

from api.server import create_app
from api.routes import register_routes

__all__ = ["create_app", "register_routes"]
