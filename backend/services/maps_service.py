import logging
import math
import os
import urllib.parse

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
_SEARCH_RADIUS_M = 50_000  # 50 km


def get_maps_link(place_name: str, city: str = "India") -> str:
    """Return a Google Maps search URL for the given place name."""
    query = urllib.parse.quote_plus(f"{place_name} {city}")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approximate distance in km between two lat/lng points."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return round(2 * math.asin(math.sqrt(a)) * 6371, 1)


async def find_nearest_cancer_center(lat: float, lng: float) -> list[dict]:
    """Find up to 3 nearest cancer screening hospitals within 50 km.

    Returns a list of dicts with keys: name, address, distance, maps_link.
    Returns an empty list on API errors.
    """
    params = {
        "location": f"{lat},{lng}",
        "radius": _SEARCH_RADIUS_M,
        "keyword": "cancer screening hospital",
        "key": _MAPS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_PLACES_NEARBY_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        places = data.get("results", [])[:3]
        centers = []

        for place in places:
            name = place.get("name", "Unknown")
            address = place.get("vicinity", "")
            loc = place.get("geometry", {}).get("location", {})
            place_lat = loc.get("lat", lat)
            place_lng = loc.get("lng", lng)
            distance_km = _haversine_km(lat, lng, place_lat, place_lng)

            maps_link = (
                "https://www.google.com/maps/search/?api=1&query="
                + urllib.parse.quote_plus(f"{name} {address}")
            )
            centers.append(
                {
                    "name": name,
                    "address": address,
                    "distance": f"{distance_km} km",
                    "maps_link": maps_link,
                }
            )

        logger.info("Found %d cancer centers near (%.4f, %.4f)", len(centers), lat, lng)
        return centers

    except Exception as exc:
        logger.error("Maps API error: %s", exc)
        return []
