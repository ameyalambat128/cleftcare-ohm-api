import os
from typing import Optional

from supabase import Client, create_client


_supabase_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """Return a cached Supabase client when configuration is available."""

    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        print("Supabase configuration missing; skipping Supabase client setup.")
        return None

    try:
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        print(f"Failed to initialize Supabase client: {exc}")
        return None

