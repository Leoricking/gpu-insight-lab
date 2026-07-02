"""GPU Insight Lab - History Controller."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from storage.database import get_database


def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    return get_database().get_sessions(limit=limit)


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    return get_database().get_session(session_id)


def delete_session(session_id: int) -> bool:
    return get_database().delete_session(session_id)


def compare_sessions(a_id: int, b_id: int) -> Optional[Dict[str, Any]]:
    return get_database().compare_sessions(a_id, b_id)


def save_session(session: Any) -> Optional[int]:
    return get_database().save_session(session)
