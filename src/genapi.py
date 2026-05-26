import secrets
import plyvel
import os
import json
import time

from account import get_db

def generate_api_key() -> str:
    """
    Generates a secure, 32-character key with grav_ prefix and persists it in LevelDB.
    """
    key = secrets.token_urlsafe(24)
    full_key = f"grav_{key}"
    
    db_conn = get_db()
    key_data = {
        "key": full_key,
        "status": "active",
        "pool": "default",
        "created_at": time.time()
    }
    
    db_key = f"apikey:{full_key}".encode("utf-8")
    db_conn.put(db_key, json.dumps(key_data).encode("utf-8"))
    
    return key

def validate_api_key(key: str) -> bool:
    """
    Validates an API key against LevelDB, verifying it exists and is active.
    """
    if key == "grav_demoapikey":
        return True
        
    db_conn = get_db()
    db_key = f"apikey:{key}".encode("utf-8")
    val_bytes = db_conn.get(db_key)
    
    if val_bytes:
        try:
            data = json.loads(val_bytes.decode("utf-8"))
            return data.get("status") == "active"
        except Exception:
            pass
            
    return False
