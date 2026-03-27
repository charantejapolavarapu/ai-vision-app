from fastapi import HTTPException
from database import create_user, get_user

def register(username, password):
    create_user(username, password)
    return {"message": "User registered"}

def login(username, password):
    user = get_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}