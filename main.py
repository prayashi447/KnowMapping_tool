from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from auth import hash_password, verify_password, create_token
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi import Body
from datasets import load_datasets

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.post("/register")
def register(username: str, email: str, password: str, db: Session = Depends(get_db)):

    # Check if user exists
    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pwd = hash_password(password)

    new_user = models.User(
        username=username,
        email=email,
        hashed_password=hashed_pwd
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}
@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):

    user = db.query(models.User).filter(models.User.username == username).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": username})

    return {"access_token": token}



@app.post("/load-dataset")
def load_dataset(
    sources: List[str] = Body(...),
    topic: str = Body(...)
):
    return load_datasets(sources, topic)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
