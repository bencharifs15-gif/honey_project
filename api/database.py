import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# =====================================
# DATABASE URL (Render PostgreSQL)
# =====================================
DATABASE_URL = "postgresql://honey_project_user:AYQ2RzeTCXtUQz2Sw4zfyRRwwzmAwajR@dpg-d7eft1f41pts73aln5b0-a.oregon-postgres.render.com/honey_project"

# =====================================
# Fix for compatibility (safety)
# =====================================
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

# =====================================
# Engine (connection to DB)
# =====================================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300
)

# =====================================
# Session (used to talk to DB)
# =====================================
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# =====================================
# Base (models inherit from this)
# =====================================
Base = declarative_base()


# =====================================
# Dependency helper (optional but useful)
# =====================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()