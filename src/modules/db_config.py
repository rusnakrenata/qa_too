import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use environment variables if set, otherwise use defaults
DB_USER = os.getenv("DB_USER", "trafficOpti")
DB_PASSWORD = os.getenv("DB_PASSWORD", "P4ssw0rd")
DB_HOST = os.getenv("DB_HOST", "147.232.204.254")
DB_NAME = os.getenv("DB_NAME", "trafficOptimization")

# Choose driver (mysqldb or pymysql)
DB_DRIVER = os.getenv("DB_DRIVER", "mysql+mysqldb")

CONNECTION_URL = f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(
    CONNECTION_URL,
    pool_recycle=3600,  # seconds (before wait_timeout)
    pool_pre_ping=True
)

# Session factory and helper
Session = sessionmaker(bind=engine, autocommit=False)
def get_session():
    return Session() 