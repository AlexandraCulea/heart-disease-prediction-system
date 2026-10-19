import os
from dotenv import load_dotenv 
from sqlalchemy import Column, Integer, Float, String, DateTime, create_engine
from sqlalchemy.orm import sessionmaker 
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()
database_url=os.getenv('DATABASE_URL')
engine=create_engine(database_url)
session=sessionmaker(autocommit=False, autoflush=False, bind=engine)
base=declarative_base() #ORM - object relational mapping - maparea obiectelor din cod cu tabelele din baza de date

class Pacient(base):
    __tablename__="pacienti"
    id=Column(Integer, primary_key=True, index=True)
    nume=Column(String, nullable=False)
    varsta=Column(Float)
    sex=Column(String)
    tensiune=Column(Float)
    colesterol=Column(Float)
    puls=Column(Float)
    diagnostic_prezis=Column(Integer)
    data_inregistrare=Column(DateTime, nullable=False)

def init_db():
    base.metadata.create_all(bind=engine)

def get_db():
    db=session()
    return db

