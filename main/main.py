import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
from fastapi.responses import HTMLResponse
import pathlib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


transformer_1 = joblib.load('model/Oby_Transformer_1.joblib')
transformer_2 = joblib.load('model/Oby_Transformer_2.joblib')
encoder = joblib.load('model/Oby_Encoder.joblib')
scaler = joblib.load('model/Oby_Scaler.joblib')
model = joblib.load('model/Oby_Model.joblib')


@app.get("/", response_class=HTMLResponse)
def get_home():
    html_path = pathlib.Path("FrontEnd/front.html").read_text()
    return HTMLResponse(content=html_path)

class User(BaseModel):
    Gender:Literal['Male','Female']
    Age:float
    Height:float
    Weight:float
    Family:Literal['yes','no']
    FAVC:Literal['yes','no']
    FCVC:float
    NCP:float
    CAEC:Literal['Sometimes','Frequently','Always','no']
    SMOKE:Literal['yes','no']
    CH2O:float
    SCC:Literal['yes','no']
    FAF:float
    TUE:float
    CALC:Literal['Sometimes','Frequently','Always','no']
    MTRANS:Literal['PublicTransportation','Walking','Automobile','Motorbike','Bike']



@app.post('/predict')
def predict(user: User):

    df = pd.DataFrame([user.model_dump()])

    
    df['Age'] = transformer_1.transform(df[['Age']])
    df['NCP'] = transformer_2.transform(df[['NCP']])

    
    object_cols = ['Gender', 'Family', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    df[object_cols] = encoder.transform(df[object_cols])

    
    df = scaler.transform(df)

    
    map = {0:'Insufficient Weight',1:'Normal Weight',2:'Obesity Type_I',
    3:'Obesity Type_II',4:'Obesity Type_III',5:'Overweight Level_I',6:'Overweight Level_II'}
    prediction = model.predict(df)[0]
    prediction = int(prediction) 
    return {
        "prediction": map[prediction]
    }
