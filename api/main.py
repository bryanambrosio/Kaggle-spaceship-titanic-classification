import pickle
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from io import BytesIO

# Inicializa a API
app = FastAPI()

# Caminho do modelo no seu GitHub (garanta que seja o caminho correto)
model_url = "https://github.com/bryanambrosio/Kaggle-spaceship-titanic-classification/raw/main/models/spaceship_titanic.pkl"  # URL do modelo no seu repositório

# Carregar o modelo diretamente do GitHub
response = requests.get(model_url)
model = pickle.load(BytesIO(response.content))

# Definir a estrutura dos dados de entrada usando Pydantic
class PassengerData(BaseModel):
    PassengerId: int
    HomePlanet: str
    CryoSleep: bool
    Cabin: str
    Destination: str
    Age: float
    VIP: bool
    RoomService: float
    FoodCourt: float
    ShoppingMall: float
    Spa: float
    VRDeck: float

@app.get("/")
def read_root():
    return {"message": "API de previsão de passageiros do Spaceship Titanic está funcionando!"}

@app.post("/predict")
def predict(data: PassengerData):
    # Converter os dados de entrada em DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Realizar o pré-processamento necessário nos dados de entrada
    input_data = pd.get_dummies(input_data)  # Exemplo de transformação de variáveis categóricas
    
    # Garantir que as colunas de input_data e o modelo correspondam
    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0
    input_data = input_data[model.feature_names_in_]

    # Fazer a previsão
    prediction = model.predict(input_data)
    
    # Retornar a previsão
    return {"prediction": bool(prediction[0])}
