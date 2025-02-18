# Visão Geral do Projeto de Predição de Transportes no Titanic Espacial

## Objetivo do Projeto

O objetivo deste projeto é construir um modelo de aprendizado de máquina para prever se os passageiros de uma nave espacial foram transportados ou não, baseado em características fornecidas. Os dados são provenientes da competição *Spaceship Titanic*, onde são apresentados dados de passageiros, incluindo informações sobre idade, destino, status de conforto e outros fatores.

## Etapas do Projeto

### 1. **Carregamento e Preparação dos Dados**
Primeiro, os dados de treinamento (`train.csv`) e de teste (`test.csv`) são carregados. Após isso, concatenamos os dois conjuntos em um único DataFrame, removemos colunas irrelevantes como 'Name' e 'PassengerId', e tratamos valores ausentes em diversas colunas.

```python
df_train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
df_test['Transported'] = False
df = pd.concat([df_train, df_test], sort=False)
df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
```

### 2. Tratamento de Dados Faltantes
Algumas colunas possuem valores ausentes. Estas lacunas são preenchidas com valores como 'U' para o planeta de origem ou usando a técnica KNNImputer para variáveis numéricas e categóricas.

```python
df['Deck'] = df['Deck'].fillna('U')
df['Num'] = df['Num'].fillna(-1)
df['Side'] = df['Side'].fillna('U')
```

### 3. Transformação de Variáveis Categóricas
Variáveis categóricas, como 'HomePlanet' e 'Destination', são transformadas em variáveis binárias usando o método one-hot encoding, para que possam ser usadas em modelos de aprendizado de máquina.

```python
df = pd.concat([df, pd.get_dummies(df['HomePlanet'], prefix='HomePlanet')], axis=1)
df = df.drop(columns=['HomePlanet'])
```

### 4. Engenharia de Atributos
Criação de novas variáveis a partir das colunas existentes para melhorar o desempenho do modelo. Por exemplo, combinamos diferentes colunas para representar o gasto total de cada passageiro com serviços a bordo.

```python
df['amt_spent'] = df[bill_cols].sum(axis=1)
```

### 5. Divisão dos Dados em Treinamento e Teste
Os dados são divididos em dois conjuntos: 80% para treinamento e 20% para teste. A variável Transported é a variável alvo.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. Treinamento do Modelo
Foram testados diversos modelos de aprendizado de máquina, como Regressão Logística, Árvore de Decisão, Random Forest, XGBoost e LightGBM. O melhor modelo foi selecionado com base na acurácia no conjunto de teste.

```python
model_5.fit(X_train, y_train)
```

### 7. Avaliação e Seleção do Melhor Modelo
Após treinar os modelos, a acurácia foi avaliada utilizando o conjunto de teste e o modelo com melhor desempenho foi selecionado para a previsão final.

### 8. Previsão e Submissão
O modelo final é utilizado para prever os valores de Transported no conjunto de dados de teste, e as previsões são salvas em um arquivo CSV para submissão.

```python
final.to_csv('submission.csv', index=False)
```

### 9. Salvar o Modelo
O modelo treinado é salvo em um arquivo .pkl usando o módulo pickle para uso futuro.

```python
with open("spaceship_titanic.pkl", 'wb') as model_file:
    pickle.dump(model_5, model_file)
```

Conclusão
Neste projeto, foi criado um pipeline completo de pré-processamento, treinamento e avaliação de modelos para prever se os passageiros do Titanic Espacial foram transportados ou não. O modelo treinado foi exportado e está disponível para uso posterior.

Tecnologias Utilizadas
- Python: Para processamento de dados e treinamento de modelos.
- pandas: Para manipulação de dados.
- scikit-learn: Para modelos de aprendizado de máquina.
- LightGBM, XGBoost: Modelos baseados em árvores de decisão com alta performance.
- pickle: Para salvar o modelo treinado.
