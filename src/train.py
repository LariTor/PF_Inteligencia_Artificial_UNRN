import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import DiabetesDataset
from model import DiabetesModel

# Cargar el dataset
data_path = '/home/lari/Documents/IA/PF_Inteligencia_Artificial_UNRN/data/diabetes.csv'
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
data = pd.read_csv(data_path, names=column_names, header=0)
data = pd.read_csv(data_path)  

# Imprime las columnas del DataFrame para verificar los nombres
#print("Columnas del DataFrame:", data.columns)

# Verifica si la columna 'Outcome' existe
if 'Outcome' in data.columns:
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values
    print("Datos cargados correctamente.")
else:
    print("La columna 'Outcome' no se encuentra en el DataFrame.")
    
# Verificar si hay valores nulos
if data.isnull().values.any():
    print("Advertencia: Hay valores nulos en los datos")
      
# Preparar datos
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Verificar si hay valores extremos
print("Estadísticas de los datos de entrada:")
print(data.describe())

# Estandarizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Verificar los tensores
print("Tamaño de X_train:", X_train.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_test:", y_test.shape)

# Definir modelo, pérdida y optimizador
model = DiabetesModel(X_train.shape[1])  # Pasar el número de características de entrada
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    
    # Verificar valores en outputs
    print(f'Epoch [{epoch+1}/{num_epochs}] - Valores de salida:', outputs.detach().numpy())
    
    loss = criterion(outputs, y_train)
    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss antes de backward: {loss.item():.4f}')
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluación
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()
    accuracy = (predicted.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f'Accuracy: {accuracy * 100:.2f}%')