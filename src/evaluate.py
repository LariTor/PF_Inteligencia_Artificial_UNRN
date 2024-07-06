import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import DiabetesDataset
from model import DiabetesModel

# Cargar el dataset
data = pd.read_csv('../data/diabetes.csv')


# Separar características y etiquetas
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear datasets y dataloaders
test_dataset = DiabetesDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Crear el modelo
model = DiabetesModel(X_train.shape[1])
model.load_state_dict(torch.load('../diabetes_model.pth'))
model.eval()

# Evaluar el modelo
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        preds = outputs.round()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{cm}')
