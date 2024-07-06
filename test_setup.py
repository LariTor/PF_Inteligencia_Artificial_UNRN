import torch
import pandas as pd
import sklearn

# Verificar versiones de las bibliotecas
print("PyTorch version:", torch.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Verificar si PyTorch puede usar la GPU
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

# Crear un tensor de ejemplo
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)

