# carregando_data.py

from datasets import load_dataset

# 1. Primeiro, carregue o dataset e atribua à variável 'ds'
ds = load_dataset("dorsar/lung-cancer")

# 2. Agora que 'ds' existe, você pode usá-la
print(ds)
primeiro_exemplo = ds['train'][0]
print(primeiro_exemplo)