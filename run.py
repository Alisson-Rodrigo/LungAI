import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset # <-- 1. Importar a biblioteca
# from architecture import ResNetLungCancer # Descomente se 'architecture.py' estiver na sua pasta

# ===============================================================
# Carregar o Dataset do Hugging Face
# ===============================================================
print("Carregando o dataset do Hugging Face...")
ds = load_dataset("dorsar/lung-cancer")
ds_test = ds['test'] # Vamos usar o conjunto de teste para a previsão
print("Dataset carregado com sucesso!")


# ===============================================================
# Carregar o Modelo (seu código original)
# ===============================================================
# Crie uma classe de modelo de exemplo se você não tiver o arquivo 'architecture.py'
class ResNetLungCancer(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetLungCancer, self).__init__()
        # Usando um resnet18 pré-treinado como exemplo
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Ajusta a última camada para o número de classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetLungCancer(num_classes=4)
# O seguinte comando vai dar erro se você não tiver o arquivo do modelo treinado.
# model.load_state_dict(torch.load('Model/lung_cancer_detection_model.pth', map_location=device))
model = model.to(device)
model.eval()


# ===============================================================
# Preprocessing (seu código original)
# ===============================================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Obtém os nomes das classes do próprio dataset (melhor prática)
class_names = ds_test.features['label'].names
print(f"Nomes das classes: {class_names}")


# ===============================================================
# Loop de Previsão sobre o Dataset
# ===============================================================
print("\n--- Iniciando previsões no conjunto de teste ---")
# Fazendo a previsão para os 5 primeiros exemplos do conjunto de teste
for i in range(5):
    # 2. Pega um exemplo do dataset
    example = ds_test[i]
    image = example['image']
    true_label_idx = example['label']
    
    # Converte para RGB, pois alguns modelos não aceitam o canal Alpha (RGBA)
    image = image.convert('RGB')
    
    # 3. Aplica o mesmo pré-processamento
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 4. Obtém a previsão do modelo
    with torch.no_grad():
        output = model(input_tensor)

    predicted_idx = torch.argmax(output, dim=1).item()

    # Pega os nomes das classes para mostrar o resultado
    predicted_class_name = class_names[predicted_idx]
    true_class_name = class_names[true_label_idx]

    print(f"\nImagem {i+1}:")
    print(f">> Rótulo Verdadeiro: {true_class_name} (índice {true_label_idx})")
    print(f">> Previsão do Modelo: {predicted_class_name} (índice {predicted_idx})")