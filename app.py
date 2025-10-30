import torch
from PIL import Image
from torchvision import transforms
# from architecture import ResNetLungCancer # Descomente se 'architecture.py' estiver na sua pasta
import gradio as gr
from datasets import load_dataset # <-- 1. Importar a biblioteca

# ===============================================================
# Crie uma classe de modelo de exemplo se você não tiver o arquivo 'architecture.py'
class ResNetLungCancer(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetLungCancer, self).__init__()
        # Usando um resnet18 pré-treinado como exemplo
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
# ===============================================================

# Carregar o Dataset do Hugging Face
print("Carregando o dataset do Hugging Face...")
# Usamos 'streaming=True' para carregar os metadados mais rápido, sem baixar tudo de uma vez.
# E 'split' para pegar diretamente o conjunto de teste.
ds_test = load_dataset("dorsar/lung-cancer", split='test', streaming=True)
print("Dataset carregado!")

# Carregar o Modelo (seu código)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetLungCancer(num_classes=4)
# O seguinte comando vai dar erro se você não tiver o arquivo do modelo treinado.
# model.load_state_dict(torch.load('Model/lung_cancer_detection_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Preprocessing (seu código)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

def predict(image):
    # Gradio passa a imagem como NumPy array, então convertemos para PIL
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Cria um dicionário de confianças para o Gradio
    confidences = {class_names[i]: float(prob) for i, prob in enumerate(torch.nn.functional.softmax(output, dim=1)[0])}
    return confidences

# <-- 2. Criar a lista de exemplos a partir do dataset
print("Preparando exemplos para a interface...")
examples_from_dataset = []
# Pega as 5 primeiras imagens do conjunto de teste para usar como exemplo
for example in ds_test.take(5):
    # A imagem já vem como objeto PIL, perfeito para o Gradio
    image = example['image'] 
    examples_from_dataset.append([image])

# Interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=1, label="Previsão"),
    title="Detector de Câncer de Pulmão",
    description="Faça o upload de uma imagem de tomografia e o modelo irá prever o tipo de anomalia.",
    examples=examples_from_dataset # <-- 3. Usar a lista de exemplos do dataset
)

iface.launch(share=True)