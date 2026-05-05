import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 기본 설정
num_classes = 150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = r"C:\vision\PokemonData"  # 포켓몬 데이터 폴더 경로


if os.path.exists(DATA_PATH):
    class_names = sorted(os.listdir(DATA_PATH))
else:
    class_names = [f"Pokemon_{i}" for i in range(num_classes)]

# 3. 전처리 정의 (학습 때와 동일해야 함)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. 모델 로드 함수
@st.cache_resource
def load_model(model_name):
    # 모델 구조 생성
    if model_name == "AlexNet":
        model = models.alexnet()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "GoogLeNet":
        model = models.googlenet(aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet-18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet-34":
        model = models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    weights_path = f"pokemon_{model_name.lower().replace('-', '')}.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.error(f"모델 파일을 찾을 수 없습니다: {weights_path}")
    
    model.to(device)
    model.eval()
    return model

# 5. GUI 화면 구성
st.title("포켓몬 분류기")

selected_model = st.sidebar.selectbox("Backbone 모델 선택", 
                                     ["AlexNet", "GoogLeNet", "ResNet-18", "ResNet-34", "EfficientNet"])

uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='업로드된 이미지', use_container_width=True)
    
    with col2:
        st.subheader(f"{selected_model} 예측 결과")
        
        with st.spinner('분석 중...'):
            # 모델 로드 및 추론
            model = load_model(selected_model)
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                top5_prob, top5_catid = torch.topk(probabilities, 5)

            # 🔥 결과를 화면에 출력하는 핵심 루프
            for i in range(5):
                idx = top5_catid[i].item()
                name = class_names[idx]
                prob = top5_prob[i].item() * 100
                
                st.write(f"**{i+1}. {name}** ({prob:.2f}%)")
                st.progress(prob / 100)