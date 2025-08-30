# SmartRecycle AI - Sistema de Classificação de Resíduos

## 🎯 Visão Geral
Sistema de classificação automatizada de resíduos utilizando Deep Learning e técnicas avançadas de Computer Vision. Este projeto implementa uma solução completa com interface webcam, métricas detalhadas e visualizações profissionais para reconhecimento de padrões em materiais recicláveis.

## 🚀 Principais Funcionalidades

### 🔬 **Arquitetura **
- **Transfer Learning**: Modelo pré-treinado no ImageNet
- **Attention Mechanism**: Módulo de atenção personalizado para melhor feature learning
- **ResNet18 Backbone**: Arquitetura robusta e eficiente

### 📊 **Sistema de Métricas Completo**
- **Accuracy, Precision, Recall, F1-Score**: Métricas detalhadas por classe
- **Matriz de Confusão**: Visualização dos resultados
- **Curvas ROC**: Análise de performance multiclasse

### 🎥 **Interface em Tempo Real**
- **Webcam Integration**: Classificação ao vivo
- **Probabilidades**: Exibição de confiança do modelo
- **Captura de Imagens**: Salvamento automático de predições

### 📈 **Visualizações Profissionais**
- Gráficos de comparação de métricas
- Heatmaps de matriz de confusão
- Curvas de performance

## 🛠️ Instalação e Configuração

### Pré-requisitos
- Python 3.7+
- PyTorch
- CUDA (opcional, para GPU)

### Instalação Rápida
```bash
# Clone o repositório
git clone https://github.com/seuusuario/SmartRecycle-AI.git
cd SmartRecycle-AI

# Instale as dependências
pip install -r requirements.txt
```

### Configuração do PyTorch
Visite o [site oficial do PyTorch](https://pytorch.org/get-started/locally/) para instalação específica do seu sistema.

## 📊 Dataset - TrashNet

### Composição dos Dados
- **Total**: 2.527 imagens distribuídas em 6 classes
  - 🥃 Glass: 501 imagens
  - 📄 Paper: 594 imagens  
  - 📦 Cardboard: 403 imagens
  - 🥤 Plastic: 482 imagens
  - 🔩 Metal: 410 imagens
  - 🗑️ Trash: 137 imagens

### Divisão dos Dados
- **Treino**: 70% dos dados
- **Validação**: 13% dos dados  
- **Teste**: 17% dos dados

## 🔄 Data Augmentation

Execute o script de augmentação com diferentes níveis de probabilidade:

```bash
python augmentation.py --root_dir dataset-resized/ --save_dir augmented/ --probability low
```

**Parâmetros disponíveis:**
- `--probability`: `low` (padrão), `mid`, `high`
- `--root_dir`: Diretório dos dados originais
- `--save_dir`: Diretório para salvar dados aumentados
## 🚀 Como Usar


## ⚙️ Configurações Avançadas

### Treinamento Personalizado

**Sem Pré-treinamento:**
```bash
python main.py --gpu 0 --arch resnet18_base --no_pretrain
```

**Sem Módulo de Atenção:**
```bash
python main.py --gpu 0 --arch resnet18_base
```

**Com Módulo de Atenção:**
```bash
python main.py --gpu 0 --arch resnet18_base --use_att --att_mode ours
```

### Parâmetros Principais
- **GPU**: `0`, `0,1`, `0,1,2` (múltiplas GPUs)
- **Arquitetura**: `resnet18_base`, `resnet34_base`, `resnet50_base`
- **Atenção**: `ours`, `cbam`, `se`

## 🏆 Resultados e Performance

### Métricas Principais
- **Accuracy**: 93.04%
- **Precision**: 93.1%
- **Recall**: 92.8%
- **F1-Score**: 93.0%


## 🤝 Contribuições

Este projeto foi desenvolvido como parte de um estudo em Reconhecimento de Padrões, com melhorias significativas em:
- Sistema completo de métricas e visualizações
- Interface webcam em tempo real
- Documentação abrangente
- Scripts automatizados para facilitar o uso
