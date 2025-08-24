# SmartRecycle AI - Sistema Inteligente de Classificação de Resíduos

## 🎯 Visão Geral
Sistema de classificação automatizada de resíduos utilizando Deep Learning e técnicas avançadas de Computer Vision. Este projeto implementa uma solução completa com interface webcam, métricas detalhadas e visualizações profissionais para reconhecimento de padrões em materiais recicláveis.

## 🚀 Principais Funcionalidades

### 🔬 **Arquitetura Avançada**
- **Transfer Learning**: Modelo pré-treinado no ImageNet
- **Attention Mechanism**: Módulo de atenção personalizado para melhor feature learning
- **ResNet18 Backbone**: Arquitetura robusta e eficiente

### 📊 **Sistema de Métricas Completo**
- **Accuracy, Precision, Recall, F1-Score**: Métricas detalhadas por classe
- **Matriz de Confusão**: Visualização profissional dos resultados
- **Curvas ROC**: Análise de performance multiclasse
- **Relatórios Detalhados**: Exportação automática de resultados

### 🎥 **Interface em Tempo Real**
- **Webcam Integration**: Classificação ao vivo
- **Probabilidades**: Exibição de confiança do modelo
- **Captura de Imagens**: Salvamento automático de predições

### 📈 **Visualizações Profissionais**
- Gráficos de comparação de métricas
- Heatmaps de matriz de confusão
- Curvas de performance
- Relatórios prontos para apresentação

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

### 🎯 Execução Completa (Recomendado)
Execute o script automatizado que inclui treinamento e avaliação completa:

```bash
python run_recyclenet.py
```

### 📊 Avaliação com Métricas Detalhadas
Execute o treinamento com sistema completo de métricas:

```bash
python main_with_metrics.py --gpu 0 --arch resnet18_base --use_att --att_mode ours
```

### 🎥 Interface Webcam
Classificação em tempo real via webcam:

```bash
python webcam_enhanced.py --resume save/model_best.pth.tar
```

### 📈 Geração de Métricas Simuladas
Para demonstração e apresentação:

```bash
python generate_simulated_metrics.py
```

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
- **Recall**: 93.0%
- **F1-Score**: 93.0%

### Comparativo de Arquiteturas

| Modelo | Accuracy | Parâmetros (M) |
|--------|----------|---------------|
| ResNet18 | 90.02% | 11.18 |
| ResNet18 + Atenção | **93.04%** | 11.24 |
| ResNet34 + Atenção | **93.97%** | 21.35 |
| ResNet50 + Atenção | **94.20%** | 24.15 |

## 🔧 Configuração do Sistema

### Otimização
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: SGD
- **Learning Rate**: 2e-4
- **Epochs**: 100
- **Scheduler**: Redução por fator de 0.1 a cada 40 épocas

### Módulo de Atenção
O sistema implementa um mecanismo de atenção personalizado que:
- Aprende pesos altos para features importantes
- Reduz a influência de features desnecessárias
- Melhora significativamente a performance de classificação

## 📁 Estrutura do Projeto

```
SmartRecycle-AI/
├── 📊 Dados e Augmentação
│   ├── data/                    # Configurações do dataset
│   ├── augmented/              # Dados aumentados
│   └── augmentation.py         # Script de augmentação
├── 🧠 Modelos e Treinamento
│   ├── main.py                 # Script principal de treinamento
│   ├── main_with_metrics.py    # Treinamento com métricas completas
│   ├── resnet.py              # Arquitetura ResNet
│   └── attention.py           # Módulo de atenção
├── 📈 Avaliação e Métricas
│   ├── metrics_evaluation.py   # Sistema completo de métricas
│   ├── metrics.py             # Funções auxiliares
│   └── generate_simulated_metrics.py
├── 🎥 Interface e Demonstração
│   ├── webcam_enhanced.py      # Interface webcam
│   └── run_recyclenet.py       # Execução automatizada
└── 📚 Documentação
    ├── README.md
    ├── GUIA_COMPLETO.md
    └── RecycleNet_Complete_Colab.ipynb
```

## 🤝 Contribuições

Este projeto foi desenvolvido como parte de um estudo em Reconhecimento de Padrões, com melhorias significativas em:
- Sistema completo de métricas e visualizações
- Interface webcam em tempo real
- Documentação abrangente
- Scripts automatizados para facilitar o uso

## 📜 Licença

Este projeto é baseado no trabalho original [RecycleNet](https://github.com/sangminwoo/RecycleNet) com melhorias e funcionalidades adicionais desenvolvidas para fins educacionais.

## 🙏 Agradecimentos

- Dataset TrashNet: [garythung/trashnet](https://github.com/garythung/trashnet)
- Trabalho original: [sangminwoo/RecycleNet](https://github.com/sangminwoo/RecycleNet)
- Bibliotecas utilizadas: PyTorch, Albumentations, scikit-learn, OpenCV
|          Max         |    92.575   |     11.24     |
|          Sum         |    **93.039**   |     11.24     |

Conclusion
----------
While proposing deep-learning model which is specialized in trash classification, there was two difficult problems faced experimentally:

*1) Insufficiency of data set*  
*2) The absence of effective feature learning methods*  
was solved by **transfer learning and attention mechanism.**

The methodology proposed through quantitative and qualitative assessments was experimentally significant. Because the proposed method exhibits significant performance improvements without significantly increasing the number of parameters, it is expected that the experimental value is also high for other applications.

Reference
----------
| # | Reference      |                    Link                      |
|---|----------------|----------------------------------------------|
| 1 | TrashNet       | https://github.com/garythung/trashnet        |
| 2 | SENet          | https://github.com/hujie-frank/SENet         |
| 3 | CBAM           | https://github.com/Jongchan/attention-module |
| 4 | Albumentations | https://github.com/albu/albumentations       |

Acknowledgement
---------------
We appreciate much the dataset [TrashNet](https://github.com/garythung/trashnet) and the well organized code [CBAM](https://github.com/Jongchan/attention-module). Our codebase is mostly built based on them.
