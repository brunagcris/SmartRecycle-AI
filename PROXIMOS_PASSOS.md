# 🚀 GUIA RÁPIDO - GOOGLE COLAB

## Abra este link no Google Colab:
https://colab.research.google.com/

## Cole este código e execute célula por célula:

```python
# 1. Instalar dependências
!pip install torch torchvision albumentations opencv-python-headless scikit-learn seaborn pandas matplotlib

# 2. Clonar repositório
!git clone https://github.com/sangminwoo/RecycleNet.git
%cd RecycleNet

# 3. Baixar dataset (opcional - para demonstração)
!wget -O dataset.zip "https://github.com/garythung/trashnet/archive/master.zip"
!unzip -q dataset.zip && mv trashnet-master/data/dataset-resized ./data/

# 4. Copiar código da classe MetricsEvaluator (do arquivo metrics_evaluation.py)

# 5. Executar simulação de resultados (como no notebook)

# 6. Gerar todos os gráficos e métricas
```

### **OPÇÃO 2: Continuar localmente**

Agora que corrigimos os problemas, você pode:

```bash
# 1. Deixar o treinamento atual terminar
# (pode parar com Ctrl+C se quiser)

# 2. Treinar modelo completo com atenção
python3.10 main.py --gpu 0 --arch resnet18_base --use_att --att_mode ours --epochs 100

# 3. Avaliar com métricas completas
python3.10 simple_runner.py --gpu 0 --evaluate --detailed_eval

# 4. Testar webcam
python3.10 webcam_enhanced.py --resume save/model_best.pth.tar --show_probabilities
```
