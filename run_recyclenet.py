#!/usr/bin/env python3
"""
🚀 SCRIPT COMPLETO PARA EXECUTAR O RECYCLENET LOCALMENTE
=====================================================

Este script permite executar o RecycleNet com todas as funcionalidades:
- Treinamento com métricas completas
- Avaliação detalhada
- Teste com webcam
- Geração de relatórios e gráficos

Autor: Assistente especializado em ML/Deep Learning
Para: Projeto de Reconhecimento de Padrões
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Verifica se todas as dependências estão instaladas"""
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'albumentations': 'albumentations',
        'scikit-learn': 'sklearn',
        'seaborn': 'seaborn',
        'pandas': 'pandas',
        'Pillow': 'PIL'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Pacotes faltando:", missing_packages)
        print("🔧 Execute: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True

def setup_data():
    """Configura e verifica os dados"""
    print("📊 Verificando dados...")
    
    data_dir = Path("./data/dataset-resized")
    if not data_dir.exists():
        print("❌ Dados não encontrados!")
        print("🔗 Baixe o dataset TrashNet:")
        print("   1. Acesse: https://github.com/garythung/trashnet")
        print("   2. Baixe e extraia em ./data/dataset-resized/")
        return False
    
    # Contar imagens por classe
    classes = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"📁 Classes encontradas: {len(classes)}")
    
    for class_dir in classes:
        count = len(list(class_dir.glob("*.jpg")))
        print(f"   • {class_dir.name}: {count} imagens")
    
    return True

def run_augmentation():
    """Executa aumento de dados"""
    print("🔄 Executando aumento de dados...")
    
    cmd = [
        sys.executable, "augmentation.py",
        "--root_dir", "data/dataset-resized/",
        "--save_dir", "augmented/",
        "--probability", "mid"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Aumento de dados concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no aumento de dados: {e}")
        return False

def run_training(args):
    """Executa treinamento do modelo"""
    print("🚀 Iniciando treinamento...")
    
    cmd = [
        sys.executable, "main_with_metrics.py",
        "--gpu", "0",
        "--arch", args.arch,
        "--epochs", str(args.epochs),
        "--b", str(args.batch_size)
    ]
    
    if args.use_attention:
        cmd.extend(["--use_att", "--att_mode", args.attention_mode])
    
    if args.no_pretrain:
        cmd.append("--no_pretrain")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Treinamento concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no treinamento: {e}")
        return False

def run_evaluation():
    """Executa avaliação com métricas completas"""
    print("📊 Executando avaliação completa...")
    
    cmd = [
        sys.executable, "main_with_metrics.py",
        "--gpu", "0",
        "--resume", "save/model_best.pth.tar",
        "--use_att",
        "--att_mode", "ours",
        "--evaluate",
        "--detailed_eval"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Avaliação concluída! Verifique a pasta results/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na avaliação: {e}")
        return False

def run_webcam():
    """Executa teste com webcam"""
    print("📹 Iniciando teste com webcam...")
    
    cmd = [
        sys.executable, "webcam_enhanced.py",
        "--resume", "save/model_best.pth.tar",
        "--use_att",
        "--att_mode", "ours",
        "--show_probabilities"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Teste com webcam concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro no teste com webcam: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="RecycleNet - Script Completo")
    
    # Argumentos principais
    parser.add_argument("--mode", choices=["setup", "train", "eval", "webcam", "all"], 
                       default="all", help="Modo de execução")
    
    # Argumentos de treinamento
    parser.add_argument("--arch", default="resnet18_base", 
                       choices=["resnet18_base", "resnet34_base", "resnet50_base"],
                       help="Arquitetura do modelo")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--use_attention", action="store_true", help="Usar módulo de atenção")
    parser.add_argument("--attention_mode", default="ours", choices=["ours", "cbam", "se"],
                       help="Tipo de atenção")
    parser.add_argument("--no_pretrain", action="store_true", help="Treinar do zero")
    
    args = parser.parse_args()
    
    print("🗂️  RECYCLENET - CLASSIFICAÇÃO DE LIXO")
    print("=" * 50)
    
    # Verificar requisitos
    if not check_requirements():
        return
    
    # Executar modo selecionado
    if args.mode in ["setup", "all"]:
        if not setup_data():
            return
        run_augmentation()
    
    if args.mode in ["train", "all"]:
        run_training(args)
    
    if args.mode in ["eval", "all"]:
        run_evaluation()
    
    if args.mode in ["webcam", "all"]:
        run_webcam()
    
    print("\n🎉 EXECUÇÃO COMPLETA!")
    print("📋 Resumo dos resultados:")
    print("   📊 Métricas: ./results/")
    print("   💾 Modelo: ./save/model_best.pth.tar")
    print("   📸 Capturas: ./webcam_captures/")

if __name__ == "__main__":
    main()
