"""
Módulo de Avaliação Completa para RecycleNet
Implementa métricas detalhadas para avaliação do modelo de classificação de lixo
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
from datetime import datetime

class MetricsEvaluator:
    """
    Classe para avaliar modelo com métricas completas de classificação
    """
    
    def __init__(self, class_names=None, save_dir='results/'):
        """
        Inicializa o avaliador de métricas
        
        Args:
            class_names: Lista com nomes das classes ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            save_dir: Diretório para salvar resultados
        """
        if class_names is None:
            self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        else:
            self.class_names = class_names
            
        self.num_classes = len(self.class_names)
        self.save_dir = save_dir
        
        # Criar diretório se não existir
        os.makedirs(save_dir, exist_ok=True)
        
        # Listas para armazenar predições e labels verdadeiros
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        
    def reset(self):
        """Reset das listas para nova avaliação"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, outputs, targets):
        """
        Atualiza com batch de predições
        
        Args:
            outputs: Tensor com logits do modelo [batch_size, num_classes]
            targets: Tensor com labels verdadeiros [batch_size]
        """
        # Converter para CPU e numpy
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
            
        # Obter probabilidades usando softmax
        probabilities = self._softmax(outputs)
        predictions = np.argmax(outputs, axis=1)
        
        self.all_predictions.extend(predictions.tolist())
        self.all_targets.extend(targets.tolist())
        self.all_probabilities.extend(probabilities.tolist())
    
    def _softmax(self, x):
        """Aplica softmax para obter probabilidades"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute_basic_metrics(self):
        """
        Calcula métricas básicas: Acurácia, Precisão, Recall, F1-Score
        
        Returns:
            dict: Dicionário com todas as métricas
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Métricas gerais
        accuracy = accuracy_score(y_true, y_pred)
        
        # Métricas por classe e médias
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_per_class': precision_per_class,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'recall_per_class': recall_per_class,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
        }
        
        return metrics
    
    def plot_confusion_matrix(self, normalize=False, save_name='confusion_matrix.png'):
        """
        Plota matriz de confusão
        
        Args:
            normalize: Se True, normaliza a matriz
            save_name: Nome do arquivo para salvar
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Matriz de Confusão Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusão'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Verdadeira')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_metrics_comparison(self, save_name='metrics_comparison.png'):
        """
        Plota comparação de métricas por classe
        """
        metrics = self.compute_basic_metrics()
        
        # Criar DataFrame para facilitar o plot
        df_data = {
            'Classe': self.class_names,
            'Precisão': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1-Score': metrics['f1_per_class']
        }
        df = pd.DataFrame(df_data)
        
        # Plot de barras agrupadas
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, df['Precisão'], width, label='Precisão', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Métricas por Classe')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_overall_metrics(self, save_name='overall_metrics.png'):
        """
        Plota métricas gerais (macro e weighted averages)
        """
        metrics = self.compute_basic_metrics()
        
        # Dados para o gráfico
        categories = ['Precisão', 'Recall', 'F1-Score']
        macro_scores = [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]
        weighted_scores = [metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, macro_scores, width, label='Média Macro', alpha=0.8)
        ax.bar(x + width/2, weighted_scores, width, label='Média Ponderada', alpha=0.8)
        
        # Adicionar linha de acurácia
        ax.axhline(y=metrics['accuracy'], color='red', linestyle='--', 
                  label=f'Acurácia: {metrics["accuracy"]:.3f}')
        
        ax.set_ylabel('Score')
        ax.set_title('Métricas Gerais do Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Adicionar valores nas barras
        for i, (macro, weighted) in enumerate(zip(macro_scores, weighted_scores)):
            ax.text(i - width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, save_name='classification_report.txt'):
        """
        Gera relatório detalhado de classificação
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     digits=4)
        
        print("RELATÓRIO DE CLASSIFICAÇÃO:")
        print("=" * 60)
        print(report)
        
        # Salvar relatório
        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w') as f:
            f.write("RELATÓRIO DE CLASSIFICAÇÃO - RecycleNet\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report)
        
        return report
    
    def plot_roc_curves(self, save_name='roc_curves.png'):
        """
        Plota curvas ROC para classificação multiclasse
        """
        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)
        
        # Binarizar labels para ROC multiclasse
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Calcular ROC para cada classe
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Linha diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Chance Aleatória')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
        plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
        plt.title('Curvas ROC por Classe')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_results(self, model_name='RecycleNet', save_name='detailed_results.txt'):
        """
        Salva resultados detalhados em arquivo
        """
        metrics = self.compute_basic_metrics()
        
        save_path = os.path.join(self.save_dir, save_name)
        
        with open(save_path, 'w') as f:
            f.write(f"RESULTADOS DETALHADOS - {model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MÉTRICAS GERAIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precisão (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Precisão (Ponderada): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (Ponderado): {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Ponderado): {metrics['f1_weighted']:.4f}\n\n")
            
            f.write("MÉTRICAS POR CLASSE:\n")
            f.write("-" * 30 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precisão: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n\n")
        
        print(f"Resultados salvos em: {save_path}")
    
    def evaluate_complete(self, model_name='RecycleNet'):
        """
        Executa avaliação completa e gera todos os gráficos e relatórios
        """
        print("Iniciando avaliação completa...")
        
        # Calcular métricas básicas
        metrics = self.compute_basic_metrics()
        
        # Imprimir resumo
        print(f"\nRESUMO DOS RESULTADOS - {model_name}")
        print("=" * 50)
        print(f"Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1-Score (Ponderado): {metrics['f1_weighted']:.4f}")
        print(f"Precisão (Ponderada): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Ponderado): {metrics['recall_weighted']:.4f}")
        
        # Gerar todos os gráficos
        print("\nGerando visualizações...")
        self.plot_confusion_matrix(normalize=False, save_name='confusion_matrix.png')
        self.plot_confusion_matrix(normalize=True, save_name='confusion_matrix_normalized.png')
        self.plot_metrics_comparison()
        self.plot_overall_metrics()
        self.plot_roc_curves()
        
        # Gerar relatórios
        print("\nGerando relatórios...")
        self.generate_classification_report()
        self.save_detailed_results(model_name)
        
        print(f"\nAvaliação completa finalizada! Resultados salvos em: {self.save_dir}")
        
        return metrics

# Função auxiliar para explicar as métricas
def explain_metrics():
    """
    Explica o significado das métricas para interpretação dos resultados
    """
    explanations = {
        "Acurácia": {
            "definição": "Proporção de predições corretas sobre o total de predições",
            "fórmula": "(VP + VN) / (VP + VN + FP + FN)",
            "interpretação": "Quanto maior, melhor. Valores próximos a 1.0 (100%) são ideais",
            "cuidados": "Pode ser enganosa em datasets desbalanceados"
        },
        
        "Precisão": {
            "definição": "Proporção de predições positivas que estão corretas",
            "fórmula": "VP / (VP + FP)",
            "interpretação": "Quanto maior, melhor. Indica quão confiáveis são as predições positivas",
            "quando_usar": "Importante quando o custo de falsos positivos é alto"
        },
        
        "Recall (Sensibilidade)": {
            "definição": "Proporção de casos positivos reais que foram identificados corretamente",
            "fórmula": "VP / (VP + FN)",
            "interpretação": "Quanto maior, melhor. Indica capacidade de encontrar todos os casos positivos",
            "quando_usar": "Importante quando o custo de falsos negativos é alto"
        },
        
        "F1-Score": {
            "definição": "Média harmônica entre precisão e recall",
            "fórmula": "2 * (Precisão * Recall) / (Precisão + Recall)",
            "interpretação": "Balança precisão e recall. Ideal quando você precisa de equilíbrio",
            "vantagem": "Melhor métrica geral para datasets desbalanceados"
        }
    }
    
    print("GUIA DE INTERPRETAÇÃO DAS MÉTRICAS")
    print("=" * 60)
    
    for metric, info in explanations.items():
        print(f"\n{metric.upper()}:")
        print("-" * 30)
        for key, value in info.items():
            print(f"{key.capitalize()}: {value}")
    
    print("\nTIPOS DE MÉDIA:")
    print("-" * 30)
    print("Macro: Média simples das métricas de cada classe (todas as classes têm peso igual)")
    print("Ponderada: Média ponderada pelo número de amostras de cada classe")
    print("Micro: Considera o total de VP, FP, FN de todas as classes")
    
    print("\nCOMO INTERPRETAR OS RESULTADOS:")
    print("-" * 30)
    print("• Acurácia > 90%: Excelente desempenho")
    print("• Acurácia 80-90%: Bom desempenho")
    print("• Acurácia 70-80%: Desempenho moderado")
    print("• Acurácia < 70%: Precisa melhorar")
    print("\nObservação: Estes valores são gerais. Para classificação de lixo,")
    print("considere também o balanceamento das classes e o contexto da aplicação.")

if __name__ == "__main__":
    # Exemplo de uso
    explain_metrics()
