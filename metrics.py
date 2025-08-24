import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os

class MetricsTracker:
    """
    Classe para calcular e acompanhar métricas de classificação
    """
    def __init__(self, num_classes=6, class_names=None):
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
        else:
            self.class_names = class_names
        
        self.reset()
    
    def reset(self):
        """Reset all tracking variables"""
        self.all_predictions = []
        self.all_targets = []
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.epoch_metrics = defaultdict(list)
    
    def update(self, predictions, targets):
        """
        Atualiza as predições e targets para cálculo das métricas
        
        Args:
            predictions: tensor de predições do modelo
            targets: tensor de targets verdadeiros
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Se predictions são logits, pegar argmax
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        self.all_predictions.extend(predictions.tolist())
        self.all_targets.extend(targets.tolist())
    
    def update_training_metrics(self, loss, accuracy):
        """Atualiza métricas de treinamento"""
        self.training_losses.append(loss)
        self.training_accuracies.append(accuracy)
    
    def update_validation_metrics(self, loss, accuracy):
        """Atualiza métricas de validação"""
        self.validation_losses.append(loss)
        self.validation_accuracies.append(accuracy)
    
    def calculate_metrics(self):
        """
        Calcula todas as métricas de classificação
        
        Returns:
            dict: Dicionário com todas as métricas calculadas
        """
        if len(self.all_predictions) == 0:
            return {}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Métricas gerais
        accuracy = accuracy_score(y_true, y_pred)
        
        # Métricas por classe e médias
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métricas por classe individual
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names
        }
        
        return metrics
    
    def print_metrics(self, metrics=None):
        """
        Imprime as métricas de forma organizada
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        if not metrics:
            print("Nenhuma métrica disponível para imprimir.")
            return
        
        print("\n" + "="*60)
        print("MÉTRICAS DE CLASSIFICAÇÃO")
        print("="*60)
        
        print(f"\n📊 MÉTRICAS GERAIS:")
        print(f"   • Acurácia (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        print(f"\n📈 MÉTRICAS MACRO (média simples entre classes):")
        print(f"   • Precisão Macro: {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)")
        print(f"   • Recall Macro: {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)")
        print(f"   • F1-Score Macro: {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
        
        print(f"\n⚖️ MÉTRICAS WEIGHTED (média ponderada por frequência):")
        print(f"   • Precisão Weighted: {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)")
        print(f"   • Recall Weighted: {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)")
        print(f"   • F1-Score Weighted: {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)")
        
        print(f"\n📋 MÉTRICAS POR CLASSE:")
        for i, class_name in enumerate(metrics['class_names']):
            if i < len(metrics['precision_per_class']):
                print(f"   {class_name}:")
                print(f"      • Precisão: {metrics['precision_per_class'][i]:.4f} ({metrics['precision_per_class'][i]*100:.2f}%)")
                print(f"      • Recall: {metrics['recall_per_class'][i]:.4f} ({metrics['recall_per_class'][i]*100:.2f}%)")
                print(f"      • F1-Score: {metrics['f1_per_class'][i]:.4f} ({metrics['f1_per_class'][i]*100:.2f}%)")
        
        print("\n" + "="*60)
    
    def save_metrics(self, filepath, metrics=None):
        """
        Salva as métricas em arquivo JSON
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        # Converter numpy arrays para listas para serialização JSON
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Métricas salvas em: {filepath}")
    
    def plot_confusion_matrix(self, save_path=None, metrics=None):
        """
        Plota a matriz de confusão
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        if not metrics:
            print("Nenhuma métrica disponível para plotar.")
            return
        
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=metrics['class_names'], 
                   yticklabels=metrics['class_names'])
        
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.xlabel('Predições', fontsize=12)
        plt.ylabel('Valores Verdadeiros', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusão salva em: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, save_path=None):
        """
        Plota curvas de treinamento (loss e accuracy)
        """
        if not self.training_losses and not self.validation_losses:
            print("Nenhum dado de treinamento disponível para plotar.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        if self.training_losses:
            epochs = range(1, len(self.training_losses) + 1)
            ax1.plot(epochs, self.training_losses, 'b-', label='Treinamento', linewidth=2)
        
        if self.validation_losses:
            epochs = range(1, len(self.validation_losses) + 1)
            ax1.plot(epochs, self.validation_losses, 'r-', label='Validação', linewidth=2)
        
        ax1.set_title('Curva de Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Épocas', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        if self.training_accuracies:
            epochs = range(1, len(self.training_accuracies) + 1)
            ax2.plot(epochs, self.training_accuracies, 'b-', label='Treinamento', linewidth=2)
        
        if self.validation_accuracies:
            epochs = range(1, len(self.validation_accuracies) + 1)
            ax2.plot(epochs, self.validation_accuracies, 'r-', label='Validação', linewidth=2)
        
        ax2.set_title('Curva de Acurácia', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Épocas', fontsize=12)
        ax2.set_ylabel('Acurácia', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curvas de treinamento salvas em: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None, metrics=None):
        """
        Plota comparação de métricas por classe
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        if not metrics:
            print("Nenhuma métrica disponível para plotar.")
            return
        
        # Preparar dados
        classes = metrics['class_names']
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        
        # Configurar gráfico
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Barras
        bars1 = ax.bar(x - width, precision, width, label='Precisão', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        # Configurações do gráfico
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparação de Métricas por Classe', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparação de métricas salva em: {save_path}")
        
        plt.show()


def calculate_accuracy(output, target, topk=(1,)):
    """
    Calcula precisão para os valores especificados de k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computa e armazena a média e valor atual"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
