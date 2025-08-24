"""
Script para gerar métricas simuladas baseadas no paper original
Útil para apresentação quando você não tem modelo treinado
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

def generate_simulated_results():
    """Gera resultados simulados baseados no paper original"""
    
    print("🎲 GERANDO RESULTADOS SIMULADOS - RECYCLENET")
    print("=" * 60)
    print("📊 Baseado nos resultados reportados no paper original")
    print("🎯 Acurácia esperada: ~93% (ResNet18 + RecycleNet)")
    print()
    
    # Configurações
    np.random.seed(42)
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Distribuição aproximada do dataset de teste
    true_distribution = [80, 100, 82, 118, 96, 27]  # Total: 503 amostras
    
    # Gerar labels verdadeiros
    y_true = []
    for i, count in enumerate(true_distribution):
        y_true.extend([i] * count)
    
    # Simular predições com ~93% de acurácia
    y_pred = y_true.copy()
    n_errors = int(len(y_true) * 0.07)  # 7% de erro para 93% de acurácia
    
    # Introduzir erros realistas entre classes similares
    error_indices = np.random.choice(len(y_true), n_errors, replace=False)
    
    confusion_pairs = {
        0: [3, 1, 2],  # cardboard confunde com paper, glass, metal
        1: [2, 0, 4],  # glass confunde com metal, cardboard, plastic
        2: [1, 4, 0],  # metal confunde com glass, plastic, cardboard
        3: [0, 4, 5],  # paper confunde com cardboard, plastic, trash
        4: [5, 2, 3],  # plastic confunde com trash, metal, paper
        5: [4, 3, 0]   # trash confunde com plastic, paper, cardboard
    }
    
    for idx in error_indices:
        true_class = y_true[idx]
        possible_errors = confusion_pairs[true_class]
        y_pred[idx] = np.random.choice(possible_errors)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return y_true, y_pred, class_names

def calculate_metrics(y_true, y_pred, class_names):
    """Calcula todas as métricas"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """Plota matriz de confusão"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matriz de Confusão Normalizada - RecycleNet'
        cmap = 'Blues'
    else:
        fmt = 'd'
        title = 'Matriz de Confusão - RecycleNet'
        cmap = 'Blues'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
               xticklabels=class_names, yticklabels=class_names,
               square=True, linewidths=0.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Classe Predita', fontsize=12)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_metrics_comparison(metrics, class_names):
    """Plota comparação de métricas por classe"""
    
    df_data = {
        'Classe': class_names,
        'Precisão': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1-Score': metrics['f1_per_class']
    }
    df = pd.DataFrame(df_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['Precisão'], width, label='Precisão', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8, color='#4ECDC4')
    bars3 = ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8, color='#45B7D1')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas por Classe - RecycleNet', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Adicionar valores nas barras
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.show()

def plot_overall_metrics(metrics):
    """Plota métricas gerais"""
    
    categories = ['Precisão', 'Recall', 'F1-Score']
    macro_scores = [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]
    weighted_scores = [metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, macro_scores, width, label='Média Macro', alpha=0.8, color='#96CEB4')
    bars2 = ax.bar(x + width/2, weighted_scores, width, label='Média Ponderada', alpha=0.8, color='#FFEAA7')
    
    # Linha de acurácia
    ax.axhline(y=metrics['accuracy'], color='red', linestyle='--', linewidth=2,
              label=f'Acurácia: {metrics["accuracy"]:.3f}')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas Gerais - RecycleNet vs Baseline', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Adicionar valores nas barras
    for i, (macro, weighted) in enumerate(zip(macro_scores, weighted_scores)):
        ax.text(i - width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison():
    """Compara diferentes modelos"""
    
    models_data = {
        'Modelo': ['ResNet18\\n(Baseline)', 'ResNet18\\n+ SE', 'ResNet18\\n+ CBAM', 'ResNet18\\n+ RecycleNet\\n(Proposto)'],
        'Acurácia': [90.02, 87.70, 79.81, 93.04],
        'Parâmetros (M)': [11.18, 11.27, 11.27, 11.24],
        'F1-Score': [0.895, 0.873, 0.790, 0.927]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Acurácia
    colors = ['#74b9ff', '#fd79a8', '#fdcb6e', '#00b894']
    bars = ax1.bar(models_data['Modelo'], models_data['Acurácia'], color=colors, alpha=0.8)
    ax1.set_ylabel('Acurácia (%)', fontsize=12)
    ax1.set_title('Comparação de Acurácia entre Modelos', fontsize=14, fontweight='bold')
    ax1.set_ylim(75, 95)
    ax1.grid(True, alpha=0.3)
    
    # Destacar o melhor
    bars[-1].set_color('#00b894')
    bars[-1].set_alpha(1.0)
    
    # Adicionar valores
    for bar, acc in zip(bars, models_data['Acurácia']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Eficiência (Acurácia vs Parâmetros)
    scatter = ax2.scatter(models_data['Parâmetros (M)'], models_data['Acurácia'], 
                         c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models_data['Modelo']):
        ax2.annotate(model.replace('\\n', ' '), 
                    (models_data['Parâmetros (M)'][i], models_data['Acurácia'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Parâmetros (Milhões)', fontsize=12)
    ax2.set_ylabel('Acurácia (%)', fontsize=12)
    ax2.set_title('Eficiência: Acurácia vs Parâmetros', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_report(y_true, y_pred, class_names, metrics):
    """Gera relatório completo"""
    
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=4)
    
    print("\\n📊 RELATÓRIO COMPLETO - RECYCLENET")
    print("=" * 70)
    print(f"🎯 RESULTADOS GERAIS:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • F1-Score (Ponderado): {metrics['f1_weighted']:.4f}")
    print(f"   • Precisão (Ponderada): {metrics['precision_weighted']:.4f}")
    print(f"   • Recall (Ponderado): {metrics['recall_weighted']:.4f}")
    
    print(f"\\n📋 RELATÓRIO DETALHADO POR CLASSE:")
    print(report)
    
    print(f"\\n📈 MÉTRICAS POR CLASSE:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name.upper()}:")
        print(f"  • Precisão: {metrics['precision_per_class'][i]:.4f}")
        print(f"  • Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"  • F1-Score: {metrics['f1_per_class'][i]:.4f}")
        print()

def main():
    """Execução principal"""
    
    print("🚀 RECYCLENET - GERADOR DE MÉTRICAS SIMULADAS")
    print("=" * 60)
    print("📊 Gerando resultados baseados no paper original...")
    print()
    
    # Gerar dados simulados
    y_true, y_pred, class_names = generate_simulated_results()
    
    # Calcular métricas
    print("📊 Calculando métricas...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Mostrar resultados principais
    print(f"\\n🎯 RESULTADOS PRINCIPAIS:")
    print(f"   • Amostras avaliadas: {len(y_true)}")
    print(f"   • Acurácia obtida: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • F1-Score (Ponderado): {metrics['f1_weighted']:.4f}")
    print()
    
    # Gerar visualizações
    print("📈 Gerando visualizações...")
    print("\\n1. Matriz de Confusão:")
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False)
    
    print("\\n2. Matriz de Confusão Normalizada:")
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
    
    print("\\n3. Métricas por Classe:")
    plot_metrics_comparison(metrics, class_names)
    
    print("\\n4. Métricas Gerais:")
    plot_overall_metrics(metrics)
    
    print("\\n5. Comparação entre Modelos:")
    plot_model_comparison()
    
    # Gerar relatório
    generate_report(y_true, y_pred, class_names, metrics)
    
    print("\\n🎉 GERAÇÃO COMPLETA!")
    print("📁 Todos os gráficos foram exibidos")
    print("📋 Relatório completo gerado")
    print("\\n💡 Use estes resultados para sua apresentação!")

if __name__ == "__main__":
    main()
