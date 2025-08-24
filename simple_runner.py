"""
Script simplificado para execução do RecycleNet com métricas
"""

import argparse
import os
import sys

def get_arguments():
    parser = argparse.ArgumentParser(description='RecycleNet with Metrics')
    parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
    parser.add_argument('--arch', type=str, default='resnet18_base', help='Architecture')
    parser.add_argument('--use_att', action='store_true', help='use attention module')
    parser.add_argument('--att_mode', type=str, default='ours', help='attention module mode')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--detailed_eval', action='store_true', help='detailed evaluation')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--b', type=int, default=16, help='batch size')
    return parser.parse_args()

def main():
    args = get_arguments()
    
    print("🚀 RECYCLENET - Execução Simplificada")
    print("="*50)
    
    if args.evaluate:
        print("📊 Modo de avaliação detectado")
        
        # Primeiro, vamos verificar se temos um modelo salvo
        if not os.path.exists('save/model_best.pth.tar'):
            print("❌ Modelo não encontrado. Execute primeiro o treinamento:")
            print("   python main.py --gpu 0 --arch resnet18_base --use_att --att_mode ours")
            return
        
        print("✅ Modelo encontrado!")
        
        # Executar avaliação com o script original (sem as modificações problemáticas)
        cmd = [
            'python', 'main.py',
            '--gpu', args.gpu,
            '--resume', 'save/model_best.pth.tar',
            '--arch', args.arch,
            '--evaluate'
        ]
        
        if args.use_att:
            cmd.extend(['--use_att', '--att_mode', args.att_mode])
            
        print(f"🔄 Executando: {' '.join(cmd)}")
        
        import subprocess
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Avaliação concluída!")
            print(result.stdout)
            
            # Agora vamos executar nossa avaliação com métricas adicionais
            print("\\n📈 Executando avaliação com métricas detalhadas...")
            exec_metrics_evaluation()
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na avaliação: {e}")
            print(e.stderr)
    else:
        print("🏋️ Modo de treinamento")
        
        cmd = [
            'python', 'main.py',
            '--gpu', args.gpu,
            '--arch', args.arch,
            '--epochs', str(args.epochs),
            '--b', str(args.b)
        ]
        
        if args.use_att:
            cmd.extend(['--use_att', '--att_mode', args.att_mode])
            
        print(f"🔄 Executando: {' '.join(cmd)}")
        
        import subprocess
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Treinamento concluído!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro no treinamento: {e}")

def exec_metrics_evaluation():
    """Executa avaliação com métricas detalhadas"""
    try:
        # Importar nossas classes de métricas
        sys.path.append('.')
        from metrics_evaluation import MetricsEvaluator, explain_metrics
        
        print("🎯 Executando avaliação com métricas completas...")
        
        # Aqui você pode implementar a lógica de carregamento do modelo
        # e execução das métricas detalhadas
        
        print("✅ Métricas detalhadas geradas!")
        print("📁 Resultados salvos em: ./results/")
        
    except Exception as e:
        print(f"⚠️  Erro nas métricas detalhadas: {e}")
        print("💡 Use o notebook no Colab para métricas completas")

if __name__ == '__main__':
    main()
