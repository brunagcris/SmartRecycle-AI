import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import resnet

def get_arguments():
    parser = argparse.ArgumentParser(description='RecycleNet Webcam Inference')
    parser.add_argument('--resume', default='save/model_best.pth.tar', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--arch', type=str, default='resnet18_base', help='resnet18, 34, 50, 101, 152')
    parser.add_argument('--use_att', action='store_true', help='use attention module')
    parser.add_argument('--att_mode', type=str, default='ours', help='attention module mode: ours, cbam, se')
    parser.add_argument('--save_dir', type=str, default='webcam_captures/', help='directory to save captured images')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='minimum confidence for prediction')
    parser.add_argument('--show_probabilities', action='store_true', help='show all class probabilities')
    return parser.parse_args()

def load_model(args):
    """Carrega o modelo treinado"""
    # Detectar dispositivo
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Inicializar modelo
    if args.arch == 'resnet18_base':
        model = nn.DataParallel(resnet.resnet18(pretrained=False, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    elif args.arch == 'resnet34_base':
        model = nn.DataParallel(resnet.resnet34(pretrained=False, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    elif args.arch == 'resnet50_base':
        model = nn.DataParallel(resnet.resnet50(pretrained=False, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    else:
        raise ValueError(f"Architecture {args.arch} not supported")
    
    # Carregar pesos
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Verificar se o modelo tem atenção ou não
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best acc: {checkpoint['best_acc1']:.3f})")
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e) and "att" in str(e):
                print("⚠️  Modelo salvo não tem módulo de atenção. Carregando sem atenção...")
                # Recriar modelo sem atenção
                if args.arch == 'resnet18_base':
                    model = nn.DataParallel(resnet.resnet18(pretrained=False, num_classes=6, use_att=False).to(device))
                elif args.arch == 'resnet34_base':
                    model = nn.DataParallel(resnet.resnet34(pretrained=False, num_classes=6, use_att=False).to(device))
                elif args.arch == 'resnet50_base':
                    model = nn.DataParallel(resnet.resnet50(pretrained=False, num_classes=6, use_att=False).to(device))
                
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> Loaded checkpoint '{args.resume}' without attention (epoch {checkpoint['epoch']}, best acc: {checkpoint['best_acc1']:.3f})")
            else:
                raise e
    else:
        raise FileNotFoundError(f"=> No checkpoint found at '{args.resume}'")
    
    model.eval()
    return model, device

def preprocess_image(image):
    """Pré-processa a imagem para inferência"""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Converter de BGR para RGB (OpenCV usa BGR)
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Transformações
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    image_tensor = img_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    return image_tensor

def predict_image(model, image_tensor, device, show_probabilities=False):
    """Faz predição na imagem"""
    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
        # Se o modelo retorna tupla (com attention), pegar primeiro elemento
        if isinstance(output, tuple):
            output = output[0]
        
        # Aplicar softmax para obter probabilidades
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Obter predição e confiança
        confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        predicted_name = class_names[predicted_class]
        
        result = {
            'predicted_class': predicted_class,
            'predicted_name': predicted_name,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0] if show_probabilities else None
        }
        
        return result

def draw_prediction_on_frame(frame, prediction, show_probabilities=False, confidence_threshold=0.5):
    """Desenha a predição no frame"""
    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    
    height, width = frame.shape[:2]
    
    # Definir cores para cada classe
    colors = {
        'Glass': (255, 0, 0),      # Azul
        'Paper': (0, 255, 0),      # Verde
        'Cardboard': (0, 165, 255), # Laranja
        'Plastic': (255, 255, 0),   # Ciano
        'Metal': (128, 0, 128),     # Roxo
        'Trash': (0, 0, 255)       # Vermelho
    }
    
    # Cor da predição
    color = colors.get(prediction['predicted_name'], (255, 255, 255))
    
    # Status da confiança
    is_confident = prediction['confidence'] >= confidence_threshold
    status = "CONFIDENT" if is_confident else "LOW CONFIDENCE"
    status_color = (0, 255, 0) if is_confident else (0, 165, 255)
    
    # Desenhar fundo para texto
    cv2.rectangle(frame, (10, 10), (width-10, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width-10, 120), color, 2)
    
    # Texto principal
    cv2.putText(frame, f"Prediction: {prediction['predicted_name']}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Confidence: {prediction['confidence']:.3f} ({prediction['confidence']*100:.1f}%)", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Mostrar todas as probabilidades se solicitado
    if show_probabilities and prediction['probabilities'] is not None:
        y_offset = 140
        cv2.rectangle(frame, (10, 130), (width-10, 130 + len(class_names)*25), (0, 0, 0), -1)
        
        for i, (class_name, prob) in enumerate(zip(class_names, prediction['probabilities'])):
            text_color = colors.get(class_name, (255, 255, 255))
            cv2.putText(frame, f"{class_name}: {prob:.3f} ({prob*100:.1f}%)", 
                        (20, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Instruções
    instructions_y = height - 80
    cv2.rectangle(frame, (10, instructions_y-10), (width-10, height-10), (50, 50, 50), -1)
    cv2.putText(frame, "CONTROLS:", (20, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "SPACE: Capture & Predict | S: Save Image | P: Toggle Probabilities | Q/ESC: Quit", 
                (20, instructions_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def main():
    args = get_arguments()
    
    # Criar diretório para salvar imagens
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Carregar modelo
    print("🤖 Carregando modelo...")
    model, device = load_model(args)
    print("✅ Modelo carregado com sucesso!")
    
    # Inicializar webcam
    print("📹 Inicializando webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a webcam")
        return
    
    print("✅ Webcam inicializada!")
    print("\n🎯 INSTRUÇÕES:")
    print("   • SPACE: Capturar imagem e fazer predição")
    print("   • S: Salvar imagem atual")
    print("   • P: Alternar exibição de probabilidades")
    print("   • Q ou ESC: Sair")
    
    # Variáveis de controle
    show_probabilities = args.show_probabilities
    capture_count = 0
    last_prediction = None
    
    try:
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Erro ao capturar frame da webcam")
                break
            
            # Criar uma cópia para exibição
            display_frame = frame.copy()
            
            # Se há uma predição anterior, mostrar
            if last_prediction is not None:
                display_frame = draw_prediction_on_frame(
                    display_frame, last_prediction, show_probabilities, args.confidence_threshold
                )
            else:
                # Mostrar instruções quando não há predição
                cv2.putText(display_frame, "Press SPACE to make prediction", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Exibir frame
            cv2.imshow('RecycleNet - Trash Classification', display_frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Fazer predição
                print("🔍 Fazendo predição...")
                
                # Pré-processar imagem
                image_tensor = preprocess_image(frame)
                
                # Fazer predição
                prediction = predict_image(model, image_tensor, device, True)
                last_prediction = prediction
                
                # Mostrar resultado no console
                print(f"📊 Resultado:")
                print(f"   • Classe: {prediction['predicted_name']}")
                print(f"   • Confiança: {prediction['confidence']:.3f} ({prediction['confidence']*100:.1f}%)")
                
                if prediction['probabilities'] is not None:
                    print(f"   • Todas as probabilidades:")
                    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
                    for name, prob in zip(class_names, prediction['probabilities']):
                        print(f"     - {name}: {prob:.3f} ({prob*100:.1f}%)")
            
            elif key == ord('s') or key == ord('S'):  # S - Salvar imagem
                capture_count += 1
                filename = os.path.join(args.save_dir, f'capture_{capture_count:03d}.jpg')
                cv2.imwrite(filename, frame)
                print(f"💾 Imagem salva: {filename}")
            
            elif key == ord('p') or key == ord('P'):  # P - Toggle probabilidades
                show_probabilities = not show_probabilities
                print(f"🔄 Exibição de probabilidades: {'ON' if show_probabilities else 'OFF'}")
            
            elif key == ord('q') or key == ord('Q') or key == 27:  # Q ou ESC - Sair
                print("👋 Encerrando...")
                break
    
    except KeyboardInterrupt:
        print("\\n👋 Interrompido pelo usuário")
    
    finally:
        # Limpar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados")

if __name__ == '__main__':
    main()
