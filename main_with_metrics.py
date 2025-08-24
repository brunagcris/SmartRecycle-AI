"""
main_with_metrics.py - Versão modificada do main.py com métricas completas
Inclui avaliação detalhada com precisão, recall, F1-score, matriz de confusão e gráficos
"""

import argparse
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from data import TrashDataset
import resnet 
from utils import indexes_to_one_hot
from metrics_evaluation import MetricsEvaluator, explain_metrics

def get_arguments():
    parser = argparse.ArgumentParser(description='RecycleNet')
    parser.add_argument('--b', '--batch', type=int, default=16)
    parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--arch', type=str, default='resnet18_base', help='resnet18, 34, 50, 101, 152')
    parser.add_argument('--att_mode', type=str, default='ours', help='attention module mode: ours, cbam, se')
    parser.add_argument('--use_att', action='store_true', help='use attention module')
    parser.add_argument('--no_pretrain', action='store_false', help='training from scratch')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--adjust-freq', type=int, default=40, help='learning rate adjustment frequency (default: 40)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
    
    # Novos argumentos para métricas
    parser.add_argument('--save_metrics', action='store_true', help='save detailed metrics')
    parser.add_argument('--plot_metrics', action='store_true', help='plot metrics graphs')
    parser.add_argument('--detailed_eval', action='store_true', help='Perform detailed evaluation with all metrics')
    parser.add_argument('--metrics_dir', type=str, default='metrics/', help='directory to save metrics')
    parser.add_argument('--metrics_dir', type=str, default='metrics/', help='directory to save metrics')
    
    return parser.parse_args()

def main():
    args = get_arguments()

    BATCH_SIZE = args.b
    GPU = args.gpu
    ROOT_DIR = args.root_dir
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Criar diretório para métricas
    if not os.path.exists(args.metrics_dir):
        os.makedirs(args.metrics_dir)

    # Inicializar evaluator de métricas
    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    metrics_evaluator = MetricsEvaluator(class_names=class_names, save_dir=args.metrics_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    def ToCudaVariable(xs, volatile=False, requires_grad=True):
        if torch.cuda.is_available():
            return [x.cuda() for x in xs]
        else:
            return xs

    # Inicializar modelo
    if args.arch == 'resnet18_base':
        model = nn.DataParallel(resnet.resnet18(pretrained=args.no_pretrain, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    elif args.arch == 'resnet34_base':
        model = nn.DataParallel(resnet.resnet34(pretrained=args.no_pretrain, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    elif args.arch == 'resnet50_base':
        model = nn.DataParallel(resnet.resnet50(pretrained=args.no_pretrain, num_classes=6, use_att=args.use_att, att_mode=args.att_mode).to(device))
    else:
        raise ValueError(f"Architecture {args.arch} not supported")

    criterion = nn.CrossEntropyLoss().to(device)

    # Se não usar pré-treinamento, alguns parâmetros de atenção podem ter LR diferente
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('=> best accuracy {}'.format(best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Preparar data loaders
    train_img_transform = transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=MEAN, std=STD)])
    train_dataset = TrashDataset(ROOT_DIR, train_img_transform, 'train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_img_transform = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=MEAN, std=STD)])
    val_dataset = TrashDataset(ROOT_DIR, val_img_transform, 'val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=True)

    test_img_transform = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=MEAN, std=STD)])
    test_dataset = TrashDataset(ROOT_DIR, test_img_transform, 'test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print("🔍 MODO AVALIAÇÃO - Calculando métricas detalhadas...")
        test_metrics = test_with_metrics(args, test_loader, model, criterion, device, metrics_evaluator)
        
        # Imprimir métricas
        metrics_evaluator.print_metrics(test_metrics)
        
        # Salvar métricas
        if args.save_metrics:
            metrics_path = os.path.join(args.metrics_dir, 'test_metrics.json')
            metrics_evaluator.save_metrics(metrics_path, test_metrics)
        
        # Plotar gráficos
        if args.plot_metrics:
            confusion_matrix_path = os.path.join(args.metrics_dir, 'confusion_matrix.png')
            metrics_comparison_path = os.path.join(args.metrics_dir, 'metrics_comparison.png')
            
            metrics_evaluator.plot_confusion_matrix(confusion_matrix_path, test_metrics)
            metrics_evaluator.plot_metrics_comparison(metrics_comparison_path, test_metrics)
        
        return

    # Treinamento
    print("🚀 INICIANDO TREINAMENTO...")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, args.adjust_freq)

        # Treinar uma época
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, epoch, device)
        metrics_evaluator.update_training_metrics(train_loss, train_acc)

        # Validar
        val_loss, val_acc = validate(args, val_loader, model, criterion, device)
        metrics_evaluator.update_validation_metrics(val_loss, val_acc)

        # Salvar melhor modelo
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        save_checkpoint({
            'epoch' : epoch + 1,
            'arch' : args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            }, is_best, args.save_dir)

    # Plotar curvas de treinamento ao final
    if args.plot_metrics:
        training_curves_path = os.path.join(args.metrics_dir, 'training_curves.png')
        metrics_evaluator.plot_training_curves(training_curves_path)

    # Teste final com métricas detalhadas
    print("🎯 TESTE FINAL - Calculando métricas detalhadas...")
    final_metrics = test_with_metrics(args, test_loader, model, criterion, device, MetricsTracker(num_classes=6, class_names=class_names))
    
    # Imprimir e salvar métricas finais
    metrics_evaluator.print_metrics(final_metrics)
    
    if args.save_metrics:
        final_metrics_path = os.path.join(args.metrics_dir, 'final_test_metrics.json')
        metrics_evaluator.save_metrics(final_metrics_path, final_metrics)
    
    if args.plot_metrics:
        final_confusion_matrix_path = os.path.join(args.metrics_dir, 'final_confusion_matrix.png')
        final_metrics_comparison_path = os.path.join(args.metrics_dir, 'final_metrics_comparison.png')
        
        metrics_evaluator.plot_confusion_matrix(final_confusion_matrix_path, final_metrics)
        metrics_evaluator.plot_metrics_comparison(final_metrics_comparison_path, final_metrics)

def train(args, train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.to(device)
        target = torch.from_numpy(np.asarray(target))
        target = target.to(device)

        output = model(input)

        loss = criterion(output[0], target)

        acc1 = accuracy(output[0], target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.to(device)

            target = torch.from_numpy(np.asarray(target))
            target = target.to(device)

            output = model(input)
            loss = criterion(output[0], target)

            acc1 = accuracy(output[0], target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Val Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg

def test_with_metrics(args, test_loader, model, criterion, device, metrics_evaluator):
    """
    Função de teste que calcula métricas detalhadas
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    metrics_evaluator.reset()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, input_path) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.to(device)

            target = torch.from_numpy(np.asarray(target))
            target = target.to(device)

            output = model(input)
            loss = criterion(output[0], target)

            # Calcular accuracy tradicional
            acc1 = accuracy(output[0], target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))

            # Atualizar métricas detalhadas
            metrics_evaluator.update(output[0], target)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Test Acc@1 {top1.avg:.3f}'.format(top1=top1))

    # Calcular e retornar métricas detalhadas
    return metrics_evaluator.calculate_metrics()

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))

def adjust_learning_rate(args, optimizer, epoch, N):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    lr = args.lr * (0.1 ** (epoch // N))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


if __name__ == '__main__':
    main()
