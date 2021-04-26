import os
from datetime import datetime
import argparse
from utility.hook import get_all_layers
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
# import visdom #used to draw loss per epoch
import numpy as np 
from utility import * ##utility is a module
from optimizer import * ##optimizer is a module

torch.set_printoptions(precision=7)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-opt', '--opt', default='SGD', type=str, 
                        help='optimizer used')
    parser.add_argument('-od', '--out_dir', default='', type=str, 
                        help='output directory')
    parser.add_argument('-p', '--target_param', default='', type=str, 
                        help='target param')
    parser.add_argument('-s', '--save_model', default=False, type=bool, 
                        help='save model or not')
    parser.add_argument('-c', '--create_contour', default=False, type=bool, 
                        help='create contour plot or not')
    
   
    args = parser.parse_args()
    args.target_param = 'layer1.0.weight'
    
    filename = os.path.basename(__file__)
    
    now = datetime.now()

    args.out_dir = os.path.join('result', args.opt, filename, now.strftime("%Y%m%d-%H%M%S"))
    
    

    tw = TensorboardWorker(args.out_dir)

    train(args, tw)


def train(args, tw):
    classes = ('0','1','2','3','4','5','6','7','8','9')
    # vis = visdom.Visdom()
    torch.manual_seed(0)
    model = ConvNet()
    device = "cuda:{}".format(args.gpus) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpus)
        model.cuda(args.gpus)
        criterion = nn.CrossEntropyLoss().cuda(args.gpus)
    else:
        criterion = nn.CrossEntropyLoss()
    
    batch_size = 100
    pytorch_optizer = pytorch_opt()
    # define loss function (criterion) and optimizer
    opt1 = pytorch_optizer.create_opt(args.opt)(model.parameters(), 1e-3)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    
    
    
    start = datetime.now()
    total_step = len(train_loader)
    best_acc = 0
    
    if args.create_contour:
        contour_res = {'cost':[],'w_a':[],'w_b':[]}
        contour_obj = contour(model, train_dataset, criterion)
        contour_obj.create_contour_surface()


    for epoch in range(args.epochs):

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            ## registration hook for viualization of the weights or gradients
            visualisation = {}
            def hook_fn(m, i, o):
                visualisation[m] = o 
            get_all_layers(model,hook_fn)


            # Forward pass
            outputs = model(images)
            
            visualisation.keys()            
            
            loss = criterion(outputs, labels)

            # Backward and optimize
            opt1.zero_grad()
            loss.backward()
            opt1.step()
            epoch_loss = loss.item()
            
            
            # ...log the running loss
            tw.writer.add_scalar('training loss',
                            epoch_loss,
                            epoch * len(train_loader) + i)

            
            ################## Evaluation ###################
            val_accuracy = tw.accuracy(model, images, labels)
            
            tw.writer.add_scalar('evaluation accuracy',
                            val_accuracy,
                            epoch * len(train_loader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            tw.writer.add_figure('predictions vs. actuals',
                            tw.plot_classes_preds(model, images, labels, classes),
                            global_step=epoch * len(train_loader) + i)
            

            ##saving the weights 


            # Print model's state_dict
            if args.create_contour:
                contour_res['cost'].append(epoch_loss)
                # print("Model's state_dict:")
                model_weights_dict = model.module.state_dict() if args.gpus > 1 else model.state_dict()
                for param_tensor in model_weights_dict:
                    # print(param_tensor, "\t", model_weights_dict[param_tensor].size())
                    if param_tensor == args.target_param:
                        weight = model_weights_dict[args.target_param].data
                        # weight = model.fc.weight.data
                        #print("weights means are", torch.mean(weight))
                        w_a_data = weight[0,0,0,1].data
                        w_b_data = weight[0,0,0,2].data 
                        contour_res['w_a'].append(w_a_data)
                        contour_res['w_b'].append(w_b_data)
            
            
            
            is_best = val_accuracy > best_acc
            best_acc = max(val_accuracy, best_acc)
        
            state = {
                'acc': val_accuracy,
                'epoch': epoch,
                'state_dict': model.module.state_dict() if args.gpus > 1 else model.state_dict(),
            }
            opt_state = {
                'optimizer': opt1.state_dict()
            }
            
            if is_best and args.save_model:
                
                model_path = os.path.join(args.out_dir, 'saved_models','model_best.t7')
                opt_path = os.path.join(args.out_dir, 'saved_models','opt_best.t7')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                os.makedirs(os.path.dirname(opt_path), exist_ok=True)
                torch.save(state, model_path)
                torch.save(opt_path, opt_state)
            # if i == 50:
            #     break 

    if args.create_contour:
        contour_obj.contour_anmiation(contour_res)

            # vis.line(np.array([epoch_loss]))
            #print(epoch_loss)

        #   epoch+1
        #   i+1 
        #     total_step 
            
        # str(datetime.now() - start)

if __name__ == '__main__':
    main()