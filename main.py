import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn  
from model import UNet
from utils import AverageMeter
import numpy as np
from torchvision.transforms.functional import resize 
import torch.nn.functional as F

# import matplotlib as plt

from args import parser 
from utils  import save_checkpoint, get_dir_name, get_data, get_file_name, get_img_size 

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

def pixel_accuray(output: torch.Tensor, target: torch.Tensor):
    pixel_accuray = 0.0
    
    with torch.no_grad():
        # target = F.softmax(target, dim=1)
        # target = torch.argmax(target, dim=1)
        
        # output = F.softmax(target, dim=1)
        output_ = torch.argmax(output, dim=1)

        correct = (output_ == target).int()
        accuracy = float(correct.sum()) / float(correct.numel()) # https://github.com/ddamddi/UNet-pytorch/blob/bfb1c47147ddeb8a85b3b50a4af06b3a2082d933/metrics.py#L7
        print(accuracy, float(correct.sum()), float(correct.numel()) )
        
    return accuracy 

def validate(model_, validloader, tb_pth_valid, epoch, batch_num_val, lr_):
    writer = SummaryWriter(tb_pth_valid)  
    
    model = model_
    optimizer = optim.SGD(model.parameters(), lr= lr_, weight_decay= 0.0001, momentum=0.99)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(tb_pth_valid) 
    
    validation_loss = 0.0    
    correct = 0.0
    total = 0.0
    
    with torch.no_grad():   
        model.eval() # check                    
        for j, val_data in enumerate(validloader, 0):
            val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_out = model(val_inputs)
            
            val_loss = criterion(val_out, val_labels)
            validation_loss += val_loss.item()
        
            _, predicted = torch.max(val_out.data, 1) 
            total+= val_labels.size(0) 
            correct += (predicted == val_labels).sum().item()               
            
            if j % (batch_num_val + 1) == batch_num_val:    
                print(f' val loss: {validation_loss / (batch_num_val + 1):.3f}, valid accuracy: {(100.0 * correct / total):.2f}%, lr: {optimizer.param_groups[0]["lr"]:.6f}')
                writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1) # revise 
                writer.add_scalar("Loss/val", validation_loss / (batch_num_val + 1.0), epoch + 1)
                writer.add_scalar("Accuracy/val", 100.0 * correct / total, epoch + 1)
                writer.add_scalar("Error/val", 100.0 - 100.0 * correct / total, epoch + 1)
                validation_loss = 0.0
                
        writer.close()
    
def train(train, valid):    
    arg = parser.parse_args()
    epoch_num = arg.epoch
    lr = arg.lr
    batch_num = arg.mini_batch
    
    path = get_dir_name()
    file_path, tb_pth_train, tb_pth_valid, tb_pth_test  = get_file_name(path)
 
    model = UNet().to(device)

    if(arg.mode == 'resume'):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # check 
        epoch_load = checkpoint['Epoch']
        lr = checkpoint['lr']
        
    optimizer = optim.SGD(model.parameters(), lr= lr, weight_decay= 0.0001, momentum=0.99)
    criterion = nn.CrossEntropyLoss().to(device)

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                          step_size=100,
                                          gamma=0.1)
    
    writer = SummaryWriter(tb_pth_train) 
    for epoch in range(epoch_num):
        if(arg.mode=='resume' and epoch_num==0):
            epoch = epoch_load
            
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        
        losses = AverageMeter()
        for i, (img, target) in enumerate(train):
            # target = target - (target == 255).int() * 255 # 255 ?

            optimizer.zero_grad() 
                        
            img, target = img.to(device), target.to(device)
            output = model(img)
            # output masking

            target = torch.argmax(target, dim=1)
            
            loss = criterion(output, target)
            
            losses.update(loss.item())

            loss.backward()
            optimizer.step() 
            
            accuracy = pixel_accuray(output, target)
            
            #print(f'[{i + 1}], train loss: {(losses.avg):.3f}, train accuracy: {(100.0 * accuracy):.2f}%,' , end=' ') 
        
        print(f'[{epoch + 1} / {epoch_num}], train loss: {(losses.avg):.3f}, train accuracy: {(100.0 * accuracy):.2f}%')     
        writer.add_scalar("Loss/train", losses.avg, epoch + 1)
        writer.add_scalar("Accuracy/train", 100.0 * accuracy)

        scheduler.step() 
    writer.close()

    save_checkpoint(epoch, model, optimizer, file_path, lr) 
    # validate(valid)    
    
def main():
    print("Start")
    torch.cuda.empty_cache()
    
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # w_max, w_min, h_max, h_min = get_img_size()

    trainload = get_data('train')
    validload = get_data('val')
    # testload = get_data('test')
        
    train(trainload, validload)
    # test(testload)
    
if __name__=="__main__":
    main()
