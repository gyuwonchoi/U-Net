import torch 
import torch.cuda
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

def pixel_accuracy(output, target): # https://github.com/ddamddi/UNet-pytorch/blob/bfb1c47147ddeb8a85b3b50a4af06b3a2082d933/metrics.py#L7
    with torch.no_grad():
        # output_ = F.softmax(output, dim=1) 
        output_ = torch.argmax(output, dim=1).long()
                
        # correct = (output_ == target).int()
        correct = torch.eq(output_, target).int()
        accuracy = float(correct.sum()) / float(correct.numel()) 
        
    return accuracy 

def meanIU(output, target):
    with torch.no_grad():  
        # output_ = F.softmax(output, dim=1)
        output_ = torch.argmax(output, dim=1).long() # check
        batch = output.size(0)

        IoU = 0.0
        
        for i in range(0, 21): # 0-20
            label = torch.ones((batch, 388, 388), dtype = torch.long).to(device)
            label = (label * i).long()
          
            output_u = torch.eq(output_ , label).int()
            # output_inter = output_u * i 
            target_u = torch.eq(target , label).int()
            # target_inter = target_u * i 
            intersection = (target_u * output_u).int()  # output_ == target == label , except zero , count True 
            region = output_u.sum() + target_u.sum() - intersection.sum()
            
            if region ==0:
                continue
            else:
                IoU += intersection.sum() / (output_u.sum() + target_u.sum() - intersection.sum()) # check : -4.8
          
        meanIoU = IoU/21.0
    return meanIoU

def validate(model_, validloader, tb_pth_valid, epoch, batch_num_val, lr_):
    writer = SummaryWriter(tb_pth_valid)  
    
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    model = model_
    optimizer = optim.SGD(model.parameters(), lr= lr_, weight_decay= 0.0001, momentum=0.99)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(tb_pth_valid) 
        
    losses = AverageMeter()
    pixel_acc = AverageMeter()   
    meanIoU = AverageMeter() 
    infer_time = AverageMeter()

    starter.record()
    with torch.no_grad():   
        # model.eval()              
        for j, (img, target) in enumerate(validloader):
            target = target - (target == 255).long() * 255
            img, target = img.to(device), target.to(device)
            
            target = torch.squeeze(target, dim=1)

            pred = model(img)
            
            loss = criterion(pred, target)
            losses.update(loss.item())
        
            pixel_acc.update(pixel_accuracy(pred, target))
            meanIoU.update(meanIU(pred, target))
            
        print(f'valid loss: {(losses.avg):.3f}, valid accuracy: {(100.0 * pixel_acc.avg):.2f}% mIoU: {(100.0 * meanIoU.avg):.2f}')     
    ender.record()
    torch.cuda.synchronize()
    infreTime = starter.elapsed_time(ender)
    infer_time.update(infreTime)
   
    writer.add_scalar("Loss/valid", losses.avg, epoch + 1)
    writer.add_scalar("Accuracy/valid", 100.0 * pixel_acc.avg, epoch+1) 
    writer.add_scalar("meanIoU/valid", 100.0* meanIoU.avg, epoch+1) 
    writer.add_scalar("InferTime/valid", infer_time.avg, epoch+1) 
    
    writer.close()
    
def train(train, valid):    
    torch.cuda.synchronize(device)
    ender = torch.cuda.Event(enable_timing=True)
    starter = torch.cuda.streams.Event(enable_timing=True)
    
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
        
    optimizer = optim.SGD(model.parameters(), lr= lr, momentum=0.99)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                          step_size=100,
                                          gamma=0.1)
    
    writer = SummaryWriter(tb_pth_train) 
    starter.record()
    for epoch in range(epoch_num):
        if(arg.mode=='resume' and epoch_num==0):
            epoch = epoch_load
    
        losses = AverageMeter()
        pixel_acc = AverageMeter()
        meanIoU = AverageMeter()
        infer_time = AverageMeter()
        for i, (img, target) in enumerate(train):
            optimizer.zero_grad() 
            
            # ***** Be aware your image does not mark boudaries!! ******
            # boundary(255) -> background(0)
            # backgroud(0), object(1-20)
            target = target - (target == 255).long() * 255
            img, target = img.to(device), target.to(device) # img: 0 ~ 1, target: 0 ~ 20
            
            output = model(img) # 0 ~ 1 by sigmoid 
                        
            target = torch.squeeze(target, dim=1)
            
            loss = criterion(output, target) # float, long , one-hot coding for target
            losses.update(loss.item())
            loss.backward()
            optimizer.step() 
            
            pixel_acc.update(pixel_accuracy(output, target))   
            meanIoU.update(meanIU(output, target)) 
            if i % 100 == 0: 
                print(f'======iteration: [{i}], train loss: {(losses.avg):.3f}, train accuracy: {(100.0 * pixel_accuracy(output, target)):.2f}%,======') 
        
        ender.record()
        torch.cuda.synchronize(device)
        infreTime = starter.elapsed_time(ender)
        infer_time.update(infreTime)
        
        print(f'[{epoch + 1} / {epoch_num}], time: {(infer_time.avg):.2f} ms, train loss: {(losses.avg):.3f}, train accuracy: {(100.0 * pixel_acc.avg):.2f}%', end =' ')     
        writer.add_scalar("Loss/train", losses.avg, epoch + 1)
        writer.add_scalar("Accuracy/train", 100.0 * pixel_acc.avg, epoch+1)
        writer.add_scalar("mIoU/train", 100.0 * meanIoU.avg, epoch+1)
        writer.add_scalar("Time/inference", infer_time.avg, epoch+1)
        
        scheduler.step() 
            
        save_checkpoint(epoch, model, optimizer, file_path, lr) 
        validate(model, valid, tb_pth_valid, epoch, batch_num, lr)  

    writer.close()
   
    
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
