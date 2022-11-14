import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn  
from model import UNet
from utils import AverageMeter

# import matplotlib as plt

from args import parser 
from utils  import save_checkpoint, get_dir_name, get_data, get_file_name, get_img_size 

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

def validate():
    print(" ")

def train(train, valid):    
    arg = parser.parse_args()
    epoch = arg.epoch
    lr = arg.lr
    batch_num = arg.mini_batch
    
    path = get_dir_name()
    file_path, tb_pth_train, tb_pth_valid, tb_pth_test  = get_file_name(path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model = UNet().to(device)

    if(arg.mode == 'resume'):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['Epoch']
        lr = checkpoint['lr']
        
    optimizer = optim.SGD(model.parameters(), lr= lr, weight_decay= 0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss() 
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                          step_size=100,
                                          gamma=0.1)
    
    for e in range(epoch):
        writer = SummaryWriter(tb_pth_train) 
        running_loss = 0.0
        correct = 0.0
        total = 0.0
    
        losses = AverageMeter()
        for i, (img, target) in enumerate(train):
            # target = target - (target == 255).int() * 255 # ?
  
            img, target = img.to(device), target.to(device)
            output = model(img)

            print(output.size(), target.size())
            
            loss = criterion(target, output)
            losses.update(loss.item())
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
            print(losses.avg)
            # # accuracy
            # _, predicted = torch.max(output.data, 1) 
            # total+= labels.size(0) 
            # correct += (predicted == labels).sum().item()

    
        # scheduler.step() 
    
    
def test(test):
    print("=======================Test=======================")
    
def main():
    print("Start")
    
    # w_max, w_min, h_max, h_min = get_img_size()
    
    trainload = get_data('train')
    validload = get_data('val')
    # testload = get_data('test')
        
    # data = trainload.__getitem__(0)

    train(trainload, validload)
    # test(testload)
    torch.cuda.empty_cache()
    
if __name__=="__main__":
    main()
