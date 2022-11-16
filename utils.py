import os
import torch 
import torchvision
import torchvision.transforms as transforms
from args import parser 

def get_img_size():
    dataset = torchvision.datasets.VOCSegmentation(root='./data', 
                                                image_set= 'trainval', 
                                                year='2011',
                                                download=False,
                                                transform=transforms.ToTensor(),
                                                target_transform=transforms.ToTensor()
                                                )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True)  
    
    h = []
    w = []
    
    for i, (img, target) in enumerate(dataloader):
        # N C H W
        # print(target.shape, img.shape) # 1 3 H W
        
        h.append(img.size(2))
        w.append(img.size(3))
        
        print(target)
    
    w = sorted(w)
    h = sorted(h)
    
    print('Width Max: ', w[-1],' Min: ', w[0])
    print('Height Max: ', h[-1],' Min: ', h[0])
    
    # Width Max:  500  Min:  174
    # Height Max:  500  Min:  112
    return w[-1], w[0], h[-1], h[-1]

def save_checkpoint(epoch, model, optimizer, filename, lr):
    state ={
        'Epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr' : lr
    }
    
    torch.save(state, filename)

def get_dir_name():
    arg = parser.parse_args()
    
    split_symbol = '~' if os.name == 'nt' else ':'
    model_name_template = split_symbol.join(['S:{}_mini_batch', '{}_layer', '{}_id'])
    model_name = model_name_template.format(arg.mini_batch, arg.layer, arg.id)
    
    dir_name = os.path.join(model_name)
    
    return dir_name 

def get_data(mode):    
    arg = parser.parse_args()

    batch_size = arg.mini_batch 

    data_transform = transforms.Compose([transforms.Resize(size= (572, 572)),
                                          transforms.ToTensor()])               # add image normalization if need 
    
    target_transform = transforms.Compose([transforms.Resize(size= (388, 388)),
                                           transforms.PILToTensor()
                                           ])  
    
    target_transform_ = transforms.Compose([transforms.Resize(size= (388, 388)),
                                           transforms.ToTensor()
                                        ])  
    
    if(mode=='train'):
        trainset = torchvision.datasets.VOCSegmentation(root='./data', 
                                                image_set= 'train', 
                                                year='2011',
                                                download=False,
                                                transform = data_transform,
                                                target_transform=target_transform
                                                )

        dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)   
        
    elif (mode=='val'):
        validset = torchvision.datasets.VOCSegmentation(root='./data', 
                                image_set= 'val', 
                                year='2011',
                                download=False,
                                transform = data_transform,
                                target_transform=target_transform
                                )
          
        dataloader = torch.utils.data.DataLoader(validset, batch_size = batch_size, shuffle=True)

    elif (mode =='test'):
        testset = torchvision.datasets.VOCSegmentation(root='./data', 
                                        image_set= 'test', 
                                        year='2011',
                                        download=False,
                                        transform = data_transform,
                                        target_transform=target_transform
                                        )  
        
        dataloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False) 
        
    return dataloader

def get_file_name(PATH):
    arg = parser.parse_args()
    
    if(arg.mode == 'resume'):
        model_path = os.path.join('./save/', PATH)
        file_path = model_path + '/UNet.pth'
        tb_pth_train = os.path.join('./logs/train/', PATH)
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        tb_pth_test = os.path.join('./logs/test/', PATH)
    
    else:
        model_path = os.path.join('./save/', PATH)
        
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        
        file_path = model_path + '/UNet.pth'
        
        # train
        tb_pth_train = os.path.join('./logs/train/', PATH)
        if not os.path.isdir(tb_pth_train):
            os.makedirs(tb_pth_train)
        
        # valid 
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        if not os.path.isdir(tb_pth_valid):
            os.makedirs(tb_pth_valid)        
        
        # test 
        tb_pth_test = os.path.join('./logs/test/', PATH)
        if not os.path.isdir(tb_pth_test):
            os.makedirs(tb_pth_test)        
    
    return file_path, tb_pth_train, tb_pth_valid, tb_pth_test


class AverageMeter():
# https://github.com/ddamddi/UNet-pytorch/blob/bfb1c47147ddeb8a85b3b50a4af06b3a2082d933/utils.py
    
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
if __name__=="__main__":
    get_img_size()
