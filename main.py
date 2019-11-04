import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataloader import *

dataset=SR_Dataloader('/MSE_SYSTEMS/Data/SuperResolution/DIV2K_train_HR',300,3)
train_loader=torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=32)
tb=SummaryWriter()

device=torch.device('cuda')
model=ESPCN(3).to(device)
model.train()
loss=torch.nn.MSELoss(reduction='mean')
learning_rate=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# sample=next(iter(train_loader))
# lr_image=sample['low_res']
# hr_image=sample['high_res']
# grid_lr=torchvision.utils.make_grid(lr_image)
# grid_hr=torchvision.utils.make_grid(hr_image)
# tb.add_image('lr_images',grid_lr,0)
# tb.add_image('lr_images',grid_hr,0)

for epoch in range(30):
    epoch_loss=0
    optimizer.zero_grad()

    if(epoch%5==0):
        for param_group in optimizer.param_groups:
            param_group['lr']=learning_rate/2

    for idx,sample in enumerate(train_loader):
        lr_image=sample['low_res'].to(device)
        hr_image=sample['high_res'].to(device)
        hr_est=model(lr_image)

        grid_lr=torchvision.utils.make_grid(lr_image)
        tb.add_image('lr_images',grid_lr,0)
        # grid_est=torchvision.utils.make_grid(hr_est)
        # tb.add_image('estHR_images',grid_est,0)

        batch_loss=loss(hr_est,hr_image)
        epoch_loss+=batch_loss
        batch_loss.backward()
        optimizer.step()
        print(batch_loss)
        tb.add_scalar('batch_loss',batch_loss,idx)
    tb.add_scalar('epoch_loss',epoch_loss,epoch)

tb.close()
