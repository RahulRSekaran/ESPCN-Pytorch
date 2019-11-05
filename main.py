import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from dataloader import *

dataset=SR_Dataloader('/MSE_SYSTEMS/Data/SuperResolution/DIV2K_train_HR',300,3)
train_loader=torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=8)
tb=SummaryWriter()

device=torch.device('cuda')
model=ESPCN(3).to(device)
loss=torch.nn.MSELoss(reduction='mean')
learning_rate=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
best_loss=99999


# sample=next(iter(train_loader))
# hr_image=sample['high_res']
# grid_hr=torchvision.utils.make_grid(hr_image)
# tb.add_image('hr_images',grid_hr,0)
# hr_image=hr_image.to(device)
#
# lr_image=sample['low_res'].to(device)
# hr_est=model(lr_image)
# print(loss(hr_est,hr_image).item())
# hr_est=hr_est.detach().cpu()
# grid_est=torchvision.utils.make_grid(hr_est)
# tb.add_image('hr_estimate',grid_est,0)

# grid_lr=torchvision.utils.make_grid(lr_image)
# tb.add_image('lr_images',grid_lr,0)

model.train()

for epoch in range(75):
    epoch_loss=0

    if(epoch%5==0):
        for param_group in optimizer.param_groups:
            param_group['lr']=learning_rate/2

    for idx,sample in enumerate(train_loader):
        optimizer.zero_grad()
        lr_image=sample['low_res'].to(device)
        hr_image=sample['high_res'].to(device)
        hr_est=model(lr_image)

        batch_loss=loss(hr_est,hr_image)
        epoch_loss+=batch_loss
        batch_loss.backward()
        optimizer.step()
        print(batch_loss)
        tb.add_scalar('batch_loss',batch_loss,idx)

        grid_hr=torchvision.utils.make_grid(hr_image.detach().cpu())
        tb.add_image('hr_images',grid_hr,0)

        grid_est=torchvision.utils.make_grid(hr_est.detach().cpu())
        tb.add_image('hr_estimate',grid_est,0)

    if(epoch_loss<best_loss):
        checkpoint={'model_state_dict':model.state_dict(),'optim_state_dict':optimizer.state_dict()}
        torch.save(checkpoint,'checkpoint.pth')

    tb.add_scalar('epoch_loss',epoch_loss,epoch)

tb.close()
