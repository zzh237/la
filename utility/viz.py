loss_window = vis.line(
Y=torch.zeros((1),device=device),
X=torch.zeros((1),device=device),
opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['Loss']))

vis.line(X=torch.ones((1,1),device=device)*epoch,Y=torch.Tensor([epoch_loss],device=device).unsqueeze(0),win=loss_window,update='append')
        