
def get_learning_rate(optimizer):
    lr = optimizer.param_groups[0]['lr']
    return lr

# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    return lr