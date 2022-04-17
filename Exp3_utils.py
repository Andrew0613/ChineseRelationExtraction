import os
import torch
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def print_networks(self, verbose):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(results,res_dir,file_name,is_prediction=False):
    """
    input: 
    results: results to save
    res_dir: path where results are gonna be saved
    file_name: name of result file
    is_prediciton: if the results are prediction
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    save_dir = os.path.join(res_dir,file_name)
    f = open(save_dir,'w')
    for result in results:
        if is_prediction:
            for r in result:
                f.writelines(str(r.item())+"\n")
        else:
            f.writelines(str(result)+"\n")
    f.close()
    print("Output finished")
def save_his(his,opt):
    acc_his, time_his,loss_his,train_acc_his= his['acc'],his['time'],his['loss'],his['train_acc']
    res_dir = os.path.join(opt.results_dir,opt.name)

    make_fig(res_dir,(train_acc_his,acc_his),opt.name,'acc')
    make_fig(res_dir,time_his,opt.name,'time')
    make_fig(res_dir,loss_his,opt.name,'loss')
    save_results(loss_his,res_dir,'loss')
    save_results(acc_his,res_dir,'acc')
    save_results(train_acc_his,res_dir,'train_acc')
    save_results(time_his,res_dir,'time')
def make_fig(res_dir,his,name,title):
    plt.figure()
    res_name = '%s_%s_his.png' % (name,title)
    res_path = os.path.join(res_dir,res_name)
    if title == 'acc':
        train_acc_his, acc_his = his
        line_train = plt.plot(range(len(acc_his)),acc_his)
        line_val = plt.plot(range(len(train_acc_his)),train_acc_his)
        lines = [line_train,line_val]
        names = ['train','val']
        plt.legend(lines,names,loc = 'lower right',fontsize = 10)
    else:
        plt.plot(range(len(his)),his)
    plt.title(title)
    plt.savefig(res_path)
    
def save_networks(model, epoch, save_dir):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        name = model.module.model_name
        if isinstance(name, str):
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(save_dir, save_filename)

            if len(model.module.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(model.module.cpu().state_dict(), save_path)
                model.module.cuda(model.module.gpu_ids[0])
            else:
                torch.save(model.module.state_dict(), save_path)
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), use the default PyTorch schedulers.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 1.0 - max(0, epoch - opt.epoch) / float(opt.epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
def update_learning_rate(optimizer,scheduler):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))