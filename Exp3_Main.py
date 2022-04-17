"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""
import time
import os
from Exp3_Config import Training_Config
from Exp3_DataSet import TextDataSet, TestDataSet, WordEmbeddingLoader, RelationLoader
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch
from Exp3_utils import *
from sklearn.metrics import f1_score
import tqdm
# import visdom
total_iters = 0
def train(epoch_num,model,loader,config):
    """
    input:
    epoch_num: which epoch it is now
    data_loader_train: a dataloader for training
    opt: hyperparameters
    output: training time for this epoch 
    """
    # 循环外可以自行添加必要内容
    global total_iters
    save_dir = os.path.join(config.checkpoints_dir, config.name)
    epoch_start_time = time.time()  # timer for entire epoch
    epoch_iter = 0 
    total_step = len(loader)
    loss_value = 0
    correct = 0
    total = 0
    for index, data in enumerate(loader, 0):
        iter_data_time = time.time()    # timer for data loading per iteration
        total_iters += config.batch_size
        epoch_iter += config.batch_size
        
        text, label, pos1, pos2 = data['text'],data['label'],data['pos1'],data['pos2']
        label = label.to(device)
        text = text.to(device)
        pos1 = pos1.to(device)
        pos2 = pos2.to(device)
        optimizer.zero_grad()
        output = model(text,pos1,pos2)
        loss = loss_function(output, label)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum().item() 
        total += label.size(0)
        loss_value += loss.item()
        loss.backward()
        optimizer.step()

         # print training losses and save logging information to the disk
        if total_iters % config.print_freq == 0:   
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}' 
                   .format(epoch_num+1, config.epoch+config.epoch_decay, index+1, total_step, loss.item(), time.time()-iter_data_time))
        
        # cache our latest model every <save_latest_freq> iterations
        if total_iters % config.save_latest_freq == 0:   
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch_num, total_iters))
            save_suffix = 'iter_%d' % total_iters if config.save_by_iter else 'latest'
            save_networks(model,save_suffix,save_dir)

    #update the learning rate
    update_learning_rate(optimizer,schedulers)

    # print out the average loss and accuracy on this epoch
    accuracy = correct/total
    epoch_time = time.time()-epoch_start_time
    print("End of epoch{},Use time:{},the accuracy is {}".format(epoch_num+1,epoch_time,accuracy))
    print("Saving the latest model....")

    #save the model
    save_networks(model,epoch='latest',save_dir=save_dir)
    return epoch_time,loss_value/total_step,accuracy
def validation(model, loader):
    """
    input:
    model: the model to be evaluated
    data_loader_val: a dataloader for validation
    output : accuracy and f1 score
    """
    correct = 0
    total = 0
    accuracy = 0
    f1_sum = 0
    with torch.no_grad():  
        for data in loader:
            text, label, pos1, pos2 = data['text'],data['label'],data['pos1'],data['pos2']
            label = label.to(device)
            text = text.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            # get output from the model
            text = text.to(device)
            label = label.to(device)
            outputs = model(text,pos1,pos2)
            _, predicted = torch.max(outputs.data, 1) 
            f1_sum += f1_score(label.cpu(),predicted.cpu(),average='macro')
            total += label.size(0) 
            correct += (predicted == label).sum().item() 
    accuracy = correct/total
    f1 = f1_sum/total
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)
    print("当前模型在验证集上的F1-score为:",f1)
    return accuracy,f1

def get_model(config,is_Train = False):
        Text_Model = TextCNN_Model(configs=config,char_vec=char_vec,char_id = char2id,class_num = class_num)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            Text_Model.to(gpu_ids[0])
            Text_Model = torch.nn.DataParallel(Text_Model, gpu_ids)  # multi-GPUs
            if not is_Train:
                model_dir = os.path.join(config.checkpoints_dir, config.name)
                model_path = os.path.join(model_dir, 'latest_net_BiLSTM_Model.pth')
                Text_Model.module.load_state_dict(torch.load(model_path))
        else:
            if not is_Train:
                model_dir = os.path.join(config.checkpoints_dir, config.name)
                model_path = os.path.join(model_dir, 'latest_net_FC.pth')
                Text_Model.load_state_dict(torch.load(model_path))
        return Text_Model
def predict(config,test_loader):
    """
    input:
    config: hyperparameters
    test_loader: a dataloader for testing
    output: prediction
    """
    model = get_model(config = config)
    results = []
    res_dir = os.path.join(config.results_dir,config.name)
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in tqdm.tqdm(test_loader):
            text, pos1, pos2 = data['text'],data['pos1'],data['pos2']
            text = text.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            outputs = model(text,pos1,pos2)
            _, predicted = torch.max(outputs.data, 1) 
            results.append(predicted)
    
    save_results(results,res_dir,"prediction",is_prediction=True)

   
if __name__ == "__main__":
    # initialize the config
    config = Training_Config()

    # make dir for checkpoints saving and results saving
    expr_dir = [os.path.join(config.checkpoints_dir, config.name),os.path.join(config.results_dir, config.name)]
    mkdirs(expr_dir)

    #set gpu ids
    str_ids = config.gpu_ids.split(',')
    config.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            config.gpu_ids.append(id)
    gpu_ids = config.gpu_ids
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    # get char_dict and relation_dict
    char2id, char_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()

    # initialize the data_loader
    train_dataset = TextDataSet(rel2id,char2id,char_vec,config,filepath="./data/data_train.txt")
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.batch_size,
        )

    val_dataset = TextDataSet(rel2id,char2id,char_vec,config,filepath="./data/data_val.txt")
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=config.batch_size,
        )

    # 测试集数据集和加载器
    test_dataset = TestDataSet(rel2id,char2id,char_vec,config,filepath="./data/test_exp3.txt")
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # initialize the model
    Text_Model = get_model(config=config,is_Train = True)

    # initialize the loss function 
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # initialize the optimizer and the scheduler
    optimizer = torch.optim.Adam(params=Text_Model.parameters(),weight_decay= config.weight_decay)  # torch.optim中的优化器进行挑选，并进行参数设置
    schedulers = get_scheduler(optimizer, config)

    # start training
    print("start Training!")
    acc_his = []
    f1_his = []
    time_his = []
    loss_his = []
    train_acc_his = []
    for i in range(config.epoch+config.epoch_decay):
        time_per_epoch,loss_per_eopch,train_acc = train(epoch_num = i,model = Text_Model,loader = train_loader,config = config)
        time_his.append(time_per_epoch)
        loss_his.append(loss_per_eopch)
        train_acc_his.append(train_acc)
        if i % config.num_val == 0:
            acc,f1 = validation(model = Text_Model,loader = val_loader)
            acc_his.append(acc)
            f1_his.append(f1)
    # save results
    his = {'acc':acc_his,'time':time_his,'loss':loss_his,'train_acc': train_acc_his}
    save_his(his,config)

    #finish training
    print("训练完成！")

    # start testing
    print("Start Testing!")
    predict(config,test_loader)

    
