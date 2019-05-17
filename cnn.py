import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt



class ClasifyModel(nn.Module):
    def __init__(self, ini_weight, in_dim, batch_size, out_dim, channels):
        super(ClasifyModel, self).__init__()
        self.weight = ini_weight
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.out_dim = out_dim
        #self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding = nn.Embedding(44988,100)
        self.embedding.weight.requires_grad = True
        self.channels = channels

        #self.channels2 = 10

        #self.cnn2 = nn.Conv2d(1,self.channels2, kernel_size=(1,3),stride=1)

        self.cnn1 = nn.Conv2d(1, 1, kernel_size = (4,100))
        self.cnn2 = nn.Conv2d(1, 1, kernel_size = (4,100))
        self.cnn3 = nn.Conv2d(1, 1, kernel_size=(3, 100))
        self.cnn4 = nn.Conv2d(1, 1, kernel_size=(3, 100))
        self.cnn5 = nn.Conv2d(1, 1, kernel_size=(2, 100))
        self.cnn6 = nn.Conv2d(1, 1, kernel_size=(2, 100))

        #self.linear1 = nn.Linear(self.channels,28)
        self.linear2 = nn.Linear(6,self.out_dim)
        #self.linear3 = nn.Linear(15,out_dim)
        #self.linear2 = nn.Linear(self.channels , self.out_dim)
    
    def forward(self, input):
        emb = self.embedding(input[0])
        out = emb.unsqueeze(0)
        out = out.permute(1,0,2,3)
        out1 = F.relu(self.cnn1(out))
        out2 = F.relu(self.cnn2(out))
        out3 = F.relu(self.cnn3(out))
        out4 = F.relu(self.cnn4(out))
        out5 = F.relu(self.cnn5(out))
        out6 = F.relu(self.cnn6(out))
        out1 = F.max_pool2d(out1, (self.in_dim - 3, 1), padding=0)
        out2 = F.max_pool2d(out2, (self.in_dim - 3, 1), padding=0)
        out3 = F.max_pool2d(out3, (self.in_dim - 2, 1), padding=0)
        out4 = F.max_pool2d(out4, (self.in_dim - 2, 1), padding=0)
        out5 = F.max_pool2d(out5, (self.in_dim - 1, 1), padding=0)
        out6 = F.max_pool2d(out6, (self.in_dim - 1, 1), padding=0)
        #print(out6.size())
        out = torch.cat((out1, out2),2)
        out = torch.cat((out, out3), 2)
        out = torch.cat((out, out4), 2)
        out = torch.cat((out, out5), 2)
        out = torch.cat((out, out6), 2)
        #print(out.size())
        #out = self.cnn2(out)
        #out = F.relu(out)
        #out = self.cnn1(out)
        #out = F.relu(out)
        #out = F.max_pool2d(out, (self.in_dim-19, 1),padding = 0)
        out = torch.squeeze(out)

        out = F.dropout(out, p=0.5)
        #out = self.linear1(out)
        #out = F.relu(out)
        #out = F.dropout(out, p=0.5)
        out = self.linear2(out)
        #out = F.relu(out)
        #out = F.dropout(out, p=0.5)
        #out = self.linear3(out)
        #print(out.size())
        out = F.softmax(out,dim=1)
        #print(out.size())
        return out

    def backward(self, out):
        out.backward()

class DataSet(Data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        data_, target_ = self.data[index],self.target[index]
        return data_, target_

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    e = 10
    train_loss = []
    test_loss = []



    weight = np.load("weight.npy")
    weight = torch.from_numpy(weight)
    in_dim = 3100
    out_dim = 8
    channel = 72
    batch_size = 50
    model = ClasifyModel(weight, in_dim, batch_size, out_dim, channel)

    data_train = np.load("training_set.npy")
    data_train = data_train.tolist()
    data_train_x = torch.from_numpy(np.array(data_train["data"]))
    data_train_y = torch.from_numpy(np.array(data_train["target"]))

    data_test = np.load("test_set.npy")
    data_test = data_test.tolist()
    data_test_x = torch.from_numpy(np.array(data_test["data"]))
    data_test_y = torch.from_numpy(np.array(data_test["target"]))

    train = DataSet(data_train_x, data_train_y)
    test = DataSet(data_test_x, data_test_y)

    train_loader = Data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_loader = Data.DataLoader(test,batch_size=len(test), shuffle=False)

    #train = Data.TensorDataset(data_tensor=data_train_x,target_tensor=data_train_y)
    #test = Data.TensorDataset(data_tensor=data_test_x,target_tensor=data_test_y)
    batch_time = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.5)
    for case in range(0, e):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(batch)
            loss = criterion(outputs,batch[1])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_time+=1
            print("batch_time" + str(batch_time)+":"+str(loss.item()))
        for batch in test_loader:
            with torch.no_grad():
                optimizer.zero_grad()
                outputs = model.forward(batch)
                loss = criterion(outputs,batch[1])
                test_loss.append(loss.item())
                print(str(case)+":"+str(loss.item()))
    f = open("result_1.txt","w",encoding="utf-8")
    f.write(str(train_loss))
    f.write("\n")
    f.write(str(test_loss))

    plt.plot(train_loss)
    plt.show()

    plt.plot(test_loss)
    plt.show()





        

