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
        self.embedding = nn.Embedding(44988,300)
        self.embedding = self.embedding.from_pretrained(self.weight, freeze=True)
        #self.embedding = nn.Embedding(44988,30)
        #self.embedding.weight.requires_grad = True
        self.channels = channels

        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=4, batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(128,8)
        #self.linear2 = nn.Linear(8,8)

        #self.channels2 = 10

        #self.cnn2 = nn.Conv2d(1,self.channels2, kernel_size=(1,3),stride=1)

        
    
    def forward(self, input):
        out = self.embedding(input[0])
        #out = out.view(out.size(0),-1)
        #print(out.size())
        #out = input[0]
        #out = nn.utils.rnn.pack_padded_sequence(out, len(out[0]), batch_first=True)
        out, (h_c,c_n) = self.lstm(out)
        out = h_c[-1,:,:]
        #out = F.dropout(output_in_last_timestep)
        out = self.linear1(out)
        #out = F.relu(out)
        #out = F.dropout(out)
        #out = self.linear2(out)
        #out = F.softmax(out, dim=1)

        return out


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
    in_dim = 500
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

    train_loader = Data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test_loader = Data.DataLoader(test,batch_size=len(test), shuffle=False)

    #train = Data.TensorDataset(data_tensor=data_train_x,target_tensor=data_train_y)
    #test = Data.TensorDataset(data_tensor=data_test_x,target_tensor=data_test_y)
    batch_time = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    for case in range(0, e):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(batch)
            #print(batch[1])
            #print(outputs)
            loss = criterion(F.softmax(outputs),batch[1])
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            batch_time += 1
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





        

