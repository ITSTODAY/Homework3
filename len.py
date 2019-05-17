import numpy as np

result = np.load("/Users/matianyi/Desktop/homework3/training_set.npy")
result = result.tolist()

data = result["data"]

max_len = 0
num_ = 0
for info in data:
    max_len += len(info)
    num_+=1

print(int(max_len/num_))

#def 3100 as the num
for i in range(0,len(data)):
    info = data[i]
    for j in range(len(info),300):
        info.append(0)
    data[i] = info[0:300]

result["data"] = data
result = np.array(result)

np.save("/Users/matianyi/Desktop/homework3/training_set.npy",result)




