import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch
import torch

from memorypytorch import MemoryDNN
from minop2 import mindelayt
def plot_rate(t_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(t_his)
    df = pd.DataFrame(t_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Time Delay')
    plt.ylim(0.65, 1)
    plt.xlabel('Time Frames')
    plt.show()
def save_to_txt(t_his, file_path):
    with open(file_path, 'w') as f:
        for t in t_his:
            f.write("%s \n" % t)

if __name__ == "__main__":
    N = 10                       # number of users
    n = 10000                    # number of time frames
    K = 10                       # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 512                 # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./gain_350_700_10000_10')['gain'][:10000, :]
    # channel = sio.loadmat('./gain_10_10000')['gain']
    # t = sio.loadmat('./gain_10_10000')['tmin']
    t = sio.loadmat('./data_10_f70712')['tmin']  # this rate is only used to plot figures; never used to train DROO.
    # rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000
    # t=t*1000000
    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size
    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size

    # 构造DRL，training_interval=10，即10个间隔训练一次
    mem = MemoryDNN(net=[N, 120, 80, N],
                    learning_rate=0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )
    t_his = []  # 存储每个时间帧内的最小时延
    t_his_ratio=[]# 存储每个时间帧内的最小时延与标签的比值
    mode_his = []  # 存储每个时间帧的最佳二进制动作
    k_idx_his = []  # 存储每个时间帧的最优动作的的有序量化动作排序序号
    K_his = []  # 存储每隔时间帧的自适应K值
    s = [130, 50, 30, 15, 75, 60, 99, 10, 164, 55]
    for i in range(n):
        print(i)
        #n除以10的结果向下取整
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        # delta自适应K的更新间隔设置为32
        # if i> 0 and i % Delta == 0:
        #     # 开始自适应K值的设置，在自适应K值的调整时，K值设置为在过去的delta个时间帧中最大值加一
        #     if Delta > 1:
        #         max_k = max(k_idx_his[-Delta:-1]) +1;
        #     else:
        #         max_k = k_idx_his[-1] +1;
        #     K = min(max_k +1, N)
        #num_test 是测试的 sample index     split_idx 是训练的 sample index
        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx,:]
        # the action selection must be either 'OP' or 'KNN'
        #m_list 存储探索生成K个二进制动作
        m_list = mem.decode(h, K, decoder_mode)
        # r_list 存储K个二进制动作下使用黄金分割法的最短通信时延之和
        # m_list.append(addm(h))
        # m_list=np.append(m_list,[[1, 1, 1, 1, 1, 1, 1, 1, 0, 1]],axis=0)
        # m_list.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        # m_list.append(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        # m_list.append(addm4(h))
        # m_list.append(addm4(h))
        # m_list.append(addm4(h))
        r_list = []
        for m in m_list:
            r_list.append(1000000*mindelayt(m,s,h / 1000000))
        # 将最好的二进制动作存储到h中对应信道增益，且每隔10次训练一次网络
        mem.encode(h, m_list[np.argmin(r_list)])
        # encode the mode with largest reward

        #以下为一些度量值
        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        #t_his 存储每个时间帧内的最小时延
        t_his.append(np.min(r_list))
        t_his_ratio.append(t[i_idx][0]/t_his[-1])
        #k_idx_his 存储每个时间帧的最优动作的有序量化动作排序序号 ，用于自适应调整K值
        k_idx_his.append(np.argmin(r_list))
        # 存储每隔时间帧的自适应K值
        K_his.append(K)
        # 存储每个时间帧的最佳二进制动作
        mode_his.append(m_list[np.argmin(r_list)])
    torch.save(mem.model,"3570mem_net10p2.pkl")
    mem.plot_cost()
    plot_rate(t_his_ratio)
    # save data into txt
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(t_his, "t_his8.txt")
    save_to_txt(t_his_ratio, "t_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")