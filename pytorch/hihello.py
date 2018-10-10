import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


idx2char = ['h', 'i', 'e', 'l', 'o']
input_size = len(idx2char)
hidden_size = len(idx2char)
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)  # batch_first guarantees the order of output = (B, S, F)

# data setting
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [1, 0, 2, 3, 3, 4]

# transform as torch tensor variable
X = Variable(torch.Tensor(x_one_hot).float())
Y = Variable(torch.Tensor(y_data).long())

# loss & optimizer setting
weights = torch.Tensor(np.ones(input_size)).float()  # weight for each class, not for position in sequence
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(rnn.parameters(), lr=0.1)

for i in range(50):

    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.squeeze(), Y)
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([idx2char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.data.numpy(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
