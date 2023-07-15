import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = output[-1, :, :]
        output = self.fc(output)
        return output

# 数据集准备
train_data = ["This movie is great", "I really like this movie", "The plot is boring", "This movie is terrible"]
train_labels = [1, 1, 0, 0]

# 构建词表
word2index = {}
for sentence in train_data:
    for word in sentence.split():
        if word not in word2index:
            word2index[word] = len(word2index)

# 超参数定义
input_size = len(word2index)
hidden_size = 64
output_size = 2
learning_rate = 0.001
epochs = 100

# 将文本转换为词袋向量
def get_input_vector(sentence):
    vector = np.zeros(input_size)
    for word in sentence.split():
        if word in word2index:
            vector[word2index[word]] += 1
    return vector

# 创建模型和优化器
model = RNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(epochs):
    epoch_loss = 0
    for i, sentence in enumerate(train_data):
        # 将文本转换为词袋向量
        input_vector = get_input_vector(sentence)
        input_tensor = torch.from_numpy(input_vector).unsqueeze(1).float()
        label_tensor = torch.tensor([train_labels[i]]).unsqueeze(0)

        # 模型前向计算并计算损失
        output = model(input_tensor)
        loss = criterion(output, label_tensor)

        # 反向传播并更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, epoch_loss))

# 测试模型
test_data = ["I love this movie", "The acting is terrible"]
for sentence in test_data:
    input_vector = get_input_vector(sentence)
    input_tensor = torch.from_numpy(input_vector).unsqueeze(1).float()
    output = model(input_tensor)
    prediction = torch.argmax(output).item()
    if prediction == 1:
        print("{}: Positive".format(sentence))
    else:
        print("{}: Negative".format(sentence))
