import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import random
import datetime as dt


BATCH_SIZE = 128
MAX_LEN = 0
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        train_data_csv = pd.read_csv("train.csv")
        test_data_csv = pd.read_csv("test.csv")
        train_values_np = np.array([train_data_csv.to_numpy()[i][1:].reshape((28, 28)) for i in range(train_data_csv.shape[0])])
        train_targets_np = np.array([train_data_csv.to_numpy()[i][0] for i in range(train_data_csv.shape[0])])
        train_values_raw = torch.from_numpy(train_values_np)

        self.train_values = train_values_raw.unsqueeze(1)
        self.train_targets = torch.from_numpy(train_targets_np)

        test_values_np = np.array([test_data_csv.to_numpy()[i].reshape((28, 28)) for i in range(test_data_csv.shape[0])])
        test_data_raw = torch.from_numpy(test_values_np)
        self.test_values = test_data_raw.unsqueeze(1)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return self.train_values.shape[0]

    def __getitem__(self, idx):
        return self.train_values[idx], self.train_targets[idx]

    def display_examples(self):
        # generate random image from tran data
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 6))

        for i, ax in enumerate(axes.flat):

            image_num = random.randint(0, self.train_values.shape[0])
            image = self.train_values[image_num].squeeze(dim=0)
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Number {self.train_targets[image_num]}')
            ax.axis('off')

        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.show()


dataset_full = Dataset()

dataset_full.display_examples()

train_test_split = int(len(dataset_full) * 0.8)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
model.to(DEVICE)

# print(model)
# summary(model, input_size=(1, 28, 28))

# Define negative log-likelihood loss
loss_func = nn.NLLLoss(reduction="sum")

# Adam optimizer
opt = optim.Adam(model.parameters(), lr=1e-4)


# helper function to compute accuracy per mini-batch
def metrics_batch(target, output):
    pred = output.argmax(dim=1, keepdim=True)
    # compare output class with target class
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# function to compute loss value per mini_batch
def loss_batch(loss_func, xb, yb, yb_h, opt=None):
    loss = loss_func(yb_h, yb)
    metric_b = metrics_batch(yb, yb_h)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), metric_b


# helper function to compute the loss and metric values for a dataset
def loss_epoch(model, loss_func, dataset_dl, opt=None):
    loss = 0.0
    metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb = xb.type(torch.float).to(DEVICE)
        yb = yb.to(DEVICE)
        yb_h = model(xb)
        loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
        loss += loss_b
        if metric_b is not None:
            metric += metric_b
    loss /= len_data
    metric /= len_data
    return loss, metric


# train_val function
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    new_best = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
            accuracy = 100 * val_metric

        if new_best > accuracy:
            time_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            torch.save(model.state_dict(), f"results/weights_{time_now}.pt")

        print(f"epoch: {epoch}, train loss: {round(train_loss, 6)}, val loss: {round(val_loss, 6)}, accuracy: {round(accuracy, 2)}")


# training model for few epoch
num_epoch = 5
train_val(num_epoch, model, loss_func, opt, data_loader_train, data_loader_test)

model.eval()
test_data = dataset_full.test_values.type(torch.float).to(DEVICE)
output_values = model(test_data)

results = output_values.argmax(dim=1).to("cpu")
print(results)

test_data_exp = pd.read_csv("sample_submission.csv")
pass_id = test_data_exp["ImageId"]

output = pd.DataFrame({'ImageId': pass_id, 'Label': results})
print(output)
output.to_csv('submission.csv', index=False)