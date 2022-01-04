import copy
import os

import seaborn
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sklearn
from torchvision.datasets import MNIST
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mnist_data(split):
    return FastMNIST('data/MNIST', train=split == "train", download=True)
    # return torchvision.datasets.MNIST("mnist",
    #                            train=(split == "train"),
    #                            download=True,
    #                            transform=torchvision.transforms.Compose([
    #                                torchvision.transforms.ToTensor(),
    #                                torchvision.transforms.Normalize(
    #                                    (0.1307,), (0.3081,))
    #                            ]))

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

class FC5(nn.Module):
    def __init__(self):
        super(FC5, self).__init__()
        self.layers = nn.Sequential(
            *self.get_bn_fc(in_dim=784),
            *self.get_bn_fc(),
            *self.get_bn_fc(),
            *self.get_bn_fc(),
            *self.get_bn_fc(),
            *self.get_bn_fc(out_dim=10, activation = "softmax")
        )

    def get_bn_fc(self, in_dim = 1024, out_dim = 1024, activation = "relu"):
        layers = [nn.BatchNorm1d(in_dim), nn.Dropout(p=0.1), nn.Linear(in_dim, out_dim)]
        fc = layers[2]
        if activation == "relu":
            torch.nn.init.kaiming_normal_(fc.weight.data)
            nn.init.constant_(fc.bias.data, 0)
            layers.append(nn.ReLU())
        elif activation == "softmax":
            torch.nn.init.xavier_normal_(fc.weight.data)
            nn.init.constant_(fc.bias.data, 0)
            layers.append(nn.Softmax())
        return layers

    def forward(self, samples):
        return self.layers(samples)

    def get_layer_output(self, samples, layer_num, before_activation = False):
        return self.layers[:4*layer_num - int(before_activation)](samples)

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))

train_ops = {
    "learn_rate": 0.01,
    "batch_size": 128,
    "epochs": 200,
    "model_path": "saved_models/mnist_fc5.model_bck",
    "save_model": True,
    "load_model": True,
    "train_model": False,
    "dropouts": []
}

def train_fc5_model(train_options):
    model = FC5()
    if train_options["load_model"]:
        load_model(model, train_options["model_path"])
    if not train_options["train_model"]:
        return model

    optimizer = torch.optim.Adam(model.parameters(), eps=train_options["learn_rate"])

    train_loader = DataLoader(dataset=get_mnist_data("train"), batch_size=train_options["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=train_options["batch_size"], shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, train_options["epochs"] + 1):
        print("Epoch: " + str(epoch))
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(X.shape[0], -1)
            pred = model(X)
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        total_correct = 0
        for X, Y in test_loader:
            X = X.view(X.shape[0], -1)
            pred = model(X)
            acc = pred.argmax(dim=-1) == Y
            total += X.size(0)
            total_correct += acc.sum().item()

        test_acc = total_correct / total
        print("Test Accuracy:" + str(test_acc))
        if test_acc >= 0.98:
            save_model(model, train_options["model_path"])
            return model

# model = train_fc5_model(train_ops)

def get_each_class_samples(model, X, layer_num=-1):
    model.eval()
    X = X.view(X.shape[0], -1)
    X = X.float().to(device)
    if layer_num != -1:
        pred = model.get_layer_output(X, layer_num)
    else:
        pred = model(X)

    pred = pred.cpu()

    if pred.size(1) > 10:
        pred = pred[..., torch.randperm(pred.size(1))[:10]]

    pred_class = pred.argmax(dim=-1)
    class_samples = {i: [] for i in range(10)}
    for k in class_samples:
        class_samples[k] = X[pred_class == k][:10].view(10, 28, 28).cpu()
    return class_samples


def plot_mnist_images(class_samples, cols=10):
    fig = plt.figure(figsize=(8, 8))
    columns = 10
    rows = 10
    for i in range(cols * 10):
        row_num = int(i / 10)
        col_num = i % 10
        if len(class_samples[row_num]) < (col_num + 1):
            continue
        img = class_samples[row_num][col_num]
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)

    plt.show()

test_loader_1000 = DataLoader(dataset=get_mnist_data("test"), batch_size=1000, shuffle=True)

def plot_10_x_10(model, test_loader, layer_num=-1):
    for X, Y in test_loader:
        class_samples = get_each_class_samples(model, X, layer_num)
        plot_mnist_images(class_samples)
        break

def reduce_dimension(samples, dim_reduce_fn, dims = 2):
    if dim_reduce_fn == "pca":
        fn = PCA(n_components=dims, random_state=42)
    elif dim_reduce_fn == "tsne":
        fn = TSNE(n_components=dims, random_state=42)
    return fn.fit_transform(samples.numpy())

def remove_noise(samples, var_multiplier = 0.05):
    samples = torch.from_numpy(samples)
    var, mean = torch.var_mean(samples[..., 0:2], dim=0)
    samples = samples[(samples[...,0] >= (mean[0] - var_multiplier*var[0])) & (samples[...,0] <= (mean[0] + var_multiplier*var[0]))
                      & (samples[...,1] >= (mean[0] - var_multiplier*var[1])) & (samples[...,1] <= (mean[1] + var_multiplier*var[1]))]
    return samples.numpy()

def plot_2d_examples_sns(data_frm, title, num_cats = 10):
    df = pd.DataFrame(data_frm[..., :2], columns=["x", "y"])
    df["cat"] = pd.Series(data_frm[..., 2])
    seaborn.scatterplot(x="x", y="y", data=df, hue=df["cat"]).plot()

def plot_2d_examples(data_frm, title, num_cats = 10, rem_noise = False):
    # markers = ['o', '.', 'd', 'x', '+', 'v', '^', '<', '>', 's', ',']

    markers = ["o" for i in range(10)]
    total_drawn = 0
    if rem_noise:
        data_frm = remove_noise(data_frm)
    plt.rcParams["figure.figsize"] = (8, 8)

    for i in range(num_cats):
        samples = data_frm[data_frm[...,2] == i]
        total_drawn += samples.shape[0]
        plt.plot(samples[..., 0], samples[..., 1], markers[i], label=str(i), markersize=3)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_reduced_dim_layers(X, Y, dim_reduce_fn, model = None, layer_num = -1, num_cats = 10, before_activation = False, rem_noise = False):

    X = X.view(-1, 784)
    if layer_num != -1:
        model.eval()
        X = model.get_layer_output(X, layer_num, before_activation=True).detach()

    pca_dims = reduce_dimension(X, dim_reduce_fn)
    data_frm = torch.cat((torch.from_numpy(pca_dims), Y.unsqueeze(1)), dim=1).numpy()
    plot_2d_examples(data_frm, ("Visualising " +dim_reduce_fn.upper() +" on Layer:" + str(layer_num), "Visualising " + dim_reduce_fn.upper() + " on original data.")[layer_num < 1], num_cats = num_cats, rem_noise = rem_noise)

def draw_reduced_dim_plots_all_layers():
    for X, Y in test_loader_1000:
        for dim_red_fn in ["pca", "tsne"]:
            for i in range(7):
                plot_reduced_dim_layers(X, Y, dim_red_fn, model, i, num_cats=10, rem_noise= (dim_red_fn == "pca"))
        break

# draw_reduced_dim_plots_all_layers()

#######################################################
################# Problem 2 ###########################
#######################################################

class FC5Custom(nn.Module):
    def __init__(self, init_fn, activation_fn, hidden_dims = 512):
        super(FC5Custom, self).__init__()
        self.layers = nn.Sequential(
            *self.get_fc_layer(activation_fn, init_fn, 784, hidden_dims),
            *self.get_fc_layer(activation_fn, init_fn),
            *self.get_fc_layer(activation_fn, init_fn),
            *self.get_fc_layer(activation_fn, init_fn),
            *self.get_fc_layer(activation_fn, init_fn),
            *self.get_fc_layer("softmax", init_fn, hidden_dims, 10)
        )

    def get_activation_fn(self, activation_fn):
        if activation_fn == "sigmoid":
            return nn.Sigmoid()
        elif activation_fn == "relu":
            return nn.ReLU()
        elif activation_fn == "softmax":
            return nn.Softmax()
        else:
            raise NotImplementedError()

    def initialize_weights(self, init_fn, fc):
        if init_fn == "xavier":
            torch.nn.init.xavier_normal_(fc.weight.data)
            nn.init.constant_(fc.bias.data, 0)
            return
        elif init_fn == "he":
            torch.nn.init.kaiming_normal_(fc.weight.data)
            nn.init.constant_(fc.bias.data, 0)
            return
        elif init_fn == "normal":
            with torch.no_grad():
                fc.weight.data.normal_(mean=0, std=0.01)
                nn.init.constant_(fc.bias.data, 0)

    def get_fc_layer(self, activation_fn, init_fn, in_dim = 512, out_dim = 512):
        layer = [nn.Linear(in_dim, out_dim)]
        fc = layer[0]
        self.initialize_weights(init_fn, fc)
        layer.append(self.get_activation_fn(activation_fn))
        return layer

    def forward(self, samples):
        return self.layers(samples)

def train_custom_model(train_options):

    print("Training: " + train_options["model_path"])
    model = FC5Custom(train_options["init_fn"], train_options["activation_fn"])
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if train_options["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_options["learn_rate"])
    elif train_options["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_options["learn_rate"])
    else:
        raise NotImplementedError()

    train_loader = DataLoader(dataset=get_mnist_data("train"), batch_size=train_options["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=train_options["batch_size"], shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_accuracy = []

    for epoch in range(1, train_options["epochs"] + 1):
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(X.shape[0], -1)
            pred = model(X)
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        total_correct = 0
        for X, Y in test_loader:
            X = X.view(X.shape[0], -1)
            pred = model(X)
            acc = pred.argmax(dim=-1) == Y
            total += X.size(0)
            total_correct += acc.sum().item()

        test_acc = total_correct / total
        epoch_accuracy.append(test_acc)
        print("Epoch: " + str(epoch) + ", Test Accuracy:" + str(test_acc))
        if test_acc >= 0.99:
            break
    save_model(model, train_options["model_path"])
    return epoch_accuracy

def get_model_path(train_options):
    return train_options["optimizer"] + "_" + train_options["activation_fn"] + "_" + train_options["init_fn"] + \
           ("", "_dropout")["dropouts" in train_options and sum(train_options["dropouts"]) > 0] + ".model"

def get_model_name(train_options):
    return train_options["activation_fn"] + "_" + train_options["init_fn"] + ("", "_dropout")["dropouts" in train_options and sum(train_options["dropouts"]) > 0]

def get_test_accuracies_for_configs(configs):
    model_accs = {get_model_name(config): [] for config in configs}
    for config in configs:
        print(config)
        model_accs[get_model_name(config)] = train_custom_model(config)
    return model_accs

def plot_test_accuracies(config_accuracies, title):
    for name in config_accuracies:
        plt.plot(config_accuracies[name], "o", label=name, markersize=2)
    plt.title(title)
    plt.legend()
    plt.show()

def prob2_train():
    train_configs_sgd = []
    train_configs_adam = []

    train_options = copy.deepcopy(train_ops)

    act_init_configs = [{"activation_fn": "sigmoid", "init_fn": "normal", "learn_rate": {"adam": 0.001, "sgd": 0.5}, "epochs": {"adam": 25, "sgd": 200}},
                        {"activation_fn": "sigmoid", "init_fn": "xavier", "learn_rate": {"adam": 0.001, "sgd": 0.5}, "epochs": {"adam": 25, "sgd": 100}},
                        {"activation_fn": "relu", "init_fn": "normal", "learn_rate": {"adam": 0.0001, "sgd": 0.25}, "epochs": {"adam": 25, "sgd": 150}},
                        {"activation_fn": "relu", "init_fn": "xavier", "learn_rate": {"adam": 0.0001, "sgd": 0.1}, "epochs": {"adam": 30, "sgd": 30}},
                        {"activation_fn": "relu", "init_fn": "he", "learn_rate": {"adam": 0.0001, "sgd": 0.5}, "epochs": {"adam": 30, "sgd": 15}}]

    train_options["optimizer"] = "sgd"
    train_options["epochs"] = 25
    train_options["load_model"] = False
    train_options["train_model"] = True

    for act_init_config in act_init_configs:
        train_options["init_fn"] = act_init_config["init_fn"]
        train_options["activation_fn"] = act_init_config["activation_fn"]
        train_options["model_path"] = get_model_path(train_options)
        train_options["learn_rate"] = act_init_config["learn_rate"]["sgd"]
        train_options["epochs"] = act_init_config["epochs"]["sgd"]
        train_configs_sgd.append(copy.deepcopy(train_options))

    for i, config in enumerate(train_configs_sgd):
        config = copy.deepcopy(config)
        config["optimizer"] = "adam"
        config["learn_rate"] = act_init_configs[i]["learn_rate"]["adam"]
        config["epochs"] = act_init_configs[i]["epochs"]["adam"]
        config["model_path"] = get_model_path(config)
        train_configs_adam.append(config)

    # lr = train_configs_sgd[0]["learn_rate"]
    # tests = []
    # for i in range(7):
    #     train_configs_sgd[0]["learn_rate"] = lr
    #     lr = lr / 2
    #     tests.append(copy.deepcopy(train_configs_sgd[0]))
    # a = get_test_accuracies_for_configs(tests)
    # if a:
    #     return

    for i in range(5):
        get_test_accuracies_for_configs(train_configs_sgd[4:5])
        train_configs_sgd[4]["learn_rate"] /= 2

    model_accuracies_sgd = get_test_accuracies_for_configs(train_configs_sgd)
    model_accuracies_adam = get_test_accuracies_for_configs(train_configs_adam)

    print(model_accuracies_sgd)
    print("-------------------------")
    print(model_accuracies_adam)

    plot_test_accuracies(model_accuracies_sgd, "SGD")
    plot_test_accuracies(model_accuracies_adam, "ADAM")

prob2_train()

#######################################################
################# Problem 3 ###########################
#######################################################
def prob3_train():
    train_configs = []

    train_options = copy.deepcopy(train_ops)

    act_init_configs = [{"activation_fn": "sigmoid", "init_fn": "xavier", "dropouts": [0,0,0,0,0], "epochs": 15, "learn_rate": 0.001}, {"activation_fn": "sigmoid", "init_fn": "xavier", "dropouts": [0.2, 0.5, 0.5, 0.5, 0.5], "epochs": 15, "learn_rate": 0.001},
                        {"activation_fn": "relu", "init_fn": "he", "dropouts": [0,0,0,0,0], "epochs": 15, "learn_rate": 0.0001}, {"activation_fn": "relu", "init_fn": "he", "dropouts": [0.2, 0.5, 0.5, 0.5, 0.5], "epochs": 15, "learn_rate": 0.0001},
                        {"activation_fn": "sigmoid", "init_fn": "xavier", "dropouts": [0, 0, 0, 0, 0], "epochs": 15, "learn_rate": 0.005},{"activation_fn": "sigmoid", "init_fn": "xavier", "dropouts": [0.2, 0.5, 0.5, 0.5, 0.5],"epochs": 15, "learn_rate": 0.005},
                        {"activation_fn": "relu", "init_fn": "he", "dropouts": [0, 0, 0, 0, 0], "epochs": 15,"learn_rate": 0.0005},{"activation_fn": "relu", "init_fn": "he", "dropouts": [0.2, 0.5, 0.5, 0.5, 0.5], "epochs": 15,"learn_rate": 0.0005}]

    train_options["epochs"] = 100
    train_options["load_model"] = False
    train_options["optimizer"] = "adam"
    train_options["hidden_dims"] = 1024
    train_options["learn_rate"] = 0.0001

    for act_init_config in act_init_configs:
        train_options["init_fn"] = act_init_config["init_fn"]
        train_options["activation_fn"] = act_init_config["activation_fn"]
        train_options["dropouts"] = act_init_config["dropouts"]
        train_options["model_path"] = get_model_path(train_options)
        train_options["epochs"] = act_init_config["epochs"]
        train_options["learn_rate"] = act_init_config["learn_rate"]
        train_configs.append(copy.deepcopy(train_options))

    model_accuracies = get_test_accuracies_for_configs(train_configs)

    print(model_accuracies)

    plot_test_accuracies(model_accuracies, "Loss")

prob3_train()
