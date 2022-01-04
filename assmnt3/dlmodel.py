import copy
import os
import random
import traceback

import torch
from torch import nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dim_after_kernels_n_stride_no_pad(in_dim, kernel_stride_seq):
    for seq in kernel_stride_seq:
        kernel_size = seq[0]
        stride = seq[1]
        last_kernel_start = in_dim - kernel_size + 1
        in_dim = int((last_kernel_start - 1) / stride) + 1
    return in_dim

def get_act_fn(act_fn):
    if act_fn == "sigmoid":
        return nn.Sigmoid()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "softmax":
        return nn.Softmax(dim=-1)
    elif act_fn == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError("Activation not implemented")

def get_cnn_layers_from_config(config, dim0 = 32, dim1 = 32):
    all_layers = []
    channels = 1
    for layer in config:
        if layer["type"] == "conv":
            padding_amt = 0 if layer["padding"] == "valid" else (layer["kernel"][0] - 1) // 2
            dim0 += padding_amt*2
            dim1 += padding_amt*2
            conv = nn.Conv2d(kernel_size = layer["kernel"], stride=layer["stride"], in_channels=layer["channels"][0], out_channels=layer["channels"][1], padding=padding_amt)
            all_layers.append(conv)
            nn.init.kaiming_normal_(conv.weight.data)
            all_layers.append(get_act_fn(layer["act_fn"]))
            channels = layer["channels"][1]
            all_layers.append(nn.BatchNorm2d(channels))
        if layer["type"] == "maxpool":
            all_layers.append(nn.MaxPool2d(kernel_size=layer["kernel"], stride=layer["stride"]))

        dim0 = get_dim_after_kernels_n_stride_no_pad(dim0, [(layer["kernel"][0], layer["stride"][0])])
        dim1 = get_dim_after_kernels_n_stride_no_pad(dim1, [(layer["kernel"][1], layer["stride"][1])])
    return all_layers, dim0, dim1, channels

def get_linear_layer(in_dim, out_dim, act_fn):
    fc = nn.Linear(in_features=in_dim, out_features=out_dim)
    if act_fn == "relu":
        nn.init.kaiming_normal_(fc.weight.data)
    if act_fn == "sigmoid":
        nn.init.xavier_normal_(fc.weight.data)
    if act_fn == "softmax":
        nn.init.xavier_normal_(fc.weight.data)

    return [fc, get_act_fn(act_fn)]

model_layers = [
    {"type": "conv", "kernel": (5, 5), "stride": (1, 1), "channels": (3, 10), "act_fn": "relu", "padding": "same"},
    {"type": "maxpool", "kernel": (2, 2), "stride": (2, 2)},
    {"type": "conv", "kernel": (5, 5), "stride": (1, 1), "channels": (10, 10), "act_fn": "relu", "padding": "same"},
    {"type": "maxpool", "kernel": (2, 2), "stride": (2, 2)}
]

def change_paths_to_absolute(train_ops):
    if train_ops["dataset_type"] == "cifar500":
        train_ops["batch_size"] = 500

    path_frags = train_ops["model_path"].split("_")
    train_ops["model_path"] = "_".join(path_frags[:-3]) +"_"+ train_ops["dataset_type"] + "_" + train_ops["model_type"] + "_" + path_frags[-1]

    if train_ops["base_path"] in train_ops["model_path"]:
        return train_ops
    train_ops["model_path"] = train_ops["base_path"] + "_".join(train_ops["model_path"].split("_")[:-3]) +"_" + train_ops["dataset_type"] + "_" + train_ops["model_type"] +"_"+ train_ops["model_path"].split("_")[-1]
    train_ops["cifar3_model_path"] = train_ops["base_path"] + train_ops["cifar3_model_path"]

    for i, file_name in enumerate(train_ops["data_files"]):
        train_ops["data_files"][i] = train_ops["base_path"] + file_name

    for i, file_name in enumerate(train_ops["test_data_files"]):
        train_ops["test_data_files"][i] = train_ops["base_path"] + file_name

base_cnn_train_options = {
    "batch_size": 256,
    "val_batch_size": 512,
    "load_model": False,
    "save_model": True,
    "lr_rate": 0.001,
    "epochs": 20,
    "base_path": "", #"drive/MyDrive/colab/dls_assmnt3/",
    "data_files": ["data/cifar10/data_batch_1", "data/cifar10/data_batch_2", "data/cifar10/data_batch_3", "data/cifar10/data_batch_4", "data/cifar10/data_batch_5"],
    "test_data_files": ["data/cifar10/test_batch"],
    "model_path": "saved_models/nn_base_base_cnn.model",
    "train_model": True,
    "val_split": 5000,
    "cifar3_model_path": "saved_models/nn_cifar3_cifar3_cnn.model",
    "feature_lr_rate": 1e-5,
    "model_type": "base", # base | cifar3 | transfer
    "dataset_type": "base" # base | augmented | cifar500 | cifar3
}

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_n_combine(files):
    all_data = torch.zeros((0, 3*32*32))
    all_labels = torch.zeros((0), dtype=torch.long)
    for file_name in files:
        dict = unpickle(file_name)
        batch_data = torch.from_numpy(dict[bytes("data", "utf8")])
        all_data = torch.cat((all_data, batch_data), dim=0)
        all_labels = torch.cat((all_labels, torch.tensor(dict[bytes("labels", "utf8")])), dim=0)

    return all_data, all_labels

def save_model(model, model_path):
    directory_path = "/".join(model_path.split("/")[:-1])
    if len(directory_path) > 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    try:
        if not os.path.exists(model_path):
            return model

        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def view_image(img):
    plt.imshow(img)
    plt.show()

class ImgDataset(nn.Module):
    def __init__(self, imgs, labels):
        super(ImgDataset, self).__init__()
        if len(imgs.shape) == 2:
            imgs = imgs.view(-1, 3, 32, 32)
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]

def plot_cifar_images(class_samples, cols=10):
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

def shift_to_zero_mean(data):
    # view_image(data[random.choice(range(10000))].view(3,32,32).permute(1,2,0) / 255)
    return (data * 2 / 255) - 1

def normalise_data(data): # Shifts to 0-1 range, expected range 0-255
    return data / 255

def brighten_pixel(data, percent = 10, type = "abs"): # expected range 0-1
    if type == "abs":
        multiplier = ((100 + percent) / 100)
        return torch.min(data * multiplier, torch.ones(data.shape))
    else:
        multuplier = (100 - percent) / 100
        data = 1. - data
        return 1. - data * multuplier

def darken_pixel(data, percent = 10): # expected range 0-1
    multiplier = (100 - percent) / 100
    return data * multiplier

def flip_images(data): # shape = N, K, H, W
    data = data.view(-1, 3, 32, 32)
    data = torch.flip(data, dims=[-1])
    return data.view(data.shape[0], -1)

def rotate_image_90(data):
    data = data.view(-1, 3, 32, 32)
    return data.rot90(dims=[-2, -1]).reshape(data.shape[0], -1)

def flip_image_vertically(data):
    data = data.view(-1, 3, 32, 32)
    return torch.flip(data, dims = [-2]).reshape(data.shape[0], -1)

def transform_data(data, labels, dataset_type = "base", split = "train"): # expected range 0-1, returned range -1, 1
    if dataset_type in ("base", "cifar500"):
        return shift_to_zero_mean(data).view(-1, 3, 32, 32), labels

    if dataset_type == "augmented":
        data = normalise_data(data)

        if split == "val": # No augmentation required for validation set
            return (data * 2) - 1, labels

        dark_imgs = darken_pixel(data)
        bright_imgs = brighten_pixel(data)
        flipped_imgs = flip_images(data)

        return torch.cat((data, dark_imgs, bright_imgs, flipped_imgs), dim=0) * 2 - 1, torch.cat((labels, labels, labels, labels), dim=0)

    if dataset_type == "cifar3":
        labels = torch.cat((torch.empty(data.shape[0], dtype=torch.long).fill_(1), torch.empty(data.shape[0], dtype=torch.long).fill_(1), torch.empty(data.shape[0], dtype=torch.long).fill_(2)), dim=0)
        data = shift_to_zero_mean(data)
        data, labels =  torch.cat((data, rotate_image_90(data), flip_image_vertically(data)), dim=0), labels
        # plot_10x10_imgs((data + 1) / 2, labels)
        return data, labels

def plot_10x10_imgs(data, labels):
    count = 0
    class_samples = {i: [] for i in range(10)}
    for i, label in enumerate(labels):
        if len(class_samples[label.item()]) < 10:
            count += 1
            class_samples[label.item()].append(data[i].view(3, 32, 32).permute(1, 2, 0))
        if count == 100:
            break
    plot_cifar_images(class_samples)

def get_train_val_data(all_data, all_labels, train_options):
    if train_options["dataset_type"] in ("base", "augmented"):
        rand_idxes = torch.randperm(all_labels.shape[0])
        val_data, val_label = all_data[rand_idxes[:train_options["val_split"]]], all_labels[rand_idxes[:train_options["val_split"]]]
        train_data, train_label = all_data[rand_idxes[train_options["val_split"]:]], all_labels[rand_idxes[train_options["val_split"]:]]
        return train_data, train_label, val_data, val_label

    if train_options["dataset_type"] == "cifar500":
        train_data, train_label = all_data[-500:, :], all_labels[-500:]
        val_data, val_label = unpickle_n_combine(train_options["test_data_files"])
        return train_data, train_label, val_data, val_label

    if train_options["dataset_type"] == "cifar3":
        train_data, train_labels = all_data[:-500, :], all_labels[:-500]
        return train_data, train_labels, train_data[:10, :], train_labels[:10] # dummy val data

def get_dataloader_for_cifar(train_options):
    all_data, all_labels = unpickle_n_combine(train_options["data_files"])

    train_data, train_label, val_data, val_label = get_train_val_data(all_data, all_labels, train_options)

    val_data, val_label = transform_data(val_data, val_label, train_options["dataset_type"], split = "val")
    train_data, train_label = transform_data(train_data, train_label, train_options["dataset_type"])

    val_dataset = ImgDataset(val_data, val_label)
    train_dataset = ImgDataset(train_data, train_label)

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_options["batch_size"])
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=train_options["val_batch_size"])

    return train_loader, val_loader

class CNNModel(nn.Module):
    def __init__(self, in_dim, classes = 10):
        super(CNNModel, self).__init__()

        conv_layers, dim0, dim1, channels = get_cnn_layers_from_config(model_layers, dim0=in_dim, dim1=in_dim)


        self.feature_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            # *get_linear_layer(dim0 * dim1 * channels, 20, "relu"),
        )

        self.last_layer = nn.Sequential(
            *get_linear_layer(dim0 * dim1 * channels, 20, "relu"),
            *get_linear_layer(20, classes, "softmax")
        )

        self.layers = nn.Sequential(
            self.feature_layers,
            self.last_layer
        )

    def forward(self, samples):
        return self.layers(samples)

    def copy_weights(self, state_dict):
        with torch.no_grad():
            self.feature_layers.weight.copy_(state_dict['feature_layers.weight'])
            self.feature_layers.bias.copy_(state_dict['feature_layers.bias'])

def calculate_val_acc_n_loss(model, loader, loss_fn):
    total_accurate = 0
    total = 0
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for X, Y in loader:
            y_pred = model(X.to(device)).cpu()
            total += y_pred.shape[0]
            total_accurate += (y_pred.argmax(dim=-1) == Y).sum().item()
            total_loss += loss_fn(y_pred, Y).item()

    return (total_accurate / total, total_loss / total) if total != 0 else (0,0)

def load_feature_layer(model: CNNModel, path):
    cifar3_model = CNNModel(32, 3)
    load_model(cifar3_model, path)
    with torch.no_grad():
        model.feature_layers.load_state_dict(cifar3_model.feature_layers.state_dict())

def train_cnn_model(train_options):
    print("Training: Dataset = " + train_options["dataset_type"] + ", model = " + train_options["model_type"])
    model = CNNModel(32, 3 if train_options["model_type"] == "cifar3" else 10)
    model.to(device)

    if train_options["load_model"]:
        if train_options["model_type"] == "transfer":
            load_feature_layer(model, train_options["cifar3_model_path"])
        else:
            load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model, None, None

    print("Learning rate: " + str(train_options["lr_rate"]))

    if train_options["model_type"] != "transfer":
        optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    else:
        optimizer = Adam([
            {"params": model.feature_layers.parameters(), "lr": train_options["feature_lr_rate"]},
            {"params": model.last_layer.parameters(), "lr": train_options["lr_rate"]}
        ], lr=train_options["lr_rate"])

    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloader_for_cifar(train_options)

    epoch_loss = {"val":[], "train":[]}
    epoch_acc = {"val":[], "train":[]}
    max_acc, val_loss = calculate_val_acc_n_loss(model, val_loader, loss_fn)

    for epoch_num in range(train_options["epochs"]):

        total_loss = 0
        total_t = 0
        total_acc = 0

        model.train()

        for imgs, labls in train_loader:
            labl_pred = model(imgs.to(device)).cpu()
            torch.cuda.empty_cache()
            loss = loss_fn(labl_pred, labls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_t += labls.shape[0]
            total_loss += loss.item()
            total_acc += (labl_pred.argmax(dim=-1) == labls).sum()

        cur_acc = total_acc / total_t
        cur_loss = total_loss / total_t

        if (train_options["model_type"] == "cifar3"):
            val_acc, val_loss = cur_acc, cur_loss # Not needed for cifar3
        else:
            val_acc, val_loss = calculate_val_acc_n_loss(model, val_loader, loss_fn)

        epoch_loss["val"].append(val_loss)
        epoch_acc["val"].append(val_acc)
        epoch_loss["train"].append(cur_loss)
        epoch_acc["train"].append(cur_acc)

        if val_acc > max_acc and train_options["save_model"]:
            save_model(model, train_options["model_path"])
            max_acc = val_acc

        if epoch_num%1 == 0:
            print("Val Accuracy: " + str(val_acc))
            print("Val Loss: " + str(val_loss))
            # print("Parameters sum: " + str(sum([a.data.sum().item() for a in model.parameters()])))
            print("---------------------------------------")

    if train_options["save_model"]:
        load_model(model, train_options["model_path"])

    return model, epoch_loss, epoch_acc

def param_search_model(train_options, model_type = "CNN"):
    train_options = copy.deepcopy(train_options)
    train_options["epochs"] = 8
    lr = 0.01

    lr_acc_map = {}

    for i in range(8):
        train_options["lr_rate"] = lr
        if model_type == "CNN":
            epoch_loss, epoch_acc = train_cnn_model(train_options)
        lr_acc_map[lr] = max(epoch_acc["val"])
        lr = lr / 5
    return lr_acc_map


# lr_acc_map = param_search_model(base_train_options)
# print(lr_acc_map)

def plot_epoch_stats(epoch_stat, label):
    for k in epoch_stat:
        plt.plot(epoch_stat[k], label = k)
    plt.legend()
    plt.title(label)
    plt.show()

def plot_losses_n_acces(losses, acces):
    plot_epoch_stats({d_type +"_"+ split: losses[d_type][split] for d_type in losses for split in ("val", "train")}, "Loss vs epoch")
    plot_epoch_stats({d_type +"_"+ split: acces[d_type][split] for d_type in acces for split in ("val", "train")}, "Accuracy vs epoch")

def train_cnn_models_all_data_types():
    base_cnn_train_options["epochs"] = 20
    base_data = copy.deepcopy(base_cnn_train_options)
    base_data["dataset_type"] = "base"
    configs = [base_data]
    aug_data = copy.deepcopy(base_data)
    aug_data["dataset_type"] = "augmented"
    configs.append(aug_data)

    data_epoch_losses = {}
    data_epoch_acces = {}

    for config in configs:
        epoch_loss, epoch_acc = train_cnn_model(config)
        data_epoch_losses[config["dataset_type"]] = epoch_loss
        data_epoch_acces[config["dataset_type"]] = epoch_acc

    plot_losses_n_acces(data_epoch_losses, data_epoch_acces)
    return data_epoch_losses, data_epoch_acces

# cnn_losses, cnn_acces = train_cnn_models_all_data_types()

# base_cnn_train_options["dataset_type"] = "augmented"
# lr_acc_map = param_search_model(base_cnn_train_options)
# print(lr_acc_map)

# BE SURE TO MENTION THAT AUGMENTING DATA IS BASICALLY EQUAL TO MORE EPOCHS

def train_cifar_models(dataset_type, model_type, epochs = 100, load_model = False):
    train_options = copy.deepcopy(base_cnn_train_options)
    train_options["load_model"] = load_model
    train_options["save_model"] = True
    train_options["epochs"] = epochs
    train_options["lr_rate"] = 0.0008
    train_options["batch_size"] = 512
    train_options["dataset_type"] = dataset_type
    train_options["model_type"] = model_type
    change_paths_to_absolute(train_options)
    return train_cnn_model(train_options)

# train_cifar_models("cifar3", "cifar3", 10, load_model = True)
# train_cifar_models("cifar500", "cifar500", 1, load_model = True)
# train_cifar_models("cifar500", "transfer", 200, load_model = True)

all_data, all_labels = unpickle_n_combine(base_cnn_train_options["data_files"])
all_data = all_data/255
plot_10x10_imgs(all_data, all_labels)
