import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mnist_data(split):
    return FastMNIST('data/MNIST', train=split == "train", download=True)

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

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

def get_bn_fc(in_dim = 1024, out_dim = 1024, activation = "relu", fc_layer_type = "linear", compress = 20):
    if fc_layer_type != "linear":
        return [nn.BatchNorm1d(in_dim), nn.Dropout(0.1),
                (SVDLinearUSV(compress, in_dim, out_dim), SVDLinearUV(compress, in_dim, out_dim), SVDLinearCustomGrad(in_dim, out_dim))[("usv_svd", "uv_svd", "svd_custom").index(fc_layer_type)],
                (nn.ReLU(), nn.Softmax(dim=-1))[activation == "softmax"]]

    layers = [nn.BatchNorm1d(in_dim), nn.Dropout(0.1), nn.Linear(in_dim, out_dim)]
    # layers = [nn.Linear(in_dim, out_dim)]
    # layers = [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dim)]
    fc = layers[2]
    if activation == "relu":
        torch.nn.init.kaiming_normal_(fc.weight.data)
        nn.init.constant_(fc.bias.data, 0)
        layers.append(nn.ReLU())
    elif activation == "softmax":
        torch.nn.init.xavier_normal_(fc.weight.data)
        nn.init.constant_(fc.bias.data, 0)
        layers.append(nn.Softmax(dim=-1))
    return layers

class FC5(nn.Module):
    def __init__(self, compress = 20, linear_layer_type = "linear"):
        super(FC5, self).__init__()
        self.layers = nn.Sequential(
            *get_bn_fc(in_dim=784, fc_layer_type = linear_layer_type),
            *get_bn_fc(compress = compress, fc_layer_type = linear_layer_type),
            *get_bn_fc(compress = compress, fc_layer_type = linear_layer_type),
            *get_bn_fc(compress = compress, fc_layer_type = linear_layer_type),
            *get_bn_fc(compress = compress, fc_layer_type = linear_layer_type),
            *get_bn_fc(out_dim=10, activation = "softmax")
        )

    def forward(self, samples):
        return self.layers(samples)

    def get_fc_layer(self, layer_num):
        for layer in self.layers:
            if type(layer) == torch.nn.modules.linear.Linear:
                if layer_num == 0:
                    return layer
                layer_num = layer_num - 1

    def get_layer_output(self, samples, layer_num, before_activation = False):
        return self.layers[:4*layer_num - int(before_activation)](samples)

    def clone_weights(self, model):
        for src_layer, dest_layer in zip(model.layers, self.layers):
            if type(dest_layer) == nn.Linear or type(src_layer) != nn.Linear:
                dest_layer.load_state_dict(src_layer.state_dict())
            else:
                dest_layer.copy_weights(src_layer)

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    return model

def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    return model

baseline_train_options = {
    "lr_rate": 0.001,
    "batch_size": 512,
    "epochs": 100,
    "base_path": "",
    "model_path": "saved_models/mnist_fc5.model",
    "save_model": True,
    "load_model": True,
    "train_model": True,
    "dropouts": [],
    "linear_layer_type": "linear",
    "compress": 20,
    "cap_accuracy": 0.97
}

def change_paths_to_absolute(train_ops):
    if train_ops["base_path"] in train_ops["model_path"]:
        return train_ops
    train_ops["model_path"] = train_ops["base_path"] + train_ops["model_path"]

def get_val_accuracy(model, loader):
    model.eval()
    with torch.no_grad():
        total = 0
        total_correct = 0
        for X, Y in loader:
            X = X.view(X.shape[0], -1)
            pred = model(X.to(device)).cpu()
            acc = pred.argmax(dim=-1) == Y
            total += X.size(0)
            total_correct += acc.sum().item()

    test_acc = total_correct / total
    return test_acc

def train_fc5_model(train_options):
    model = FC5(compress=train_options["compress"], linear_layer_type=train_options["linear_layer_type"])

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = torch.optim.Adam(model.parameters(), eps=train_options["lr_rate"])

    train_loader = DataLoader(dataset=get_mnist_data("train"), batch_size=train_options["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=train_options["batch_size"], shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    max_acc = get_val_accuracy(model, test_loader)
    print("Initial Acc: " + str(max_acc))

    for epoch in range(1, train_options["epochs"] + 1):
        print("Epoch: " + str(epoch))
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(X.shape[0], -1)
            pred = model(X.to(device)).cpu()
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = get_val_accuracy(model, test_loader)
        if test_acc > max_acc:
            max_acc = test_acc
            save_model(model, train_options["model_path"])

        print("Test Accuracy:" + str(test_acc))
        if test_acc >= train_options["cap_accuracy"]:
            save_model(model, train_options["model_path"])
            return model

def get_all_fc_layer_weights(model: FC5, num_layers = 5):
    layer_weights = []
    for layer_idx in range(num_layers):
        layer = model.get_fc_layer(layer_idx)
        # print(layer.weight.data.shape)
        layer_weights.append(torch.clone(layer.weight.data, memory_format=torch.preserve_format))

    return layer_weights

def get_svd_weights(layer_weights):
    U, S, V = [], [], []
    for weights in layer_weights:
        u,s,v = torch.linalg.svd(weights, full_matrices = False)
        U.append(u)
        S.append(s)
        V.append(v)

    # total_weights = sum([s.abs().sum() for s in S])
    # for a in (10, 20, 50, 100, 200):
    #     print(str(a) + " : " + str(sum([s[:a].abs().sum() for s in S]) / total_weights))
    return U,S,V

def save_svd_comp_model_reduced(model: FC5, U, S, V, compressed_rows, model_path, linear_layer_type = "usv_svd"):
    new_model = FC5(compress=compressed_rows, linear_layer_type=linear_layer_type)
    decomp_idx = 0

    with torch.no_grad():
        for src, dest in zip(model.layers, new_model.layers):
            if type(dest) == nn.Linear or type(src) != nn.Linear:
                dest.load_state_dict(src.state_dict())
            else:
                dest.load_from_decomposition(U[decomp_idx], S[decomp_idx], V[decomp_idx], src.bias.data)
                decomp_idx += 1

    save_model(new_model, model_path + "_" + str(compressed_rows) + "_" + linear_layer_type)
    return new_model

def save_svd_comp_model(model, U, S, V, compress_rows, model_path):
    new_model = FC5()
    new_model.load_state_dict(model.state_dict()) # Load all other params for batch norm
    all_layers = [new_model.get_fc_layer(i) for i in range(len(S))]

    with torch.no_grad():
        for i, layer in enumerate(all_layers):
            if compress_rows == -1:
                W_i = torch.clone(torch.matmul(torch.matmul(U[i],torch.diag(S[i])), V[i]))
            else:
                W_i = torch.clone(torch.matmul(torch.matmul(U[i][..., :compress_rows],torch.diag(S[i])[:compress_rows, :compress_rows]), V[i][:compress_rows, ...]))
            layer.weight.data = W_i

    save_model(new_model, model_path + "_" + str(compress_rows) + "_usv_svd")
    return new_model

def load_svd_model(model_path, compression_rows = None, fc_layer_type ="linear"):
    model = FC5(compression_rows, linear_layer_type=fc_layer_type)
    # model = FC5()
    return load_model(model, model_path + ("", "_" + str(compression_rows))[compression_rows is not None] + ("", "_" + fc_layer_type)[fc_layer_type != "linear"])

def save_svd_compressed_models(model: FC5, compression_rows, model_path, num_layers = 5):
    layer_weights = get_all_fc_layer_weights(model, num_layers)

    U, S, V = get_svd_weights(layer_weights)

    for num_rows in compression_rows:
        model_same = save_svd_comp_model_reduced(model, U, S, V, num_rows, model_path)

def prob1_execute():
    # baseline_train_options["train_model"] = False
    # model = train_fc5_model(baseline_train_options)

    model = load_svd_model(baseline_train_options["model_path"])
    print("Original parameters:" + str(sum([torch.tensor(parameter.shape).prod() for parameter in model.parameters()])))
    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=baseline_train_options["batch_size"], shuffle=False)
    all_compressions = [10, 20, 50, 100, 200, -1]
    save_svd_compressed_models(model, all_compressions, baseline_train_options["model_path"])

    for compress_rows in all_compressions:
        model = load_svd_model(baseline_train_options["model_path"], compress_rows, fc_layer_type="usv_svd")
        print("Parameters: " + str(sum([torch.tensor(parameter.shape).prod() for parameter in model.parameters()])))
        print("Compressed to rows:" + str(compress_rows))
        print("Test accuracy: " + str(get_val_accuracy(model, test_loader)))
        print("------------------------------------------")

class SVDLinearUV(nn.Module):
    def __init__(self, compress, in_dim = 1024, out_dim = 1024):
        super(SVDLinearUV, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        compress = compress if compress > 0 else out_dim
        self.compress = compress
        self.U = nn.Parameter(torch.zeros(out_dim, compress))
        self.V = nn.Parameter(torch.zeros(compress, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, samples):
        return samples @ self.V.T @ self.U.T + self.bias

    def load_from_decomposition(self, U, S, V, bias):
        self.bias.data = torch.clone(bias)
        self.U.data = torch.clone(U[..., :self.compress] @ torch.diag(S)[:self.compress, :self.compress])
        self.V.data = torch.clone(V[:self.compress, ...])

    def copy_weights(self, linear: nn.Linear):
        with torch.no_grad():
            U, S, V = torch.linalg.svd(linear.weight.data, full_matrices=False)
            self.load_from_decomposition(U,S,V,linear.bias.data)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        print("Doing Nothing...")
        pass

class SVDLinearUSV(nn.Module):
    def __init__(self, compress, in_dim = 1024, out_dim = 1024):
        super(SVDLinearUSV, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        compress = compress if compress > 0 else out_dim
        self.compress = compress
        self.U = nn.Parameter(torch.zeros(out_dim, compress))
        self.S = nn.Parameter(torch.zeros(compress, compress))
        self.V = nn.Parameter(torch.zeros(compress, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, samples):
        return samples @ (self.U @ (self.S @ self.V)).T + self.bias

    def load_from_decomposition(self, U, S, V, bias):
        self.bias.data = torch.clone(bias)
        self.U.data = torch.clone(U[..., :self.compress])
        self.S.data = torch.clone(torch.diag(S)[:self.compress, :self.compress])
        self.V.data = torch.clone(V[:self.compress, ...])

    def copy_weights(self, linear: nn.Linear):
        with torch.no_grad():
            U,S,V = torch.linalg.svd(linear.weight.data, full_matrices = False)
            self.load_from_decomposition(U, S, V, linear.bias.data)

def prob2_execute():
    baseline_train_options["cap_accuracy"] = 0.98
    baseline_train_options["load_model"] = True
    # model = train_fc5_model(baseline_train_options)
    model = load_svd_model(baseline_train_options["model_path"])
    svd_model = FC5(compress=20, linear_layer_type="uv_svd")
    svd_model.clone_weights(model)

    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=baseline_train_options["batch_size"], shuffle=False)
    print(get_val_accuracy(svd_model, test_loader))

    model_path = baseline_train_options["model_path"] + "_20_uv_svd"
    save_model(svd_model, model_path)
    train_options = copy.deepcopy(baseline_train_options)
    train_options["batch_size"] = 1024
    train_options["model_path"] = model_path

    train_options["cap_accuracy"] = 1.0
    train_options["lr_rate"] = 1e-6
    train_options["load_model"] = True
    train_options["linear_layer_type"] = "uv_svd"
    train_fc5_model(train_options)

# prob2_execute()

class SVDCompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, bias):
        u,s,v = torch.linalg.svd(weights)
        s = torch.diag(s)
        ctx.save_for_backward(input, weights, bias)
        return input @ (u[:, :20] @ (s[:20, :20] @ v[:20, :])).T + bias

    @staticmethod
    def backward(ctx, grad_outputs):
        input, weights, bias = ctx.saved_tensors
        grad_input = grad_outputs @ weights
        grad_weight = grad_outputs.T @ input
        grad_bias = torch.clone(grad_outputs)
        return grad_input, grad_weight, grad_bias

class SVDLinearCustomGrad(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SVDLinearCustomGrad, self).__init__()
        self.weight = nn.Parameter(torch.rand(out_dim, in_dim))
        self.bias = nn.Parameter(torch.rand(out_dim))
        self.calc = SVDCompute.apply
        self.reuse_decomposition = False

    def forward(self, samples):
        if self.training:
            # Turn off USV
            self.reuse_decomposition = False
            return self.calc(samples, self.weight, self.bias)
        else:
            # Turn on USV
            if self.reuse_decomposition:
                # To reduce the computations of u, s, v for test data
                return samples @ (self.u[:, :20] @ (self.s[:20, :20] @ self.v[:20, :])).T + self.bias

            self.reuse_decomposition = True
            self.u, self.s, self.v = torch.linalg.svd(self.weight)
            self.s = torch.diag(self.s)
            return samples @ (self.u[:, :20] @ (self.s[:20, :20] @ self.v[:20, :])).T + self.bias

    def copy_weights(self, linear: nn.Linear):
        with torch.no_grad():
            self.weight.data = torch.clone(linear.weight.data)
            self.bias.data = torch.clone((linear.bias.data))

def prob3_execute():
    print("Prob3")
    model = load_svd_model(baseline_train_options["model_path"])
    svd_model = FC5(compress=20, linear_layer_type="svd_custom")
    svd_model.clone_weights(model)

    model_path = baseline_train_options["model_path"] + "_20_svd_custom"
    save_model(svd_model, model_path)
    train_options = copy.deepcopy(baseline_train_options)
    train_options["batch_size"] = 1024
    train_options["model_path"] = model_path

    train_options["cap_accuracy"] = 1.0
    train_options["lr_rate"] = 1e-6
    train_options["load_model"] = True
    train_options["save_model"] = True
    train_options["linear_layer_type"] = "svd_custom"
    train_fc5_model(train_options)

prob3_execute()
