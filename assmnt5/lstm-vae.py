import os
import pickle
import traceback

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mnist_data(split):
    return SectionalMNIST('data/MNIST', train=split == "train", download=True)

def get_mnist_sections(data):
    src = torch.zeros(data.shape[0], 0, 49)
    target = torch.zeros(data.shape[0], 0, 49)
    for num in range(15):
        i = int(num / 4)
        j = num % 4
        src = torch.cat((src, data[..., i*7:i*7 + 7, j*7:j*7 + 7].reshape(-1, 1, 49)), dim= 1)
        num = num + 1
        i = int(num / 4)
        j = num % 4
        target = torch.cat((target, data[..., i*7:i*7 + 7, j*7:j*7 + 7].reshape(-1, 1, 49)), dim= 1)

    return src, target

class SectionalMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.targets = self.targets.to(device)
        self.cur_fr, self.next_fr = get_mnist_sections(self.data.squeeze())
        self.cur_fr = self.cur_fr.to(device)
        self.next_fr = self.next_fr.to(device)
        self.data = self.data.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.cur_fr[index], self.next_fr[index], self.targets[index]

def plot_epoch_stats(epoch_stat, label):
    for k in epoch_stat:
        plt.plot(epoch_stat[k], label = k)
    plt.legend()
    plt.title(label)
    plt.show()

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

        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def plot_mnist_images(class_samples, class_titles = None, cols=10):
    f, axarr = plt.subplots(len(class_samples.keys()), cols, figsize=(8, 8), squeeze = False)
    for i in range(len(class_samples.keys()) * 10):
        row_num = int(i / 10)
        col_num = i % 10
        if len(class_samples[row_num]) < (col_num + 1):
            continue
        img = class_samples[row_num][col_num]
        axarr[row_num, col_num].imshow(class_samples[row_num][col_num])
        axarr[row_num, col_num].set_title("" if class_titles is None else class_titles[row_num][col_num])
        plt.imshow(img)

    plt.show()

class LSTMVae(nn.Module):
    def __init__(self):
        super(LSTMVae, self).__init__()
        self.hidden_dim = 64
        self.rnn = nn.LSTM(input_size=49, hidden_size=self.hidden_dim, num_layers=2, dropout=0., batch_first=True)
        self.gen_mode = False
        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_dim),
            nn.Linear(self.hidden_dim, 49),
            nn.Linear(49, 49),
            nn.Sigmoid()
        )

        for layer in self.linear_layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    def forward(self, samples):
        if self.gen_mode:
            return self.gan_forward(samples)
        o, (h,c) = self.rnn(samples)
        return self.linear_layers(o.reshape(-1, self.hidden_dim)).reshape(samples.shape[0], -1, 49)

    def gan_forward(self, samples):
        o, (h,c) = self.rnn(samples)
        new_sample = self.linear_layers(o.reshape(-1, self.hidden_dim)).reshape(samples.shape[0], -1, 49)[:, -1:, :]
        output_samples = new_sample

        for i in range(7):
            o, (h,c) = self.rnn(new_sample, (h,c))
            new_sample = self.linear_layers(o.reshape(-1, self.hidden_dim)).reshape(samples.shape[0], 1, -1)
            output_samples = torch.cat((output_samples, new_sample), dim=1)

        return output_samples

def get_test_loss(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X,Y,nums in loader:
            pred = model(X.to(device))
            total_loss += loss_fn(pred, Y).item()

    return total_loss / loader.dataset.__len__()

def convert_section_to_images(sections): # batch, 16, 49
    imgs = torch.zeros(sections.shape[0], 28, 28)
    for sec_num in range(16):
        i = int(sec_num / 4)
        j = sec_num % 4
        imgs[:, i*7: i*7 + 7, j*7:j*7+7] = sections[:, sec_num, :].reshape(-1,7,7)
    return imgs


def generate_images(model, loader):
    model.eval()
    model.gen_mode = True
    images_10 = {i: {"true":[], "pred":[]} for i in range(10)}
    with torch.no_grad():
        for X,Y,nums in loader:
            pred = model(X[:,:8,:].to(device))
            full_img_pred = torch.cat((X[:,:8,:], pred), dim=1)
            full_img_true = torch.cat((X[:,:,:], Y[:,-1:,:]), dim=1)

            full_img_true = convert_section_to_images(full_img_true)
            full_img_pred = convert_section_to_images(full_img_pred)
            full_img_pred = full_img_pred > 0.5

            for i, num in enumerate(nums):
                num = num.item()
                if len(images_10[num]["pred"]) >= 10:
                    continue
                images_10[num]["pred"].append(full_img_pred[i])
                images_10[num]["true"].append(full_img_true[i])

            if sum([len(images_10[k]["pred"])for k in images_10]) == 100:
                break
    plot_mnist_images({k:images_10[k]["true"] for k in images_10})
    plot_mnist_images({k:images_10[k]["pred"] for k in images_10})

    model.gen_mode = False


def train_lstm_gan_model(train_options):
    model = LSTMVae()
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = torch.optim.Adam(model.parameters(), eps=train_options["lr_rate"])

    train_loader = DataLoader(dataset=get_mnist_data("train"), batch_size=train_options["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=get_mnist_data("test"), batch_size=train_options["batch_size"], shuffle=True)

    loss_fn = torch.nn.BCELoss()

    min_test_loss = get_test_loss(model, test_loader, loss_fn)

    generate_images(model, test_loader)

    for epoch in range(0, train_options["epochs"]):
        print("Epoch: " + str(epoch))
        model.train()
        total_loss = 0
        for i, (X, Y, nums) in enumerate(train_loader):
            if (Y.view(-1) > 1).sum() >=1 or (Y.view(-1) <0).sum() >=1:
                print("INPUT MALFORMED")
            pred = model(X.to(device))
            if (pred.view(-1) > 1).sum() >= 1 or (pred.view(-1) < 0).sum() >= 1:
                print("INPUT MALFORMED")
            loss = loss_fn(pred, Y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = get_test_loss(model, test_loader, loss_fn)
        print(f"Test Loss: {test_loss}")
        print(f"Train Loss: {total_loss / train_loader.dataset.__len__()}")

        if epoch % 10 == 0:
            generate_images(model, test_loader)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_model(model, train_options["model_path"])
    return model

class MnistVaeDEF(nn.Module):
    def __init__(self):
        super(MnistVaeDEF, self).__init__()
        self.latent_dim = 3
        self.net = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
            nn.ReLU(),
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 784),
            nn.Sigmoid()
        )

        self.norm1d = nn.BatchNorm1d(num_features=3)

    def decoder(self, sample):
        return self.net[6:](sample)


    def forward(self, samples):

        return self.net(samples), torch.zeros(samples.shape[0], 3), torch.zeros(samples.shape[0], 3)


class MnistVaeCNN(nn.Module):
    def __init__(self):
        super(MnistVaeCNN, self).__init__()
        self.latent_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_size=(3,3), in_channels=1, out_channels=2),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2), #13x13

            nn.Conv2d(kernel_size=(3, 3), in_channels=2, out_channels=3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=3), # 5x5

            nn.Flatten(),
            nn.Linear(in_features=75, out_features=self.latent_dim * 2),
        )

        self.norm1d = nn.BatchNorm1d(num_features=self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(3, 75),
            nn.ReLU(),
            nn.Unflatten(1, (3, 5, 5)),
            nn.BatchNorm2d(num_features=3),

            nn.ConvTranspose2d(kernel_size=(3,3), stride=(2,2), in_channels=3, out_channels=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=3),

            nn.ConvTranspose2d(kernel_size=(3,3), stride=(2,2), in_channels=3, out_channels=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1),

            nn.Flatten(),
            nn.Linear(in_features=529, out_features=784),
            nn.Sigmoid()
        )

        for layer in self.decoder:
            if type(layer) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(layer.weight.data)

        for layer in self.encoder:
            if type(layer) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(layer.weight.data)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar * 0.5)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, samples):  # batch, 784
        mean_var = self.encoder(samples.view(-1, 1, 28, 28)).view(samples.shape[0], 2, -1)

        # mean_l = self.mean_fc(mean_var)
        # var_l = self.var_fc(mean_var)
        mean_l = mean_var[:, 0, :]
        var_l = mean_var[:, 1, :]

        decoder_inp = self.reparametrize(mean_l, var_l)
        self.norm1d(decoder_inp)
        return self.decoder(decoder_inp), mean_l, var_l


class MnistVae(nn.Module):
    def __init__(self):
        super(MnistVae, self).__init__()
        self.fc1_dim = 100
        self.fc2_dim = 30
        self.latent_dim = 3
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=self.fc1_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.fc1_dim, out_features=self.fc2_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.fc2_dim, out_features=self.latent_dim * 2),
            # nn.ReLU()
        )

        # self.mean_fc = nn.Linear(in_features=self.fc1_dim, out_features=self.latent_dim)
        # self.var_fc = nn.Linear(in_features=self.fc1_dim, out_features=self.latent_dim)

        # nn.init.xavier_normal_(self.mean_fc.weight.data)
        # nn.init.xavier_normal_(self.var_fc.weight.data)

        # To calculate running mean and variance...
        self.norm1d_mean = nn.BatchNorm1d(num_features=self.latent_dim)
        self.norm1d_var = nn.BatchNorm1d(num_features=self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, self.fc1_dim),
            nn.ReLU(),
            nn.Linear(self.fc1_dim, 784),
            nn.Sigmoid()
        )

        for layer in self.encoder:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

        for layer in self.decoder:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar * 0.5)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, samples):  # batch, 784
        mean_var = self.encoder(samples).view(samples.shape[0], 2, -1)

        # mean_l = self.mean_fc(mean_var)
        # var_l = self.var_fc(mean_var)
        mean_l = mean_var[:, 0, :]
        var_l = mean_var[:, 1, :]

        self.norm1d_mean(mean_l)
        self.norm1d_var(var_l)

        decoder_inp = self.reparametrize(mean_l, var_l)
        return self.decoder(decoder_inp), mean_l, var_l


def calculate_vae_loss(loss_fn, pred, target, mu, sigma):
    bce_loss = loss_fn(pred, target)
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return bce_loss + kl_loss

def get_audio_data(file_path):
    s = pickle.load(open(file_path, "rb"))
    return torch.from_numpy(s).reshape(-1, 784)

class PoorSevenDataset(nn.Module):
    def __init__(self, data):
        super(PoorSevenDataset, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

def get_poor7_dataloader(file_path, split, batch_size):
    data = get_audio_data(file_path)
    dataset = PoorSevenDataset(data)
    return DataLoader(dataset = dataset, shuffle=split == "train", batch_size = batch_size)

def get_vae_test_loss(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X in loader:
            pred, mu, sigma = model(X.to(device))
            total_loss += calculate_vae_loss(loss_fn, pred.cpu(), X, mu.cpu(), sigma.cpu())

    return total_loss / loader.dataset.__len__()

def generate_images_with_latent_dim(model, loader = None, latent_means = None, latent_vars = None):
    model.eval()
    latent_dim = model.latent_dim
    gen_imgs = {i: [] for i in range(latent_dim)}
    gen_titles = {i: [] for i in range(latent_dim)}
    with torch.no_grad():
        if loader is not None:
            for X in loader:
                X, mu, sigma = model(X.to(device))
                plot_mnist_images({0: X[:10].view(10, 28, 28)}, cols=10)
                return
        img_per_dim = 10
        for i in range(latent_dim):
            if latent_means is not None:
                latent_vec = latent_means[i]
            else:
                latent_vec = torch.clone(model.norm1d_mean.running_mean)

            if latent_vars is not None:
                var = latent_vars[i]
            else:
                var = model.norm1d_var.running_var[i]
            for multiplier in range(int(-img_per_dim / 2), int(img_per_dim / 2), 1):
                new_l_vec = torch.clone(latent_vec)
                new_l_vec[i] = new_l_vec[i] + var * multiplier * 0.1
                pred = model.decoder(new_l_vec.unsqueeze(0))
                # pred = pred > 0.3
                gen_imgs[i].append(pred.view(28, 28).cpu())
                gen_titles[i].append(str(round(new_l_vec[i].item(), 2)))
        plot_mnist_images(gen_imgs, gen_titles, cols=10)


def train_vae1_model(train_options):
    model = MnistVae()
    # model = MnistVaeCNN()
    # model = MnistVaeDEF()
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = torch.optim.Adam(model.parameters(), eps=train_options["lr_rate"])

    train_loader = get_poor7_dataloader(train_options["train_data_path"], "train", train_options["batch_size"])
    test_loader = get_poor7_dataloader(train_options["test_data_path"], "test", train_options["batch_size"])

    loss_fn = torch.nn.BCELoss(reduction="sum")

    min_test_loss = get_vae_test_loss(model, test_loader, loss_fn)

    for epoch in range(0, train_options["epochs"]):
        print("Epoch: " + str(epoch))
        model.train()
        total_loss = 0
        for i, X in enumerate(train_loader):
            pred, mu, sigma = model(X.to(device))
            loss = calculate_vae_loss(loss_fn, pred.cpu(), X, mu.cpu(), sigma.cpu())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = get_vae_test_loss(model, test_loader, loss_fn)

        print(f"test loss: {test_loss}")
        print(f"train loss: {total_loss / train_loader.dataset.__len__()}")

        if epoch % 100 == 0:
            generate_images_with_latent_dim(model, train_loader)
            generate_images_with_latent_dim(model)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_model(model, train_options["model_path"])

    return model

class MnistDisDataset():
    def __init__(self, batch_size, rand_dim = 100):
        if batch_size % 10 != 0:
            raise AttributeError("batch size not a multiple of 10.")
        self.batch_size = batch_size # Multiple of 10
        self.rand_dim = rand_dim
        self.per_digit_exemplars = batch_size // 10

    def get_data(self):
        rand_dim = torch.randn(self.batch_size, self.rand_dim)
        digits = torch.tile(torch.eye(10), (self.per_digit_exemplars, 1))
        # digits = torch.tile(torch.tensor([1,0,0,0,0,0,0,0,0,0]), (self.batch_size, 1))
        return torch.cat((rand_dim, digits), dim=1)

class MnistGenerator(nn.Module):
    def __init__(self, in_dim = 110):
        super(MnistGenerator, self).__init__()
        self.in_dim = in_dim

        self.gen_layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=784),
            nn.Tanh()
        )

        for i, layer in enumerate(self.gen_layers):
            if type(layer) != nn.Linear:
                continue
            if i == (len(self.gen_layers) - 1):
                nn.init.xavier_normal_(layer.weight.data)
                continue
            nn.init.kaiming_normal_(layer.weight.data)

    def forward(self, samples):
        gen_labels = torch.clone(samples[:, -10:])
        gen_digits = self.gen_layers(samples)
        return torch.cat((gen_digits, gen_labels), dim=1)

def plot_generator_imgs(model_gen):
    model_gen.eval()
    class_samples = {i:[] for i in range(10)}
    with torch.no_grad():
        for i in range(10):
            class_samples[i] = model_gen(torch.cat((torch.randn(10, 100), torch.nn.functional.one_hot((torch.ones(10) * i).long(), num_classes = 10)), dim = 1).to(device)).cpu()[:, :784].view(-1, 28, 28)
    plot_mnist_images(class_samples)

class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        self.dis_layers = nn.Sequential(
            nn.Linear(in_features=794, out_features=400),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=400, out_features=200),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
        )

        for i, layer in enumerate(self.dis_layers):
            if type(layer) != nn.Linear:
                continue
            if i == (len(self.dis_layers) - 1):
                nn.init.xavier_normal_(layer.weight.data)
                continue
            nn.init.kaiming_normal_(layer.weight.data)

    def forward(self, samples):
        return self.dis_layers(samples)

class MnistGanDataset3(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data * 2 - 1 # scale to [1,-1]

        self.data_with_digit = torch.cat((self.data.view(-1, 784), torch.nn.functional.one_hot(self.targets, num_classes = 10)), dim=1).to(device)
        self.labels = torch.ones(self.data_with_digit.shape[0], 1).to(device)

    def __getitem__(self, index):
        return self.data_with_digit[index], self.labels[index]

    def __len__(self):
        return self.data_with_digit.shape[0]

def get_dataloader_gen_dataset(batch_size):
    dataset = MnistGanDataset3('data/MNIST', train ="train", download=True)
    return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)

def train_mnist_gan(train_options):
    model_gen = MnistGenerator()
    model_dis = MnistDiscriminator()
    model_dis.to(device)
    model_gen.to(device)

    if train_options["load_model"]:
        load_model(model_gen, train_options["gen_model_path"])
        load_model(model_dis, train_options["dis_model_path"])

    # plot_generator_imgs(model_gen)

    if not train_options["train_model"]:
        return model_gen, model_dis

    optimizer_gen = torch.optim.Adam(model_gen.parameters(), eps=train_options["lr_rate"])
    optimizer_dis = torch.optim.Adam(model_dis.parameters(), eps=train_options["lr_rate"])

    true_loader = get_dataloader_gen_dataset(train_options["batch_size"])
    gen_loader = MnistDisDataset(batch_size=train_options["batch_size"])

    loss_fn = torch.nn.BCELoss()

    print(sum([param.sum().item() for param in model_dis.parameters()]))
    print(sum([param.sum().item() for param in model_gen.parameters()]))

    for epoch in range(1, train_options["epochs"] + 1):
        print("Epoch: " + str(epoch))
        model_gen.train()
        model_dis.train()
        total_loss_dis = 0
        total_loss_gen = 0
        dis_acc = 0
        gen_acc = 0
        total_gen = 0
        total_dis = 0

        for true_imgs, true_labels in true_loader:
            gen_data = gen_loader.get_data()
            gen_imgs = model_gen(gen_data.to(device))

            dis_samples = torch.cat((true_imgs, gen_imgs), dim=0)
            dis_labels = torch.cat((true_labels, torch.zeros(gen_imgs.shape[0], 1).to(device)), dim=0)
            dis_pred = model_dis(dis_samples)

            total_dis += true_imgs.shape[0]
            total_gen += gen_imgs.shape[0]
            dis_acc += (dis_pred[:gen_data.shape[0]] > 0.5).sum().item()
            gen_acc += (dis_pred[gen_data.shape[0]:] < 0.5).sum().item()

            loss_dis = loss_fn(dis_pred, dis_labels)
            total_loss_dis += loss_dis.item()
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            # dis_acc += (dis_labels == (dis_pred > 0.5)).sum().item()
            # total_dis += dis_labels.shape[0]

            d_param_sum = sum([param.sum().item() for param in model_dis.parameters()])
            ############################
            model_dis.eval()
            for x in range(1):
                gen_data = gen_loader.get_data()
                for d in range(4):
                    gen_data = torch.cat((gen_data, gen_loader.get_data()), dim=0)
                gen_imgs = model_gen(gen_data.to(device))
                dis_pred = model_dis(gen_imgs)
                loss_dis = loss_fn(dis_pred, torch.ones(dis_pred.shape).to(device))
                optimizer_gen.zero_grad()
                loss_dis.backward()
                optimizer_gen.step()

                total_loss_gen += loss_dis.item()
                # gen_acc += (dis_pred > 0.5).sum().item()
                # total_gen += dis_pred.shape[0]
            model_dis.train()
            new_d_param_sum = sum([param.sum().item() for param in model_dis.parameters()])
            if new_d_param_sum > d_param_sum * 1.0001 or new_d_param_sum < d_param_sum * 0.9999:
                print("----------------___________________XXxxxx---------ABBBOOOOORT------------------------------____________________")
        print(f"Generator Accuracy: {gen_acc / total_gen}")
        print(f"Discriminator Accuracy: {dis_acc / total_dis}")
        # print(sum([param.sum().item() for param in model_dis.parameters()]))
        # print(sum([param.sum().item() for param in model_gen.parameters()]))
        print(f"discriminator loss {total_loss_dis / total_dis}")
        print(f"generator loss {total_loss_gen / total_gen}")

        if train_options["save_model"]:
            save_model(model_dis, train_options["dis_model_path"])
            save_model(model_gen, train_options["gen_model_path"])

        if epoch % 10 == 0:
            plot_generator_imgs(model_gen)

    return model_gen, model_dis

train_options = {
    "lr_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "model_path": "saved_models/mnist_gan_lstm.model",
    "save_model": True,
    "load_model": False,
    "train_model": True,
}

# train_lstm_gan_model(train_options)

vae_train_options = {
    "lr_rate": 0.0001,
    "batch_size": 512,
    "epochs": 5000,
    "model_path": "saved_models/mnist_vae1.model",
    "train_data_path": "data/hw5_tr7.pkl",
    "test_data_path": "data/hw5_te7.pkl",
    "save_model": True,
    "load_model": True,
    "train_model": False,
}
model = train_vae1_model(vae_train_options)
generate_images_with_latent_dim(model)

train_options_gan3 = {
    "lr_rate": 1e-2,
    "batch_size": 100,
    "epochs": 500,
    "dis_model_path": "saved_models/mnist_gan_3_dis.model",
    "gen_model_path": "saved_models/mnist_gan_3_gen.model",
    "train_data_path": "data/hw5_tr7.pkl",
    "test_data_path": "data/hw5_te7.pkl",
    "save_model": True,
    "load_model": True,
    "train_model": True,
}

# train_mnist_gan(train_options_gan3)

class MnistMissingGenerator(nn.Module):
    def __init__(self, in_dim = 200):
        super(MnistMissingGenerator, self).__init__()

        self.gen_layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=784),
            nn.Tanh()
        )

        for layer in self.gen_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, samples):
        return self.gen_layers(samples)

class MnistMissingDiscriminator(nn.Module):
    def __init__(self):
        super(MnistMissingDiscriminator, self).__init__()

        self.dis_layers = nn.Sequential(
            nn.Linear(in_features=784, out_features=400),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=400, out_features=200),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
        )

        for layer in self.dis_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, samples):
        return self.dis_layers(samples)

class MnistMissingDataset(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data * 2 - 1 # scale to [1,-1]

        self.data = self.data.view(-1, 784).to(device)
        self.labels = torch.ones(self.data.shape[0], 1).to(device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

def convert_to_center_patch_rand(mnist_data):
    return torch.cat((mnist_data.view(-1, 28, 28)[:, 9:19, 9:19].reshape(-1, 100), torch.randn(mnist_data.shape[0], 100).to(device)), dim = -1)

def plot_missing_generator_imgs(model, loader):
    class_samples = {i:[] for i in range(10)}
    gen_class_samples = {i:[] for i in range(10)}

    model.eval()
    with torch.no_grad():
        for imgs, _, clses in loader:
            clses = [cls.item() for cls in clses]
            center_patch = convert_to_center_patch_rand(imgs)
            gen_imgs = model(center_patch)

            for i, cls in enumerate(clses):
                if len(class_samples[cls]) >= 10:
                    continue
                class_samples[cls].append(imgs[i].view(28, 28))
                gen_class_samples[cls].append(gen_imgs[i].view(28, 28))
            if sum([len(class_samples[k]) for k in class_samples]) >=100:
                break

    plot_mnist_images(gen_class_samples)
    plot_mnist_images(class_samples)


def train_missing_generator(train_options):
    model_gen = MnistMissingGenerator()
    model_dis = MnistMissingDiscriminator()
    model_dis.to(device)
    model_gen.to(device)

    if train_options["load_model"]:
        load_model(model_gen, train_options["gen_model_path"])
        load_model(model_dis, train_options["dis_model_path"])

    # plot_generator_imgs(model_gen)

    if not train_options["train_model"]:
        return model_gen, model_dis

    optimizer_gen = torch.optim.Adam(model_gen.parameters(), eps=train_options["lr_rate"])
    optimizer_dis = torch.optim.Adam(model_dis.parameters(), eps=train_options["lr_rate"])

    mnist_dataset = MnistMissingDataset('data/MNIST', train ="train", download=True)
    mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=train_options["batch_size"], shuffle=True)

    bc_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    reg_param = 5

    # print(sum([param.sum().item() for param in model_dis.parameters()]))
    # print(sum([param.sum().item() for param in model_gen.parameters()]))

    for epoch in range(1, train_options["epochs"] + 1):
        print("Epoch: " + str(epoch))
        model_gen.train()
        model_dis.train()
        total_loss_dis = 0
        total_loss_gen = 0
        dis_acc = 0
        gen_acc = 0
        total_gen = 0
        total_dis = 0

        for mnist_imgs, mnist_labels, _ in mnist_loader:
            gen_data = convert_to_center_patch_rand(mnist_imgs)
            gen_imgs = model_gen(gen_data.to(device))

            dis_samples = torch.cat((mnist_imgs, gen_imgs), dim=0)
            dis_labels = torch.cat((mnist_labels, torch.zeros(gen_imgs.shape[0], 1).to(device)), dim=0)
            dis_pred = model_dis(dis_samples)

            total_dis += mnist_imgs.shape[0]
            total_gen += gen_imgs.shape[0]
            dis_acc += (dis_pred[:gen_data.shape[0]] > 0.5).sum().item()
            gen_acc += (dis_pred[gen_data.shape[0]:] < 0.5).sum().item()

            loss_dis = bc_loss(dis_pred, dis_labels)
            total_loss_dis += loss_dis.item()
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            # dis_acc += (dis_labels == (dis_pred > 0.5)).sum().item()
            # total_dis += dis_labels.shape[0]

            d_param_sum = sum([param.sum().item() for param in model_dis.parameters()])
            ############################
            model_dis.eval()
            gen_imgs = model_gen(gen_data)
            gen_pred = model_dis(gen_imgs)
            loss_gen_bc = bc_loss(gen_pred, torch.ones(gen_pred.shape).to(device))
            loss_gen_mse = mse_loss(gen_imgs.view(-1, 28, 28)[:, 9:19, 9:19].reshape(-1, 100), gen_data[:, :100])
            gen_loss = loss_gen_mse * reg_param + loss_gen_bc
            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()

            model_dis.train()
            new_d_param_sum = sum([param.sum().item() for param in model_dis.parameters()])
            if new_d_param_sum > d_param_sum * 1.0001 or new_d_param_sum < d_param_sum * 0.9999:
                print("----------------___________________XXxxxx---------ABBBOOOOORT------------------------------____________________")
        print(f"Generator Accuracy: {gen_acc / total_gen}")
        print(f"Discriminator Accuracy: {dis_acc / total_dis}")
        # print(sum([param.sum().item() for param in model_dis.parameters()]))
        # print(sum([param.sum().item() for param in model_gen.parameters()]))
        print(f"discriminator loss {total_loss_dis / total_dis}")
        print(f"generator loss {total_loss_gen / total_gen}")

        if train_options["save_model"]:
            save_model(model_dis, train_options["dis_model_path"])
            save_model(model_gen, train_options["gen_model_path"])

        if epoch % 1 == 0:
            plot_missing_generator_imgs(model_gen, mnist_loader)

    return model_gen, model_dis

# train_options_missing_gan = {
#     "lr_rate": 1e-3,
#     "batch_size": 100,
#     "epochs": 500,
#     "dis_model_path": "saved_models/mnist_gan_mis_dis.model",
#     "gen_model_path": "saved_models/mnist_gan_mis_gen.model",
#     "save_model": True,
#     "load_model": True,
#     "train_model": True,
# }
#
# train_missing_generator(train_options_missing_gan)
