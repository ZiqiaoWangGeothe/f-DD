
from tqdm import tqdm
from fDAL import fDALLearner
from torchvision import transforms
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
from data_list import ImageList, ForeverDataIterator
import torch.optim as optim
import numpy as np
import random
import fire
import os
import torch.nn.utils.spectral_norm as sn


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from copy import deepcopy

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # False


def build_network():
    # network encoder...
    lenet = nn.Sequential(
        nn.Conv2d(1, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(20, 50, kernel_size=5),
        nn.Dropout2d(p=0.5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
    )

    # create a bootleneck layer. it usually helps
    bottleneck_dim = 500
    bottleneck = nn.Sequential(
        nn.Linear(800, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.LeakyReLU(),
        nn.Dropout(0.5)
    )

    backbone = nn.Sequential(
        lenet,
        bottleneck
    )

    # classification head
    num_classes = 10
    taskhead = nn.Sequential(
        sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        sn(nn.Linear(bottleneck_dim, num_classes)),
    )

    return backbone, taskhead, num_classes


def build_data_loaders(source_list, target_list, test_list, batch_size):
    # source_list = './data_demo/usps2mnist/mnist_train.txt'
    # target_list = './data_demo/usps2mnist/usps_train.txt'
    # test_list = './data_demo/usps2mnist/usps_test.txt'
    # batch_size = 128

    # training loaders....
    train_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)

    train_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

    return train_source, train_target, test_loader


def scheduler(optimizer_, init_lr_, decay_step_, gamma_):
    class DecayLRAfter:
        def __init__(self, optimizer, init_lr, decay_step, gamma):
            self.init_lr = init_lr
            self.gamma = gamma
            self.optimizer = optimizer
            self.iter_num = 0
            self.decay_step = decay_step

        def get_lr(self) -> float:
            if ((self.iter_num + 1) % self.decay_step) == 0:
                lr = self.init_lr * self.gamma
                self.init_lr = lr

            return self.init_lr

        def step(self):
            """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

            self.iter_num += 1

        def __str__(self):
            return str(self.__dict__)

    return DecayLRAfter(optimizer_, init_lr_, decay_step_, gamma_)


def test_accuracy(model, loader, loss_fn, device):
    avg_acc = 0.
    avg_loss = 0.
    n = len(loader.dataset)
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            yhat = model(x)
            avg_loss += (loss_fn(yhat, y).item() / n)

            pred = yhat.max(1, keepdim=True)[1]
            avg_acc += (pred.eq(y.view_as(pred)).sum().item() / n)

    return avg_acc, avg_loss




def sample_batch(train_source, train_target, device):
    x_s, labels_s = next(train_source)
    x_t, _ = next(train_target)
    x_s = x_s.to(device)
    x_t = x_t.to(device)
    labels_s = labels_s.to(device)
    return x_s, x_t, labels_s

def tsne_plot(model, train_source, train_target, test_loader, divergence, device, seed):


    import seaborn as sns
    train_target = ForeverDataIterator(train_target)
    train_source = ForeverDataIterator(train_source)
    test_loader = ForeverDataIterator(test_loader)
    # sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # palette = sns.color_palette("bright", 10)
    tsne = TSNE(n_components=2, random_state=42)
    # colors = np.random.rand(num_classes)
    # n = len(loader.dataset)
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        # tsne = TSNE(n_components=2, init='pca', learning_rate='auto')

        # for (x, _), (xx, _) in zip(next(train_source), next(train_target)):
        x, _ = next(train_source)
        xx, _ = next(train_target)
        features_train = model(x.to(device)).cpu().detach().numpy()
        tsne_train = tsne.fit_transform(features_train)
        features_test = model(xx.to(device)).cpu().detach().numpy()
        tsne_test = tsne.fit_transform(features_test)

        plt.close()
        plt.figure(figsize=(8, 6))
        plt.xticks([])
        plt.yticks([])


        sns.scatterplot(x=tsne_train[:, 0], y=tsne_train[:, 1], color='blue', label='source', s=5, alpha=0.8)
        sns.scatterplot(x=tsne_test[:, 0], y=tsne_test[:, 1], color='orange', label='target', s=5, alpha=0.8)
        # sns.scatterplot(x=X1[:, 0], y=X1[:, 1], color='blue', label='Set 1', alpha=0.7)
        # sns.scatterplot(x=X2[:, 0], y=X2[:, 1], color='red', label='Set 2', alpha=0.7)

        # plt.title('t-SNE Visualization of Iris Data')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        plt.legend(loc='best')

        # plt.scatter(tsne_train[:, 0], tsne_train[:, 1], s=1,  color='blue', label='source')
        # plt.scatter(tsne_test[:, 0], tsne_test[:, 1], s=1, color='orange', label='target')
        # sns.scatterplot(tsne_train[:, 0], tsne_train[:, 1], hue=y, legend='full', palette=palette)
        # plt.legend()
        # plt.gca().set_facecolor('white')
        print('Saving figure...')
        plt.tight_layout()
        plt.savefig('Digits_{}_source_target_train_{}.pdf'.format(divergence, seed), bbox_inches='tight')
        plt.savefig('Digits_{}_source_target_train_{}.png'.format(divergence, seed), bbox_inches='tight')
        plt.show()
        # break

        # for (x, _), (xx, _) in zip(next(train_source), next(test_loader)):
        # for xx, _ in test_loader:
        xx, _ = next(test_loader)
        # features_train = model(x.to(device)).cpu().detach().numpy()
        # tsne_train = tsne.fit_transform(features_train)
        features_test = model(xx.to(device)).cpu().detach().numpy()
        tsne_test = tsne.fit_transform(features_test)

        plt.close()
        # plt.figure()
        plt.figure(figsize=(8, 6))
        plt.xticks([])
        plt.yticks([])

        sns.scatterplot(x=tsne_train[:, 0], y=tsne_train[:, 1], color='blue', label='source', s=5, alpha=0.8)
        sns.scatterplot(x=tsne_test[:, 0], y=tsne_test[:, 1], color='orange', label='target', s=5, alpha=0.8)

        # plt.scatter(tsne_train[:, 0], tsne_train[:, 1], s=5, color='blue', label='source')
        # plt.scatter(tsne_test[:, 0], tsne_test[:, 1], s=5, color='orange', label='target')
        # sns.scatterplot(tsne_train[:, 0], tsne_train[:, 1], hue=y, legend='full', palette=palette)
        # plt.legend()
        plt.legend(loc='best')
        # plt.gca().set_facecolor('white')
        print('Saving figure...')
        plt.tight_layout()
        plt.savefig('{}_source_target_test_{}.pdf'.format(divergence, seed), bbox_inches='tight')
        plt.savefig('{}_source_target_test_{}.png'.format(divergence, seed), bbox_inches='tight')
        plt.show()
            # break
#wd=0.0005   batch_size = 128,  wd=0.002

def main(divergence='pearson',
         root_src='/scratch/usps2mnist/', src=0, trg=1,  batch_size = 64,
         n_epochs=30, iter_per_epoch=3000, lr=0.01, wd=0.0005, reg_coef=0.5, lam=0, seed=2):
    seed_all(seed)

    # unzip datasets if this is first run.
    # if prepare_data_if_first_time() is False:
    #     return False
    if src == 0:
        decay_step = 5
        batch_size = 128
        wd = 0.002
    else:
        decay_step = 6
        batch_size = 64
        wd = 0.0005
        # decay_step = 5


    domain = ['mnist_train.txt', 'usps_train.txt']
    # train_target_domain = ['usps_train.txt', 'mnist_train.txt']
    test_domain = ['mnist_test.txt', 'usps_test.txt']

    source = domain[src]
    target = domain[trg]

    print(f"Divergence:{divergence} Source: {source} Target: {target} Seed: {seed}  "
          f"batch_size:{batch_size:.4f} | lr: {lr:.4f} | wd: {wd:.4f} | reg_coef: {reg_coef:.4f} | lam: {lam:.4f}")

    src_list = root_src + source
    trg_list = root_src + target
    test_list = root_src + test_domain[trg]
    # build the network.
    backbone, taskhead, num_classes = build_network()

    # build the dataloaders.
    train_source, train_target, test_loader = build_data_loaders(src_list, trg_list, test_list, batch_size)

    # define the loss function....
    taskloss = nn.CrossEntropyLoss()

    # fDAL ----
    train_target = ForeverDataIterator(train_target)
    train_source = ForeverDataIterator(train_source)
    learner = fDALLearner(backbone, taskhead, taskloss, divergence=divergence, reg_coef=reg_coef, lam=lam, n_classes=num_classes,
                          grl_params={"max_iters": 3000, "hi": 0.6, "auto_step": True}  # ignore for defaults.
                          )
    # end fDAL---

    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learner = learner.to(device)

    # define the optimizer.
    best_acc = -np.inf
    # Hyperparams and scheduler follows CDAN.
    opt = optim.SGD(learner.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    opt_schedule = scheduler(opt, lr, decay_step_=iter_per_epoch * int(decay_step), gamma_=0.5)

    print('Starting training...')
    for epochs in range(n_epochs):
        learner.train()
        for i in range(iter_per_epoch):
            opt_schedule.step()
            # batch data loading...
            x_s, x_t, labels_s = sample_batch(train_source, train_target, device)
            # forward and loss
            loss, others = learner((x_s, x_t), labels_s)
            # opt stuff
            opt.zero_grad()
            loss.backward()
            # avoid gradient issues if any early on training.
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10)
            opt.step()
            if i % 1500 == 0:
                # print(f"Epoch:{epochs} Iter:{i}. Task Loss:{others['taskloss']}")
                print(
                    f"Epoch:{epochs} Iter:{i}. Task Loss:{others['taskloss']:.4f} | \t f-DAL Src: {others['fdal_src']:.4f} \t f-DAL Trg: {others['fdal_trg']:.4f}"
                    f" | \t Src-Trg: {others['st_dis']:.4f} \t Trg-Src: {others['ts_dis']:.4f} \t f-Loss: {others['fdal_loss']:.4f}")

        test_acc, test_loss = test_accuracy(learner.get_reusable_model(True), test_loader, taskloss, device)


        if test_acc >= best_acc:
            # backbone_save, _ = learner.get_reusable_model(False)
            best_feature = deepcopy(learner.backbone)
        # print(f"Epoch:{epochs} Test Acc: {test_acc} Test Loss: {test_loss}")
        best_acc = max(test_acc, best_acc)
        print(f"Epoch:{epochs} Test Acc: {test_acc} Test Loss: {test_loss}. Best Acc: {best_acc}")

    # save the model.
    # torch.save(learner.get_reusable_model(True).state_dict(), './checkpoint.pt')
    print(f'best_acc:{best_acc}')
    print('t-SNE plotting...')
    train_source, train_target, test_loader = build_data_loaders(src_list, trg_list, test_list, 5000)

    model = best_feature
    tsne_plot(model, train_source, train_target, test_loader, divergence, device, seed)

    print('done.')


if __name__ == "__main__":
    fire.Fire(main)
