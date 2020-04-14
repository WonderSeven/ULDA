import numpy as np
import torch
from torchvision.transforms import functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from preprocess.tools import tensor2pil

def show_Images(imgs, max_cols, title=None, show=True,  filename=None, norm=True, figsize=(20, 20)):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu()

    gray = False
    if imgs.dim() == 3:
        length, H, W = imgs.shape
        gray = True
    elif imgs.dim() == 4:
        length, C, H, W = imgs.shape
    else:
        assert ValueError('Dims must in [3, 4]')

    # Plot graph
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=48)
    max_line = np.ceil(length / max_cols)
    for i in range(1, length+1):
        ax = fig.add_subplot(max_line, max_cols, i)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if norm:
            img = tensor2pil(imgs[i-1])
        else:
            img = F.to_pil_image(imgs[i-1])
        # clip image
        if gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    else:
         plt.savefig(filename, bbox_inches='tight')


def parse_reward_file(file_path):
    task_ids = []
    rewards = []
    with open(file_path, 'r') as textFile:
        context = textFile.readlines()
        for line in context:
            line = line.strip()
            task_idx = line.find('Task idx')
            if task_idx > 0:
                end_task_idx = line.index('||') - 1
                start_reward = line.index('reward:') + 7
                end_reward = line.index(',', start_reward)
                # print(line[task_idx+9:end_task_idx], line[start_reward:end_reward])
                task_ids.append(int(line[task_idx+9:end_task_idx]))
                rewards.append(float(line[start_reward:end_reward]))
    return task_ids, rewards


def parse_accuracy_file(file_path):
    epoch_ids = []
    val_accuracies = []
    test_accuracies = []
    with open(file_path, 'r') as textFile:
        context = textFile.readlines()
        for line in context:
            line = line.strip()
            epoch_idx = line.find('Epoch:')
            if epoch_idx > 0:
                end_epoch_idx = line.index('||') - 1
                epoch_idx += 6
                start_accuracy = line.find('Vac Accuracy:')
                if start_accuracy > 0:
                    start_accuracy += len('Vac Accuracy:')
                    end_accuracy = line.index(',', start_accuracy)
                    epoch_ids.append(int(line[epoch_idx: end_epoch_idx]))
                    val_accuracies.append(float(line[start_accuracy: end_accuracy]))
                else:
                    start_accuracy = line.index('Test Accuracy:')
                    start_accuracy += len('Test Accuracy:')
                    end_accuracy = line.index(',', start_accuracy)
                    test_accuracies.append(float(line[start_accuracy: end_accuracy]))
    return epoch_ids, val_accuracies, test_accuracies


def show_accuracies(epoch_ids, val_accuracies=None, test_accuracies=None, name=None, show=True):
    plt.figure()
    if val_accuracies is not None:
        plt.plot(epoch_ids, val_accuracies, color='blue', linewidth=1, label='Val')
    if test_accuracies is not None:
        plt.plot(epoch_ids, test_accuracies, color='red', linewidth=1, label='Test')

    plt.title(name, fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=15)
    if show:
        plt.show()
    else:
        plt.savefig('accuracies.png')


def show_reward(task_idx, rewards, name=None):
    plt.figure()
    # plt.plot(task_idx, rewards, linewidth=0.5)
    plt.scatter(task_idx, rewards, s=1, edgecolors='green')
    plt.title(name, fontsize=20)
    plt.xlabel('task id', fontsize=15)
    plt.ylabel('reward', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()


def show_by_task(task_idx, rewards, episode=10000, title=None, show=False):
    rewards = np.array(rewards, dtype=np.float32)
    length = len(rewards)

    new_len = int(length / episode)
    new_rewards = np.zeros(new_len, dtype=np.float32)
    for i in range(new_len):
        new_rewards[i] = np.mean(rewards[i*episode:(i+1)*episode])
        # print(i)
    plt.figure()
    plt.plot(new_rewards, linewidth=0.5)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig('average_reward.png')

def show_task_by_count(task_idx, rewards, episode=10000, title=None, show=True):
    rewards = np.array(rewards, dtype=np.float32)
    length = len(rewards)
    new_len = int(length / episode)
    reward_count = np.zeros(new_len, dtype=np.float32)
    for i in range(new_len):
        tmp = rewards[i*episode:(i+1)*episode]
        reward_count[i] = np.sum(tmp > 0)
        # reward_count[i] = np.sum(np.where(rewards[i*episode:(i+1)*episode]))
    plt.figure()
    plt.plot(reward_count, linewidth=0.5)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig('positive_reward.png')

def show_meta_feature(imgs, max_cols, title=None, show=True,  filename=None, figsize=(20, 20)):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.squeeze(0)
        imgs = imgs.cpu().detach().numpy()

    length, H, W = imgs.shape

    # Plot graph
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=48)
    max_line = np.ceil(length / max_cols)
    for i in range(1, length+1):
        ax = fig.add_subplot(max_line, max_cols, i)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        img = imgs[i-1]
        # clip image
        ax.imshow(img, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    else:
         plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    file_path = '/data/tiexin/data/Meta_augmentor/ProtoNet_ResNet_NFT_lr=0.001/DAG_ResNet256F_train_2019-11-07-14:33.txt'
    task_ids, rewards = parse_reward_file(file_path)
    epoch_ids, val_accuracies, test_accuracies = parse_accuracy_file(file_path)
    # show_reward(task_ids, rewards, 'replace')
    show_accuracies(epoch_ids, val_accuracies, test_accuracies, show=False)
    show_by_task(task_ids, rewards, 10000, 'replace', False)
    show_task_by_count(task_ids, rewards, 10000, 'replace', False)
