import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms
import math

class FIFTY_DATASET(Dataset):
    def __init__(self, root, dim, subset='train', block_size='512', scenario='1', transform=None, target_transform=None):
        super(FIFTY_DATASET, self).__init__()
        self.root = root
        self.dim = dim
        self.transform = transform
        self.target_transform = target_transform
        self.scenario = scenario
        self.block_size = block_size
        self.train = subset
        self.data, self.targets, self.filename, self.labels = self.load(block_size, scenario, subset)
        self.targets = self.targets.astype(np.int64)
        self.max_targets = self.targets.max() + 1
        print(f"已经加载{subset}数据，数据维度为{self.data.shape}，dim为{dim}")


    def data_add_bit(self, data, ngram):
        data_arr = []
        for shift_bit in range(8):
            tmp = self.getshift(shift_bit, data).astype(np.uint8)
            data_arr.append(tmp)

        data = np.array(data_arr)
        re = data
        for i in range(1, ngram):
            tmp = np.roll(data, -i)
            re = np.concatenate((re, tmp), axis=0)

        data = re
        data = data[:, :-ngram]
        data = np.expand_dims(data, 2)
        data = data.astype(np.uint8)
        return data

    def __getitem__(self, index):
        data, target, raw_data = self.data[index], self.targets[index], self.data[index]
        data_ngram = {}
        for d in self.dim:

            if self.block_size == '512':
                data = self.data_add_bit(raw_data, d)
                t = transforms.Compose([
                    transforms.ToPILImage(mode='L'),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
                data = self.transform(data) if self.transform is not None else t(data)
            else:
                L = [512*i for i in range(9)]
                t = transforms.Compose([
                    transforms.ToPILImage(mode='L'),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
                data_bit = []
                for i in range(8):
                    tmp = data[L[i]:L[i+1]]
                    tmp_bit = self.data_add_bit(tmp, d)
                    tmp_bit = self.transform(tmp_bit) if self.transform is not None else t(tmp_bit)
                    data_bit.append(tmp_bit)
                data_bit = torch.cat(data_bit, axis=0)
                data = data_bit

            data_ngram[str(d)] = data
        target = self.target_transform(target) if self.target_transform is not None else target

        return data_ngram, target, raw_data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def getshift(self, shift, arr):
        arr_s = np.roll(arr, -1)
        arr_s[len(arr_s) - 1] = 0

        arr = arr << shift & 255
        arr_s = arr_s >> (8-shift) & 255

        arr = arr + arr_s
        return arr

    def getlabels(self):
        return self.labels

    def getfilename(self, index):
        return self.filename[index]

    def load(self, block_size='512', scenario='1', subset='train'):
        if block_size not in ['512', '4k']:
            raise ValueError('Invalid block size!')
        if subset not in ['train', 'val', 'test']:
            raise ValueError('Invalid subset!')

        data_dir = os.path.join(self.root, '{:s}_{:s}'.format(block_size, scenario))
        print(data_dir)
        if subset=='train':
            print(os.path.join(data_dir, '{}.npz'.format('train')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('train')))
        elif subset=='val':
            print(os.path.join(data_dir, '{}.npz'.format('val')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('val')))
        else:
            print(os.path.join(data_dir, '{}.npz'.format('test')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('test')))

        data_x, data_y, data_z = data['x'], data['y'], []

        labels = {
          "1": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
               "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar",
               "msi", "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
               "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt",
               "tex", "json", "html", "xml", "log", "csv", "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap",
               "ttf", "dwg", "sqlite"],
         "2": ["bmp", "raw", "vec", "vid", "arc", "exe", "off", "pub", "hr", "aud", "oth"],
         "3": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
               "bmp", "gif", "png", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "oth"],
         "4": ["jpg", "raw", "vid", "5_bmps", "oth"],
         "5": ["jpg", "oth"],
         "6": ["jpg", "oth"],
         "tags": ["bitmap", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "bitmap",
                  "bitmap", "bitmap", "bitmap", "bitmap", "vector", "vector", "vector", "video", "video", "video",
                  "video", "video", "video", "video", "archive", "archive", "archive", "archive", "archive", "archive",
                  "archive", "archive", "archive", "archive", "archive", "archive", "archive", "executable",
                  "executable", "executable", "executable", "office", "office", "office", "office", "office", "office",
                  "office", "published", "published", "published", "published", "human-readable", "human-readable",
                  "human-readable", "human-readable", "human-readable", "human-readable", "human-readable",
                  "human-readable", "human-readable", "audio", "audio", "audio", "audio", "audio", "audio", "audio",
                  "misc", "misc", "misc", "misc"]
         }

        return data_x, data_y, data_z, labels[scenario]


def prepare_datasets(root, scenario=1, block_size=512, batch_size=512, dim=[8, 16, 32, 64]):
    train_transform = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing()
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = FIFTY_DATASET(root=root, dim=dim, subset='train', block_size='512', scenario=scenario, transform=train_transform)
    val_dataset = FIFTY_DATASET(root=root, dim=dim, subset='val', block_size='512', scenario=scenario, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    return train_loader, val_loader


class CustomLRScheduler:
    def __init__(self, optimizer, heating_epochs, cooling_epochs, top_lr, bottom_lr, total_epochs):
        self.optimizer = optimizer
        self.heating_epochs = heating_epochs
        self.cooling_epochs = cooling_epochs  # 新增
        self.top_lr = top_lr
        self.bottom_lr = bottom_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.heating_epochs:
            # 线性增长阶段：从0到TOP_LR
            lr = self.top_lr * (self.current_epoch + 1) / self.heating_epochs
        elif self.current_epoch > self.cooling_epochs:
            # 余弦退火阶段：从TOP_LR到BOTTOM_LR
            #progress = (self.current_epoch - self.heating_epochs) / (self.total_epochs - self.heating_epochs)
            progress = (self.current_epoch - self.cooling_epochs) / (self.total_epochs - self.cooling_epochs)  #修改
            lr = self.bottom_lr + (self.top_lr - self.bottom_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # 持续加温阶段：保持不变
            lr = self.top_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
    
    def state_dict(self):
        return {
            'heating_epochs': self.heating_epochs,
            'cooling_epochs': self.cooling_epochs,  # 新增
            'top_lr': self.top_lr,
            'bottom_lr': self.bottom_lr,
            'total_epochs': self.total_epochs,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict):
        self.heating_epochs = state_dict['heating_epochs']
        self.cooling_epochs = state_dict['cooling_epochs']  # 新增
        self.top_lr = state_dict['top_lr']
        self.bottom_lr = state_dict['bottom_lr']
        self.total_epochs = state_dict['total_epochs']
        self.current_epoch = state_dict['current_epoch']


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_loader, val_loader = prepare_datasets(root='/root/', scenario='2', block_size=512, batch_size=512, dim=[1,2,4,8])
