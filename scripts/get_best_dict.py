import sys
from glob import glob
import torch
import os


def get_best_dict(BASE_PATH):

    best_dict = {}
    for i in range(5):
        best_dict[i] = torch.load(glob(f'{BASE_PATH}/{i}/*.pth')[0])['model_state_dict']

    save_dir = f'{BASE_PATH}/datasets'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(best_dict, f'{save_dir}/best_dict.pth')

if __name__ == '__main__':
    args = sys.argv
    base_path = args[1]
    get_best_dict(base_path)
