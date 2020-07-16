import os
from shutil import copy
from pathlib import Path


def get_arm():
    with open('arm.txt', 'r') as f:
        lines = f.read().split()
    dirs = ['malware', 'benign']
    for line in lines:
        for dir in dirs:
            path = f'opcode/{dir}/{line}.txt'
            if Path(path).exists():
                copy(path, Path(f'opcode_arm/{dir}'))


def get_iotpot():
    with open('iotpot.txt', 'r') as f:
        lines = f.read().split()
    for line in lines:
        path = f'opcode/malware/{line}.txt'
        if Path(path).exists():
            copy(path, Path(f'opcode_iotpot/malware/'))


def main():
    # get_arm()
    get_iotpot()


if __name__ == "__main__":
    main()