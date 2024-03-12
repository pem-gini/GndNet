import argparse
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--logs', default='training.log', type=str, metavar='PATH', help='path to logfile')
parser.add_argument('--date', default='2024-03-03 23:33:46', type=str, metavar='PATH', help='path to logfile')
parser.add_argument('-min', '--min', dest='show_min', action='store_true', help='Show minimum')
parser.add_argument('-max', '--max', dest='show_max', action='store_true', help='Show maximum')
args = parser.parse_args()

def parse_log_file(log_file_path, start_datetime):
    val_avg_losses = []
    val_min_losses = []
    val_max_losses = []
    train_avg_losses = []
    train_min_losses = []
    train_max_losses = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        train_epoch = -1
        val_epoch = -1
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch: \[(\d+)\]\[\d+\/\d+\].*Loss (\d+\.\d+) \((\d+\.\d+)\)$', line)
            if match:
                timestamp_str, epoch, loss, avg_loss = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                if timestamp >= start_datetime:
                    loss = float(loss)
                    avg_loss = float(avg_loss)
                    if epoch == train_epoch:
                        train_avg_losses[-1] = avg_loss
                        train_min_losses[-1] = min(train_min_losses[-1], loss)
                        train_max_losses[-1] = max(train_max_losses[-1], loss)
                    else:
                        train_avg_losses.append(avg_loss)
                        train_min_losses.append(loss)
                        train_max_losses.append(loss)
                        train_epoch = epoch
            else:
                match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Test: \[\d+\/\d+\].*Loss (\d+\.\d+) \((\d+\.\d+)\)$', line)
                if match:
                    timestamp_str, loss, avg_loss = match.groups()
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if timestamp >= start_datetime:
                        loss = float(loss)
                        avg_loss = float(avg_loss)
                        if val_epoch == train_epoch:
                            val_avg_losses[-1] = avg_loss
                            val_min_losses[-1] = min(val_min_losses[-1], loss)
                            val_max_losses[-1] = max(val_max_losses[-1], loss)
                        else:
                            val_avg_losses.append(avg_loss)
                            val_min_losses.append(loss)
                            val_max_losses.append(loss)
                            val_epoch = train_epoch
    return (
        ['val. avg.', 'val. min', 'val. max', 'train avg.', 'train min', 'train max'],
        [val_avg_losses, val_min_losses, val_max_losses, train_avg_losses, train_min_losses, train_max_losses]
    )

def plot_epoch_losses(labels, epoch_losses):
    for label, losses in zip(labels, epoch_losses):
        if 'min' in label and not args.show_min:
            continue
        if 'max' in label and not args.show_max:
            continue
        plt.plot(losses, marker='o', label=label)
    
    plt.legend()
    plt.title('Average Loss of Last "Test" of Each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_file_path = args.logs  # Replace with the actual path to your log file
    start_datetime =  datetime.strptime(args.date, '%Y-%m-%d %H:%M:%S')  # Define the start date and time

    labels, epoch_losses = parse_log_file(log_file_path, start_datetime)
    print(epoch_losses)
    plot_epoch_losses(labels, epoch_losses)