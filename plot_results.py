import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

def collect_experiment_dicts(target_dir, test_flag=False):
    experiment_dicts = dict()
    for subdir, dir, files in os.walk(target_dir):
        for file in files:
            filepath = None
            if not test_flag:
                if file == 'summary.csv':
                    filepath = os.path.join(subdir, file)
            
            elif test_flag:
                if file == 'test_summary.csv':
                    filepath = os.path.join(subdir, file)
            
            if filepath is not None:
                
                with open(filepath, 'r') as read_file:
                    lines = read_file.readlines()
                    
                current_experiment_dict = {key: [] for key in lines[0].replace('\n', '').split(',')}
                idx_to_key = {idx: key for idx, key in enumerate(lines[0].replace('\n', '').split(','))}
                
                for line in lines[1:]:
                    for idx, value in enumerate(line.replace('\n', '').split(',')):
                        current_experiment_dict[idx_to_key[idx]].append(float(value))
                
                experiment_dicts[subdir.split('/')[-2]] = current_experiment_dict
                
    return experiment_dicts

def plot_result_graphs(plot_name, stats, name, notebook=True):

    
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['train_loss', 'val_loss']:
        item = stats[name][k]
        ax_1.plot(np.arange(0, len(item)), 
                    item, label='{}_{}'.format(name, k))
            
    ax_1.legend(loc=0)
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['train_acc', 'val_acc']:
        item = stats[name][k]
        print(item)
        ax_2.plot(np.arange(0, len(item)), 
                    item, label='{}_{}'.format(name, k))
            
    ax_2.legend(loc=0)
    ax_2.set_ylabel('Accuracy')
    ax_2.set_xlabel('Epoch number')
    
    fig_1.savefig('{}_loss_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    fig_2.savefig('{}_accuracy_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

experiment_dir = 'VGG_38_batch_norm'
result_dict_BN = collect_experiment_dicts(target_dir=experiment_dir)
for key, value in result_dict_BN.items():
    print(key, list(value.keys()))
print(result_dict_BN['VGG_38_batch_norm']['train_loss'])
plot_result_graphs('figure4_BN', result_dict_BN, name='VGG_38_batch_norm')
"""
experiment_dir = 'VGG_38_BNRC'
result_dict_BN = collect_experiment_dicts(target_dir=experiment_dir)
plot_result_graphs('figure4_BNRC', result_dict_BN, name='VGG_38_BNRC')
"""