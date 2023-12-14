#{'optim_args': {'learning_rate': 0.5, 'weight_decay': 0.0001, 'momentum': 0.9,
# 'batch_size': 512, 'use_sgd': True, 'ff_lr_scaling': None}, 'data_args': {'k':
#                                                                           5,
#                                                                           'seq_length':
#                                                                           256,
#                                                                           'show_latents':
#                                                                           False,
#                                                                           'fixed_special_toks':
#                                                                           True,
#                                                                           'special_toks_offset':
#                                                                           0,
#                                                                           'output_counter':
#                                                                           True,
#                                                                           'no_repeat':
#                                                                           False},
# 'model_args': {'vocab_size': 65, 'dim': 128, 'max_length': 256, 'final_ffn':
#                False, 'first_ffn': False, 'linear_final_ffn': True,
#                'linear_first_ffn': True, 'freeze_embeddings': True,
#                'freeze_output': True, 'tie_output': False, 'use_rope': False,
#                'sqrtd_embeddings': False, 'no_sqrtd': False, 'sin_cos': False},
# 'max_iters': 1000, 'eval_delta': 5, 'log_norms': False, 'log_probes': True,
# 'freeze_until': '', 'loss_head_only': True, 'bigram_outs_train': False,
# 'bigram_outs_test': False, 'num_data_workers': 1, 'seed': 42, 'save_dir': None,
# 'root_dir': '', 'train_stepped': False}
# Re-importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import json
import ast
import os

def plot_csv_data(file_path, out_name = None):
    """
    Read a CSV file, skip the first line for plotting data, and use the first line as the plot title.
    :param file_path: Path to the CSV file
    """
    # Read the first line for the title
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    first_line_dict = ast.literal_eval(first_line) #json.loads(first_line)
    title = ""
    #title += "lr: "+str(first_line_dict["optim_args"]["learning_rate"])
    title += "loss_head_only: "+str(first_line_dict["loss_head_only"])
    title += ", final_ffn: "+str(first_line_dict["model_args"]["final_ffn"])
    title += ", fixed_special_toks: "+str(first_line_dict["data_args"]["fixed_special_toks"])
    title += ", train_stepped: "+str(first_line_dict["train_stepped"])

    # Read the rest of the file into a pandas DataFrame, skipping the first row
    df = pd.read_csv(file_path, skiprows=1)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['batch_num'], df['loss_item'], marker='o', label='loss_item')
    plt.plot(df['batch_num'], df['loss_bigram'], marker='o', label='loss_bigram')
    plt.plot(df['batch_num'], df['loss_head'], marker='o', label='loss_head')
    plt.plot(df['batch_num'], df['wk1_acc'], marker='o', label='wk1_acc')
    plt.plot(df['batch_num'], df['wo1_acc'], marker='o', label='wo1_acc')
    plt.plot(df['batch_num'], df['wk0_acc'], marker='o', label='wk0_acc')
    plt.plot(df['batch_num'], df['wk0_64_acc'], marker='o', label='wk0_64_acc')
    plt.plot(df['batch_num'], df['ff1_loss'], marker='o', label='ff1_loss')
    plt.xlabel('Batch Number')
    plt.ylabel('')
    plt.title(title)  # Using the first line as the title
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if out_name == None:
        plt.show()
    else:
        plt.savefig(out_name)

# You need to run this function with the path to your CSV file like this:
# plot_csv_data('data2/out_combined2.csv')

def process_all_csv_in_folder(folder_path):
    """
    Process all CSV files in the given folder.
    :param folder_path: Path to the folder containing CSV files.
    """
    for i,filename in enumerate(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            plot_csv_data(file_path, "out"+str(i)+".png")
            print(f"Processed and plotted data from {filename}")

process_all_csv_in_folder('./data2')
