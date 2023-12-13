import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('out.csv')  # Replace 'your_file.csv' with your CSV file path

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
plt.title('Some plot over Batches')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()
