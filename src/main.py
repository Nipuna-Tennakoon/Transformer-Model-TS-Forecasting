import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt



input_window = 10 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
batch_size = 250

# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("your device is: ", device)

# read facebook dataset
df = pd.read_csv("data/facebook.csv")
df.head()



# logarithmic normalization
open = df["open"].fillna(method="ffill")
open = np.array(open)
open_logreturn = np.diff(np.log(open))
open_csum_logreturn = open_logreturn.cumsum()


close = df["close"].fillna(method="ffill")
close = np.array(close)
close_logreturn = np.diff(np.log(close))
close_csum_logreturn = close_logreturn.cumsum()


average = df["average"].fillna(method="ffill")
average = np.array(average)
average_logreturn = np.diff(np.log(average))
average_csum_logreturn = average_logreturn.cumsum()



fig = plt.figure(figsize=(10, 15))

plt.subplot(3, 2, 1)
plt.plot(open, color="blue")
plt.title("Raw Open df")
plt.xlabel("Time Steps")
plt.ylabel("Open Price")

plt.subplot(3, 2, 2)
plt.plot(open_csum_logreturn, color="red")
plt.title("Normalized Open df")
plt.xlabel("Time Steps")
plt.ylabel("Open Price")

plt.subplot(3, 2, 3)
plt.plot(close, color="orange")
plt.title("Raw Close df")
plt.xlabel("Time Steps")
plt.ylabel("Close Price")

plt.subplot(3, 2, 4)
plt.plot(close_csum_logreturn, color="green")
plt.title("Normalized Close df")
plt.xlabel("Time Steps")
plt.ylabel("Close Price")

plt.subplot(3, 2, 5)
plt.plot(average, color="magenta")
plt.title("Raw average df")
plt.xlabel("Time Steps")
plt.ylabel("average Price")

plt.subplot(3, 2, 6)
plt.plot(average_csum_logreturn, color="black")
plt.title("Normalized average df")
plt.xlabel("Time Steps")
plt.ylabel("average Price")






class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    



class transformer(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(transformer, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if(self.src_mask is None or self.src_mask.size(0) != len(src)):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask
    


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)



def get_data(data, split):

    series = data

    split = round(split * len(series))
    train_data = series[:split]
    test_data = series[split:]

    train_data = train_data.cumsum()

    # Training data augmentation, increase amplitude for the model to better generalize.(Scaling by 2 is aribitrary)
    # Similar to image transformation to allow model to train on wider data sets
    train_data = 2 * train_data

    test_data = test_data.cumsum()

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target



def train(train_data):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    # Initialize metrics accumulators
    mse_sum = 0.0
    mae_sum = 0.0
    batch_count = 0

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            mse_batch = loss.item()
            rmse_batch = np.sqrt(mse_batch)
            mae_batch = torch.mean(torch.abs(output - targets)).item()

            mse_sum += mse_batch
            mae_sum += mae_batch
            batch_count += 1

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if(batch % log_interval == 0 and batch > 0):
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | "
                  "lr {:02.10f} | {:5.2f} ms | "
                  "loss {:5.7f}".format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

    # Calculate average metrics for the epoch
    avg_mse = mse_sum / batch_count
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = mae_sum / batch_count

    # Evaluate validation metrics
    val_mse, val_rmse, val_mae = evaluate(model, val_data)

    # Store metrics
    train_mse_history.append(avg_mse)
    train_rmse_history.append(avg_rmse)
    train_mae_history.append(avg_mae)
    val_mse_history.append(val_mse)
    val_rmse_history.append(val_rmse)
    val_mae_history.append(val_mae)

    # wandb.log({
    #     "MSE training loss": avg_mse,
    #     "RMSE training loss": avg_rmse,
    #     "MAE training loss": avg_mae,
    #     "MSE validation loss": val_mse,
    #     "RMSE validation loss": val_rmse,
    #     "MAE validation loss": val_mae
    # })

    return avg_mse, avg_rmse, avg_mae, val_mse, val_rmse, val_mae



def evaluate(eval_model, data_source):
    eval_model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total_samples = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)

            # Calculate MSE
            mse_batch = criterion(output, targets).cpu().item()
            batch_size = len(data[0])

            # Calculate MAE
            mae_batch = torch.mean(torch.abs(output - targets)).cpu().item()

            mse_sum += mse_batch * batch_size
            mae_sum += mae_batch * batch_size
            total_samples += batch_size

    # Calculate average metrics
    avg_mse = mse_sum / total_samples
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = mae_sum / total_samples

    return avg_mse, avg_rmse, avg_mae



def model_forecast(model, seqence):
    model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    seq = np.pad(seqence, (0, 3), mode="constant", constant_values=(0, 0))
    seq = create_inout_sequences(seq, input_window)
    seq = seq[:-output_window].to(device)

    seq, _ = get_batch(seq, 0, 1)
    with torch.no_grad():
        for i in range(0, output_window):
            output = model(seq[-output_window:])
            seq = torch.cat((seq, output[-1:]))

    seq = seq.cpu().view(-1).numpy()

    return seq



def forecast_seq(model, sequences):

    start_timer = time.time()
    model.eval()
    forecast_seq = torch.Tensor(0)
    actual = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(sequences) - 1):
            data, target = get_batch(sequences, i, 1)
            output = model(data)
            forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
            actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
    timed = time.time()-start_timer
    print(f"{timed} sec")

    return forecast_seq, actual





train_data, val_data = get_data(close_logreturn, 0.6) # 60% train, 40% test split
model = transformer().to(device)




criterion = nn.MSELoss()
lr = 0.00005
epochs = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Initialize metric history lists to track training and validation metrics
train_mse_history = []
train_rmse_history = []
train_mae_history = []
val_mse_history = []
val_rmse_history = []
val_mae_history = []




for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    # Metrics are already calculated and stored in train() function
    if(epoch % epochs == 0): # Valid model after last training epoch
        print("-" * 80)
        print("| end of epoch {:3d} | time: {:5.2f}s".format(epoch, (time.time() - epoch_start_time)))
        if len(train_mse_history) > 0:
            print("| Train - MSE: {:5.7f} | RMSE: {:5.7f} | MAE: {:5.7f}".format(
                train_mse_history[-1], train_rmse_history[-1], train_mae_history[-1]))
            print("| Valid - MSE: {:5.7f} | RMSE: {:5.7f} | MAE: {:5.7f}".format(
                val_mse_history[-1], val_rmse_history[-1], val_mae_history[-1]))
        print("-" * 80)

    else:
        print("-" * 80)
        print("| end of epoch {:3d} | time: {:5.2f}s".format(epoch, (time.time() - epoch_start_time)))
        if len(train_mse_history) > 0:
            print("| Train - MSE: {:5.7f} | RMSE: {:5.7f} | MAE: {:5.7f}".format(
                train_mse_history[-1], train_rmse_history[-1], train_mae_history[-1]))
            print("| Valid - MSE: {:5.7f} | RMSE: {:5.7f} | MAE: {:5.7f}".format(
                val_mse_history[-1], val_rmse_history[-1], val_mae_history[-1]))
        print("-" * 80)

    scheduler.step()



test_result, truth = forecast_seq(model, val_data)



plt.plot(truth, color="red", alpha=0.7)
plt.plot(test_result, color="blue", linestyle="dashed", linewidth=0.7)
plt.title("Actual vs Forecast")
plt.legend(["Actual", "Forecast"])
plt.xlabel("Time Steps")
plt.show()


# Plot training and validation loss metrics
if len(train_mse_history) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_range = range(1, len(train_mse_history) + 1)

    # Plot MSE
    axes[0].plot(epochs_range, train_mse_history, label='Train MSE', color='blue', linewidth=2)
    axes[0].plot(epochs_range, val_mse_history, label='Validation MSE', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot RMSE
    axes[1].plot(epochs_range, train_rmse_history, label='Train RMSE', color='blue', linewidth=2)
    axes[1].plot(epochs_range, val_rmse_history, label='Validation RMSE', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('RMSE Loss', fontsize=12)
    axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot MAE
    axes[2].plot(epochs_range, train_mae_history, label='Train MAE', color='blue', linewidth=2)
    axes[2].plot(epochs_range, val_mae_history, label='Validation MAE', color='red', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('MAE Loss', fontsize=12)
    axes[2].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final metrics
    print("\n" + "="*80)
    print("FINAL METRICS SUMMARY")
    print("="*80)
    print(f"Training   - MSE: {train_mse_history[-1]:.7f}, RMSE: {train_rmse_history[-1]:.7f}, MAE: {train_mae_history[-1]:.7f}")
    print(f"Validation - MSE: {val_mse_history[-1]:.7f}, RMSE: {val_rmse_history[-1]:.7f}, MAE: {val_mae_history[-1]:.7f}")
    print("="*80)
else:
    print("No training metrics available. Please run the training loop first.")




r = np.random.randint(100000, 160000)
test_forecast = model_forecast(model, close_csum_logreturn[r: r+10]) # random 10 sequence length

print(f"forecast sequence: {test_forecast}\n")
print(f"Actual sequence: {close_csum_logreturn[r: r+11]}")



torch.save(model.state_dict(), "model/time_forecasting_transformer.pth")





model_val = transformer()
model_val.load_state_dict(torch.load("model/time_forecasting_transformer.pth"))
model_val.to(device)



df2 = pd.read_csv("data/boeing.csv")
close2 = df2["close"].fillna(method = "ffill")
close2 = np.array(close2)
logreturn2 = np.diff(np.log(close2))


train_data2, val_data2 = get_data(logreturn2, 0.6)
test2_eval_mse, test2_eval_rmse, test2_eval_mae = evaluate(model_val, val_data2)
print(f"boeing test - MSE: {test2_eval_mse:.5f}, RMSE: {test2_eval_rmse:.5f}, MAE: {test2_eval_mae:.5f}")



test_result2, truth2 = forecast_seq(model_val, val_data2)

plt.plot(truth2, color="red", alpha=0.7)
plt.plot(test_result2, color="blue", linestyle="dashed", linewidth=0.7)
plt.title("Actual vs Forecast")
plt.legend(["Actual", "Forecast"])
plt.xlabel("Time Steps")
plt.show()


df3 = pd.read_csv("data/jp_morgan.csv")
close3 = df3["close"].fillna(method = "ffill")
close3 = np.array(close3)
logreturn3 = np.diff(np.log(close3))


train_data3, val_data3 = get_data(logreturn3, 0.6)
test3_eval_mse, test3_eval_rmse, test3_eval_mae = evaluate(model_val, val_data3)
print(f'jp morgan test - MSE: {test3_eval_mse:.5f}, RMSE: {test3_eval_rmse:.5f}, MAE: {test3_eval_mae:.5f}')


test_result3, truth3 = forecast_seq(model_val, val_data3)

plt.plot(truth3, color="red", alpha=0.7)
plt.plot(test_result3, color="blue", linestyle="dashed", linewidth=0.7)
plt.title("Actual vs Forecast")
plt.legend(["Actual", "Forecast"])
plt.xlabel("Time Steps")
plt.show()

test_result3, truth3 = forecast_seq(model_val, val_data3)

plt.plot(truth3, color="red", alpha=0.7)
plt.plot(test_result3, color="blue", linestyle="dashed", linewidth=0.7)
plt.title("Actual vs Forecast")
plt.legend(["Actual", "Forecast"])
plt.xlabel("Time Steps")
plt.show()