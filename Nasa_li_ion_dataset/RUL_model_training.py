from imports_libraries import *
from rul_model_defined import *
from torch.utils.data import DataLoader, TensorDataset

## Importing the datasets
from rul_data_processing import train_dataset,test_dataset,val_dataset


## We clip the maximum value of gradients to ensure stability
max_norm = 0.6


def train_model(model, train_dataset, val_dataset, num_epochs=10,
    learning_rate=0.001, batch_size=48, device='cpu', save_path='model.pth',
    best_save_path = 'model_best.pth'):

    criterion = nn.MSELoss() ## MSE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1) # We use step LR scheduler
    # Create DataLoader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 0 )

    # Create DataLoader for validation data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers = 0 )

    model = model.to(device)
    ## The losses below are initialized to later save the best loss
    val_loss_min = float('inf')
    train_loss_best = float('inf')

    train_loss_epoch = []
    valid_loss_epoch = []
    epoch_best = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        i = 0
        batch_count = 0
        train_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for current, voltage, temperature,scaled_time, cycle in train_tqdm:

            i+=1 ## Counting number of minibatches
            current, voltage, temperature, cycle = current.to(device), voltage.to(device), temperature.to(device), cycle.to(device)
            scaled_time = scaled_time.to(device)

            ## We scale the numbers by 100 so that our neural network based model is able to predict the
            cycle = cycle/100

            # Combine input features
            input_data = torch.cat((voltage, temperature,scaled_time), dim=2)


            output = model(input_data,current)

            #output = output.mean(dim=1).squeeze()
            cycle = cycle.mean(dim=1).squeeze()
            loss = criterion(output, cycle)
            loss.backward()
            batch_count+=1
            # optimizer.step()
            # We are using gradient accumulation for training
            if batch_count == 4:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0
                #break


            train_loss += loss.item()
            train_tqdm.set_postfix({'Batch Loss': f'{loss.item():.4f}', 'Avg Loss': f'{train_loss/i}'})

            # if(i==30):
            #     break
        # torch.cuda.empty_cache()
        # Validation
        model.eval()
        val_loss = 0.0
        j=0
        val_tqdm = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for current, voltage, temperature, scaled_time,cycle in val_tqdm:
                j+=1 ## Counting minibatches
                current, voltage, temperature, cycle = current.to(device), voltage.to(device), temperature.to(device), cycle.to(device)
                cycle = cycle/100
                # Combine input features
                scaled_time = scaled_time.to(device)
                input_data = torch.cat((voltage, temperature,scaled_time), dim=2)

                val_output = model(input_data,current)

                #val_output = val_output.mean(dim=1).squeeze()
                cycle = cycle.mean(dim=1).squeeze()

                val_loss += criterion(val_output, cycle).item()
                val_tqdm.set_postfix({'Batch Loss': f'{val_loss:.4f}', 'Avg Loss': f'{val_loss/j}'})
                #break
                # if(j==30):
                #     break

        train_loss /= i
        val_loss /= j

        train_loss_epoch.append(train_loss)
        valid_loss_epoch.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # We save the best model on the validation set
        torch.save(model.state_dict(), save_path)
        if val_loss < val_loss_min:
            torch.save(model.state_dict(), best_save_path)
            val_loss_min = val_loss
            train_loss_best = train_loss
            print(f'Model saved to {best_save_path}')
            epoch_best = epoch
        print(f'Best Val loss is {val_loss_min:.4f} and corresponding train loss is {train_loss_best:.4f}')
        print(f'Best epoch was {epoch_best}')
        scheduler.step(val_loss)

        torch.cuda.empty_cache() # We clear the cache to optimize memory usage
    return model, train_loss_epoch, valid_loss_epoch
    # Save the model


# Example usage
batch_size = 16
sequence_length = 150

model = v10_rescaling_adaptive_TransformerModel3Encoder(input_dim=3+1, hidden_dim=16, output_dim=1, nhead=4, num_layers=1, dropout=0,
                                                        max_len=sequence_length)

# Load pre-trained model if needed

# saved_model_path = 'RUL_Model/_minutes.pth'
# error_above_model = 55.27 ## This is the one sided error for the dummy model
# torch.save(model.state_dict(), saved_model_path)
# model_state_dict = torch.load(saved_model_path)
# model.load_state_dict(model_state_dict)

best_saved_model_path = 'RUL_Model/150seconds.pth'

#device = torch.device("mps")# for mac, note that mps trained model cannot be used with cuda
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_model(model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            num_epochs=15,
            learning_rate=1e-4,
            save_path='RUL_Model/temp.pth',
            best_save_path = best_saved_model_path)
