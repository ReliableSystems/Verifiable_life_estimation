from imports_libraries import *
from simulator_model_defined import *
from torch.utils.data import DataLoader, TensorDataset
from simulator_data_processing import sim_train_dataset,sim_test_dataset,sim_val_dataset
import mlflow
import mlflow.pytorch
from utils import calculate_dimensional_mape


train_1_mape= [ ]
train_2_mape= [ ]
train_3_mape= [ ]

valid_1_mape= []
valid_2_mape= []
valid_3_mape= []

test_1_mape = []
test_2_mape = []

max_norm = 0.6 # gradient clipping for stabilization
def train_model(model,
                train_dataset,test_dataset,val_dataset,
                num_epochs=10, learning_rate=0.001, batch_size=48,
                device='cpu',
                  sequence_length=10, save_path='model.pth',temp_storage='temp.pth'):
    mlflow.start_run()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    train_dataset=train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset=val_dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset=test_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    val_loss_min =np.inf
    train_loss_best =np.inf
    train_loss_epoch = []
    valid_loss_epoch = []
    for epoch in range(num_epochs):
        print(epoch)
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            state, action, next_state = batch
            state, action, next_state =state.to(device), action.to(device), next_state.to(device)

            optimizer.zero_grad()
            output = model(state, action)

            #loss = criterion(output.clone().detach(), next_state.clone().detach())

            loss_normalized = (criterion(output[:,:,0], next_state[:,:,0]))+(criterion(output[:,:,1], next_state[:,:,1])/100)
            loss= loss_normalized

            loss_normalized.backward()

            mape1,mape2 = calculate_dimensional_mape(target = next_state,
                                                           prediction=output)
            mlflow.log_metric("train_voltage_mape", mape1)
            mlflow.log_metric("train_temperature_mape", mape2)

            mlflow.log_metric("Train_Normalized_loss", loss_normalized.item())


            train_1_mape.append(mape1)
            train_2_mape.append(mape2)
            #break


            optimizer.step()
            train_loss += loss.item()
        average_train_voltage_mape = np.mean(train_1_mape)
        average_train_temperature_mape = np.mean(train_2_mape)
        mlflow.log_metric("average_train_voltage_mape", float(average_train_voltage_mape))
        mlflow.log_metric("average_train_temperature_mape", float(average_train_temperature_mape))
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                val_state, val_action, val_next_state = batch
                val_state, val_action, val_next_state =val_state.to(device), val_action.to(device), val_next_state.to(device)
                val_output = model(val_state, val_action)
                # val_loss += criterion(val_output, val_next_state).item()
                mape1,mape2 = calculate_dimensional_mape(target = val_next_state,
                                                           prediction=val_output)
                valid_loss_normalized = (criterion(val_output[:,:,0], val_next_state[:,:,0]))+(criterion(val_output[:,:,1], val_next_state[:,:,1])/100)
                val_loss = valid_loss_normalized
                valid_1_mape.append(mape1)
                valid_2_mape.append(mape2)

                mlflow.log_metric("valid_voltage_mape", mape1)
                mlflow.log_metric("valid_temperature_mape", mape2)
                mlflow.log_metric("valid_loss_normalized", valid_loss_normalized)
                #break

            average_valid_voltage_mape = np.mean(valid_1_mape)
            average_valid_temperature_mape = np.mean(valid_2_mape)
            mlflow.log_metric("average_valid_voltage_mape", float(average_valid_voltage_mape))
            mlflow.log_metric("average_valid_temperature_mape", float(average_valid_temperature_mape))



        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_loss_epoch.append(train_loss)
        valid_loss_epoch.append(val_loss)
        # np.save('train_mape.npy', train_loss_epoch)
        # np.save('valid_1_mape.npy', valid_loss_epoch)
        torch.save(model.state_dict(), temp_storage) # Storing the model temporarily incase the training run stops in between
        if(val_loss<val_loss_min):
            torch.save(model.state_dict(), save_path)
            val_loss_min = val_loss
            train_loss_best = train_loss
            print(f'Model saved to {save_path}')
            print(f'Best Val loss is {val_loss_min} and corresponding train loss is {train_loss_best}')

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            for batch in tqdm(test_loader):
                test_state, test_action, test_next_state = batch
                test_state, test_action, test_next_state =test_state.to(device), test_action.to(device), test_next_state.to(device)
                test_output = model(test_state, test_action)
                val_loss += criterion(test_output, test_next_state).item()
                mape1,mape2 = calculate_dimensional_mape(target = test_next_state,
                                                            prediction=test_output)
                test_loss_normalized = (criterion(test_output[:,:,0], test_next_state[:,:,0]))+(criterion(test_output[:,:,1], test_next_state[:,:,1])/100)

                test_1_mape.append(mape1)
                test_2_mape.append(mape2)

                mlflow.log_metric("test_voltage_mape", mape1)
                mlflow.log_metric("test_temperature_mape", mape2)
                mlflow.log_metric("test_loss_normalized", test_loss_normalized)
                # break

            average_test_voltage_mape = np.mean(test_1_mape)
            average_test_temperature_mape = np.mean(test_2_mape)
            mlflow.log_metric("average_test_voltage_mape", float(average_valid_voltage_mape))
            mlflow.log_metric("average_test_temperature_mape", float(average_valid_temperature_mape))

            print(f'Average train voltage mape : {average_train_voltage_mape} || Average train temperature mape:{average_train_temperature_mape}')
            print(f'Average valid voltage mape : {average_valid_voltage_mape} || Average train temperature mape:{average_valid_temperature_mape}')
            print(f'Average test voltage mape : {average_test_voltage_mape} || Average train temperature mape:{average_test_temperature_mape}')
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=")


        #break
        scheduler.step(val_loss)
    mlflow.end_run()


    # Save the model


# Example usage
batch_size = 64
sequence_length = 150
# Initialize random tensors for training and validation
# Create model
# model = TransformerModel(time_embedding_size=0,state_input_size=3,state_output_size=3,
#                          action_input_size=1, output_size=4,
#                          sequence_length=sequence_length)
# model = Transformer_Model(state_input_size=3,action_input_size=1,output_size=3,
#                           sequence_length=10)

# model = TransformerMode_Sequential(time_embedding_size=0,state_input_size=3,state_output_size=3,
#                          action_input_size=1, output_size=4,
#                          sequence_length=sequence_length)

# model = Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = rescaling_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = rescaling_current_additive_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)
# model = rescaling2_current_additive_Correct_TransformerModel3(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)
# model = v2_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=1,
#                           dropout=0.3)

# model = v3_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=4,num_layers=8,
#                           dropout=0.3)

# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=32,output_dim=2,nhead=9,num_layers=4,
#                           dropout=0.3) #
# path for above '/Users/paarthsachan/simulator/v3_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth'
# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=10,num_layers=2,
#                           dropout=0.3)

# model = v4_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=5,num_layers=1,
#                           dropout=0.1)

# model = v5_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=146,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

# model = v6_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

# model = v8_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=10,num_layers=1,
#                           dropout=0.1)

model = v9_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=20,num_layers=1,
                          dropout=0.1)
## Path for above v4_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3
##########################################################################################
### RELOADING MODEL FOR RETRAINING THE SAME MODEL
##########################################################################################
#ssaved_model_path = 'Simulator_model/v11_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth'

#model_state_dict = torch.load(saved_model_path)
#model.load_state_dict(model_state_dict)
############################################################################################################
# model = LSTMModel(input_size_state=3, input_size_action=1,
#                   hidden_size=256, num_layer=4,output_size=2)

#device = torch.device("mps")# for mac, note that mps trained model cannot be used with cuda
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Train the model and save it

train_model(model,
            train_dataset=sim_train_dataset,test_dataset=sim_test_dataset,val_dataset=sim_val_dataset,
            batch_size=16,
            device = device,num_epochs=15,learning_rate=5e-5,
            sequence_length=sequence_length, save_path='Simulator_model/New_rul_input_trained.pth',
            temp_storage='Simulator_model/temp.pth')
