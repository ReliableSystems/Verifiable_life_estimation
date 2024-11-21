
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import numpy as np
import torch.optim as optim
from numba import jit
import torch.nn.functional as F



class TransformerModel(nn.Module):
    def __init__(self, action_input_size, state_input_size, state_output_size,output_size, sequence_length,time_embedding_size=20, num_layers=1):
        super(TransformerModel, self).__init__()
        self.state_embedding = nn.Linear(time_embedding_size+state_input_size, output_size)
        self.action_embedding = nn.Linear(action_input_size, output_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*output_size, nhead=1),
            num_layers=num_layers
        )

        self.fc = nn.Linear(2*output_size, state_output_size)

    def forward(self, state, action, num_steps=10):
        if num_steps > action.shape[1]:
            raise ValueError("Cannot predict more than the given sequence length")
        embedded_state = self.state_embedding(state)
        action_reshaped = action.view(action.shape[0] * action.shape[1], -1)
        embedded_action = self.action_embedding(action_reshaped)
        embedded_action = embedded_action.view(action.shape[0], action.shape[1], -1)
        embedded_state = embedded_state.unsqueeze(1).repeat(1, embedded_action.shape[1], 1)

        # Apply Transformer Encoder step by step
        outputs = []

        for i in range(num_steps):
            # Apply Transformer Encoder to the current output
            embedded_action_single = embedded_action[:,i,:]
            embedded_state_single  = embedded_state[:,i,:]
            embedding_combined_single=torch.cat((embedded_action_single
                                , embedded_state_single), dim=1).unsqueeze(1)


            transformer_output = self.transformer_encoder(embedding_combined_single)

            # Permute back to original dimensions
            transformer_output_squeezed = transformer_output.squeeze()
            # Apply fully connected layer
            external_state = self.fc(transformer_output_squeezed)

            # Append current output to the list of outputs
            outputs.append(external_state.unsqueeze(0))

        # Concatenate outputs along the time dimension

        outputs = torch.stack(outputs, dim=2).squeeze()

        return outputs

    def inference_single(self, state, action, num_steps=1):
        embedded_state = self.state_embedding(state)
        action_reshaped = action.view(action.shape[0] * action.shape[1], -1)
        embedded_action = self.action_embedding(action_reshaped)
        embedded_action = embedded_action.view(action.shape[0], action.shape[1], -1)
        embedded_state = embedded_state.unsqueeze(1).repeat(1, embedded_action.shape[1], 1)
        embedded_action_single = embedded_action[:,0,:]
        embedded_state_single  = embedded_state[:,0,:]
        embedding_combined_single=torch.cat((embedded_action_single
                            , embedded_state_single), dim=1).unsqueeze(1)


        transformer_output = self.transformer_encoder(embedding_combined_single)

        # Permute back to original dimensions
        transformer_output_squeezed = transformer_output.squeeze()
        # Apply fully connected layer
        external_state = self.fc(transformer_output_squeezed)
        # Apply Transformer Encoder step by step
        return external_state



class Transformer_Model(nn.Module):
    def __init__(self, action_input_size,state_input_size, output_size, sequence_length):
        super(Transformer_Model, self).__init__()
        self.state_embedding = nn.Linear(state_input_size, output_size)
        self.action_embedding =  nn.Linear(action_input_size, output_size)
        self.transformer = nn.Transformer(d_model=output_size, nhead=1,batch_first=True)
        self.fc = nn.Linear(output_size, state_input_size)

    def forward(self, state, action):
        embedded_state = self.state_embedding(state)
        action_reshaped = action.reshape(action.shape[0]*action.shape[1],-1)
        embedded_action = self.action_embedding(action_reshaped)
        embedded_action=embedded_action.reshape(action.shape[0],action.shape[1],-1)
        embedded_state=embedded_state.unsqueeze(1).repeat(1, embedded_action.shape[1], 1)

        output = self.transformer(embedded_state, embedded_action)
        output = self.fc(output)

        return output

class TransformerMode_Sequential(nn.Module):
    def __init__(self, action_input_size, state_input_size, state_output_size,output_size, sequence_length,time_embedding_size=20, num_layers=1):
        super(TransformerMode_Sequential, self).__init__()
        self.state_embedding = nn.Linear(time_embedding_size+state_input_size, output_size)
        self.action_embedding = nn.Linear(action_input_size, output_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*output_size, nhead=1),
            num_layers=num_layers
        )

        self.fc = nn.Linear(2*output_size, state_output_size)

    def forward(self, state, action, num_steps=10):
        if num_steps > action.shape[1]:
            raise ValueError("Cannot predict more than the given sequence length")
        embedded_state = self.state_embedding(state)
        action_reshaped = action.view(action.shape[0] * action.shape[1], -1)
        embedded_action = self.action_embedding(action_reshaped)
        embedded_action = embedded_action.view(action.shape[0], action.shape[1], -1)
        embedded_state = embedded_state.unsqueeze(1).repeat(1, embedded_action.shape[1], 1)

        # Apply Transformer Encoder step by step
        outputs = []

        for i in range(num_steps):
            # Apply Transformer Encoder to the current output
            embedded_action_single = embedded_action[:,i,:]
            embedded_state_single  = embedded_state[:,i,:]
            embedding_combined_single=torch.cat((embedded_action_single
                                , embedded_state_single), dim=1).unsqueeze(1)


            transformer_output = self.transformer_encoder(embedding_combined_single)

            # Permute back to original dimensions
            transformer_output_squeezed = transformer_output.squeeze()
            # Apply fully connected layer
            external_state = self.fc(transformer_output_squeezed)

            # Append current output to the list of outputs
            outputs.append(external_state.unsqueeze(0))

        # Concatenate outputs along the time dimension

        outputs = torch.stack(outputs, dim=2).squeeze()

        return outputs

    def inference_single(self, state, action, num_steps=1):
        embedded_state = self.state_embedding(state)
        action_reshaped = action.view(action.shape[0] * action.shape[1], -1)
        embedded_action = self.action_embedding(action_reshaped)
        embedded_action = embedded_action.view(action.shape[0], action.shape[1], -1)
        embedded_state = embedded_state.unsqueeze(1).repeat(1, embedded_action.shape[1], 1)
        embedded_action_single = embedded_action[:,0,:]
        embedded_state_single  = embedded_state[:,0,:]
        embedding_combined_single=torch.cat((embedded_action_single
                            , embedded_state_single), dim=1).unsqueeze(1)


        transformer_output = self.transformer_encoder(embedding_combined_single)

        # Permute back to original dimensions
        transformer_output_squeezed = transformer_output.squeeze()
        # Apply fully connected layer
        external_state = self.fc(transformer_output_squeezed)
        # Apply Transformer Encoder step by step
        return external_state

class TransformerModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=5000, dropout=0.5):
        super(TransformerModel3, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,0]= initial_state[:,1]/30
        actions = actions/5

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        #print(transformer_input.shape,self.positional_encoding.shape,self.positional_encoding[:transformer_input.size(1), :].shape," SHAPES BEFORE ADDING")
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]
        # print(jdd)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)

        return predicted_states

class LSTMModel(nn.Module):
    def __init__(self, input_size_state, input_size_action, hidden_size, output_size,num_layer):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size_state + input_size_action,
                            hidden_size=hidden_size,num_layers=num_layer,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        # Expand state along the last dimension to match action size
        state[:,1]= state[:,1]/3
        state[:,0]= state[:,1]/30
        action = action/5
        #print(state.shape,action.shape,"#$$$$$$$$$$$$$$$$$$")
        state = state.unsqueeze(1).repeat(1, action.size(1), 1)
        #print(state.shape,"#$$$$$$$$$$$$$$$$$$")



        # Concatenate state and action along the last dimension
        combined_input = torch.cat((state, action), dim=-1)

        # Initialize hidden state and cell state
        #h0 = torch.zeros(1, combined_input.size(0), self.hidden_size).to(state.device)
        #c0 = torch.zeros(1, combined_input.size(0), self.hidden_size).to(state.device)

        # Forward pass through LSTM
        lstm_out, _ = self.lstm(combined_input)

        # Apply fully connected layer to the output of LSTM
        #print(lstm_out.shape,"#################@@@@@@@@@@@@@@@@@@")
        output = self.fc(lstm_out[:, :, :])
        #print(output.shape,"$$$$$$$$@@@@@@@@@@@@@@@@")
        #print(jd)

        return output



class Correct_TransformerModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=5000, dropout=0.5):
        super(Correct_TransformerModel3, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30
        # print(initial_state.shape,actions.shape,"##################")
        # print(jd)
        actions = actions/5

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        #print(transformer_input.shape,self.positional_encoding.shape,self.positional_encoding[:transformer_input.size(1), :].shape," SHAPES BEFORE ADDING")
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]
        # print(jdd)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)

        return predicted_states

class Correct_LSTMModel(nn.Module):
    def __init__(self, input_size_state, input_size_action, hidden_size, output_size,num_layer):
        super(Correct_LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size_state + input_size_action,
                            hidden_size=hidden_size,num_layers=num_layer,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        # Expand state along the last dimension to match action size
        state[:,1]= state[:,1]/3
        state[:,0]= state[:,1]/30
        action = action/5
        #print(state.shape,action.shape,"#$$$$$$$$$$$$$$$$$$")
        state = state.unsqueeze(1).repeat(1, action.size(1), 1)
        #print(state.shape,"#$$$$$$$$$$$$$$$$$$")



        # Concatenate state and action along the last dimension
        combined_input = torch.cat((state, action), dim=-1)

        # Initialize hidden state and cell state
        #h0 = torch.zeros(1, combined_input.size(0), self.hidden_size).to(state.device)
        #c0 = torch.zeros(1, combined_input.size(0), self.hidden_size).to(state.device)

        # Forward pass through LSTM
        lstm_out, _ = self.lstm(combined_input)

        # Apply fully connected layer to the output of LSTM
        #print(lstm_out.shape,"#################@@@@@@@@@@@@@@@@@@")
        output = self.fc(lstm_out[:, :, :])
        #print(output.shape,"$$$$$$$$@@@@@@@@@@@@@@@@")
        #print(jd)

        return output

class rescaling_Correct_TransformerModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(rescaling_Correct_TransformerModel3, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30
        # print(cloned_initial_state)
        # print(jd)
        # print(initial_state.shape,actions.shape,"##################")
        # print(jd)
        actions = actions/5
        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        #print(transformer_input.shape,self.positional_encoding.shape,self.positional_encoding[:transformer_input.size(1), :].shape," SHAPES BEFORE ADDING")
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]
        # print(jdd)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)
        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        predicted_states = predicted_states/100


        # print(predicted_states[:,:,0].shape,"Voltage dimension")
        # print(predicted_states[:,:,1].shape,"Temp dimension")
        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states[:,:,0] += repeated_original_voltage
        predicted_states[:,:,1] += repeated_original_temprature
        # print(repeated_original_initial_state.shape,"Original")
        # print(repeated_original_voltage.shape,"Voltage shape")
        # print(repeated_original_temprature.shape,"Temperature shape")
        # print(repeated_original_voltage)
        # print(repeated_original_temprature)
        # print(predicted_states.shape,"shape of the predicted states")
        # print(jd)

        return predicted_states


class rescaling_current_additive_Correct_TransformerModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(rescaling_current_additive_Correct_TransformerModel3, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30
        # print(cloned_initial_state)
        # print(jd)
        # print(initial_state.shape,actions.shape,"##################")
        # print(jd)
        actions = (actions+5)/10
        # print(actions)
        # print(jd)
        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        #print(transformer_input.shape,self.positional_encoding.shape,self.positional_encoding[:transformer_input.size(1), :].shape," SHAPES BEFORE ADDING")
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]
        # print(jdd)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)
        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        predicted_states = predicted_states/100


        # print(predicted_states[:,:,0].shape,"Voltage dimension")
        # print(predicted_states[:,:,1].shape,"Temp dimension")
        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states[:,:,0] += repeated_original_voltage
        predicted_states[:,:,1] += repeated_original_temprature
        # print(repeated_original_initial_state.shape,"Original")
        # print(repeated_original_voltage.shape,"Voltage shape")
        # print(repeated_original_temprature.shape,"Temperature shape")
        # print(repeated_original_voltage)
        # print(repeated_original_temprature)
        # print(predicted_states.shape,"shape of the predicted states")
        # print(jd)

        return predicted_states

class rescaling2_current_additive_Correct_TransformerModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(rescaling2_current_additive_Correct_TransformerModel3, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30
        # print(cloned_initial_state)
        # print(jd)
        # print(initial_state.shape,actions.shape,"##################")
        # print(jd)
        actions = (actions+5)/10
        # print(actions)
        # print(jd)
        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        #print(transformer_input.shape,self.positional_encoding.shape,self.positional_encoding[:transformer_input.size(1), :].shape," SHAPES BEFORE ADDING")
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]
        # print(jdd)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)
        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        predicted_states[:,:,0] = predicted_states[:,:,0]/100
        predicted_states[:,:,1] = predicted_states[:,:,1]/1000



        # print(predicted_states[:,:,0].shape,"Voltage dimension")
        # print(predicted_states[:,:,1].shape,"Temp dimension")
        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states[:,:,0] += repeated_original_voltage
        predicted_states[:,:,1] += repeated_original_temprature
        # print(repeated_original_initial_state.shape,"Original")
        # print(repeated_original_voltage.shape,"Voltage shape")
        # print(repeated_original_temprature.shape,"Temperature shape")
        # print(repeated_original_voltage)
        # print(repeated_original_temprature)
        # print(predicted_states.shape,"shape of the predicted states")
        # print(jd)
        return predicted_states


class rescaling2_current_additive_Correct_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(rescaling2_current_additive_Correct_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        # Output linear layer
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions+5)/10

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        # Concatenate initial state with actions
        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        # Transform input dimensions
        transformer_input = self.linear_in(transformer_input)

        # Add learnable positional encoding
        transformer_input = transformer_input + self.positional_encoding[:transformer_input.size(1), :]

        # Transformer Decoder
        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        # Output layer to get predicted states
        predicted_states = self.linear_out(transformer_output)

        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        predicted_states[:,:,0] = predicted_states[:,:,0]/100
        predicted_states[:,:,1] = predicted_states[:,:,1]/1000

        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states[:,:,0] += repeated_original_voltage
        predicted_states[:,:,1] += repeated_original_temprature

        return predicted_states

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v2_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v2_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 4))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim+4, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(hidden_dim+4, output_dim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states = self.linear_out1(transformer_output)

        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        #print(actions.squeeze(dim=2).shape,"@@@@@@@@@")
        spacing_array = torch.linspace(1, 16, 150).to(actions.device)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array

        #print(current_based_spacing_temporal,power_based_spacing_temporal)
        #print(jd)
        predicted_states[:,:,0] = -(predicted_states[:,:,0]*current_based_spacing_temporal)#/10
        predicted_states[:,:,1] = (predicted_states[:,:,1]*power_based_spacing_temporal)#/10

        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states[:,:,0] += repeated_original_voltage
        predicted_states[:,:,1] += repeated_original_temprature

        return predicted_states

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v3_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v3_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 4))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim+4, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(hidden_dim+4, 3*output_dim)
        self.linear_out2 = nn.Linear(3*output_dim,output_dim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states = self.linear_out1(transformer_output)
        predicted_states = self.gelu(predicted_states)
        #predicted_states = self.linear_out2(predicted_states)

        initial_state_volt_temp = initial_state_repeated[:,:,1:]
        #print(actions.squeeze(dim=2).shape,"@@@@@@@@@")
        spacing_array = torch.linspace(0.1, 15, 150).to(actions.device)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array

        # print("Scaling shapes",current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print("Output Shapes",predicted_states[:,:,0].shape,predicted_states[:,:,1].shape)
        # print(spacing_array.shape,predicted_states.shape,"###")
        # print(spacing_array.shape,predicted_states[:,:,2].shape,"!!!!!!!!!!!!!!!")
        # print((predicted_states[:,:,2]*spacing_array).shape,"SPacing ####")
        # print(jd)
        # v1->v2[lr change]
        # v2->v3 [normal scaling + physical quantity based scaling, sogmoid an 3.7 based addition]

        predicted_states_voltage = -(predicted_states[:,:,0]*(current_based_spacing_temporal))+predicted_states[:,:,2]*spacing_array+predicted_states[:,:,4]

        predicted_states_temperature = (predicted_states[:,:,1]*(power_based_spacing_temporal))+predicted_states[:,:,3]*spacing_array+predicted_states[:,:,5]


        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states_voltage += repeated_original_voltage #okayish for previous iter
        predicted_states_temperature += repeated_original_temprature


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class v4_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v4_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 4))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim+4, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(hidden_dim+4, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,output_dim)

        self.direct_prediction_before_scaling_1 = nn.Linear(hidden_dim+4,5*output_dim)
        self.direct_prediction_before_scaling_2 = nn.Linear(5*output_dim,2*output_dim)
        self.direct_prediction_before_scaling_3 = nn.Linear(2*output_dim,output_dim)


        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual = self.linear_out1(transformer_output)
        predicted_states_residual = self.gelu(predicted_states_residual)
        predicted_states_residual = self.linear_out2(predicted_states_residual)
        predicted_states_residual = self.gelu(predicted_states_residual)
        predicted_states_residual = self.linear_out3(predicted_states_residual)

        predicted_direct_scaling = self.direct_prediction_before_scaling_1(transformer_output)
        predicted_direct_scaling = self.gelu(predicted_direct_scaling)
        predicted_direct_scaling = self.direct_prediction_before_scaling_2(predicted_direct_scaling)
        predicted_direct_scaling = self.gelu(predicted_direct_scaling)
        predicted_direct_scaling = self.direct_prediction_before_scaling_3(predicted_direct_scaling)

        spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array
        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(jd)
        predicted_direct_scaling[:,:,0] = predicted_direct_scaling[:,:,0]*current_based_spacing_temporal
        predicted_direct_scaling[:,:,1] = predicted_direct_scaling[:,:,1]*power_based_spacing_temporal
        # print(predicted_states_residual.shape,predicted_direct.shape,"#$$$$")
        # print(jd)
        # print(predicted_direct_scaling.shape,predicted_states_residual.shape)
        scaled_residual = predicted_states_residual*predicted_direct_scaling


        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states_voltage = repeated_original_voltage +(scaled_residual[:,:,0]/100)
        predicted_states_temperature = repeated_original_temprature+(scaled_residual[:,:,1]/1000)


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class v5_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v5_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 4))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim+4, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(hidden_dim+4, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,output_dim)

        self.direct_prediction_before_scaling_1 = nn.Linear(hidden_dim+4,5*output_dim)
        self.direct_prediction_before_scaling_2 = nn.Linear(5*output_dim,2*output_dim)
        self.direct_prediction_before_scaling_3 = nn.Linear(2*output_dim,output_dim)


        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)

        predicted_direct_scaling_1 = self.direct_prediction_before_scaling_1(transformer_output)
        predicted_direct_scaling_1 = self.gelu(predicted_direct_scaling_1)
        predicted_direct_scaling_2 = self.direct_prediction_before_scaling_2(predicted_direct_scaling_1)
        predicted_direct_scaling_2 = self.gelu(predicted_direct_scaling_2)
        predicted_direct_scaling_3 = self.direct_prediction_before_scaling_3(predicted_direct_scaling_2)

        spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array
        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(jd)
        predicted_direct_scaling_3[:,:,0] = predicted_direct_scaling_3[:,:,0]*current_based_spacing_temporal
        predicted_direct_scaling_3[:,:,1] = predicted_direct_scaling_3[:,:,1]*power_based_spacing_temporal
        # print(predicted_states_residual.shape,predicted_direct.shape,"#$$$$")
        # print(jd)
        # print(predicted_direct_scaling.shape,predicted_states_residual.shape)
        scaled_residual_linear = predicted_states_residual_3*predicted_direct_scaling_3

        bias_voltage = predicted_states_residual_2[:,:,0]
        bias_temperature = predicted_states_residual_2[:,:,1]

        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states_voltage = repeated_original_voltage +bias_voltage+((scaled_residual_linear[:,:,0])/100)
        predicted_states_temperature = repeated_original_temprature+bias_temperature+((bias_temperature+scaled_residual_linear[:,:,1])/1000)


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class v6_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v6_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(2*hidden_dim, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,2*output_dim)
        self.conv1d_volt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_temp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)



        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)


        spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array

        current_based_spacing_temporal = current_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        current_based_spacing_temporal = self.conv1d_volt(current_based_spacing_temporal)
        current_based_spacing_temporal = current_based_spacing_temporal.squeeze(1)  # Remove channel dimension

        power_based_spacing_temporal = power_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        power_based_spacing_temporal = self.conv1d_temp(power_based_spacing_temporal)
        power_based_spacing_temporal = power_based_spacing_temporal.squeeze(1)  # Remove channel dimension


        voltage_deg_one = predicted_states_residual_3[:,:,0]*current_based_spacing_temporal
        voltage_bias = predicted_states_residual_3[:,:,1]

        temp_deg_one = predicted_states_residual_3[:,:,2]*power_based_spacing_temporal
        temp_bias = predicted_states_residual_3[:,:,3]

        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(predicted_states_residual_3.shape)
        # print(jd)

        repeated_original_voltage = repeated_original_initial_state[:,:,1]+voltage_deg_one+voltage_bias
        repeated_original_temprature = repeated_original_initial_state[:,:,2]+temp_deg_one+temp_bias

        predicted_states_voltage = repeated_original_voltage #+bias_voltage+((scaled_residual_linear[:,:,0])/100)
        predicted_states_temperature = repeated_original_temprature#+bias_temperature+((bias_temperature+scaled_residual_linear[:,:,1])/1000)


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class v7_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v7_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(2*hidden_dim, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,2*output_dim)
        self.conv1d_volt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_temp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)



        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)


        spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array

        current_based_spacing_temporal = current_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        current_based_spacing_temporal = self.conv1d_volt(current_based_spacing_temporal)
        current_based_spacing_temporal = current_based_spacing_temporal.squeeze(1)  # Remove channel dimension

        power_based_spacing_temporal = power_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        power_based_spacing_temporal = self.conv1d_temp(power_based_spacing_temporal)
        power_based_spacing_temporal = power_based_spacing_temporal.squeeze(1)  # Remove channel dimension


        voltage_deg_one = predicted_states_residual_3[:,:,0]*current_based_spacing_temporal
        voltage_bias = predicted_states_residual_3[:,:,1]

        temp_deg_one = predicted_states_residual_3[:,:,2]*power_based_spacing_temporal
        temp_bias = predicted_states_residual_3[:,:,3]

        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(predicted_states_residual_3.shape)
        # print(jd)

        repeated_original_voltage = repeated_original_initial_state[:,:,1]+voltage_deg_one+voltage_bias
        repeated_original_temprature = repeated_original_initial_state[:,:,2]+temp_deg_one+temp_bias

        predicted_states_voltage = repeated_original_voltage #+bias_voltage+((scaled_residual_linear[:,:,0])/100)
        predicted_states_temperature = repeated_original_temprature#+bias_temperature+((bias_temperature+scaled_residual_linear[:,:,1])/1000)


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v8_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v8_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(2*hidden_dim, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,2*output_dim)
        self.conv1d_volt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_temp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)



        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        # print(actions.shape,"#######")
        actions_clone_1= actions.clone()
        actions_clone_2= actions.clone()
        actions_delta_shifted = actions_clone_2
        actions_delta_shifted[:,0:-1,:]-=actions_clone_1[:,1:,:]
        actions_delta_shifted[:,-1,:]=0
        # print(actions_delta_shifted[:,0:-1,:].shape,actions_clone_1[:,1:,:].shape,"####")
        # print(actions_delta_shifted)
        # print(jd)

        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        transformer_input = torch.cat([initial_state_repeated, actions], dim=-1)

        transformer_input = self.linear_in(transformer_input)


        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)


        spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        #current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        current_based_spacing_temporal = actions_delta_shifted.squeeze(dim=2)*spacing_array
        power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array
        #############
        # v9 had convolution v10 won't
        # current_based_spacing_temporal = current_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        # current_based_spacing_temporal = self.conv1d_volt(current_based_spacing_temporal)
        # current_based_spacing_temporal = current_based_spacing_temporal.squeeze(1)  # Remove channel dimension

        # power_based_spacing_temporal = power_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        # power_based_spacing_temporal = self.conv1d_temp(power_based_spacing_temporal)
        # power_based_spacing_temporal = power_based_spacing_temporal.squeeze(1)  # Remove channel dimension
        #############


        voltage_deg_one = predicted_states_residual_3[:,:,0]*current_based_spacing_temporal
        voltage_bias = predicted_states_residual_3[:,:,1]

        temp_deg_one = predicted_states_residual_3[:,:,2]*power_based_spacing_temporal
        temp_bias = predicted_states_residual_3[:,:,3]

        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(predicted_states_residual_3.shape)
        # print(jd)

        repeated_original_voltage = repeated_original_initial_state[:,:,1]+voltage_deg_one+voltage_bias*10
        repeated_original_temprature = repeated_original_initial_state[:,:,2]+temp_deg_one+temp_bias*10

        predicted_states_voltage = repeated_original_voltage #+bias_voltage+((scaled_residual_linear[:,:,0])/100)
        predicted_states_temperature = repeated_original_temprature#+bias_temperature+((bias_temperature+scaled_residual_linear[:,:,1])/1000)


        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class v9_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v9_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_in_1 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm_final = nn.LayerNorm(output_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        # Output linear layers
        # self.linear_out1 = nn.Linear(2*hidden_dim, hidden_dim)
        # self.linear_out2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear_out1 = nn.Linear(2*hidden_dim, hidden_dim//10)
        self.linear_out2 = nn.Linear(hidden_dim//10, hidden_dim//20)

        self.linear_out3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out4 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear_out5 = nn.Linear(hidden_dim//2, hidden_dim//5)
        self.linear_out6 = nn.Linear(hidden_dim//5, hidden_dim//10)
        # self.linear_out7 = nn.Linear(hidden_dim//10, hidden_dim//30)
        # self.linear_out8 = nn.Linear(hidden_dim//30, 1)
          # Final output layer

        self.gelu = nn.GELU()

    def forward(self, initial_state, current):
        # Normalize inputs
        initial_state_normalized = initial_state.clone()
        initial_state_normalized[:,:,0] = initial_state[:,:,0] / 3  # Voltage normalization
        initial_state_normalized[:,:,1] = initial_state[:,:,1] / 30  # Temperature normalization

        current_normalized = current / 5

        # Concatenate inputs
        transformer_input = torch.cat([initial_state_normalized, current_normalized], dim=-1)

        # Linear layers
        #print(transformer_input.shape,"@@@@@@@@")
        transformer_input = self.linear_in(transformer_input)
        #print(transformer_input.shape,"@@@@@@@@")
        transformer_input = self.linear_in_1(transformer_input)
        #print(transformer_input.shape,"@@@@@@@@")
        # Add positional encoding
        pos_encoding = self.positional_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)
        transformer_input = torch.cat((transformer_input, pos_encoding), dim=-1)
        #print(transformer_input.shape,"@@@@@@@@")
        # Transformer decoder
        tgt_mask = self.generate_square_subsequent_mask(current.size(1)).to(current.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        # Output layers
        output = self.linear_out1(transformer_output)
        output = self.gelu(output)
        output = self.linear_out2(output)
        # output = self.gelu(output)
        # output = self.linear_out3(output)
        # output = self.gelu(output)

        # output = self.gelu(output)
        # output = self.linear_out4(output)

        # output = self.gelu(output)
        # output = self.linear_out5(output)

        # output = self.gelu(output)
        # output = self.linear_out6(output)

        output = output.mean(dim=2).unsqueeze(-1)
        output = 900*torch.tanh(output)+450
        #output -torch.log(output)
        # output = output*(1000)
        # output = torch.exp(output/10)
        #output = torch.exp(output*10)

        return output  # Return shape: (batch_size,)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v10_rescaling_adaptive_TransformerModel3Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v10_rescaling_adaptive_TransformerModel3Encoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_in_1 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=[300, 1])


        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout, batch_first=True,
                                                    dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output linear layers
        self.linear_out1 = nn.Linear(2*hidden_dim*max_len, max_len*hidden_dim//10)
        self.linear_out2 = nn.Linear(max_len*hidden_dim//10, max_len)
        self.linear_out3 = nn.Linear(max_len, max_len//10)
        self.linear_out4 = nn.Linear(max_len//10, 1)

        # self.linear_out3 = nn.Linear(hidden_dim//20, hidden_dim)
        # self.linear_out4 = nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear_out5 = nn.Linear(hidden_dim//2, hidden_dim//5)
        # self.linear_out6 = nn.Linear(hidden_dim//5, hidden_dim//10)
        # self.linear_out7 = nn.Linear(hidden_dim//10, hidden_dim//30)
        # self.linear_out8 = nn.Linear(hidden_dim//30, 1)

        self.gelu = nn.GELU()


        self.seq_lin1 = nn.Linear(600,50)
        self.seq_lin2 = nn.Linear(50,10)
        self.seq_lin3 = nn.Linear(10,1)

    def forward(self, initial_state, current):
        # Normalize inputs
        initial_state_normalized = initial_state.clone()
        initial_state_normalized[:,:,0] = initial_state[:,:,0] / 3  # Voltage normalization
        initial_state_normalized[:,:,1] = initial_state[:,:,1] / 30  # Temperature normalization

        current_normalized = current / 5

        # Concatenate inputs
        #print(initial_state.shape, current.shape,"$$$$$$$$$$$$$$$$")
        transformer_input = torch.cat([initial_state_normalized, current_normalized], dim=-1)
        # print(transformer_input.shape)
        # print(jd)
        # Linear layers
        #print(transformer_input.shape,"inp after cat")
        transformer_input = self.linear_in(transformer_input)
        #print(transformer_input.shape,"inp after lin")
        transformer_input = self.linear_in_1(transformer_input)
        #print(transformer_input.shape,"inp after lin_1")
        # Add positional encoding
        pos_encoding = self.positional_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)
        #print(transformer_input.shape,pos_encoding.shape,"#####")
        transformer_input = torch.cat((transformer_input, pos_encoding), dim=-1)

        # Transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Output layers

        transformer_output = torch.flatten(transformer_output, start_dim=1)

        # print(transformer_output.shape,"#######")
        # print(jd)
        output = self.linear_out1(transformer_output)
        output = self.gelu(output)
        output = self.linear_out2(output)
        output = self.gelu(output)
        output = self.linear_out3(output)
        output = self.gelu(output)

        output = 4*self.linear_out4(output)
        # output = torch.round(output)

        # output = self.linear_out3(output)
        # output = self.gelu(output)
        # output = self.linear_out4(output)
        # output = self.gelu(output)
        # output = self.linear_out5(output)
        # output = self.gelu(output)
        # output = self.linear_out6(output)
        # output = self.gelu(output)
        # output = self.linear_out7(output)
        # output = self.gelu(output)
        # output = self.linear_out8(output)

        # Scale the output
        #output = torch.abs(output) * 10
        # output = self.layer_norm(output)
        # # print(output.shape,"@#######")
        # # print(Jd)
        # output = output.squeeze()

        # output =self.seq_lin1(output)
        # output = self.gelu(output)

        # output =self.seq_lin2(output)
        # output = self.gelu(output)

        # output =self.seq_lin3(output)
        # output = 100*output
        # output = output//100

        # print(output.shape,"$$$$$$$$$$ OUTPUT SHAPE BEFORE TRNSFORMATION")
        # print(jd)
        # output = torch.sigmoid(output)*3
        # output = torch.exp(output)
        # output = output - torch.exp(torch.tensor(1.5))




        return output.squeeze()  # Return shape: (batch_size, seq_len)
