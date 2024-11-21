from imports_libraries import *
from simulator_model_defined import *
from torch.utils.data import DataLoader, TensorDataset

model = v9_rescaling_adaptive_TransformerModel3Decoder(input_dim=3+1,hidden_dim=150,output_dim=2,nhead=20,num_layers=1,
                          dropout=0.1)

saved_model_path = 'Simulator_model/v11_hid32_4hd_l1_decoder_transformer_diff_rescaled_additive_3.pth'

model_state_dict = torch.load(saved_model_path)
model.load_state_dict(model_state_dict)


batch_size = 32
state = torch.randn(batch_size, 3)
action = torch.randn(batch_size, 150, 1)
output = model(state, action)
print(output.shape,"####")
