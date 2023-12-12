# LSTM

# Model Path
The model files can be found at:
https://drive.google.com/drive/folders/1Q4lNVWl6gutG3tmtFqS81jByPCLGrxoE?usp=sharing
save it in the models/ folder to use the model
e.g. project_directory/models/ft3/pytorch_model.bin

# Repo Structure
figure contains the loss overtime graphs
loss_record contains the losses overtime in training
models contains the fine-tuned model, it's contents can be ignored because it's too large and can't be uploaded
raw_data contains the raw data files and the data processing script data_process.py, and the extracted dataset backstory_large.pkl

demo.py is the script for the gradio UI
generate.py is a script for generating descriptions for development purposes
generate_skill_test.py is also a script for generating descriptions for development purposes
plot_loss_bs.py, plot_loss_lr.py, plot_loss_opt.py are plotting scripts
train.py is the script to finr-tune the GPT-2 small model
