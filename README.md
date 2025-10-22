# LPTNet+
LPTNet+ repository


For easy evaluation please insert released model LPTNet+ M200 into a folder named saved_models_1d and run Sim_example_evaluate.py.
Also possible to train with Sim_main_train_script.py to see how the model interacts with data.

Example training run:
python /YOUR_PATH/Sim_Main_train_script.py \
    --data_path='/YOUR_Path/spectrogram_dataset' \
    --save_dir='/YOUR_Path/saved_models_1d' \
    --starting_epoch 10  --epochs 15 --lr 4.11e-4 --M 200 --ending_epoch 4 --entropy_lambda 11e-2 --sum_lambda 5e-8 --tv_lambda 2e-1

Example inference:
python /YOUR_PATH/Sim_example_evaluate.py \
    --data_path='/YOUR_PATH/spectrogram_dataset_small/spectrogram_dataset_small_test.pt' \
    --save_dir='/YOUR_PATH/saved_models_1d/lpt_sim_final_M200 with 29.87dB.pth'
