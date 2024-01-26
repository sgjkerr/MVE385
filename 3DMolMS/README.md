**You can find the original work here:**
https://github.com/JosieHong/3DMolMS/tree/main


**Before running** 

Please install the required packages by creating a new `Anaconda` environment.

Navigate to `/3DMolMS` and run:
````bash
conda env create -f environment.yml
````
This was tested with:
 - OS: Ubuntu 22.04.3 LTS x86_64
 - Kernel: 6.5.0-14-generic
 - Shell: bash 5.1.16
 - CPU: 12th Gen Intel i7-12700K (20) @ 4.900GHz
 - GPU: NVIDIA 01:00.0 NVIDIA Corporation Device 2684 
  
 - Nvidia version:
   - Driver Version: 535.154.05
   - CUDA Version: 12.2 




**Steps to create datasets for MassFormer and AttentiveFP**

Step 1: Follow steps 1-2 to create the datasets under **Run 3DMolMS**

Step 2: Navigate to the directory `MassFormer`and run the conversion script:
````bash
cd MassFormer
python ConvertToMassFormer.py
````
To create datasets for `AttentiveFP` navigate to the directory `AttentiveFP` do the following:
````bash
cd MassFormer
python ConvertToMassFormer.py
````

# Run 3DMolMS

Step 1: Download the datasets provided by https://www.nature.com/articles/s41597-023-02408-4#Sec10. Go here for instructions to transfer data: https://docs.olcf.ornl.gov/data/index.html#data-transferring-data 

Gather the datasets as follows:
```bash
|- datasets
  |- 10.13139_OLCF_1890227
    |- dftb_gdb9_electronic_excitation_spectrum
      |- mol_000008
      |- mol_000009
      |- ...
  |- 10.13139_OLCF_1907919
    |- ornl_aisd_ex_csv
      |- ornl_aisd_ex_1.csv
      |- ornl_aisd_ex_2.csv
      |- ...
    |- ornl_aisd_ex_1.tar.gz
    |- ornl_aisd_ex_2.tar.gz
    |- ...

```

Step 2: Create data for pre-training and training by running the following:
```bash
python preprocess_large_dataset.py --dataset_pre_size 2000 --dataset_size 1000 --dataset_val_size 500 --do_parallel
python preprocess_small_dataset.py --dataset_pre_size 2000 --dataset_size 1000 --dataset_val_size 500 --do_parallel
```
This will put data files in the `./data` directory as `hdf5` files. 

Step 3: Run `pretrain.py` to pre-train the encoder:
````bash
python pretrain.py
````
Step 4 (a): Train the full model (to use the pre-trained encoder skip this step):
````bash
python train.py
````
Step 4 (b): Train the decoder and use the pre-trained encoder:
````bash
python train.py --checkpoint_path ./check_point/molnet_pre_uv-vis_all_features.pt --transfer
````

Step 5: Make predictions on validation set by running `pred.ipynb`

**Design you own network**
The number of nodes and number of layers of the network can be changed in `config/molnet_pre.yml` and `config/molnet.yml` for pre-training and training respectively. The encoder in `config/molnet.yml` must match the encoder in `config/molnet_pre.yml` if you wish to transfer the encoder.

