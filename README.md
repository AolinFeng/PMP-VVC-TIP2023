# PMP-VVC-TIP2023

The project page for the paper:

[Partition Map Prediction for Fast Block Partitioning in VVC Intra-frame Coding](https://ieeexplore.ieee.org/abstract/document/10102791?casa_token=AZrFglBKhj4AAAAA:l_CLVvP08dXwlI8OyH9B_0wUoNnegpKJYYKpPb13bS-p3F2zQrYwsm5XJvjjOBkQ84C9KjlTpg)

If our paper and codes are useful for your research, please cite:
```
@article{feng2023partition,
  title={Partition map prediction for fast block partitioning in VVC intra-frame coding},
  author={Feng, Aolin and Liu, Kang and Liu, Dong and Li, Li and Wu, Feng},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE},
  volume={32},
  number={},
  pages={2237-2251},
  doi={10.1109/TIP.2023.3266165}}
```
If you have any questions, please feel free to contact:

Aolin Feng (aolinf19@gmail.com)

## Preceding note:

This project involves the VVC codec, CNN training/inference, post-processing, and so on. We tried our best to provide the code about dataset preparation, network training/inference, and fast algorithm implementation in the VVC encoder. But forgive us that we cannot provide a complete demo that can be run with one click. Please feel free to contact us if you encounter difficulties. 

## Folder Instructions
* codec: Include the source files and exe of VTM-10.0 implemented with the proposed fast algorithm
* trained models: Include the trained models of QT net and MTT net (details in the paper).

## File Instructions

### Dataset preparation
* CreateDataSet.py: Generate dataset for network training and testing, including yuv blocks, qt depth map, direction map, and the last layer of bt depth map.
* GenMSBtMap.py: Generate complete multi-layer bt depth map. Combine with the results from CreateDataSet.py to get the complete dataset.

### Train and inference
* Train_QBD.py: Down-Up-CNN training
* Model_QBD.py: Down-Up-CNN model
* Inference_QBD.py: Network inference + Post-processing
* Metrics.py: Related metrics used in other files
* Map2Partition.py: From partition map (network-output) to partition structure ( able to be used in the VVC encoder)
* VVC_Test_Sequences.txt: Information of VVC test sequences, which is used in some functions and whose format can be referenced if needing to test other sequences.

## Note
* There is a README file in each folder. Please refer to it for instructions.
* The note at the beginning of the code file is useful. It tells the function and noteworthy part of the code file. 
* The annotated code may be unused in the running process, but can be referenced for further modification.
* The dataset is generated from the VVC encoder. It can be quite different if using different versions of VTM or different configurations. 
* The name "QBD" is from "QT" + "BT" + "Direction" in the begining stage. We modified the expression of "BT" into "MTT" later in the paper writing, but we did not modify the naming in the code.


