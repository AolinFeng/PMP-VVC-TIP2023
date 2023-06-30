# PMP-VVC-TIP2023

The project page for the paper:

[Partition Map Prediction for Fast Block Partitioning in VVC Intra-frame Coding](https://ieeexplore.ieee.org/abstract/document/10102791?casa_token=AZrFglBKhj4AAAAA:l_CLVvP08dXwlI8OyH9B_0wUoNnegpKJYYKpPb13bS-p3F2zQrYwsm5XJvjjOBkQ84C9KjlTpg)

If our paper and codes are useful for your research, please cite:
```
@ARTICLE{10102791,
  author={Feng, Aolin and Liu, Kang and Liu, Dong and Li, Li and Wu, Feng},
  journal={IEEE Transactions on Image Processing}, 
  title={Partition Map Prediction for Fast Block Partitioning in VVC Intra-Frame Coding}, 
  year={2023},
  volume={32},
  number={},
  pages={2237-2251},
  doi={10.1109/TIP.2023.3266165}}
```
If you have any question, please feel free to contact:

Aolin Feng (fal1997@mail.ustc.edu.cn)

## Preceding note:

This project involves the VVC codec, CNN training/inference, post-processing and son on. We tried our best to provide the code about dataset preparation, network training/inference, and acceleration algorithm implementation in the VVC encoder. But forgive us that we cannot provide a demo that can be runned with one click. Please feel free to contact us if you encounter difficulties. 

## File Instructions

### Dataset preparation
* CreateDataSet.py: Generate dataset for network training and testing, including yuv blocks, qt depth map, direction map, and the last layer of bt depth map.
* GenMSBtMap.py: Generate complete multi-layer bt depth map. Combine with the results from CreateDataSet.py to get complete dataset.

### Train and inference
* Train_QBD.py: Down-Up-CNN training
* Model_QBD.py: Down-Up-CNN model
* Inference_QBD.py: Network inference + Post processing
* Metrics.py: Related metrics used in other file
* Map2Partition.py: From partition map (network-output) to partition structure ( able to be used in the VVC encoder)
* VVC_Test_Sequences.txt: Information of VVC test sequences, which is used in some functions and whose format can be refereced if needing to test other sequences.

## Note

* The note in the biegining of the code file is useful. It tells the function and noteworthy part of the code file. 
* The annotated code may be unused in the runing process, but can be refereced for further modification.
* The dataset is generated from VVC encoder. It can be quite different if using different version of VTM or different configurations. So We recommend to generate the dataset according to the version of VTM you plan to use.
* The name "QBD" is from "QT" + "BT" + "Direction" in the begining stage. We modefied the expression of "BT" into "MTT" later in the paper writting, but we did not modify the naming in the code.

