This codec is for the paper: 
A. Feng, K. Liu, D. Liu, L. Li and F. Wu, "Partition Map Prediction for Fast Block Partitioning in VVC Intra-Frame Coding," in IEEE Transactions on Image Processing, vol. 32, pp. 2237-2251, 2023, doi: 10.1109/TIP.2023.3266165.

* Folder "vtm-exe-with-pmp-fast-alg" includes VTM-10.0 and VTM-17.2 exe implemented with the partition map-based fast algorithm. Each codec version contains four configurations. For example, "EncoderApp_FastPartition100L3.exe" means it is the VTM-10.0 encoder with the acceleration configuration L3 (details in the paper).

* Folder "vtm10.0-source-with-pmp-fast-alg" includes the source files of the VTM-10.0 implemented with the partition map-based fast algorithm.

* For the source files, you can start with the ".\Lib\CommonLib\TypeDef.h". In the "TypeDef.h", 

    - Macro definition "Save_Depth_fal" is related to the dataset creation. For dataset creation, there should be a folder named "DepthSaving" in the working directory. The saved partition information will be stored in the "DepthSaving" folder.

    - Macro definition "Partition_Map_Acceleration_fal" is related to the fast algorithm. 
	
	- I suggest using the "find all reference" function of visual studio to see how the implementation works. 
	
* Folder "demo" provides a VTM-10.0 codec implemented with the partition map-based fast partition algorithm.