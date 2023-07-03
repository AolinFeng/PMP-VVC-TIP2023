This demo is for the paper: 
A. Feng, K. Liu, D. Liu, L. Li and F. Wu, "Partition Map Prediction for Fast Block Partitioning in VVC Intra-Frame Coding," in IEEE Transactions on Image Processing, vol. 32, pp. 2237-2251, 2023, doi: 10.1109/TIP.2023.3266165.

* This demo provides a VTM-10.0 codec implemented with the partition map-based fast partition algorithm.
* The partition information output by the post-processing algorithm should be stored in the folder "PartitionMat". The folder "PartitionMat" should be in the working directory.
* The referenced command line: 

	- Encoding:
	```
	.\EncoderApp_FastPartition100L3.exe -c .\mycfg\RaceHorses_416x240_30.cfg -c .\cfg\encoder_intra_vtm.cfg -v 6 -f 1 -ts 8 -q 27 -b .\out_data\enc_out.bin -o .\out_data\enc_rec.yuv --SEIDecodedPictureHash=1 > .\out_data\Enc_Info.log
	```
	- Decoding:
	```
	.\DecoderApp_FastPartition100L3.exe -b .\out_data\enc_out.bin -o .\out_data\dec_rec.yuv >.\out_data\Dec_Info.log
	```