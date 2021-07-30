
This is the repository regarding the PhD of Alexander Hunt at the University of Edinburgh

# Requirements

Python3, gcc/glang and bash required 
  Nvidia CUDA GPU or AMD RadeonOpenCompute GPU prefereable unless using Apple M ARM CPU
    Intel CPU also prefered over AMD (`SSE4.1` `SSE4.2` `AVX` `AVX2` `FMA`)

# Install 

```bash
$ conda create -n env tensorflow-gpu==2.4.1
$ conda activate env
$ pip install -r requirements.txt
# Or use this auto setup script to install it for you (optional)
$ sudo chmod +x installer.sh
$ ./installer.sh
```
# Roadmap

 - [ ] basic data manipulation
	- [x] image conversion
	- [ ] image renaming
	- [x] image resizing 
	- [ ] autolabelling 
 - [ ] recognition of sample components
	- [ ] test with different architectures of ML
		- [ ] SMV
		- [ ] Random forests
		- [ ] image classification
		- [ ] object detection 
		- [ ] metadata analysis 
 - [ ] enumeration of components within field of view
	- [ ] use of IDs and bash scripting for calculation (may migrate to cpp)
 - [ ] detection of signs of infection (TBD)

