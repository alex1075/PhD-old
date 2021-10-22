
This is the repository regarding the PhD of Alexander Hunt at the University of Edinburgh

# Requirements

Python3, gcc/glang and bash required 
  Nvidia CUDA GPU or AMD RadeonOpenCompute GPU prefereable unless using Apple ARM SoC
    , Intel CPU also prefered over AMD (`SSE4.1` `SSE4.2` `AVX` `AVX2` `FMA`)

# Install 

```bash 
$ conda create -n PhD python=3.9 -y
$ conda activate PhD
$ conda install -c apple tensorflow-deps -y
$ conda install opencv-python
$ conda install matplotlib
$ conda install scikit-learn
$ pip install --no-dependencies imgaug
$ pip install -r requirements-$platform$.txt
```
Double check $pip list with requiem.txt to check all has been installed successfully.


# Roadmap

 - [ ] basic data manipulation
	- [x] image conversion
	- [ ] image renaming
	- [x] image resizing 
	- [x] folder creation
	- [x] video to image conversion
 - [ ] recognition of sample components
	- [ ] Autolabelling labelling of subjects within an image
		- [x] Implementation of image manipulation
			- [x] Blurr
			- [x] Brighness 
			- [x] Contrast
		- [ ] Implementaton of cyclegan for image manipulation
			- [x] Getting all the packages to work together
			- [ ] Getting the cycleGan to work 
		- [ ] auto-cropping of subjects out of original image
			- [ ] TBD
		- [ ] use of other ML technique to automatically sort out the cropped images
			- [ ] TBD
	- [ ] test with different architectures of ML
		- [ ] SMV
		- [ ] Random forests
		- [ ] image classification
		- [ ] object detection 
		- [ ] metadata analysis 
 - [ ] enumeration of components within field of view
	- [ ] use of IDs and bash scripting for calculation (may migrate to cpp)
 - [ ] detection of signs of infection (TBD)

