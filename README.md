# Coding task
## Logo detector: 
Detects bosch and benz logo symbols in images.

### Requirements:
python: > 3.6 <br />
opencv <br />
numpy <br />
skimage <br />
pyyaml <br />
matplotlib <br />

### Quickly create an anaconda environment with all dependencies
Run <br />
conda env create -f environment.yml <br />
The yaml file is available [here](https://github.com/NareshGuru77/logo_detector/blob/master/environment.yml). The environment name is "logo_det".

### Running the code
python logo_detector.py [-h] [--yaml_path YAML_PATH]

### Changing parameters
The parameter and their values are listed in [parameters.yml](https://github.com/NareshGuru77/logo_detector/blob/master/parameters.yml). The parameter values can be updated in the yaml.

### About the detector
The detector works in two stages. <br />
1. Region proposals generation.
2. Region classification.

#### 1. Region proposals generation
A set of square regions where a circle exists in the input image is detected and proposed as potential regions where the logo could be found. <br />
An example where the detected circles are shown in pink and the proposed square regions are shown in green:

* **Circle proposals**<br/><br/>
![Circle proposals](https://github.com/NareshGuru77/logo_detector/blob/master/results/region_proposals.jpg)

#### 2. Region classification
Each of the proposed regions is compared with a template image and a score is calculated using ssim (Structural Similarity Index). Regions with scores above a certain threshold are classified as regions with the required logo. <br />
Region proposal            , SSIM,   &nbsp;&nbsp;&nbsp;&nbsp; Template, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Result <br />
:---------------------:|:------:|:--------------------:|:-----------------------------: <br />
<img src="https://github.com/NareshGuru77/logo_detector/blob/master/results/region_w.jpg" width="100" height="100"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/NareshGuru77/logo_detector/blob/master/results/template_w.jpg" width="100" height="100"> < threshold : ignored <br />
<br />
<img src="https://github.com/NareshGuru77/logo_detector/blob/master/results/region_c.jpg" width="100" height="100"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/NareshGuru77/logo_detector/blob/master/results/template_c.jpg" width="100" height="100"> > threshold : detection

#### Results
The detected logos are indicated with green squares and the ssim score is written inside the square in brown.
##### 1. Bosch logo
![Circle proposals](https://github.com/NareshGuru77/logo_detector/blob/master/results/bosch_result.jpg)

##### 2. Benz logo
![Circle proposals](https://github.com/NareshGuru77/logo_detector/blob/master/results/benz_result.jpg)

#### Limitations
This is a limited example with a naive approach. Several limitations could be identified. Some limitations are listed here: <br />
1. Only the front facing viewpoint of the logos can be detected.
2. Different lightings, brightness/contrast adjustments will lead to failures.
3. Only logos in silver color as in template image is likely to be detected.
4. Finding the right set of parameter values is difficult. For instance, bosch logo in bosch_1.jpg and bosch_2.jpg will be detected if elliptical kernel size is changed to 5, 5 (anchor parameter in parameters.yml). This same change, will also avoid false positives in benz_1.jpg and benz_4.jpg.

