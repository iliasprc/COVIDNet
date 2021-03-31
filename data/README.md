<!-- Datasets -->
## Datasets
### 1) COVID-CT-Dataset

The COVID-CT-Dataset has 288 CT images containing clinical findings of COVID-19. We are continuously adding more COVID CTs.

The images are collected from medRxiv and bioRxiv papers about COVID-19. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to medRxiv and bioRxiv.

Please refer to the preprint for details: COVID-CT-Dataset: A CT Scan Dataset about COVID-19

### 2) COVIDx  dataset 


The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

We especially thank the Radiological Society of North America and others involved in the RSNA Pneumonia Detection Challenge, and Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project, for making data available to the global community.

### Steps to generate the COVIDx dataset

Download the datasets listed above
 * `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
 * go to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to download the RSNA pneumonia dataset
2. Create a `data` directory and within the data directory, create a `train` and `test` directory
3. Use [COVIDNet.ipynb](COVIDNet.ipynb) to combine the two dataset to create COVIDx. Make sure to remember to change the file paths.
4. We provide the train and test txt files with patientId, image path and label (normal, pneumonia or COVID-19). The description for each file is explained below:
 * [train\_COVIDx.txt](train_COVIDx.txt): This file contains the samples used for training.
 * [test\_COVIDx.txt](test_COVIDx.txt): This file contains the samples used for testing.


COVIDx  dataset  Chest radiography images distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    66    | 16546 |
|  test |   100  |     100   |    10    |   210 |


