# LR-TS
**This code repository is the implementation of "LR-TS: A Research Method for Long-term Action Prediction Focusing on Local Motion Feature Processing".**
# Algorithm description
Due to the inefficiency and computational limitations of traditional Transformer architectures in handling short-term or local action details, their performance can be significantly affected when actions change rapidly within a short duration or involve multiple complex variations. Inspired by the success of global and local feature processing in the field of image processing, this paper proposes an LR-TS model based on the Transformer Encoder-Decoder architecture to address these inefficiencies and computational challenges. 
The attention module has been innovatively redesigned.In the attention mechanism, we introduce a local window attention mechanism to capture relationships between local video frames. This mechanism is integrated with the multi-head self-attention mechanism, enabling the model to preserve global contextual awareness while effectively processing local details. For positional encoding, we adopt a rotary positional encoding strategy to enhance the model’s ability to encode local features and improve spatial awareness.
Extensive experiments on the Breakfast and 50Salads datasets demonstrate that our proposed model achieves significant performance improvements compared to baseline models.
# Environment setup
Our experiments are conducted on a Tesla V100-PCIE-32GB GPU.
The code runs in an environment with Pytorch==2.1.2 and python==3.8. Use the following commands to install the dependencies:
   ```txt
   conda create --name env_name --file path_of_requirements.txt
   ```
# Datasets
The dataset can be obtained from the following link: [https://zenodo.org/records/3625992](https://zenodo.org/records/3625992).
# Dataset storage path
The dataset consists of features extracted from the Breakfast and 50Salads datasets using I3D. It should be placed under the "datasets" folder, ensuring the folder structure and file paths are as follows:
   ```txt
   datasets
   ->breakfast
   -->features --->featuresFiles.npy
   -->groundTruth --->gtFiles.txt
   -->splits   --->filesSplit.bundle
   -->mapping.txt

   ->50Salads
   -->features --->featuresFiles.npy
   -->groundTruth --->gtFiles.txt
   -->splits   --->filesSplit.bundle
   -->mapping.txt
   ```
#  Calibration & Prediction
If the code is deployed on a Linux system, you can simply run the following command. 
However, if you attempt to debug the code on a Windows system using the VSCode editor, you need to create a terminal window and enter the following command in the bash window to run the code.
## Breakfast
### Train
   ```txt
   ./scripts/bf_train.sh 
   ```
### Predict
   ```txt
   ./scripts/bf_predict.sh 
   ```
## 50Salads
### Train
   ```txt
   ./scripts/50s_train.sh 
   ```
### Predict
   ```txt
   ./scripts/50s_predict.sh 
   ```
The parameter tuning for the code can be done by modifying the parameter values in the bash file.
