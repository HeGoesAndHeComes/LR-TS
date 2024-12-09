# LR-TS
**本代码库是论文 "LR-TS: A Research Method for Long-term Action Prediction Focusing on Local Motion Feature Processing"的代码实现**
## 运行前的一些准备

需要将数据集放入到datasets文件夹下，数据集可以在"https://zenodo.org/records/3625992#.Xiv9jGhKhPY"链接下获得。
这里的数据集为利用I3D提取的Breakfast数据集和50Salads数据集的特征。文件存放路径格式应为：

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
## 运行
在Linux中或在bash命令窗口执行scripts中的bash命令文件，即可开始训练，训练参数可以自行调整
