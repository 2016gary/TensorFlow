# 记录本人成为TF boys的心路历程：
# CNN、RNN、GAN

---
# CNN
### CNN模型所使用的数据集：
    @misc{e-VDS,
      author = {Culurciello, Eugenio and Canziani, Alfredo},
      title = {{e-Lab} Video Data Set},
      howpublished = {\url{https://engineering.purdue.edu/elab/eVDS/}},
      year={2017}
    }

## TensorBoard：
## 原始数据集训练结果：
### Graph:
![](https://i.imgur.com/SAVk6O5.png)

### SCALARS:
![](https://i.imgur.com/BYdLgiE.png)

### IMAGES:
![](https://i.imgur.com/cmLOn1r.png)

### DISTRIBUTIONS:
![](https://i.imgur.com/O9Usde0.png)

### HISTOGRAMS:
![](https://i.imgur.com/IvcLASj.png)

### EMBEDDINGS:
![](https://i.imgur.com/fAHgRl4.png)



### （可以看到模型已经过拟合了，由于把笔记本当服务器跑，GPU烧坏了还没时间去修，所以精确度徘徊在70%，接下来做了一下数据增强，按道理讲应该能到80%，再使用regularization loss、dropout、调一下learning rate应该能到90%，本人的美好愿望。。。）