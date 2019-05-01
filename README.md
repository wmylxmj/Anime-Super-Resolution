# Anime-Super-Resolution
动漫图片超分辨率——基于WDSR (2019-4-30)
- utils.py -- 图像降采样与数据导入
- model.py -- wdsr模型
- optimizer.py -- 权重归一化Adam优化器
- train.py -- 模型训练
- predict.py -- 测试集预测
- evaluate.py -- 在不同难度等级下测试网络表现
### 难度测试
#### Easy
<div align="center">
  <img src="images/easy_lr.jpg" width="420" >
  <img src="images/easy_sr.jpg" width="420" >
</div>

#### Normal
<div align="center">
  <img src="images/normal_lr.jpg" width="420" >
  <img src="images/normal_sr.jpg" width="420" >
</div>

#### Hard
<div align="center">
  <img src="images/hard_lr.jpg" width="420" >
  <img src="images/hard_sr.jpg" width="420" >
</div>

#### Lunatic
<div align="center">
  <img src="images/lunatic_lr.jpg" width="420" >
  <img src="images/lunatic_sr.jpg" width="420" >
</div>

### 测试图片放大
![](outputs/lr_10.jpg)
![](outputs/sr_10.jpg)
![](outputs/lr_11.jpg)
![](outputs/sr_11.jpg)
