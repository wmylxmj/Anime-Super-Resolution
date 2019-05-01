# Anime-Super-Resolution
动漫图片超分辨率——基于WDSR (2019-4-30)
#### 实现动漫图片4倍的图片放大及超分辨率

- utils.py -- 图像降采样与数据导入
- model.py -- wdsr模型
- optimizer.py -- 权重归一化Adam优化器
- train.py -- 模型训练
- predict.py -- 测试集预测
- evaluate.py -- 在不同难度等级下测试网络表现

#### 图像降采样
- 使用NEAREST, BICUBIC, BILINEAR, HAMMING, LANCZOS插值方法将图像缩减4~12倍
- 使用半径1~3的核进行高斯模糊

### Demo(LR, SR)
<div align="center">
  <img src="images/demo_lr_1.jpg" width="420" >
  <img src="images/demo_sr_1.jpg" width="420" >
  <img src="images/demo_lr_2.jpg" width="420" >
  <img src="images/demo_sr_2.jpg" width="420" >
  <img src="images/demo_lr_3.jpg" width="420" >
  <img src="images/demo_sr_3.jpg" width="420" >
  <img src="images/demo_lr_4.jpg" width="420" >
  <img src="images/demo_sr_4.jpg" width="420" >
</div>

### 难度测试

#### Easy（缩减倍率 4 ~ 6，模糊半径 1 ~ 1.5）
<div align="center">
  <img src="images/easy_lr.jpg" width="420" >
  <img src="images/easy_sr.jpg" width="420" >
</div>

#### Normal（缩减倍率 6 ~ 8，模糊半径 1.5 ~ 2）
<div align="center">
  <img src="images/normal_lr.jpg" width="420" >
  <img src="images/normal_sr.jpg" width="420" >
</div>

#### Hard（缩减倍率 8 ~ 10，模糊半径 2 ~ 2.5）
<div align="center">
  <img src="images/hard_lr.jpg" width="420" >
  <img src="images/hard_sr.jpg" width="420" >
</div>

#### Lunatic（缩减倍率 10 ~ 12，模糊半径 2.5 ~ 3）
<div align="center">
  <img src="images/lunatic_lr.jpg" width="420" >
  <img src="images/lunatic_sr.jpg" width="420" >
</div>

### 测试集表现
<div align="center">
  <img src="outputs/lr_22.jpg" width="420" >
  <img src="outputs/sr_22.jpg" width="420" >
</div>

<div align="center">
  <img src="outputs/lr_11.jpg" width="420" >
  <img src="outputs/sr_11.jpg" width="420" >
</div>
