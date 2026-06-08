# 模式识别与机器学习正式样题参考答案

---

## 第 1 题 基本概念、泛化与 HMM

### 1. 学习类型判断

- 已有大量带诊断标签的病例，训练模型判断新病例类别：监督学习。
- 没有用户标签，仅根据购买行为把用户分群：无监督学习。
- 只有少量标注图像和大量未标注图像，希望提升分类器性能：半监督学习。
- 智能体在游戏环境中通过奖励信号学习策略：强化学习。

### 2. 过拟合、欠拟合与泛化

- 模型 A：训练错误率 2%，验证错误率 20%。训练集很好但验证集明显差，属于过拟合。
- 模型 B：训练错误率 25%，验证错误率 27%。训练和验证都差，属于欠拟合。
- 模型 C：训练错误率 8%，验证错误率 10%。训练和验证都较低且接近，泛化较好。

### 3. HMM 三个基本问题

隐马尔可夫模型的三个基本问题是：

1. 评估问题：已知模型参数和观测序列，求该观测序列出现的概率。
2. 解码问题：已知模型参数和观测序列，求最可能的隐藏状态序列。
3. 学习问题：已知观测序列，估计模型参数。

常见算法对应为：前向算法、Viterbi 算法、Baum-Welch 算法。

---

## 第 2 题 贝叶斯决策与高斯判别

### 1. 最小错误率贝叶斯决策规则

最小错误率贝叶斯决策选择后验概率最大的类别：

$$
\text{若 }P(\omega_1\mid x)>P(\omega_2\mid x),\text{ 则判为 }\omega_1;
$$

否则判为 $\omega_2$。

由于

$$
P(\omega_i\mid x)\propto p(x\mid\omega_i)P(\omega_i),
$$

也可以比较

$$
p(x\mid\omega_1)P(\omega_1)
\quad\text{和}\quad
p(x\mid\omega_2)P(\omega_2).
$$

### 2. 决策边界

因为两类先验相等，所以只需要比较两个类条件密度：

$$
p(x\mid\omega_1)
\quad\text{和}\quad
p(x\mid\omega_2).
$$

两类方差相同，密度分别为

$$
p(x\mid\omega_1)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2},
$$

$$
p(x\mid\omega_2)=\frac{1}{\sqrt{2\pi}}e^{-(x-2)^2/2}.
$$

决策边界满足

$$
\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
=
\frac{1}{\sqrt{2\pi}}e^{-(x-2)^2/2}.
$$

消去相同系数，并取对数：

$$
-\frac{x^2}{2}=-\frac{(x-2)^2}{2}.
$$

所以

$$
x^2=(x-2)^2.
$$

展开：

$$
x^2=x^2-4x+4.
$$

得到

$$
x=1.
$$

因此决策边界为 $x=1$。

### 3. 样本分类

两个高斯均值分别为 0 和 2，边界在中点 $x=1$。

- $x=0.3<1$，更靠近均值 0，判为 $\omega_1$。
- $x=1.7>1$，更靠近均值 2，判为 $\omega_2$。

### 4. 最小风险决策

在 $x=1$ 处，已知

$$
P(\omega_1\mid x)=P(\omega_2\mid x)=\frac12.
$$

若判为 $\omega_1$，条件风险为

$$
R(\alpha_1\mid x)
=0\cdot\frac12+4\cdot\frac12
=2.
$$

若判为 $\omega_2$，条件风险为

$$
R(\alpha_2\mid x)
=1\cdot\frac12+0\cdot\frac12
=\frac12.
$$

因为

$$
R(\alpha_2\mid x)<R(\alpha_1\mid x),
$$

所以按最小风险贝叶斯决策，应判为 $\omega_2$。

---

## 第 3 题 MLE、Parzen 窗、KNN 密度估计与 EM

### 1. 伯努利分布的最大似然估计

样本为

$$
1,0,1,1,0.
$$

其中取值为 1 的样本有 3 个，总样本数为 5。伯努利参数 $\theta=P(X=1)$ 的最大似然估计为

$$
\hat\theta_{MLE}=\frac{3}{5}.
$$

### 2. 方窗 Parzen 密度估计

一维 Parzen 窗估计为

$$
\hat p(x)=\frac{1}{Nh}\sum_{i=1}^N
\varphi\left(\frac{x-x_i}{h}\right).
$$

这里 $N=4,h=1,x=1$。

方窗满足

$$
\varphi(u)=1
\quad\Longleftrightarrow\quad
|u|\le\frac12.
$$

因为 $h=1$，所以要统计满足

$$
|1-x_i|\le\frac12
$$

的样本点。

逐个计算距离：

$$
|1-0.2|=0.8,
$$

$$
|1-0.9|=0.1,
$$

$$
|1-1.4|=0.4,
$$

$$
|1-2.0|=1.0.
$$

落入方窗的样本为 $0.9$ 和 $1.4$，共有 2 个。因此

$$
\hat p(1)=\frac{2}{4\times1}=\frac12.
$$

### 3. 高斯窗 Parzen 密度估计

高斯窗为

$$
\varphi(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}.
$$

代入 $N=4,h=1,x=1$：

$$
\hat p(1)
=\frac14\left[
\varphi(0.8)+\varphi(0.1)+\varphi(-0.4)+\varphi(-1.0)
\right].
$$

也就是

$$
\hat p(1)
=\frac{1}{4\sqrt{2\pi}}
\left[
e^{-0.8^2/2}
+e^{-0.1^2/2}
+e^{-0.4^2/2}
+e^{-1^2/2}
\right].
$$

### 4. KNN 密度估计

KNN 密度估计的一般形式为

$$
\hat p(x)=\frac{k}{NV}.
$$

其中 $V$ 是包含 $x$ 附近 $k$ 个近邻的区域体积。

对 $x=1$，到四个样本点的距离为

$$
0.8,\quad0.1,\quad0.4,\quad1.0.
$$

第 2 近邻距离为

$$
R=0.4.
$$

在一维情况下，以 $x=1$ 为中心、半径为 $R$ 的区间长度为

$$
V=2R=0.8.
$$

所以

$$
\hat p(1)=\frac{2}{4\times0.8}
=\frac{2}{3.2}
=0.625.
$$

### 5. EM 算法中 E 步和 M 步的含义

EM 是 Expectation-Maximization 的缩写。

- E 步：在当前参数下，估计隐变量的后验分布，或者计算完整数据对数似然的条件期望。
- M 步：固定 E 步得到的期望形式，重新最大化它，从而更新模型参数。

直观上，E 步是在“补全看不见的信息”，M 步是在“用补全后的信息重新估计参数”。二者交替进行，直到参数或似然函数基本稳定。

---

## 第 4 题 线性模型与感知器算法

### 1. 一轮感知器更新

初始：

$$
w=\begin{pmatrix}0\\0\end{pmatrix},\quad b=0.
$$

先看样本 $x_1=(2,1)^T,y_1=+1$：

$$
y_1(w^Tx_1+b)=1\times0=0.
$$

满足更新条件，所以

$$
w\leftarrow w+y_1x_1
=
\begin{pmatrix}0\\0\end{pmatrix}
+
\begin{pmatrix}2\\1\end{pmatrix}
=
\begin{pmatrix}2\\1\end{pmatrix},
$$

$$
b\leftarrow b+y_1=1.
$$

更新后：

$$
w=\begin{pmatrix}2\\1\end{pmatrix},\quad b=1.
$$

再看样本 $x_2=(1,2)^T,y_2=-1$：

$$
w^Tx_2+b
=
\begin{pmatrix}2&1\end{pmatrix}
\begin{pmatrix}1\\2\end{pmatrix}
+1
=2+2+1=5.
$$

所以

$$
y_2(w^Tx_2+b)=-1\times5=-5\le0.
$$

继续更新：

$$
w\leftarrow w+y_2x_2
=
\begin{pmatrix}2\\1\end{pmatrix}
-
\begin{pmatrix}1\\2\end{pmatrix}
=
\begin{pmatrix}1\\-1\end{pmatrix},
$$

$$
b\leftarrow b+y_2=1-1=0.
$$

一轮更新后：

$$
w=\begin{pmatrix}1\\-1\end{pmatrix},\quad b=0.
$$

### 2. 判断点 $x=(1,0)^T$

$$
w^Tx+b=
\begin{pmatrix}1&-1\end{pmatrix}
\begin{pmatrix}1\\0\end{pmatrix}
+0
=1.
$$

因为结果大于 0，所以判为正类 $+1$。

### 3. 逻辑回归模型、交叉熵和梯度

二分类逻辑回归写作

$$
p=P(y=1\mid x)=\sigma(w^Tx+b)
=\frac{1}{1+e^{-(w^Tx+b)}}.
$$

其中 $y\in\{0,1\}$。单样本交叉熵损失为

$$
\ell(w,b)=-y\log p-(1-y)\log(1-p).
$$

损失对 $w$ 的梯度为

$$
\frac{\partial \ell}{\partial w}=(p-y)x.
$$

对偏置项的梯度为

$$
\frac{\partial \ell}{\partial b}=p-y.
$$

---

## 第 5 题 K-均值聚类

### 1. 第一次样本分配

初始中心为

$$
\mu_1^{(0)}=(0,0),\quad \mu_2^{(0)}=(5,5).
$$

显然前三个点

$$
(0,0),(0,2),(1,1)
$$

更靠近 $\mu_1^{(0)}$，后三个点

$$
(5,5),(6,5),(5,6)
$$

更靠近 $\mu_2^{(0)}$。

因此第一次分配为

$$
C_1=\{(0,0),(0,2),(1,1)\},
$$

$$
C_2=\{(5,5),(6,5),(5,6)\}.
$$

### 2. 更新聚类中心

对 $C_1$：

$$
\mu_1^{(1)}
=\frac{(0,0)+(0,2)+(1,1)}{3}
=\left(\frac13,1\right).
$$

对 $C_2$：

$$
\mu_2^{(1)}
=\frac{(5,5)+(6,5)+(5,6)}{3}
=\left(\frac{16}{3},\frac{16}{3}\right).
$$

### 3. 计算 SSE

对 $C_1$：

$$
\|(0,0)-(\frac13,1)\|^2
=\frac19+1=\frac{10}{9},
$$

$$
\|(0,2)-(\frac13,1)\|^2
=\frac19+1=\frac{10}{9},
$$

$$
\|(1,1)-(\frac13,1)\|^2
=\frac49.
$$

所以

$$
SSE_1=\frac{10}{9}+\frac{10}{9}+\frac49
=\frac{24}{9}
=\frac83.
$$

对 $C_2$：

$$
\|(5,5)-(\frac{16}{3},\frac{16}{3})\|^2
=\frac19+\frac19=\frac29,
$$

$$
\|(6,5)-(\frac{16}{3},\frac{16}{3})\|^2
=\frac49+\frac19=\frac59,
$$

$$
\|(5,6)-(\frac{16}{3},\frac{16}{3})\|^2
=\frac19+\frac49=\frac59.
$$

所以

$$
SSE_2=\frac29+\frac59+\frac59
=\frac{12}{9}
=\frac43.
$$

总 SSE 为

$$
SSE=SSE_1+SSE_2
=\frac83+\frac43
=4.
$$

### 4. K-均值目标函数

K-均值聚类的目标是最小化簇内平方误差和：

$$
J=\sum_{j=1}^{K}\sum_{x_i\in C_j}\|x_i-\mu_j\|^2.
$$

其中 $\mu_j$ 是第 $j$ 个簇的中心。算法交替进行两步：

1. 固定中心，把样本分给最近的中心；
2. 固定分配，用每个簇内样本均值更新中心。

---

## 第 6 题 PCA、LDA 与特征选择

### 1. PCA 主成分方向

协方差矩阵为

$$
\Sigma=
\begin{pmatrix}
2&0\\
0&1
\end{pmatrix}.
$$

它已经是对角矩阵。特征值和特征向量为：

$$
\lambda_1=2,\quad v_1=\begin{pmatrix}1\\0\end{pmatrix},
$$

$$
\lambda_2=1,\quad v_2=\begin{pmatrix}0\\1\end{pmatrix}.
$$

第一主成分方向是最大特征值对应的特征向量：

$$
v_1=(1,0)^T.
$$

第二主成分方向为

$$
v_2=(0,1)^T.
$$

### 2. 保留的方差比例

总方差为

$$
\lambda_1+\lambda_2=2+1=3.
$$

只保留第一主成分时，保留方差比例为

$$
\frac{\lambda_1}{\lambda_1+\lambda_2}
=\frac{2}{3}.
$$

### 3. 投影与重构

样本

$$
x=\begin{pmatrix}2\\1\end{pmatrix}.
$$

投影到第一主成分上：

$$
z=v_1^Tx=
\begin{pmatrix}1&0\end{pmatrix}
\begin{pmatrix}2\\1\end{pmatrix}
=2.
$$

只用第一主成分重构：

$$
\hat x=zv_1
=2\begin{pmatrix}1\\0\end{pmatrix}
=\begin{pmatrix}2\\0\end{pmatrix}.
$$

### 4. LDA 投影方向

两类 LDA 的投影方向满足

$$
w\propto S_w^{-1}(\mu_1-\mu_2).
$$

这里

$$
S_w=I_2,\quad
\mu_1-\mu_2=
\begin{pmatrix}0\\0\end{pmatrix}
-
\begin{pmatrix}2\\0\end{pmatrix}
=
\begin{pmatrix}-2\\0\end{pmatrix}.
$$

因此

$$
w\propto
\begin{pmatrix}-2\\0\end{pmatrix}.
$$

方向的正负不影响一维投影的判别能力，所以也可以写成

$$
w\propto
\begin{pmatrix}1\\0\end{pmatrix}.
$$

### 5. 特征选择方法举例

- 过滤式特征选择：先按统计指标给特征打分，再选特征。例如方差筛选、相关系数、互信息、卡方检验。
- 包裹式特征选择：把模型训练效果作为评价标准，反复尝试不同特征子集。例如递归特征消除 RFE。
- 嵌入式特征选择：特征选择发生在模型训练过程中。例如 L1 正则化逻辑回归、Lasso、决策树基于分裂自动选择特征。

---

## 第 7 题 支持向量机与核方法

### 1. 硬间隔 SVM 原问题

硬间隔 SVM 的原问题为

$$
\min_{w,b}\frac12w^2
$$

满足

$$
y_i(wx_i+b)\ge1,\quad i=1,2.
$$

### 2. 求最优 $w,b$

对正类样本 $x_2=2,y_2=+1$：

$$
2w+b\ge1.
$$

对负类样本 $x_1=-2,y_1=-1$：

$$
-1\cdot(w(-2)+b)\ge1.
$$

即

$$
2w-b\ge1.
$$

所以约束为

$$
2w+b\ge1,\quad 2w-b\ge1.
$$

等价于

$$
b\ge1-2w,\quad b\le2w-1.
$$

要存在可行的 $b$，需要

$$
1-2w\le2w-1.
$$

所以

$$
4w\ge2,\quad w\ge\frac12.
$$

目标函数为

$$
\frac12w^2.
$$

为了最小化它，取最小可行值

$$
w^\star=\frac12.
$$

此时

$$
b\ge1-2\cdot\frac12=0,
$$

$$
b\le2\cdot\frac12-1=0.
$$

所以

$$
b^\star=0.
$$

### 3. 决策边界和间隔边界

最终决策函数为

$$
f(x)=\frac12x.
$$

决策边界满足

$$
\frac12x=0,
$$

所以

$$
x=0.
$$

两条间隔边界满足

$$
wx+b=1
\quad\text{和}\quad
wx+b=-1.
$$

代入 $w=\frac12,b=0$：

$$
\frac12x=1 \Rightarrow x=2,
$$

$$
\frac12x=-1 \Rightarrow x=-2.
$$

所以间隔边界为

$$
x=2,\quad x=-2.
$$

### 4. 支持向量

支持向量是满足

$$
y_i(w^\star x_i+b^\star)=1
$$

的样本。

对 $x_1=-2,y_1=-1$：

$$
y_1(w^\star x_1+b^\star)
=-1\cdot\left(\frac12(-2)+0\right)
=1.
$$

对 $x_2=2,y_2=+1$：

$$
y_2(w^\star x_2+b^\star)
=1\cdot\left(\frac12(2)+0\right)
=1.
$$

因此两个样本都是支持向量。

### 5. 常见核函数与核技巧作用

常见核函数包括：

线性核：

$$
K(x,z)=x^Tz.
$$

多项式核：

$$
K(x,z)=(x^Tz+c)^d.
$$

高斯核，也称 RBF 核：

$$
K(x,z)=\exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right).
$$

核技巧的作用是：在 SVM 的对偶问题中，样本只以内积 $x_i^Tx_j$ 的形式出现。把内积替换成核函数 $K(x_i,x_j)$，就相当于在不显式写出高维映射 $\phi(x)$ 的情况下，计算高维空间中的内积：

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j).
$$

这样可以用线性 SVM 的对偶形式处理非线性分类问题。

---

## 第 8 题 决策树、集成学习与综合应用

### 1. 划分前类别熵

总共有 6 个样本，正类 4 个、负类 2 个。所以

$$
P(+)=\frac46=\frac23,\quad P(-)=\frac26=\frac13.
$$

类别熵为

$$
H(Y)
=-\frac23\log_2\frac23-\frac13\log_2\frac13.
$$

数值约为

$$
H(Y)\approx0.918.
$$

### 2. 条件熵与信息增益

当 $A=0$ 时，有 3 个样本，其中正类 1 个、负类 2 个：

$$
H(Y\mid A=0)
=-\frac13\log_2\frac13-\frac23\log_2\frac23
\approx0.918.
$$

当 $A=1$ 时，有 3 个样本，其中正类 3 个、负类 0 个：

$$
H(Y\mid A=1)=0.
$$

由于 $A=0$ 和 $A=1$ 各有 3 个样本，所以

$$
H(Y\mid A)
=\frac36H(Y\mid A=0)+\frac36H(Y\mid A=1)
$$

$$
=\frac12\times0.918+\frac12\times0
=0.459.
$$

信息增益为

$$
Gain(Y,A)=H(Y)-H(Y\mid A)
=0.918-0.459
=0.459.
$$

### 3. ID3 的特征选择

ID3 优先选择信息增益更大的特征。

因为

$$
Gain(Y,A)=0.459>Gain(Y,B)=0.10,
$$

所以优先选择特征 $A$。

### 4. 随机森林的优势

随机森林由多棵决策树组成，通常使用 bootstrap 采样和随机特征子集。相比单棵决策树，它的主要优势是：

- 降低方差，缓解单棵树容易过拟合的问题；
- 对噪声和样本扰动更稳健；
- 可以输出特征重要性，便于分析。

### 5. 门禁人脸识别可行方案

一种可行方案是：

1. 对采集图像做人脸检测、对齐、裁剪、归一化。
2. 使用预训练人脸识别网络提取人脸特征向量。
3. 注册阶段为每位职员保存一个或多个模板，例如把该职员多张注册图像的特征向量取平均得到模板。
4. 识别阶段提取待识别人脸特征，与各职员模板计算余弦相似度或欧氏距离。
5. 若最高相似度超过阈值，则输出对应职员；否则判为未知人员或进入人工复核。

这个方案通常不需要从零训练网络；在数据充足时，也可以对预训练网络或分类器进行微调。

