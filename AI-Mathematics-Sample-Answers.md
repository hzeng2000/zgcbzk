# 人工智能数学基础样题参考答案

## 第 1 题 线性代数基础

已知

$$
u_1=\begin{pmatrix}1\\1\\0\end{pmatrix},\quad
u_2=\begin{pmatrix}1\\-1\\2\end{pmatrix},\quad
v=\begin{pmatrix}2\\0\\1\end{pmatrix}
$$

令 $U=\operatorname{span}\{u_1,u_2\}$。

### (a) 判断 $u_1$ 与 $u_2$ 是否正交

计算内积：

$$
u_1^Tu_2=1\cdot1+1\cdot(-1)+0\cdot2=0
$$

因此 $u_1$ 与 $u_2$ 正交。

### (b) 单位化并写出标准正交向量组

先计算长度：

$$
\lVert u_1\rVert=\sqrt{1^2+1^2+0^2}=\sqrt2
$$

$$
\lVert u_2\rVert=\sqrt{1^2+(-1)^2+2^2}=\sqrt6
$$

单位化得到：

$$
e_1=\frac{u_1}{\lVert u_1\rVert}
=\begin{pmatrix}
\frac{1}{\sqrt2}\\[2mm]
\frac{1}{\sqrt2}\\[2mm]
0
\end{pmatrix}
$$

$$
e_2=\frac{u_2}{\lVert u_2\rVert}
=\begin{pmatrix}
\frac{1}{\sqrt6}\\[2mm]
-\frac{1}{\sqrt6}\\[2mm]
\frac{2}{\sqrt6}
\end{pmatrix}
$$

因为原来的 $u_1,u_2$ 已经正交，所以分别单位化后，$e_1,e_2$ 就是一组标准正交向量组。

### (c) 求 $v$ 在 $U$ 上的正交投影

因为 $u_1,u_2$ 正交，所以可以直接使用：

$$
\operatorname{proj}_U(v)
=\frac{v^Tu_1}{u_1^Tu_1}u_1+\frac{v^Tu_2}{u_2^Tu_2}u_2
$$

分别计算：

$$
v^Tu_1=2\cdot1+0\cdot1+1\cdot0=2,\quad
u_1^Tu_1=2
$$

$$
v^Tu_2=2\cdot1+0\cdot(-1)+1\cdot2=4,\quad
u_2^Tu_2=6
$$

所以：

$$
\operatorname{proj}_U(v)=1\cdot u_1+\frac{2}{3}u_2
$$

$$
=\begin{pmatrix}1\\1\\0\end{pmatrix}
+\frac{2}{3}
\begin{pmatrix}1\\-1\\2\end{pmatrix}
=
\begin{pmatrix}
\frac{5}{3}\\[1mm]
\frac{1}{3}\\[1mm]
\frac{4}{3}
\end{pmatrix}
$$

最终答案：

$$
\boxed{
\operatorname{proj}_U(v)=
\begin{pmatrix}
\frac{5}{3}\\[1mm]
\frac{1}{3}\\[1mm]
\frac{4}{3}
\end{pmatrix}}
$$

## 第 2 题 概率论基础

概率密度函数为

$$
f(x)=
\begin{cases}
cx, & 0<x<1,\\
c(2-x), & 1\le x<2,\\
0, & \text{其他}.
\end{cases}
$$

### (a) 求 $c$ 与 $P(\frac12<X<\frac32)$

概率密度函数在全体实数上的积分必须等于 1：

$$
\int_0^1 cx\,dx+\int_1^2 c(2-x)\,dx=1
$$

$$
\frac{c}{2}+\frac{c}{2}=1
$$

所以：

$$
c=1
$$

因此

$$
f(x)=
\begin{cases}
x, & 0<x<1,\\
2-x, & 1\le x<2,\\
0, & \text{其他}.
\end{cases}
$$

计算概率：

$$
P\left(\frac12<X<\frac32\right)
=\int_{1/2}^{1}x\,dx+\int_1^{3/2}(2-x)\,dx
$$

$$
\int_{1/2}^{1}x\,dx
=\left.\frac{x^2}{2}\right|_{1/2}^{1}
=\frac12-\frac18
=\frac38
$$

$$
\int_1^{3/2}(2-x)\,dx
=\left.\left(2x-\frac{x^2}{2}\right)\right|_1^{3/2}
=\frac38
$$

所以：

$$
\boxed{
P\left(\frac12<X<\frac32\right)=\frac34}
$$

### (b) 求 $E[X]$ 与 $\operatorname{Var}(X)$

期望：

$$
E[X]=\int_{-\infty}^{+\infty}xf(x)\,dx
$$

$$
E[X]=\int_0^1x^2\,dx+\int_1^2x(2-x)\,dx
$$

$$
=\frac13+\int_1^2(2x-x^2)\,dx
$$

$$
=\frac13+\frac23=1
$$

所以：

$$
\boxed{E[X]=1}
$$

方差：

$$
\operatorname{Var}(X)=E[X^2]-(E[X])^2
$$

先求 $E[X^2]$：

$$
E[X^2]=\int_0^1x^3\,dx+\int_1^2x^2(2-x)\,dx
$$

$$
=\frac14+\int_1^2(2x^2-x^3)\,dx
$$

$$
=\frac14+\frac{11}{12}=\frac76
$$

因此：

$$
\operatorname{Var}(X)=\frac76-1^2=\frac16
$$

最终答案：

$$
\boxed{\operatorname{Var}(X)=\frac16}
$$

## 第 3 题 概率论基础

已知：

$$
P(S_A)=0.6,\quad P(S_B)=0.4
$$

$$
P(\text{错}\mid S_A)=0.1,\quad P(\text{错}\mid S_B)=0.2
$$

### (a) 求随机抽到一个样本被错误分类的概率

用全概率公式：

$$
P(\text{错})
=P(S_A)P(\text{错}\mid S_A)+P(S_B)P(\text{错}\mid S_B)
$$

$$
=0.6\cdot0.1+0.4\cdot0.2=0.14
$$

答案：

$$
\boxed{0.14}
$$

### (b) 求 $P(X=0)$

$X=0$ 表示两个样本都被正确分类。

来自数据源 A 时：

$$
P(X=0\mid S_A)=0.9^2=0.81
$$

来自数据源 B 时：

$$
P(X=0\mid S_B)=0.8^2=0.64
$$

所以：

$$
P(X=0)=0.6\cdot0.81+0.4\cdot0.64=0.742
$$

答案：

$$
\boxed{P(X=0)=0.742}
$$

### (c) 求 $P(S_B\mid X=1)$

先计算：

$$
P(X=1\mid S_A)=2\cdot0.1\cdot0.9=0.18
$$

$$
P(X=1\mid S_B)=2\cdot0.2\cdot0.8=0.32
$$

再用全概率公式求 $P(X=1)$：

$$
P(X=1)=0.6\cdot0.18+0.4\cdot0.32=0.236
$$

由贝叶斯公式：

$$
P(S_B\mid X=1)=\frac{P(X=1\mid S_B)P(S_B)}{P(X=1)}
$$

$$
=\frac{0.32\cdot0.4}{0.236}
=\frac{0.128}{0.236}
=\frac{32}{59}
$$

答案：

$$
\boxed{P(S_B\mid X=1)=\frac{32}{59}\approx0.5424}
$$

### (d) 求 $E[X]$

来自数据源 A 时，两个样本中错误个数的期望为：

$$
E[X\mid S_A]=2\cdot0.1=0.2
$$

来自数据源 B 时：

$$
E[X\mid S_B]=2\cdot0.2=0.4
$$

所以：

$$
E[X]=P(S_A)E[X\mid S_A]+P(S_B)E[X\mid S_B]
$$

$$
=0.6\cdot0.2+0.4\cdot0.4=0.28
$$

答案：

$$
\boxed{E[X]=0.28}
$$

## 第 4 题 最优化基础

目标函数：

$$
f(x_1,x_2)=\frac12(x_1+x_2-2)^2+\frac12(2x_1-x_2)^2
$$

令

$$
r_1=x_1+x_2-2,\quad r_2=2x_1-x_2
$$

则

$$
f(x_1,x_2)=\frac12r_1^2+\frac12r_2^2
$$

### (a) 求梯度与 Hessian 矩阵

对 $x_1$ 求偏导：

$$
\frac{\partial f}{\partial x_1}
=r_1\frac{\partial r_1}{\partial x_1}
+r_2\frac{\partial r_2}{\partial x_1}
=r_1+2r_2
$$

$$
=(x_1+x_2-2)+2(2x_1-x_2)
=5x_1-x_2-2
$$

对 $x_2$ 求偏导：

$$
\frac{\partial f}{\partial x_2}
=r_1\frac{\partial r_1}{\partial x_2}
+r_2\frac{\partial r_2}{\partial x_2}
=r_1-r_2
$$

$$
=(x_1+x_2-2)-(2x_1-x_2)
=-x_1+2x_2-2
$$

因此：

$$
\nabla f(x)=
\begin{pmatrix}
5x_1-x_2-2\\
-x_1+2x_2-2
\end{pmatrix}
$$

Hessian 矩阵为：

$$
\nabla^2f(x)=
\begin{pmatrix}
5 & -1\\
-1 & 2
\end{pmatrix}
$$

### (b) 判断是否为凸函数

Hessian 矩阵为常数矩阵：

$$
H=
\begin{pmatrix}
5 & -1\\
-1 & 2
\end{pmatrix}
$$

判断正定性：

$$
\Delta_1=5>0
$$

$$
\Delta_2=\det(H)=5\cdot2-(-1)(-1)=9>0
$$

两个顺序主子式都大于 0，所以 $H$ 正定。因此 $f$ 是严格凸函数，最优解唯一。

### (c) 求最优解

无约束可微凸优化问题的最优解满足：

$$
\nabla f(x)=0
$$

即：

$$
\begin{cases}
5x_1-x_2-2=0,\\
-x_1+2x_2-2=0.
\end{cases}
$$

由第一式得：

$$
x_2=5x_1-2
$$

代入第二式：

$$
-x_1+2(5x_1-2)-2=0
$$

$$
9x_1-6=0
$$

$$
x_1=\frac23
$$

再代回：

$$
x_2=5\cdot\frac23-2=\frac43
$$

所以：

$$
\boxed{
x^\star=
\begin{pmatrix}
\frac23\\[1mm]
\frac43
\end{pmatrix}}
$$

### (d) 梯度下降迭代格式，并求 $x^{(1)},x^{(2)}$

梯度下降迭代格式：

$$
x^{(k+1)}=x^{(k)}-\eta\nabla f(x^{(k)})
$$

题目给：

$$
x^{(0)}=\begin{pmatrix}0\\0\end{pmatrix},\quad \eta=\frac14
$$

先算：

$$
\nabla f(x^{(0)})
=
\begin{pmatrix}
-2\\
-2
\end{pmatrix}
$$

所以：

$$
x^{(1)}
=
\begin{pmatrix}0\\0\end{pmatrix}
-\frac14
\begin{pmatrix}-2\\-2\end{pmatrix}
=
\begin{pmatrix}\frac12\\[1mm]\frac12\end{pmatrix}
$$

再算：

$$
\nabla f\left(\frac12,\frac12\right)
=
\begin{pmatrix}
0\\
-\frac32
\end{pmatrix}
$$

所以：

$$
x^{(2)}
=
\begin{pmatrix}\frac12\\[1mm]\frac12\end{pmatrix}
-\frac14
\begin{pmatrix}0\\[1mm]-\frac32\end{pmatrix}
=
\begin{pmatrix}\frac12\\[1mm]\frac78\end{pmatrix}
$$

最终：

$$
\boxed{
x^{(1)}=
\begin{pmatrix}\frac12\\[1mm]\frac12\end{pmatrix},
\quad
x^{(2)}=
\begin{pmatrix}\frac12\\[1mm]\frac78\end{pmatrix}}
$$

## 第 5 题 最优化基础：硬间隔 SVM

原问题：

$$
\min_{w,b}\frac12\lVert w\rVert_2^2
$$

约束为：

$$
y_i(w^Tx_i+b)\ge 1,\quad i=1,\ldots,m
$$

### (a) 拉格朗日函数

先把约束改写为：

$$
1-y_i(w^Tx_i+b)\le 0
$$

引入拉格朗日乘子：

$$
\alpha_i\ge0,\quad i=1,\ldots,m
$$

拉格朗日函数为：

$$
L(w,b,\alpha)
=\frac12\lVert w\rVert_2^2
+\sum_{i=1}^m\alpha_i\left[1-y_i(w^Tx_i+b)\right]
$$

### (b) 驻点条件

对 $w$ 求导：

$$
\frac{\partial L}{\partial w}
=w-\sum_{i=1}^m\alpha_i y_i x_i
$$

令其为 0：

$$
\boxed{
w=\sum_{i=1}^m\alpha_i y_i x_i}
$$

对 $b$ 求导：

$$
\frac{\partial L}{\partial b}
=-\sum_{i=1}^m\alpha_i y_i
$$

令其为 0：

$$
\boxed{
\sum_{i=1}^m\alpha_i y_i=0}
$$

### (c) 对偶问题

把驻点条件代回拉格朗日函数，得到对偶问题：

$$
\max_{\alpha}
\quad
\sum_{i=1}^m\alpha_i
-\frac12
\sum_{i=1}^m\sum_{j=1}^m
\alpha_i\alpha_j y_i y_j x_i^Tx_j
$$

约束为：

$$
\alpha_i\ge0,\quad i=1,\ldots,m
$$

$$
\sum_{i=1}^m\alpha_i y_i=0
$$

### (d) 恢复 $w^\star$，并说明支持向量

由对偶最优解 $\alpha^\star$ 恢复：

$$
\boxed{
w^\star=\sum_{i=1}^m\alpha_i^\star y_i x_i}
$$

若需要恢复 $b^\star$，可选取任意一个满足 $\alpha_i^\star>0$ 的样本点，利用：

$$
y_i\left((w^\star)^Tx_i+b^\star\right)=1
$$

得到：

$$
b^\star=y_i-(w^\star)^Tx_i
$$

支持向量是满足：

$$
\boxed{\alpha_i^\star>0}
$$

的样本点。它们通常正好落在间隔边界上，即：

$$
y_i\left((w^\star)^Tx_i+b^\star\right)=1
$$

## 第 6 题 大模型数学基础：单头自注意力

### (a) 单头自注意力公式

设输入矩阵：

$$
X\in\mathbb{R}^{n\times d}
$$

其中 $n$ 是序列长度，$d$ 是每个 token 的表示维度。

计算 Query、Key、Value：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

相似度得分矩阵：

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

其中 $d_k$ 是 key 的维度。

对 $S$ 的每一行做 softmax：

$$
P=\operatorname{softmax}(S)
$$

输出：

$$
O=PV
$$

整体公式：

$$
\boxed{
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
$$

### (b) 代入具体矩阵计算

题目给：

$$
X=
\begin{pmatrix}
1 & 0\\
1 & 1
\end{pmatrix},
\quad
W_Q=W_K=W_V=I_2
$$

因此：

$$
Q=K=V=X=
\begin{pmatrix}
1 & 0\\
1 & 1
\end{pmatrix}
$$

先计算：

$$
K^T=
\begin{pmatrix}
1 & 1\\
0 & 1
\end{pmatrix}
$$

$$
QK^T=
\begin{pmatrix}
1 & 0\\
1 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1\\
0 & 1
\end{pmatrix}
=
\begin{pmatrix}
1 & 1\\
1 & 2
\end{pmatrix}
$$

因为 $d_k=2$，所以：

$$
S=\frac{QK^T}{\sqrt2}
=
\begin{pmatrix}
\frac{1}{\sqrt2} & \frac{1}{\sqrt2}\\[1mm]
\frac{1}{\sqrt2} & \frac{2}{\sqrt2}
\end{pmatrix}
$$

对每一行做 softmax。第一行两个元素相同，所以：

$$
P_{1,:}=
\begin{pmatrix}
\frac12 & \frac12
\end{pmatrix}
$$

第二行：

$$
P_{2,:}=
\begin{pmatrix}
\frac{1}{1+e^{1/\sqrt2}} &
\frac{e^{1/\sqrt2}}{1+e^{1/\sqrt2}}
\end{pmatrix}
$$

因此：

$$
P=
\begin{pmatrix}
\frac12 & \frac12\\[2mm]
\frac{1}{1+e^{1/\sqrt2}} &
\frac{e^{1/\sqrt2}}{1+e^{1/\sqrt2}}
\end{pmatrix}
$$

最后：

$$
O=PV
$$

第一行：

$$
\frac12(1,0)+\frac12(1,1)=(1,\frac12)
$$

令

$$
a=\frac{e^{1/\sqrt2}}{1+e^{1/\sqrt2}}
$$

则第二行权重为 $(1-a,a)$，所以：

$$
(1-a)(1,0)+a(1,1)=(1,a)
$$

最终：

$$
\boxed{
O=
\begin{pmatrix}
1 & \frac12\\[2mm]
1 & \frac{e^{1/\sqrt2}}{1+e^{1/\sqrt2}}
\end{pmatrix}}
$$

### (c) 长序列时的计算和存储瓶颈

主要瓶颈是计算并存储注意力得分矩阵：

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

因为 $S$ 的形状是 $n\times n$。当序列长度 $n$ 很大时，注意力矩阵的规模按 $n^2$ 增长。

因此：

$$
\text{计算复杂度约为 }O(n^2d_k)
$$

$$
\text{存储复杂度约为 }O(n^2)
$$

结论：长序列场景下，自注意力的主要计算瓶颈和存储瓶颈都来自 $QK^T$ 以及后续的注意力权重矩阵 $P$。

## 第 7 题 大模型数学基础：LoRA

原模型：

$$
h=W_1x,\quad z=\sigma(h),\quad y=W_2z,\quad L=f(y)
$$

LoRA 修改后：

$$
h=(W_1+A_1B_1)x
$$

$$
z=\sigma(h)
$$

$$
y=(W_2+A_2B_2)z
$$

$$
L=f(y)
$$

其中 $W_1,W_2$ 冻结不训练，只训练 $A_1,B_1,A_2,B_2$。

### (a) 前向传播、反向传播与梯度

记：

$$
M_1=W_1+A_1B_1,\quad M_2=W_2+A_2B_2
$$

前向传播：

$$
h=M_1x
$$

$$
z=\sigma(h)
$$

$$
y=M_2z
$$

$$
L=f(y)
$$

定义输出层梯度：

$$
\delta_y=\frac{\partial L}{\partial y}=\nabla f(y)
$$

反向传播：

$$
\delta_z=\frac{\partial L}{\partial z}=M_2^T\delta_y
$$

$$
\delta_h=\frac{\partial L}{\partial h}=\delta_z\odot\sigma'(h)
$$

其中 $\odot$ 表示逐元素相乘。

对第二层 LoRA 参数求梯度。因为：

$$
y=W_2z+A_2B_2z
$$

所以：

$$
\boxed{
\frac{\partial L}{\partial A_2}
=\delta_y(B_2z)^T}
$$

$$
\boxed{
\frac{\partial L}{\partial B_2}
=A_2^T\delta_y z^T}
$$

对第一层 LoRA 参数求梯度。因为：

$$
h=W_1x+A_1B_1x
$$

所以：

$$
\boxed{
\frac{\partial L}{\partial A_1}
=\delta_h(B_1x)^T}
$$

$$
\boxed{
\frac{\partial L}{\partial B_1}
=A_1^T\delta_h x^T}
$$

形状检查：

$$
\frac{\partial L}{\partial A_2}\in\mathbb{R}^{p\times s},\quad
\frac{\partial L}{\partial B_2}\in\mathbb{R}^{s\times m}
$$

$$
\frac{\partial L}{\partial A_1}\in\mathbb{R}^{m\times r},\quad
\frac{\partial L}{\partial B_1}\in\mathbb{R}^{r\times n}
$$

补充说明：初始化时只需要让乘积 $A_1B_1$、$A_2B_2$ 为零即可，不一定要把两个因子都初始化为零。实际 LoRA 中常见做法是一个因子随机初始化，另一个因子初始化为零。

### (b) 参数量比较

预训练阶段要学习 $W_1,W_2$。

$$
W_1\in\mathbb{R}^{m\times n}
$$

所以 $W_1$ 的参数量为：

$$
mn
$$

$$
W_2\in\mathbb{R}^{p\times m}
$$

所以 $W_2$ 的参数量为：

$$
pm
$$

预训练参数量为：

$$
\boxed{mn+pm}
$$

LoRA 微调阶段只学习：

$$
A_1\in\mathbb{R}^{m\times r},\quad B_1\in\mathbb{R}^{r\times n}
$$

$$
A_2\in\mathbb{R}^{p\times s},\quad B_2\in\mathbb{R}^{s\times m}
$$

所以 LoRA 微调参数量为：

$$
mr+rn+ps+sm
$$

也就是：

$$
\boxed{r(m+n)+s(p+m)}
$$

由于：

$$
r\ll\min\{m,n\},\quad s\ll\min\{p,m\}
$$

通常有：

$$
r(m+n)+s(p+m)\ll mn+pm
$$

因此，LoRA 能显著减少微调阶段需要学习的参数量。
