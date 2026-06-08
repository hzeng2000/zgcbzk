# 人工智能数学基础正式样题

总分：100 分。  
建议时间：120 分钟。  
说明：闭卷部分侧重基本计算，开卷部分侧重综合推导。除特别说明外，计算过程需要写出关键步骤。

---

## 第 1 题 线性代数基础（15 分）

已知

$$
u_1=\begin{pmatrix}1\\1\\0\end{pmatrix},\quad
u_2=\begin{pmatrix}1\\-1\\0\end{pmatrix},\quad
v=\begin{pmatrix}3\\1\\2\end{pmatrix}.
$$

令

$$
U=\operatorname{span}\{u_1,u_2\}.
$$

又给定矩阵

$$
A=
\begin{pmatrix}
2&1&0\\
1&2&0\\
0&0&1
\end{pmatrix}.
$$

回答下列问题：

1. 判断 $u_1,u_2$ 是否正交，并将它们单位化。
2. 求 $v$ 在子空间 $U$ 上的正交投影 $\operatorname{proj}_U(v)$。
3. 求矩阵 $A$ 的特征值与一组对应特征向量。
4. 判断 $A$ 是否正定，并写出一个对角化形式 $A=PDP^{-1}$。

---

## 第 2 题 常用分布与连续型随机变量（15 分）

设随机变量 $X\sim Uniform(1,5)$。

1. 写出 $X$ 的概率密度函数和累积分布函数。
2. 求 $E[X]$、$\operatorname{Var}(X)$ 和 $P(2<X<4)$。

设随机变量 $T\sim Exponential(\lambda)$，其中 $\lambda=\frac12$。

3. 写出 $T$ 的概率密度函数和累积分布函数。
4. 求 $P(T>3)$ 和 $P(T>5\mid T>2)$，并说明这个结果体现了指数分布的什么性质。

---

## 第 3 题 联合分布、边缘分布与协方差（15 分）

设离散型随机变量 $(X,Y)$ 的联合分布如下：

|       | $Y=0$ | $Y=1$ | $Y=2$ |
|-------|------:|------:|------:|
| $X=0$ | 0.10  | 0.20  | 0.10  |
| $X=1$ | 0.20  | 0.15  | 0.25  |

回答下列问题：

1. 求 $X$ 和 $Y$ 的边缘分布。
2. 求 $P(X=1\mid Y=2)$。
3. 求 $E[X]$、$E[Y]$、$E[XY]$ 和 $\operatorname{Cov}(X,Y)$。
4. 判断 $X$ 与 $Y$ 是否独立，并说明理由。

---

## 第 4 题 最小二乘、梯度与凸性（15 分）

考虑线性回归模型

$$
\hat y=ax+b.
$$

现有两个样本：

$$
(x_1,y_1)=(1,1),\quad (x_2,y_2)=(2,2).
$$

采用平方损失，定义经验损失函数

$$
J(a,b)=\frac12(a+b-1)^2+\frac12(2a+b-2)^2.
$$

回答下列问题：

1. 求 $\nabla J(a,b)$ 和 Hessian 矩阵 $\nabla^2J(a,b)$。
2. 判断 $J(a,b)$ 是否为严格凸函数。
3. 求最优解 $(a^\star,b^\star)$。
4. 写出梯度下降迭代格式，并在初始点 $(a^{(0)},b^{(0)})=(0,0)$、学习率 $\eta=0.1$ 时，求 $(a^{(1)},b^{(1)})$ 和 $(a^{(2)},b^{(2)})$。

---

## 第 5 题 拉格朗日函数与对偶问题（15 分）

考虑约束优化问题

$$
\min_{x_1,x_2}\frac12(x_1^2+x_2^2)
\quad
\text{s.t.}\quad
x_1+2x_2=1.
$$

回答下列问题：

1. 写出拉格朗日函数。
2. 写出关于 $x_1,x_2$ 的驻点条件，并求原问题最优解。
3. 将驻点条件代回拉格朗日函数，推导对偶函数 $g(\lambda)$。
4. 写出对偶问题，并求对偶最优值。说明它与原问题最优值的关系。

---

## 第 6 题 前反向传播与自注意力（25 分）

### (a) 简单神经网络的前反向传播（13 分）

考虑如下两层网络：

$$
h=W_1x+b_1,\quad z=ReLU(h),\quad y=W_2z+b_2,\quad L=\frac12(y-t)^2.
$$

给定

$$
x=\begin{pmatrix}1\\2\end{pmatrix},\quad
W_1=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix},\quad
b_1=\begin{pmatrix}0\\-3\end{pmatrix},
$$

$$
W_2=\begin{pmatrix}1&-1\end{pmatrix},\quad
b_2=0,\quad
t=0.
$$

回答下列问题：

1. 计算 $h,z,y,L$。
2. 计算 $\frac{\partial L}{\partial W_2}$、$\frac{\partial L}{\partial b_2}$、$\frac{\partial L}{\partial W_1}$、$\frac{\partial L}{\partial b_1}$。

### (b) 单头自注意力计算（12 分）

设输入矩阵

$$
X=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix},
$$

并设

$$
W_Q=W_K=W_V=I_2.
$$

采用缩放点积注意力：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

$$
S=\frac{QK^T}{\sqrt{d_k}},\quad P=\operatorname{softmax}(S),\quad O=PV.
$$

回答下列问题：

1. 计算 $Q,K,V,S,P,O$。其中 softmax 可以保留指数形式。
2. 当序列长度为 $n$、表示维度为 $d$ 时，自注意力的主要计算和存储瓶颈是什么？为什么？

