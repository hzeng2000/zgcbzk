# 人工智能数学基础正式样题参考答案

---

## 第 1 题 线性代数基础

### 1. 正交性与单位化

计算内积：

$$
u_1^Tu_2=1\cdot1+1\cdot(-1)+0\cdot0=0.
$$

所以 $u_1,u_2$ 正交。

两个向量的范数都是

$$
\|u_1\|=\|u_2\|=\sqrt{1^2+1^2}=\sqrt2.
$$

单位化后：

$$
e_1=\frac{1}{\sqrt2}\begin{pmatrix}1\\1\\0\end{pmatrix},
\quad
e_2=\frac{1}{\sqrt2}\begin{pmatrix}1\\-1\\0\end{pmatrix}.
$$

### 2. 正交投影

因为 $u_1,u_2$ 正交，所以

$$
\operatorname{proj}_U(v)
=\frac{v^Tu_1}{u_1^Tu_1}u_1+\frac{v^Tu_2}{u_2^Tu_2}u_2.
$$

计算：

$$
v^Tu_1=3+1=4,\quad u_1^Tu_1=2,
$$

$$
v^Tu_2=3-1=2,\quad u_2^Tu_2=2.
$$

所以

$$
\operatorname{proj}_U(v)=2u_1+u_2
=2\begin{pmatrix}1\\1\\0\end{pmatrix}
+\begin{pmatrix}1\\-1\\0\end{pmatrix}
=\begin{pmatrix}3\\1\\0\end{pmatrix}.
$$

### 3. 特征值与特征向量

矩阵 $A$ 的前两维是

$$
\begin{pmatrix}2&1\\1&2\end{pmatrix}.
$$

对向量 $(1,1)^T$：

$$
\begin{pmatrix}2&1\\1&2\end{pmatrix}
\begin{pmatrix}1\\1\end{pmatrix}
=\begin{pmatrix}3\\3\end{pmatrix}
=3\begin{pmatrix}1\\1\end{pmatrix}.
$$

所以一个特征值为 $3$，对应三维特征向量可取

$$
p_1=\begin{pmatrix}1\\1\\0\end{pmatrix}.
$$

对向量 $(1,-1)^T$：

$$
\begin{pmatrix}2&1\\1&2\end{pmatrix}
\begin{pmatrix}1\\-1\end{pmatrix}
=\begin{pmatrix}1\\-1\end{pmatrix}.
$$

所以一个特征值为 $1$，对应三维特征向量可取

$$
p_2=\begin{pmatrix}1\\-1\\0\end{pmatrix}.
$$

第三个坐标方向满足

$$
A\begin{pmatrix}0\\0\\1\end{pmatrix}
=\begin{pmatrix}0\\0\\1\end{pmatrix}.
$$

所以另一个特征值也是 $1$，对应特征向量可取

$$
p_3=\begin{pmatrix}0\\0\\1\end{pmatrix}.
$$

因此特征值为

$$
3,\quad 1,\quad 1.
$$

### 4. 正定性与对角化

矩阵 $A$ 是实对称矩阵，且所有特征值都大于 0，因此 $A$ 正定。

令

$$
P=
\begin{pmatrix}
1&1&0\\
1&-1&0\\
0&0&1
\end{pmatrix},
\quad
D=
\begin{pmatrix}
3&0&0\\
0&1&0\\
0&0&1
\end{pmatrix}.
$$

其中 $P$ 的每一列是一个特征向量。于是

$$
A=PDP^{-1}.
$$

---

## 第 2 题 常用分布与连续型随机变量

### 1. 均匀分布的密度函数和分布函数

若 $X\sim Uniform(1,5)$，则概率密度函数为

$$
f_X(x)=
\begin{cases}
\frac14,&1<x<5,\\
0,&\text{其他}.
\end{cases}
$$

累积分布函数为

$$
F_X(x)=
\begin{cases}
0,&x\le 1,\\
\frac{x-1}{4},&1<x<5,\\
1,&x\ge 5.
\end{cases}
$$

### 2. 期望、方差和区间概率

均匀分布 $Uniform(a,b)$ 的期望和方差为

$$
E[X]=\frac{a+b}{2},\quad \operatorname{Var}(X)=\frac{(b-a)^2}{12}.
$$

这里 $a=1,b=5$，所以

$$
E[X]=3,\quad \operatorname{Var}(X)=\frac{16}{12}=\frac43.
$$

又因为区间长度为 $5-1=4$，所以

$$
P(2<X<4)=\frac{4-2}{5-1}=\frac12.
$$

### 3. 指数分布的密度函数和分布函数

若 $T\sim Exponential(\lambda)$，且 $\lambda=\frac12$，则

$$
f_T(t)=
\begin{cases}
\frac12 e^{-t/2},&t\ge 0,\\
0,&t<0.
\end{cases}
$$

累积分布函数为

$$
F_T(t)=
\begin{cases}
0,&t<0,\\
1-e^{-t/2},&t\ge 0.
\end{cases}
$$

### 4. 概率计算与无记忆性

指数分布的生存函数为

$$
P(T>t)=e^{-\lambda t}.
$$

所以

$$
P(T>3)=e^{-3/2}.
$$

条件概率为

$$
P(T>5\mid T>2)
=\frac{P(T>5)}{P(T>2)}
=\frac{e^{-5/2}}{e^{-1}}
=e^{-3/2}.
$$

这说明在已经等待 2 个单位时间以后，还需要再等待超过 3 个单位时间的概率，和从一开始等待超过 3 个单位时间的概率相同。这体现了指数分布的无记忆性。

---

## 第 3 题 联合分布、边缘分布与协方差

### 1. 边缘分布

对 $Y$ 求和得到 $X$ 的边缘分布：

$$
P(X=0)=0.10+0.20+0.10=0.40,
$$

$$
P(X=1)=0.20+0.15+0.25=0.60.
$$

对 $X$ 求和得到 $Y$ 的边缘分布：

$$
P(Y=0)=0.10+0.20=0.30,
$$

$$
P(Y=1)=0.20+0.15=0.35,
$$

$$
P(Y=2)=0.10+0.25=0.35.
$$

### 2. 条件概率

$$
P(X=1\mid Y=2)=\frac{P(X=1,Y=2)}{P(Y=2)}
=\frac{0.25}{0.35}
=\frac57.
$$

### 3. 期望与协方差

$$
E[X]=0\cdot0.40+1\cdot0.60=0.60.
$$

$$
E[Y]=0\cdot0.30+1\cdot0.35+2\cdot0.35=1.05.
$$

因为 $X=0$ 时 $XY=0$，只需要看 $X=1$ 这一行：

$$
E[XY]=0\cdot0.20+1\cdot0.15+2\cdot0.25=0.65.
$$

因此

$$
\operatorname{Cov}(X,Y)=E[XY]-E[X]E[Y]
=0.65-0.60\times1.05
=0.02.
$$

### 4. 独立性判断

若 $X,Y$ 独立，则应有

$$
P(X=x,Y=y)=P(X=x)P(Y=y)
$$

对所有 $x,y$ 都成立。

取 $(X=0,Y=0)$：

$$
P(X=0,Y=0)=0.10,
$$

但

$$
P(X=0)P(Y=0)=0.40\times0.30=0.12.
$$

二者不相等，所以 $X,Y$ 不独立。

---

## 第 4 题 最小二乘、梯度与凸性

记

$$
e_1=a+b-1,\quad e_2=2a+b-2.
$$

则

$$
J(a,b)=\frac12e_1^2+\frac12e_2^2.
$$

### 1. 梯度与 Hessian

对 $a$ 求偏导：

$$
\frac{\partial J}{\partial a}=e_1\cdot1+e_2\cdot2
=a+b-1+2(2a+b-2)
=5a+3b-5.
$$

对 $b$ 求偏导：

$$
\frac{\partial J}{\partial b}=e_1\cdot1+e_2\cdot1
=a+b-1+2a+b-2
=3a+2b-3.
$$

所以

$$
\nabla J(a,b)=
\begin{pmatrix}
5a+3b-5\\
3a+2b-3
\end{pmatrix}.
$$

Hessian 矩阵为

$$
\nabla^2J(a,b)=
\begin{pmatrix}
5&3\\
3&2
\end{pmatrix}.
$$

### 2. 凸性判断

判断 Hessian 是否正定：

$$
5>0,\quad
\det\begin{pmatrix}5&3\\3&2\end{pmatrix}=10-9=1>0.
$$

两个顺序主子式都大于 0，所以 Hessian 正定。因此 $J(a,b)$ 是严格凸函数。

### 3. 最优解

严格凸函数的驻点就是唯一全局最优点。令梯度为零：

$$
\begin{cases}
5a+3b-5=0,\\
3a+2b-3=0.
\end{cases}
$$

即

$$
\begin{cases}
5a+3b=5,\\
3a+2b=3.
\end{cases}
$$

由第二式乘 3 得

$$
9a+6b=9.
$$

第一式乘 2 得

$$
10a+6b=10.
$$

相减得

$$
a=1.
$$

代回

$$
3a+2b=3
$$

得

$$
b=0.
$$

因此

$$
(a^\star,b^\star)=(1,0).
$$

### 4. 梯度下降

梯度下降格式为

$$
\begin{pmatrix}
a^{(k+1)}\\
b^{(k+1)}
\end{pmatrix}
=
\begin{pmatrix}
a^{(k)}\\
b^{(k)}
\end{pmatrix}
-\eta
\begin{pmatrix}
5a^{(k)}+3b^{(k)}-5\\
3a^{(k)}+2b^{(k)}-3
\end{pmatrix}.
$$

初始点为 $(0,0)$：

$$
\nabla J(0,0)=
\begin{pmatrix}
-5\\
-3
\end{pmatrix}.
$$

所以

$$
\begin{pmatrix}
a^{(1)}\\
b^{(1)}
\end{pmatrix}
=
\begin{pmatrix}
0\\
0
\end{pmatrix}
-0.1
\begin{pmatrix}
-5\\
-3
\end{pmatrix}
=
\begin{pmatrix}
0.5\\
0.3
\end{pmatrix}.
$$

继续计算：

$$
\nabla J(0.5,0.3)=
\begin{pmatrix}
5(0.5)+3(0.3)-5\\
3(0.5)+2(0.3)-3
\end{pmatrix}
=
\begin{pmatrix}
-1.6\\
-0.9
\end{pmatrix}.
$$

所以

$$
\begin{pmatrix}
a^{(2)}\\
b^{(2)}
\end{pmatrix}
=
\begin{pmatrix}
0.5\\
0.3
\end{pmatrix}
-0.1
\begin{pmatrix}
-1.6\\
-0.9
\end{pmatrix}
=
\begin{pmatrix}
0.66\\
0.39
\end{pmatrix}.
$$

---

## 第 5 题 拉格朗日函数与对偶问题

### 1. 拉格朗日函数

原问题为

$$
\min_{x_1,x_2}\frac12(x_1^2+x_2^2)
\quad
\text{s.t.}\quad
x_1+2x_2=1.
$$

写成约束 $x_1+2x_2-1=0$。拉格朗日函数为

$$
L(x_1,x_2,\lambda)
=\frac12(x_1^2+x_2^2)+\lambda(x_1+2x_2-1).
$$

### 2. 驻点条件与原问题最优解

对 $x_1$ 求偏导：

$$
\frac{\partial L}{\partial x_1}=x_1+\lambda=0,
$$

所以

$$
x_1=-\lambda.
$$

对 $x_2$ 求偏导：

$$
\frac{\partial L}{\partial x_2}=x_2+2\lambda=0,
$$

所以

$$
x_2=-2\lambda.
$$

代回约束：

$$
x_1+2x_2=1
$$

得

$$
-\lambda+2(-2\lambda)=1.
$$

即

$$
-5\lambda=1,\quad \lambda=-\frac15.
$$

因此

$$
x_1^\star=\frac15,\quad x_2^\star=\frac25.
$$

原问题最优值为

$$
\frac12\left[\left(\frac15\right)^2+\left(\frac25\right)^2\right]
=\frac12\cdot\frac{5}{25}
=\frac{1}{10}.
$$

### 3. 对偶函数

由驻点条件

$$
x_1=-\lambda,\quad x_2=-2\lambda.
$$

把它们代回拉格朗日函数：

$$
g(\lambda)
=L(-\lambda,-2\lambda,\lambda).
$$

先算目标函数部分：

$$
\frac12(x_1^2+x_2^2)
=\frac12(\lambda^2+4\lambda^2)
=\frac52\lambda^2.
$$

再算约束乘子部分：

$$
\lambda(x_1+2x_2-1)
=\lambda(-\lambda+2(-2\lambda)-1)
=\lambda(-5\lambda-1)
=-5\lambda^2-\lambda.
$$

所以

$$
g(\lambda)=\frac52\lambda^2-5\lambda^2-\lambda
=-\frac52\lambda^2-\lambda.
$$

### 4. 对偶问题与最优值

对偶问题为

$$
\max_{\lambda\in\mathbb R}
g(\lambda)
=
\max_{\lambda\in\mathbb R}
\left(-\frac52\lambda^2-\lambda\right).
$$

对 $\lambda$ 求导：

$$
g'(\lambda)=-5\lambda-1.
$$

令 $g'(\lambda)=0$，得

$$
\lambda^\star=-\frac15.
$$

对偶最优值为

$$
g\left(-\frac15\right)
=-\frac52\cdot\frac1{25}+\frac15
=-\frac1{10}+\frac2{10}
=\frac1{10}.
$$

它与原问题最优值相等。这是因为该问题是凸二次目标加仿射等式约束，满足强对偶。

---

## 第 6 题 前反向传播与自注意力

### (a) 简单神经网络的前反向传播

#### 1. 前向传播

先算第一层：

$$
h=W_1x+b_1
=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
\begin{pmatrix}
1\\
2
\end{pmatrix}
+
\begin{pmatrix}
0\\
-3
\end{pmatrix}
=
\begin{pmatrix}
1\\
-1
\end{pmatrix}.
$$

经过 ReLU：

$$
z=ReLU(h)=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

输出：

$$
y=W_2z+b_2
=
\begin{pmatrix}1&-1\end{pmatrix}
\begin{pmatrix}
1\\
0
\end{pmatrix}
=1.
$$

损失：

$$
L=\frac12(y-t)^2=\frac12(1-0)^2=\frac12.
$$

#### 2. 反向传播

记

$$
\delta_y=\frac{\partial L}{\partial y}=y-t=1.
$$

对第二层参数：

$$
\frac{\partial L}{\partial W_2}
=\delta_y z^T
=1\begin{pmatrix}1&0\end{pmatrix}
=\begin{pmatrix}1&0\end{pmatrix}.
$$

$$
\frac{\partial L}{\partial b_2}=\delta_y=1.
$$

向前一层传回：

$$
\frac{\partial L}{\partial z}
=W_2^T\delta_y
=
\begin{pmatrix}
1\\
-1
\end{pmatrix}.
$$

因为

$$
h=\begin{pmatrix}1\\-1\end{pmatrix},
$$

所以

$$
ReLU'(h)=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

于是

$$
\frac{\partial L}{\partial h}
=
\frac{\partial L}{\partial z}\odot ReLU'(h)
=
\begin{pmatrix}
1\\
-1
\end{pmatrix}
\odot
\begin{pmatrix}
1\\
0
\end{pmatrix}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

对第一层参数：

$$
\frac{\partial L}{\partial W_1}
=
\frac{\partial L}{\partial h}x^T
=
\begin{pmatrix}
1\\
0
\end{pmatrix}
\begin{pmatrix}
1&2
\end{pmatrix}
=
\begin{pmatrix}
1&2\\
0&0
\end{pmatrix}.
$$

$$
\frac{\partial L}{\partial b_1}
=
\frac{\partial L}{\partial h}
=
\begin{pmatrix}
1\\
0
\end{pmatrix}.
$$

### (b) 单头自注意力计算

因为

$$
W_Q=W_K=W_V=I_2,
$$

所以

$$
Q=K=V=X=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}.
$$

此时 $d_k=2$，因此

$$
S=\frac{QK^T}{\sqrt2}
=\frac{1}{\sqrt2}
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}.
$$

令

$$
c=e^{1/\sqrt2}.
$$

对 $S$ 的每一行做 softmax：

$$
P=
\begin{pmatrix}
\frac{c}{c+1}&\frac{1}{c+1}\\
\frac{1}{c+1}&\frac{c}{c+1}
\end{pmatrix}.
$$

因为 $V=I_2$，所以

$$
O=PV=P=
\begin{pmatrix}
\frac{c}{c+1}&\frac{1}{c+1}\\
\frac{1}{c+1}&\frac{c}{c+1}
\end{pmatrix}.
$$

当序列长度为 $n$、表示维度为 $d$ 时，主要瓶颈来自注意力得分矩阵

$$
S=QK^T.
$$

这是一个 $n\times n$ 矩阵。计算 $QK^T$ 通常需要 $O(n^2d)$ 的计算量，保存注意力权重矩阵需要 $O(n^2)$ 的存储量。因此序列长度变长时，自注意力的计算和显存开销会按序列长度的平方增长。

