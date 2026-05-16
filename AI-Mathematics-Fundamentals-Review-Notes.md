# 人工智能数学基础复习笔记

## 2026 新版考纲更新提示

- 新版指南同时适用于闭卷与开卷：闭卷更重基础概念与基本计算，开卷更重信息检索、跨知识点整合与应用。
- 线性代数部分重点是向量与矩阵的基本性质与计算。
- 概率部分重点是概率与随机变量的基本性质与计算，对应第 3-4 章；本笔记保留的 `MLE / MAP` 主要用于旧版衔接与开卷拓展，不占用新版考纲章节编号。
- 最优化部分重点是凸集 / 凸函数判定、机器学习常见损失及其导数、常见优化算法尤其前反向传播的迭代格式；新版明确不考收敛性证明。
- 大模型数学基础是新版新增重点：循环神经网络、自注意力、Transformer、Decoder-only 大模型的参数量 / 计算量 / 内存量计算，以及 LoRA。

---

# 第一部分：线性代数

## 第 1 章 向量基础

### 1.1 向量的线性相关与线性无关 (#)

**线性组合**：给定向量组 $v_1, v_2, \dots, v_n$，形如 $c_1v_1 + c_2v_2 + \dots + c_nv_n$ 的表达式

**线性相关**：存在不全为零的系数 $c_1, \dots, c_n$，使得
$$c_1v_1 + c_2v_2 + \dots + c_nv_n = 0$$

**线性无关**：只有当所有系数都为零时，上述等式才成立

---

**基础知识补充：如何判断线性相关？**

在继续之前，我们需要先学习两个工具：**行列式**和**矩阵的秩**。这两个概念将在后面详细介绍，但为了判断线性相关性，我们先了解它们的计算方法。

#### 工具 1：行列式 (determinant)

**$2 \times 2$ 行列式**：
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**例题**：$\det\begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} = 3 \times 4 - 1 \times 2 = 10$

**$3 \times 3$ 行列式**（按第一行展开）：
$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a(ei-fh) - b(di-fg) + c(dh-eg)$$

**例题**：计算 $\det\begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{pmatrix}$

$$= 1 \cdot \det\begin{pmatrix} 1 & 4 \\ 6 & 0 \end{pmatrix} - 2 \cdot \det\begin{pmatrix} 0 & 4 \\ 5 & 0 \end{pmatrix} + 3 \cdot \det\begin{pmatrix} 0 & 1 \\ 5 & 6 \end{pmatrix}$$
$$= 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1$$

**沙路法（Sarrus Rule）** - 仅适用于 $3 \times 3$ 的快捷方法：

```
    1  2  3     1  2
    0  1  4     0  1
    5  6  0     5  6
```

主对角线方向（相加）：$1\times1\times0 + 2\times4\times5 + 3\times0\times6 = 0 + 40 + 0 = 40$

副对角线方向（相减）：$3\times1\times5 + 1\times4\times6 + 2\times0\times0 = 15 + 24 + 0 = 39$

结果：$40 - 39 = 1$ ✓

---

#### 工具 2：矩阵的秩 (rank)

**定义**：矩阵中线性无关的行（或列）的最大数目。

**计算方法**：高斯消元法

将矩阵化为行阶梯形，非零行的数量就是秩。

**例题**：求 $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix}$ 的秩

$$\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix} \xrightarrow{R_2 - 2R_1} \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{pmatrix} \xrightarrow{R_3 - R_1} \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & -1 & -2 \end{pmatrix}$$

交换第 2、3 行：
$$\xrightarrow{R_2 \leftrightarrow R_3} \begin{pmatrix} 1 & 2 & 3 \\ 0 & -1 & -2 \\ 0 & 0 & 0 \end{pmatrix}$$

有 2 个非零行，故 $\text{rank}(A) = 2$

---

**回到线性相关性判定**

**判定方法**：
- **矩阵法**：将向量组成矩阵 $A$，若 $\text{rank}(A) < n$（向量个数）则线性相关
- **行列式法**：$n$ 个 $n$ 维向量，若 $\det(A) \neq 0$ 则线性无关

**例题 1**：判断向量 $v_1 = [1, 2, 3]^T$, $v_2 = [2, 4, 6]^T$, $v_3 = [1, 0, 1]^T$ 是否线性相关

**解**：观察发现 $v_2 = 2v_1$，因此 $2v_1 - v_2 + 0v_3 = 0$，系数 $(2, -1, 0)$ 不全为零，故**线性相关**。

**例题 2**：用行列式法判断 $v_1 = [1, 2, 1]^T$, $v_2 = [2, 1, 0]^T$, $v_3 = [1, 1, 1]^T$

$$A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$

$$\det(A) = 1(1\times1 - 1\times1) - 2(2\times1 - 1\times1) + 1(2\times1 - 1\times1) = 1(0) - 2(1) + 1(1) = -1 \neq 0$$

故**线性无关**。

**例题 3**：用秩判定 $v_1 = [1, 2, 3]^T$, $v_2 = [2, 4, 6]^T$, $v_3 = [1, 1, 1]^T$

$$A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 1 \\ 3 & 6 & 1 \end{pmatrix} \xrightarrow{R_2-2R_1} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 3 & 6 & 1 \end{pmatrix} \xrightarrow{R_3-3R_1} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 0 & 0 & -2 \end{pmatrix} \xrightarrow{R_3-2R_2} \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 0 & 0 & 0 \end{pmatrix}$$

秩 = 2 < 3（向量个数），故**线性相关**。

---

### 1.2 向量空间 (#)

**定义**：向量空间 $V$ 是对加法和数乘封闭的集合，满足 8 条公理。

**常见向量空间**：
- $\mathbb{R}^n$：$n$ 维实向量空间
- $C[a,b]$：连续函数空间
- $P_n$：次数不超过 $n$ 的多项式空间

**子空间**：向量空间的非空子集，对加法和数乘封闭。

**例题**：验证 $W = \{[x, y, 0]^T \mid x, y \in \mathbb{R}\}$ 是 $\mathbb{R}^3$ 的子空间

**验证**：
1. 零向量：$[0, 0, 0]^T \in W$ ✓
2. 加法封闭：$[x_1, y_1, 0]^T + [x_2, y_2, 0]^T = [x_1+x_2, y_1+y_2, 0]^T \in W$ ✓
3. 数乘封闭：$c[x, y, 0]^T = [cx, cy, 0]^T \in W$ ✓

故 $W$ 是 $\mathbb{R}^3$ 的子空间（实际上是 $xy$ 平面）。

---

### 1.3 基、维数与坐标变换 (#)

**基**：向量空间中线性无关且能张成整个空间的向量组。

**维数**：基中向量的个数，记作 $\dim(V)$。

**坐标**：向量 $v$ 在基 $\{e_1, \dots, e_n\}$ 下的表示
$$v = x_1e_1 + x_2e_2 + \dots + x_ne_n$$
坐标向量为 $[x_1, x_2, \dots, x_n]^T$。

**例题**：求向量 $v = [5, 3]^T$ 在基 $\{e_1 = [1, 1]^T, e_2 = [1, -1]^T\}$ 下的坐标。

**解**：设 $v = x_1e_1 + x_2e_2$，即
$$\begin{pmatrix} 5 \\ 3 \end{pmatrix} = x_1\begin{pmatrix} 1 \\ 1 \end{pmatrix} + x_2\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} x_1 + x_2 \\ x_1 - x_2 \end{pmatrix}$$

解方程组：
$$\begin{cases} x_1 + x_2 = 5 \\ x_1 - x_2 = 3 \end{cases} \Rightarrow x_1 = 4, x_2 = 1$$

故坐标为 $[4, 1]^T$。

**坐标变换公式**：从基 $\{e_i\}$ 到基 $\{e'_i\}$ 的变换
$$[v]_{e'} = P^{-1}[v]_e$$
其中 $P$ 是从 $\{e_i\}$ 到 $\{e'_i\}$ 的过渡矩阵（列向量为新基在旧基下的坐标）。

**例题**：标准基 $\{e_1=[1,0]^T, e_2=[0,1]^T\}$ 到新基 $\{e'_1=[1,1]^T, e'_2=[1,-1]^T\}$ 的坐标变换。

过渡矩阵 $P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$，$P^{-1} = \frac{1}{-2}\begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$

若 $[v]_e = [5, 3]^T$，则 $[v]_{e'} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 5 \\ 3 \end{pmatrix} = \begin{pmatrix} 4 \\ 1 \end{pmatrix}$

---

### 1.4 内积、范数与正交性 (#)

**内积**：$\mathbb{R}^n$ 中向量 $x, y$ 的标准内积
$$\langle x, y \rangle = x^T y = \sum_{i=1}^{n} x_i y_i$$

**内积性质**：
1. 对称性：$\langle x, y \rangle = \langle y, x \rangle$
2. 线性：$\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle$
3. 正定性：$\langle x, x \rangle \geq 0$，等号当且仅当 $x=0$

**例题**：计算 $x = [1, 2, 3]^T$, $y = [4, 5, 6]^T$ 的内积
$$\langle x, y \rangle = 1\times4 + 2\times5 + 3\times6 = 4 + 10 + 18 = 32$$

**范数**（向量长度）：
$$\|x\| = \sqrt{\langle x, x \rangle} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**常用范数**：
| 范数 | 公式 | 例题 $x=[3,-4]^T$ |
|------|------|------------------|
| $L_1$ | $\|x\|_1 = \sum \|x_i\|$ | $\lvert3\rvert+\lvert-4\rvert = 7$ |
| $L_2$ | $\|x\|_2 = \sqrt{\sum x_i^2}$ | $\sqrt{9+16} = 5$ |
| $L_\infty$ | $\|x\|_\infty = \max_i \|x_i\|$ | $\max(3,4) = 4$ |

**正交**：$\langle x, y \rangle = 0$

**正交基**：基向量两两正交。

**标准正交基**：正交基且每个基向量长度为 1。

**Gram-Schmidt 正交化**：将线性无关向量组转化为正交向量组。

**核心思想**：逐个减去"平行部分"，保留"垂直部分"。

**为什么要正交化？**
- 原始向量可能不正交，计算复杂
- 正交向量内积为 0，很多交叉项消失，计算更简单
- 正交化后张成的空间不变（新向量仍在这个空间内）

---

**详细例题**：将 $v_1 = [1, 1, 0]^T$, $v_2 = [1, 0, 1]^T$ 正交化。

**问题分析**：
- $v_1$ 和 $v_2$ 线性无关，但不正交（内积不为 0）
- 目标：找 $u_1, u_2$，使得 $u_1 \perp u_2$，且张成的平面不变

**直观理解**（只看 v₁ 和 v₂ 张成的平面）：

```
从原点 O 出发的向量：

              ● P (v₂的终点)
             /│
            / │
           /  │
        v₂/   │ u₂ = QP (垂直部分，待求的新向量)
         /    │
        /θ    │
       O──────●──────→ 方向：u₁ = v₁
              Q
           投影 = OQ
                 (v₂在 u₁方向上的分量)

OP = v₂ = [1, 0, 1]ᵗ（原向量）
OQ = 投影 = (v₂在 u₁方向上的分量)
QP = u₂ = v₂ - 投影（与 u₁垂直的新向量）
u₁ = v₁ = [1, 1, 0]ᵗ（方向向右，与 OQ 同方向）
```

**关键关系**：
- u₁ 和 v₁ 是同一条向量（第一步不变）
- 投影的方向与 u₁ 相同（OQ 的方向）
- u₂ 从 Q 指向 P，与 u₁ 垂直
- 三角形 OQP 是直角三角形（∠OQP = 90°）

---

**Step 1：第一个向量保持不变**
$$u_1 = v_1 = [1, 1, 0]^T$$

**Step 2：第二个向量减去在 u₁ 上的投影**

关键问题：如何从 v₂ 中减去它在 u₁ 方向上的分量？

设 $u_2 = v_2 - k \cdot u_1$，其中 k 是待定系数。

因为 $u_2$ 要与 $u_1$ 垂直，所以它们的内积为 0：
$$\langle u_2, u_1 \rangle = 0$$

代入 $u_2 = v_2 - k \cdot u_1$：
$$\langle v_2 - k \cdot u_1, u_1 \rangle = 0$$

展开（内积的线性性质）：
$$\langle v_2, u_1 \rangle - k \cdot \langle u_1, u_1 \rangle = 0$$

解出 k：
$$k = \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle}$$

**所以公式是**：
$$u_2 = v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle}u_1$$

---

**具体计算**：

1. 计算 $\langle v_2, u_1 \rangle$（分子）：
   $$1\times1 + 0\times1 + 1\times0 = 1$$

2. 计算 $\langle u_1, u_1 \rangle$（分母）：
   $$1\times1 + 1\times1 + 0\times0 = 2$$

3. 计算系数 k：
   $$k = \frac{1}{2} = 0.5$$

4. 从 v₂ 中减去 u₁ 的 k 倍：
   $$u_2 = [1, 0, 1]^T - 0.5 \times [1, 1, 0]^T$$
   $$u_2 = [1, 0, 1]^T - [0.5, 0.5, 0]^T$$
   $$u_2 = [0.5, -0.5, 1]^T$$

---

**验证正交性**：
$$\langle u_1, u_2 \rangle = 1\times0.5 + 1\times(-0.5) + 0\times1 = 0.5 - 0.5 + 0 = 0$$

内积为 0，说明 $u_1 \perp u_2$，正交化成功！

---

**为什么张成的空间不变？**

原始向量组：$\{v_1, v_2\}$
正交化后：$\{u_1, u_2\}$

- $u_1 = v_1$（没变）
- $u_2 = v_2 - 0.5v_1$（用 v₁, v₂ 线性组合得到）

所以 $u_2$ 仍在 $v_1, v_2$ 张成的平面内。

反过来：
- $v_1 = u_1$
- $v_2 = u_2 + 0.5u_1$

所以 $v_2$ 也在 $u_1, u_2$ 张成的平面内。

**结论**：两个向量组张成同一个平面，但 $\{u_1, u_2\}$ 是正交的。

---

**标准化（得到标准正交基）**：

将正交向量单位化（除以长度）：

$$e_1 = \frac{u_1}{\|u_1\|} = \frac{1}{\sqrt{1^2+1^2+0^2}}[1, 1, 0]^T = \frac{1}{\sqrt{2}}[1, 1, 0]^T$$

$$e_2 = \frac{u_2}{\|u_2\|} = \frac{1}{\sqrt{0.5^2+(-0.5)^2+1^2}}[0.5, -0.5, 1]^T = \frac{1}{\sqrt{1.5}}[0.5, -0.5, 1]^T = \frac{\sqrt{2}}{\sqrt{3}}[0.5, -0.5, 1]^T$$

---

**三个向量的 Gram-Schmidt 正交化**：

对于 $v_1, v_2, v_3$，步骤如下：

1. $u_1 = v_1$
2. $u_2 = v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle}u_1$
3. $u_3 = v_3 - \frac{\langle v_3, u_1 \rangle}{\langle u_1, u_1 \rangle}u_1 - \frac{\langle v_3, u_2 \rangle}{\langle u_2, u_2 \rangle}u_2$

每个新向量都要减去它在**之前所有已正交化的向量**上的投影。

---

## 第 2 章 矩阵基础

### 2.1 矩阵的基本运算 (#)

**转置**：$(A^T)_{ij} = A_{ji}$

**例题**：$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$，则 $A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$

**性质**：$(A^T)^T = A$，$(AB)^T = B^T A^T$

---

**矩阵乘法详解**

**矩阵乘法的条件**：
- $A$ 是 $m \times n$ 矩阵，$B$ 是 $n \times p$ 矩阵 → $AB$ 是 $m \times p$ 矩阵
- **关键**：$A$ 的列数 = $B$ 的行数（否则无法相乘）

**矩阵乘法的计算**：
$$C = AB, \quad c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

**直观理解**：$C$ 的第 $(i,j)$ 个元素 = $A$ 的第 $i$ 行 与 $B$ 的第 $j$ 列 的点积。

**图解**：
```
    A              ×         B              =         C
┌─────────┐              ┌─────────┐              ┌─────────┐
│ a₁₁ a₁₂ │              │ b₁₁ b₁₂ │              │ c₁₁ c₁₂ │
│ a₂₁ a₂₂ │      ×       │ b₂₁ b₂₂ │      =       │ c₂₁ c₂₂ │
└─────────┘              └─────────┘              └─────────┘

c₁₁ = a₁₁×b₁₁ + a₁₂×b₂₁   (第 1 行 · 第 1 列)
c₁₂ = a₁₁×b₁₂ + a₁₂×b₂₂   (第 1 行 · 第 2 列)
c₂₁ = a₂₁×b₁₁ + a₂₂×b₂₁   (第 2 行 · 第 1 列)
c₂₂ = a₂₁×b₁₂ + a₂₂×b₂₂   (第 2 行 · 第 2 列)
```

**详细例题**：
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

**逐步计算**：
- $c_{11} = 1\times5 + 2\times7 = 5 + 14 = 19$
- $c_{12} = 1\times6 + 2\times8 = 6 + 16 = 22$
- $c_{21} = 3\times5 + 4\times7 = 15 + 28 = 43$
- $c_{22} = 3\times6 + 4\times8 = 18 + 32 = 50$

$$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$$

**验证 $BA$**（体验不满足交换律）：
$$BA = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 5+18 & 10+24 \\ 7+24 & 14+32 \end{pmatrix} = \begin{pmatrix} 23 & 34 \\ 31 & 46 \end{pmatrix} \neq AB$$

---

**矩阵乘法在机器学习中的应用**：

神经网络前向传播：$z = Wx + b$

$$W = \begin{pmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{pmatrix}, \quad x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

$$z = Wx = \begin{pmatrix} w_{11}x_1 + w_{12}x_2 \\ w_{21}x_1 + w_{22}x_2 \end{pmatrix}$$

每个输出 = 权重与输入的加权求和！

---

**行列式**（已在 1.1 节介绍过计算方法）

**性质**：
- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$

**迹**：$\text{tr}(A) = \sum_i A_{ii}$（主对角线元素之和）

**例题**：$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$，$\text{tr}(A) = 1 + 4 = 5$

**迹的性质**：
- $\text{tr}(AB) = \text{tr}(BA)$
- $\text{tr}(A) = \sum_i \lambda_i$（特征值之和）

---

### 2.2 矩阵的秩与可逆矩阵 (#)

**秩**（已在 1.1 节介绍过）：矩阵中线性无关的行（或列）的最大数目。

**性质**：
- $\text{rank}(A) = \text{rank}(A^T)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

---

**逆矩阵详解**

**定义**：对于 $n \times n$ 方阵 $A$，若存在矩阵 $B$ 使得 $AB = BA = I$，则称 $B$ 为 $A$ 的逆矩阵，记作 $A^{-1}$。

**单位矩阵** $I$：主对角线为 1，其余为 0
$$I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

**性质**：$AI = IA = A$

---

**求逆矩阵的一般方法：高斯 - 约旦消元法**

这是适用于任意 $n \times n$ 矩阵的通用方法。

**核心思想**：对增广矩阵 $[A | I]$ 做初等行变换，把 $A$ 变成 $I$，此时右边的 $I$ 就变成了 $A^{-1}$。

$$[A | I] \xrightarrow{\text{行变换}} [I | A^{-1}]$$

**为什么有效？**

行变换相当于左乘一个初等矩阵。一系列行变换把 $A$ 变成 $I$，即：
$$E_k \cdots E_2 E_1 A = I$$

那么 $E_k \cdots E_2 E_1 = A^{-1}$。

同样的变换作用在 $I$ 上：
$$E_k \cdots E_2 E_1 I = A^{-1}$$

---

**详细例题**：用高斯消元法求 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ 的逆。

**Step 1**：写出增广矩阵 $[A | I]$
$$\begin{pmatrix} 1 & 2 & | & 1 & 0 \\ 3 & 4 & | & 0 & 1 \end{pmatrix}$$

**Step 2**：消去第 2 行第 1 列的 3（第 2 行 - 3×第 1 行）
$$\xrightarrow{R_2 - 3R_1} \begin{pmatrix} 1 & 2 & | & 1 & 0 \\ 0 & -2 & | & -3 & 1 \end{pmatrix}$$

**Step 3**：把第 2 行第 2 列的 -2 变成 1（第 2 行除以 -2）
$$\xrightarrow{R_2 \div (-2)} \begin{pmatrix} 1 & 2 & | & 1 & 0 \\ 0 & 1 & | & 1.5 & -0.5 \end{pmatrix}$$

**Step 4**：消去第 1 行第 2 列的 2（第 1 行 - 2×第 2 行）
$$\xrightarrow{R_1 - 2R_2} \begin{pmatrix} 1 & 0 & | & -2 & 1 \\ 0 & 1 & | & 1.5 & -0.5 \end{pmatrix}$$

**完成！** 左边是 $I$，右边就是 $A^{-1}$：
$$A^{-1} = \begin{pmatrix} -2 & 1 \\ 1.5 & -0.5 \end{pmatrix}$$

---

**$2 \times 2$ 矩阵的逆（公式法）**

对于 $2 \times 2$ 矩阵，有现成的公式可以直接使用。

**公式**：
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**记忆口诀**："主对调，副变号，除以行列式"

---

**公式推导**（用高斯消元法推导）：

对一般 $2 \times 2$ 矩阵做高斯消元：

$$\begin{pmatrix} a & b & | & 1 & 0 \\ c & d & | & 0 & 1 \end{pmatrix}$$

假设 $a \neq 0$，先消去 $c$：
$$\xrightarrow{R_2 - \frac{c}{a}R_1} \begin{pmatrix} a & b & | & 1 & 0 \\ 0 & d-\frac{bc}{a} & | & -\frac{c}{a} & 1 \end{pmatrix} = \begin{pmatrix} a & b & | & 1 & 0 \\ 0 & \frac{ad-bc}{a} & | & -\frac{c}{a} & 1 \end{pmatrix}$$

把第 2 行除以 $\frac{ad-bc}{a}$（即乘以 $\frac{a}{ad-bc}$）：
$$\xrightarrow{R_2 \cdot \frac{a}{ad-bc}} \begin{pmatrix} a & b & | & 1 & 0 \\ 0 & 1 & | & \frac{-c}{ad-bc} & \frac{a}{ad-bc} \end{pmatrix}$$

消去第 1 行的 $b$：
$$\xrightarrow{R_1 - bR_2} \begin{pmatrix} a & 0 & | & 1+\frac{bc}{ad-bc} & \frac{-ab}{ad-bc} \\ 0 & 1 & | & \frac{-c}{ad-bc} & \frac{a}{ad-bc} \end{pmatrix}$$

第 1 行除以 $a$：
$$\xrightarrow{R_1 \div a} \begin{pmatrix} 1 & 0 & | & \frac{ad-bc+bc}{a(ad-bc)} & \frac{-b}{ad-bc} \\ 0 & 1 & | & \frac{-c}{ad-bc} & \frac{a}{ad-bc} \end{pmatrix} = \begin{pmatrix} 1 & 0 & | & \frac{d}{ad-bc} & \frac{-b}{ad-bc} \\ 0 & 1 & | & \frac{-c}{ad-bc} & \frac{a}{ad-bc} \end{pmatrix}$$

所以：
$$A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**这就是公式的来源！**

---

**详细例题**：$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$

**Step 1**：计算行列式 $\det(A) = 1\times4 - 2\times3 = -2$

**Step 2**：主对调 → $\begin{pmatrix} 4 & ? \\ ? & 1 \end{pmatrix}$

**Step 3**：副变号 → $\begin{pmatrix} 4 & -2 \\ -3 & 1 \end{pmatrix}$

**Step 4**：除以行列式 → $A^{-1} = \frac{1}{-2}\begin{pmatrix} 4 & -2 \\ -3 & 1 \end{pmatrix} = \begin{pmatrix} -2 & 1 \\ 1.5 & -0.5 \end{pmatrix}$

**验证**：
$$AA^{-1} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix} -2 & 1 \\ 1.5 & -0.5 \end{pmatrix} = \begin{pmatrix} -2+3 & 1-1 \\ -6+6 & 3-2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I$$

验证成功！

---

**$3 \times 3$ 矩阵的逆（伴随矩阵法）**

**公式**：
$$A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$$

其中 $\text{adj}(A)$ 是伴随矩阵。

---

**步骤 1：理解代数余子式**

**余子式** $M_{ij}$：删除第 $i$ 行第 $j$ 列后，剩下元素的行列式。

**代数余子式** $A_{ij}$：带符号的余子式
$$A_{ij} = (-1)^{i+j} M_{ij}$$

**符号规律**（"棋盘格"）：
```
+  -  +
-  +  -
+  -  +
```
位置 $(i,j)$ 的符号由 $(-1)^{i+j}$ 决定：
- $i+j$ 为偶数 → 正号 (+)
- $i+j$ 为奇数 → 负号 (-)

---

**步骤 2：构造伴随矩阵**

**伴随矩阵** = 代数余子式矩阵的**转置**

$$\text{adj}(A) = \begin{pmatrix} A_{11} & A_{21} & A_{31} \\ A_{12} & A_{22} & A_{32} \\ A_{13} & A_{23} & A_{33} \end{pmatrix}$$

**注意**：伴随矩阵是**转置**后的结果！$A_{ij}$ 放在第 $j$ 行第 $i$ 列。

---

**详细例题**：求 $A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{pmatrix}$ 的逆。

**Step 1**：计算行列式 $\det(A)$

按第一行展开：
$$\det(A) = 1 \cdot \det\begin{pmatrix} 1 & 4 \\ 6 & 0 \end{pmatrix} - 2 \cdot \det\begin{pmatrix} 0 & 4 \\ 5 & 0 \end{pmatrix} + 3 \cdot \det\begin{pmatrix} 0 & 1 \\ 5 & 6 \end{pmatrix}$$
$$= 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1$$

$\det(A) = 1 \neq 0$，故 $A$ 可逆。

---

**Step 2**：计算 9 个代数余子式

**第一行**：
- $A_{11} = (+1) \cdot \det\begin{pmatrix} 1 & 4 \\ 6 & 0 \end{pmatrix} = 1(0-24) = -24$
- $A_{12} = (-1) \cdot \det\begin{pmatrix} 0 & 4 \\ 5 & 0 \end{pmatrix} = -1(0-20) = 20$
- $A_{13} = (+1) \cdot \det\begin{pmatrix} 0 & 1 \\ 5 & 6 \end{pmatrix} = 1(0-5) = -5$

**第二行**：
- $A_{21} = (-1) \cdot \det\begin{pmatrix} 2 & 3 \\ 6 & 0 \end{pmatrix} = -1(0-18) = 18$
- $A_{22} = (+1) \cdot \det\begin{pmatrix} 1 & 3 \\ 5 & 0 \end{pmatrix} = 1(0-15) = -15$
- $A_{23} = (-1) \cdot \det\begin{pmatrix} 1 & 2 \\ 5 & 6 \end{pmatrix} = -1(6-10) = 4$

**第三行**：
- $A_{31} = (+1) \cdot \det\begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix} = 1(8-3) = 5$
- $A_{32} = (-1) \cdot \det\begin{pmatrix} 1 & 3 \\ 0 & 4 \end{pmatrix} = -1(4-0) = -4$
- $A_{33} = (+1) \cdot \det\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix} = 1(1-0) = 1$

---

**Step 3**：构造代数余子式矩阵并转置

代数余子式矩阵（按原位置排列）：
$$\begin{pmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \\ A_{31} & A_{32} & A_{33} \end{pmatrix} = \begin{pmatrix} -24 & 20 & -5 \\ 18 & -15 & 4 \\ 5 & -4 & 1 \end{pmatrix}$$

**伴随矩阵**（转置后）：
$$\text{adj}(A) = \begin{pmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{pmatrix}$$

**注意**：$A_{21}=18$ 放在第 1 行第 2 列，$A_{31}=5$ 放在第 1 行第 3 列，以此类推。

---

**Step 4**：代入公式

$$A^{-1} = \frac{1}{\det(A)}\text{adj}(A) = \frac{1}{1}\begin{pmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{pmatrix} = \begin{pmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{pmatrix}$$

---

**验证**（可选）：
$$AA^{-1} = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{pmatrix}\begin{pmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{pmatrix}$$

计算第一行第一列：$1(-24) + 2(20) + 3(-5) = -24 + 40 - 15 = 1$ ✓

计算第一行第二列：$1(18) + 2(-15) + 3(4) = 18 - 30 + 12 = 0$ ✓

可以验证其他元素，结果为单位矩阵 $I$。

---

**另一种方法：高斯 - 约旦消元法（适用于 3×3）**

对 $[A | I]$ 做行变换：

$$\left(\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 4 & 0 & 1 & 0 \\ 5 & 6 & 0 & 0 & 0 & 1 \end{array}\right)$$

逐步变换（过程略），最终得到：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & -24 & 18 & 5 \\ 0 & 1 & 0 & 20 & -15 & -4 \\ 0 & 0 & 1 & -5 & 4 & 1 \end{array}\right)$$

右边即为 $A^{-1}$，结果与伴随矩阵法一致！

---

**可逆矩阵的充要条件**（以下条件等价）：
- $\det(A) \neq 0$
- $\text{rank}(A) = n$（满秩）
- 所有特征值非零
- 列向量线性无关

**奇异矩阵**（不可逆）：$\det(A) = 0$

**例题**：$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$，$\det(A) = 4-4 = 0$，故**不可逆**。

几何解释：这个变换把平面"压扁"成一条线，丢失了信息，无法恢复。

---

### 2.3 特征值与特征向量 (#)

**直观理解**：

矩阵乘法 $Ax$ 可以看作对向量 $x$ 的**变换**。大多数向量经过变换后，方向和长度都会改变。

但有些特殊的向量，经过变换后**方向不变**（或反向），只改变长度。这些向量就是**特征向量**，伸缩的比例就是**特征值**。

```
        Ax（变换后）
        ↑
        │ 长度变为原来的 λ 倍
        │ 方向不变
   x（原向量）
        ↗

A 是变换矩阵，x 是特征向量，λ 是特征值
```

---

**定义**：对于 $n \times n$ 矩阵 $A$，若存在非零向量 $x$ 和标量 $\lambda$，使得

$$Ax = \lambda x$$

则称：
- $\lambda$ 是 **特征值**（eigenvalue）
- $x$ 是对应于 $\lambda$ 的 **特征向量**（eigenvector）

---

**如何求特征值和特征向量？**

**Step 1：从定义出发**

$$Ax = \lambda x$$

移项：
$$Ax - \lambda x = 0$$

提取 $x$（注意要写成矩阵形式）：
$$(A - \lambda I)x = 0$$

其中 $I$ 是单位矩阵。

---

**Step 2：什么时候有非零解？**

$(A - \lambda I)x = 0$ 是一个齐次线性方程组。

- 如果 $A - \lambda I$ 可逆（行列式不为 0），只有零解 $x = 0$，但特征向量要求非零，不符合！
- 如果 $A - \lambda I$ **不可逆**（行列式为 0），则有**非零解**！

**所以条件**：
$$\det(A - \lambda I) = 0$$

这个方程称为**特征方程**。

---

**Step 3：解特征方程得到特征值**

$\det(A - \lambda I)$ 是关于 $\lambda$ 的多项式，称为**特征多项式**。

解特征方程得到的根就是特征值。

---

**Step 4：对每个特征值，求对应的特征向量**

把 $\lambda$ 代入 $(A - \lambda I)x = 0$，解这个齐次方程组。

---

**详细例题**：求 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ 的特征值和特征向量。

---

**求特征值**：

**Step 1**：写出 $A - \lambda I$
$$A - \lambda I = \begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix}$$

**Step 2**：计算行列式，令其为 0
$$\det(A - \lambda I) = \det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix} = (2-\lambda)^2 - 1 = 0$$

**Step 3**：解方程
$$(2-\lambda)^2 - 1 = 0$$
$$(2-\lambda)^2 = 1$$
$$2-\lambda = \pm 1$$

解得：
- $\lambda_1 = 2 + 1 = 3$
- $\lambda_2 = 2 - 1 = 1$

---

**求特征向量**：

**对于 $\lambda_1 = 3$**：

代入 $(A - 3I)x = 0$：
$$\begin{pmatrix} 2-3 & 1 \\ 1 & 2-3 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

展开：
$$\begin{cases} -x_1 + x_2 = 0 \\ x_1 - x_2 = 0 \end{cases}$$

两个方程等价，都得到 $x_1 = x_2$。

取 $x_1 = 1$，则 $x_2 = 1$。

所以对应于 $\lambda_1 = 3$ 的特征向量为：
$$x_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

（任何 $c\begin{pmatrix} 1 \\ 1 \end{pmatrix}$，$c \neq 0$ 都是特征向量）

---

**对于 $\lambda_2 = 1$**：

代入 $(A - I)x = 0$：
$$\begin{pmatrix} 2-1 & 1 \\ 1 & 2-1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

展开：
$$\begin{cases} x_1 + x_2 = 0 \\ x_1 + x_2 = 0 \end{cases}$$

得到 $x_1 = -x_2$。

取 $x_1 = 1$，则 $x_2 = -1$。

所以对应于 $\lambda_2 = 1$ 的特征向量为：
$$x_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

---

**验证**（可选）：

验证 $\lambda_1 = 3$，$x_1 = [1, 1]^T$：
$$Ax_1 = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 3 \end{pmatrix} = 3\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \lambda_1 x_1$$

验证成功！

验证 $\lambda_2 = 1$，$x_2 = [1, -1]^T$：
$$Ax_2 = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \\ -1 \end{pmatrix} = 1\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \lambda_2 x_2$$

验证成功！

---

**几何解释**：

```
矩阵 A = [2 1; 1 2] 的变换效果：

y
↑
│    ● Ax₁ = 3x₁ (伸长 3 倍，方向不变)
│   /
│  / x₁ 方向
│ /
●───────●────→ x
    x₂/ Ax₂ = x₂ (长度不变，方向不变)
      /
     ●

特征向量 x₁ = [1,1]ᵗ：沿 y=x 方向，变换后伸长 3 倍
特征向量 x₂ = [1,-1]ᵗ：沿 y=-x 方向，变换后长度不变
```

---

**重要性质**：

| 性质 | 说明 | 例题验证 |
|------|------|----------|
| 特征值之和 = 迹 | $\lambda_1 + \lambda_2 = \text{tr}(A) = a_{11} + a_{22}$ | $3 + 1 = 4 = 2 + 2$ ✓ |
| 特征值之积 = 行列式 | $\lambda_1 \cdot \lambda_2 = \det(A)$ | $3 \times 1 = 3 = \det(A)$ ✓ |
| 对称矩阵的特征值是实数 | 若 $A = A^T$，则特征值都是实数 | $A$ 是对称的，特征值 3 和 1 都是实数 ✓ |
| 不同特征值的特征向量线性无关 | 不同特征值对应的特征向量方向不同 | $x_1$ 和 $x_2$ 线性无关 ✓ |

---

**特征值和特征向量的用途**：
- 矩阵对角化（简化矩阵幂的计算）
- 主成分分析（PCA）
- 振动分析（固有频率）
- 马尔可夫链的稳态
- Google 的 PageRank 算法

---

### 2.4 矩阵对角化 (#)

**核心问题**：计算 $A^k$（矩阵的 $k$ 次幂）很麻烦，尤其是 $k$ 很大的时候。

**对角化的目的**：把 $A$ 变成一个"简单形式"，方便计算。

---

#### 什么是对角化？

**对角矩阵**：只有主对角线上有非零元素的矩阵
$$D = \begin{pmatrix} \lambda_1 & 0 & \cdots \\ 0 & \lambda_2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

**对角化**：找到可逆矩阵 $P$ 和对角矩阵 $D$，使得
$$A = PDP^{-1}$$

或等价地：
$$P^{-1}AP = D$$

---

#### 为什么能这样分解？

**关键观察**：

假设 $A$ 有 $n$ 个线性无关的特征向量 $v_1, v_2, \dots, v_n$，对应的特征值是 $\lambda_1, \lambda_2, \dots, \lambda_n$。

根据定义：
$$Av_1 = \lambda_1 v_1$$
$$Av_2 = \lambda_2 v_2$$
$$\vdots$$
$$Av_n = \lambda_n v_n$$

把这些式子写成矩阵形式：

$$A[v_1 \ v_2 \ \cdots \ v_n] = [\lambda_1 v_1 \ \lambda_2 v_2 \ \cdots \ \lambda_n v_n]$$

令 $P = [v_1 \ v_2 \ \cdots \ v_n]$（用特征向量组成矩阵），则：
$$AP = [Av_1 \ Av_2 \ \cdots \ Av_n] = [\lambda_1 v_1 \ \lambda_2 v_2 \ \cdots \ \lambda_n v_n]$$

另一方面：
$$PD = [v_1 \ v_2 \ \cdots \ v_n]\begin{pmatrix} \lambda_1 & 0 & \cdots \\ 0 & \lambda_2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix} = [\lambda_1 v_1 \ \lambda_2 v_2 \ \cdots \ \lambda_n v_n]$$

**所以**：
$$AP = PD$$

两边同时左乘 $P^{-1}$：
$$P^{-1}AP = D$$

或者：
$$A = PDP^{-1}$$

---

#### 什么时候可以对角化？

**可对角化条件**：$n$ 阶矩阵 $A$ 有 $n$ 个**线性无关**的特征向量。

**注意**：不是所有矩阵都可以对角化！

**例子**：若某个特征值的几何重数（对应线性无关特征向量的个数）小于代数重数（特征方程中该特征值作为根的重数），则不能对角化。

---

#### 对角化后有什么用？

**最大好处**：计算 $A^k$ 变得超级简单！

$$A = PDP^{-1}$$

$$A^2 = (PDP^{-1})(PDP^{-1}) = PD(P^{-1}P)DP^{-1} = PDI DP^{-1} = PD^2P^{-1}$$

$$A^3 = A^2 \cdot A = PD^2P^{-1} \cdot PDP^{-1} = PD^3P^{-1}$$

以此类推：
$$A^k = PD^kP^{-1}$$

而 $D^k$ 超简单，只需要把对角线上的元素分别求 $k$ 次幂：
$$D^k = \begin{pmatrix} \lambda_1^k & 0 & \cdots \\ 0 & \lambda_2^k & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

---

#### 详细例题：对角化 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$

#### Step 1：求特征值和特征向量

由上节可知：
- 特征值：$\lambda_1 = 3$，$\lambda_2 = 1$
- 对应特征向量：$v_1 = [1, 1]^T$，$v_2 = [1, -1]^T$

#### Step 2：构造矩阵 $P$ 和 $D$

$P$ 的列是特征向量：
$$P = [v_1 \ v_2] = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$D$ 的对角线是对应特征值：
$$D = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$$

**注意顺序**：$P$ 的第 1 列对应 $D$ 的第 1 个对角元，第 2 列对应第 2 个对角元。

#### Step 3：求 $P^{-1}$

对于 $2 \times 2$ 矩阵 $P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$：

行列式：$\det(P) = 1(-1) - 1(1) = -2$

用公式法：
$$P^{-1} = \frac{1}{-2}\begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

#### Step 4：验证 $A = PDP^{-1}$

计算 $PDP^{-1}$：

首先计算 $PD$：
$$PD = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1\times3 + 1\times0 & 1\times0 + 1\times1 \\ 1\times3 + (-1)\times0 & 1\times0 + (-1)\times1 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix}$$

再乘以 $P^{-1}$：
$$PDP^{-1} = \begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 3+1 & 3-1 \\ 3-1 & 3+1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} = A$$

验证成功！

---

#### 计算 $A^{10}$ 的例子

**直接算**：需要 9 次矩阵乘法，非常麻烦！

**用对角化**：
$$A^{10} = PD^{10}P^{-1}$$

$$D^{10} = \begin{pmatrix} 3^{10} & 0 \\ 0 & 1^{10} \end{pmatrix} = \begin{pmatrix} 59049 & 0 \\ 0 & 1 \end{pmatrix}$$

$$A^{10} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 59049 & 0 \\ 0 & 1 \end{pmatrix}\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 59049 & 1 \\ 59049 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 59050 & 59048 \\ 59048 & 59050 \end{pmatrix} = \begin{pmatrix} 29525 & 29524 \\ 29524 & 29525 \end{pmatrix}$$

快得多！

---

#### 对称矩阵的正交对角化

如果 $A$ 是**实对称矩阵**（$A = A^T$），则有更强的性质：

**定理**：实对称矩阵一定可以对角化，且可以用**正交矩阵** $Q$ 对角化：
$$A = Q\Lambda Q^T$$

其中：
- $Q$ 是正交矩阵，满足 $Q^T = Q^{-1}$
- $Q$ 的列是**单位化**后的特征向量
- $\Lambda$ 是对角矩阵，对角线是特征值

**为什么特殊？** 因为 $Q^T = Q^{-1}$，所以不需要单独求逆矩阵！

---

#### 对称矩阵的例子：正交对角化

仍用 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$。

之前得到的特征向量：
- $v_1 = [1, 1]^T$（对应 $\lambda_1 = 3$）
- $v_2 = [1, -1]^T$（对应 $\lambda_2 = 1$）

**Step 1：单位化**

$$\|v_1\| = \sqrt{1^2 + 1^2} = \sqrt{2}$$
$$e_1 = \frac{1}{\sqrt{2}}[1, 1]^T = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}$$

$$\|v_2\| = \sqrt{1^2 + (-1)^2} = \sqrt{2}$$
$$e_2 = \frac{1}{\sqrt{2}}[1, -1]^T = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{pmatrix}$$

**Step 2：构造正交矩阵 $Q$**

$$Q = [e_1 \ e_2] = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 3：对角矩阵**

$$\Lambda = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$$

**验证**：$A = Q\Lambda Q^T$

$$Q\Lambda = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix}$$

$$Q\Lambda Q^T = \frac{1}{\sqrt{2}}\begin{pmatrix} 3 & 1 \\ 3 & -1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} = A$$

验证成功！

**注意**：$Q^TQ = I$，所以 $Q^{-1} = Q^T$，不需要额外求逆。

---

#### 总结

| 概念 | 公式/条件 |
|------|----------|
| 可对角化条件 | 有 $n$ 个线性无关的特征向量 |
| 对角化 | $A = PDP^{-1}$ |
| $P$ | 列是特征向量 |
| $D$ | 对角线是特征值（与 $P$ 的列顺序对应） |
| $A^k$ | $A^k = PD^kP^{-1}$ |
| 对称矩阵 | $A = Q\Lambda Q^T$，$Q$ 是正交矩阵 |

---

### 2.5 正定矩阵与半正定矩阵 (#)

**核心问题**：给定一个对称矩阵 $A$，如何判断它是否是"正定"的？为什么这个概念重要？

---

#### 什么是二次型？

对于对称矩阵 $A$ 和向量 $x$，表达式 $x^T A x$ 称为**二次型**。

**计算示例**：
$$A = \begin{pmatrix} a & b \\ b & c \end{pmatrix}, \quad x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

$$x^T A x = \begin{pmatrix} x_1 & x_2 \end{pmatrix}\begin{pmatrix} a & b \\ b & c \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

$$= \begin{pmatrix} x_1 & x_2 \end{pmatrix}\begin{pmatrix} ax_1 + bx_2 \\ bx_1 + cx_2 \end{pmatrix}$$

$$= x_1(ax_1 + bx_2) + x_2(bx_1 + cx_2) = ax_1^2 + 2bx_1x_2 + cx_2^2$$

这就是一个**二次多项式**！

---

#### 正定矩阵的定义

**定义**：对称矩阵 $A$ 是**正定矩阵**，如果对任意非零向量 $x \neq 0$，都有：
$$x^T A x > 0$$

**半正定矩阵**：对任意向量 $x$，都有：
$$x^T A x \geq 0$$

**负定矩阵**：对任意非零向量 $x \neq 0$，都有：
$$x^T A x < 0$$

**不定矩阵**：存在 $x_1, x_2$ 使得 $x_1^T A x_1 > 0$ 且 $x_2^T A x_2 < 0$

---

#### 几何意义（二维情况）

#### 正定矩阵的例子

考虑 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$，对应的二次型：
$$x^T A x = 2x_1^2 + 2x_1x_2 + 2x_2^2$$

**配方**：
$$= x_1^2 + x_2^2 + (x_1 + x_2)^2$$

这是三个平方项之和，任何非零向量都会得到正数！

```
        z = x^T A x (二次曲面)
        ↑
        │      ● (最高点)
        │     / \
        │    /   \
        │   /     \
        │  /       \
        │ /         \
        ·/___________\────→ x₂
         \           /
          \         /
           \       /
            \_____/
             x₁

这是一个向上开口的椭球面（碗状），最小值在原点 (0,0)
所有方向都是上升的 → 正定
```

#### 不定矩阵的例子

考虑 $A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$，对应的二次型：
$$x^T A x = x_1^2 - x_2^2$$

- 沿 $x_1$ 轴：$x_1^2 > 0$（向上）
- 沿 $x_2$ 轴：$-x_2^2 < 0$（向下）

```
        z = x₁² - x₂² (马鞍面)
        ↑
        │       /‾‾‾‾‾‾‾‾\
        │      /          \
        │     /            \
        │    ──●───────────\──→ x₂
        │     \            /
        │      \          /
        │       \________/

这是个"马鞍面"：某些方向向上，某些方向向下 → 不定
```

---

#### 判定条件（对称矩阵）

对于 $n \times n$ 对称矩阵 $A$，以下条件**等价**：

#### 方法 1：特征值判定

| 类型 | 特征值条件 |
|------|----------|
| **正定** | 所有特征值 $\lambda_i > 0$ |
| **半正定** | 所有特征值 $\lambda_i \geq 0$ |
| **负定** | 所有特征值 $\lambda_i < 0$ |
| **不定** | 有正有负的特征值 |

**原理**：对称矩阵可以正交对角化 $A = Q\Lambda Q^T$，则：
$$x^T A x = x^T Q \Lambda Q^T x = y^T \Lambda y = \sum_{i=1}^{n} \lambda_i y_i^2$$

其中 $y = Q^T x$。因为 $Q$ 可逆，当 $x \neq 0$ 时 $y \neq 0$。所以 $x^T A x$ 的符号由 $\lambda_i$ 决定。

---

#### 方法 2：顺序主子式（Sylvester 准则）

**顺序主子式**：删除最后 $k$ 行和最后 $k$ 列后剩下的 $k \times k$ 子矩阵的行列式。

对于 $A = \begin{pmatrix} a_{11} & a_{12} & \cdots \\ a_{21} & a_{22} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$：

- 一阶主子式：$\det(a_{11}) = a_{11}$
- 二阶主子式：$\det\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$
- 三阶主子式：$\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$
- ...

| 类型 | 顺序主子式条件 |
|------|--------------|
| **正定** | 全部 $> 0$ |
| **半正定** | 全部 $\geq 0$（需要验证所有主子式，不仅是顺序的） |
| **负定** | 符号交替：$(-1)^k \cdot \text{第 }k\text{ 个主子式} > 0$ |
| **不定** | 不满足上述条件 |

---

#### 详细例题：判断 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ 是否正定

#### 方法 1：特征值法

由上节可知，$A$ 的特征值为 $\lambda_1 = 3$，$\lambda_2 = 1$。

两个特征值都 $> 0$，故**正定**。✓

---

#### 方法 2：顺序主子式法

- 一阶主子式：$a_{11} = 2 > 0$ ✓
- 二阶主子式：$\det(A) = 2\times2 - 1\times1 = 3 > 0$ ✓

全部为正，故**正定**。✓

---

#### 方法 3：直接验证定义

对任意 $x = [x_1, x_2]^T \neq 0$：
$$x^T A x = 2x_1^2 + 2x_1x_2 + 2x_2^2$$

**配方**：
$$= x_1^2 + x_2^2 + x_1^2 + 2x_1x_2 + x_2^2$$
$$= x_1^2 + x_2^2 + (x_1 + x_2)^2$$

这是三个平方项之和：
- 每一项都 $\geq 0$
- 只有当 $x_1 = x_2 = 0$ 且 $x_1 + x_2 = 0$ 时才等于 0
- 但已知 $x \neq 0$，所以结果严格 $> 0$

故**正定**。✓

---

#### 另一个例题：$B = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$

#### 方法 1：特征值法

特征方程：$\det(B - \lambda I) = (1-\lambda)(4-\lambda) - 4 = \lambda^2 - 5\lambda = 0$

特征值：$\lambda_1 = 5$，$\lambda_2 = 0$

有一个特征值为 0，故**不是正定**，但可能是**半正定**。

---

#### 方法 2：顺序主子式法

- 一阶主子式：$1 > 0$ ✓
- 二阶主子式：$\det(B) = 0$ ✗（不正定要求严格大于 0）

不是正定矩阵。

---

#### 方法 3：直接验证

$$x^T B x = x_1^2 + 4x_1x_2 + 4x_2^2 = (x_1 + 2x_2)^2 \geq 0$$

对任意 $x$ 都 $\geq 0$，且当 $x_1 = -2x_2 \neq 0$ 时等于 0。

故**半正定**（但不是正定）。

---

#### 三维矩阵的例子

$C = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$

#### 方法 1：顺序主子式法

- 一阶：$2 > 0$ ✓
- 二阶：$\det\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} = 4 - 1 = 3 > 0$ ✓
- 三阶：$\det(C)$

计算三阶行列式（按第一行展开）：
$$\det(C) = 2\det\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} - 1\det\begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix} + 0$$
$$= 2(3) - 1(2) = 6 - 2 = 4 > 0$$

验证：三阶主子式为 $4 > 0$，符合要求。

---

#### 不定的例子

$D = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

- 一阶主子式：$1 > 0$
- 二阶主子式：$\det(D) = -1 < 0$

主元符号不一致，故**不定**。

验证：
- 取 $x = [1, 0]^T$：$x^T D x = 1 > 0$
- 取 $x = [0, 1]^T$：$x^T D x = -1 < 0$

确实有正有负 → 不定。

---

#### 半正定的判定

**注意**：半正定用顺序主子式判定时，需要验证**所有主子式**（不仅仅是顺序的），而不仅限于顺序主子式。

例子：$E = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$

- 顺序主子式：一阶 $0 \geq 0$，二阶 $0 \geq 0$
- 但这个矩阵确实是半正定的（特征值为 0 和 1）

---

#### 总结表

| 类型 | 特征值 | 顺序主子式（正定/负定） | 定义 |
|------|--------|----------------------|------|
| **正定** | 全部 $> 0$ | 全部 $> 0$ | $x \neq 0 \Rightarrow x^T A x > 0$ |
| **半正定** | 全部 $\geq 0$ | 所有主子式 $\geq 0$ | 对所有 $x$: $x^T A x \geq 0$ |
| **负定** | 全部 $< 0$ | $(-1)^k \Delta_k > 0$ | $x \neq 0 \Rightarrow x^T A x < 0$ |
| **不定** | 有正有负 | 不满足以上 | 存在 $x_1, x_2$ 使符号相反 |

---

#### 应用

#### 1. 优化理论中的海森矩阵

如果函数 $f(x)$ 在点 $x^*$ 处的海森矩阵 $H_f(x^*)$ 是**正定**的，则 $x^*$ 是**局部极小值点**。

**例子**：$f(x,y) = x^2 + y^2$ 的海森矩阵是 $\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$，正定，故 $(0,0)$ 是最小值点。

#### 2. 协方差矩阵一定是半正定的

统计中，随机向量的协方差矩阵总是半正定的。这是因为：
$$\text{Var}(a^T X) = a^T \Sigma a \geq 0$$

方差永远非负！

#### 3. 机器学习中的正则化

添加 $L_2$ 正则化相当于在损失函数的海森矩阵上加一个正定矩阵（$\lambda I$），使优化问题更稳定。

---

### 2.6 奇异值分解 (SVD)

**核心价值**：SVD 是线性代数中最强大的工具之一，可以把任意（甚至非方阵）分解为简单的正交矩阵和对角矩阵的乘积，应用极其广泛。

---

#### 什么是 SVD？

**定理**：任意 $m \times n$ 矩阵 $A$ 都可以分解为：
$$A = U\Sigma V^T$$

其中：
- $U$：$m \times m$ 正交矩阵，列向量称为**左奇异向量**
- $\Sigma$：$m \times n$ 对角矩阵，对角线元素 $\sigma_i \geq 0$，称为**奇异值**（按从大到小排列）
- $V$：$n \times n$ 正交矩阵，列向量称为**右奇异向量**
- $V^T$ 是 $V$ 的转置（因为 $V$ 正交，所以 $V^{-1} = V^T$）

---

#### SVD 与特征值分解的区别

**特征值分解**：$A = PDP^{-1}$
- 只适用于**方阵**（$n \times n$）
- $P$ 不一定是正交矩阵

**SVD**：$A = U\Sigma V^T$
- 适用于**任意形状**的矩阵（$m \times n$，可以不是方阵）
- $U, V$ 都是正交矩阵（更稳定）
- 奇异值永远非负（$\sigma_i \geq 0$），特征值可能为负

---

#### 几何意义（二维示例）

矩阵 $A$ 对向量 $x$ 的变换 $Ax$ 可以分解为三步：
1. $V^T$：在原坐标系下旋转/反射（不改变长度）
2. $\Sigma$：在各坐标轴上缩放（拉伸/压缩，奇异值是缩放比例）
3. $U$：在新坐标系下旋转/反射（不改变长度）

```
x → 旋转 → 缩放 → 旋转 → Ax
    Vᵀ     Σ      U
```

---

#### 与特征值的关系

SVD 可以通过两个对称半正定矩阵的特征值分解得到：

1. **右奇异向量 $V$**：$A^TA$ 的特征向量
2. **左奇异向量 $U$**：$AA^T$ 的特征向量
3. **奇异值 $\sigma_i$**：$A^TA$（或 $AA^T$）的特征值的平方根

即：
$$\sigma_i = \sqrt{\lambda_i(A^TA)} = \sqrt{\lambda_i(AA^T)}$$

---

#### 为什么这样是对的？

从正交性推导：
$$A = U\Sigma V^T$$
两边右乘 $V$：
$$AV = U\Sigma$$
$$Av_i = \sigma_i u_i$$

两边左乘 $A^T$：
$$A^T A v_i = \sigma_i A^T u_i = \sigma_i \cdot \sigma_i v_i = \sigma_i^2 v_i$$

所以 $v_i$ 是 $A^TA$ 的特征向量，对应特征值 $\sigma_i^2$。

---

#### 算法步骤（如何手动计算 SVD）

**目标**：对任意 $m \times n$ 矩阵 $A$，求 $U, \Sigma, V$。

#### Step 1：计算 $A^TA$ 和 $AA^T$
- $A^TA$ 是 $n \times n$ 对称半正定矩阵
- $AA^T$ 是 $m \times m$ 对称半正定矩阵

#### Step 2：求 $A^TA$ 的特征值和特征向量
- 特征值：$\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n \geq 0$
- 对应单位正交特征向量：$v_1, v_2, \dots, v_n$ → 组成右奇异矩阵 $V = [v_1 \ v_2 \ \dots \ v_n]$

#### Step 3：计算奇异值
$$\sigma_i = \sqrt{\lambda_i}, \quad i = 1, \dots, \min(m,n)$$
奇异值按降序排列，组成对角矩阵 $\Sigma$ 的对角线，其余元素为 0。

#### Step 4：求左奇异向量 $U$
对每个非零奇异值 $\sigma_i > 0$：
$$u_i = \frac{1}{\sigma_i} A v_i$$
剩下的 $u_i$ 补全为正交基。

---

#### 详细例题 1：2×2 非对角矩阵

求 $A = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$ 的 SVD。

---

#### Step 1：计算 $A^TA$
$$A^T = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}$$
$$A^TA = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

---

#### Step 2：求 $A^TA$ 的特征值和特征向量

**特征方程**：
$$\det(A^TA - \lambda I) = \det\begin{pmatrix} 1-\lambda & 1 \\ 1 & 1-\lambda \end{pmatrix} = (1-\lambda)^2 - 1 = \lambda^2 - 2\lambda = 0$$

解得特征值：
$$\lambda_1 = 2, \quad \lambda_2 = 0$$

**求特征向量**：

- 对 $\lambda_1 = 2$：
  $$(A^TA - 2I)x = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = 0 \Rightarrow x_1 = x_2$$
  单位化：$v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

- 对 $\lambda_2 = 0$：
  $$(A^TA - 0I)x = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = 0 \Rightarrow x_1 = -x_2$$
  单位化：$v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$

右奇异矩阵：
$$V = [v_1 \ v_2] = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

---

#### Step 3：计算奇异值
$$\sigma_1 = \sqrt{\lambda_1} = \sqrt{2}, \quad \sigma_2 = \sqrt{\lambda_2} = 0$$

奇异值矩阵：
$$\Sigma = \begin{pmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{pmatrix}$$

---

#### Step 4：求左奇异向量 $U$

**对 $\sigma_1 = \sqrt{2}$**：
$$u_1 = \frac{1}{\sigma_1} A v_1 = \frac{1}{\sqrt{2}} \cdot \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

**补充 $u_2$**：需要找一个与 $u_1$ 正交的单位向量，取 $u_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$。

左奇异矩阵：
$$U = [u_1 \ u_2] = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

---

#### Step 5：最终 SVD
$$A = U\Sigma V^T = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**验证**：
$$U\Sigma = \begin{pmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{pmatrix}$$
$$U\Sigma V^T = \begin{pmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = A$$

验证成功！

---

#### 详细例题 2：非方阵（2×3 矩阵）

求 $A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$ 的 SVD。

---

#### Step 1：计算 $A^TA$
$$A^TA = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$$

#### Step 2：求特征值
特征方程：$\det(A^TA - \lambda I) = 0$，解得：
$$\lambda_1 = 3, \quad \lambda_2 = 1, \quad \lambda_3 = 0$$

#### Step 3：奇异值
$$\sigma_1 = \sqrt{3}, \quad \sigma_2 = \sqrt{1} = 1$$

#### Step 4：右奇异向量
单位特征向量：
$$v_1 = \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ 1 \\ 2 \end{pmatrix}, \quad v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \\ 0 \end{pmatrix}, \quad v_3 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 \\ 1 \\ -1 \end{pmatrix}$$

#### Step 5：左奇异向量
$$u_1 = \frac{1}{\sqrt{3}} A v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$
$$u_2 = \frac{1}{1} A v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

#### 最终 SVD
$$A = \underbrace{\begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}}_{U} \underbrace{\begin{pmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}}_{\Sigma} \underbrace{\begin{pmatrix} \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{3}} \end{pmatrix}}_{V^T}$$

---

#### SVD 的核心性质

#### 1. 秩
矩阵 $A$ 的秩 = 非零奇异值的个数。

#### 2. 范数
- 算子 2 范数：$\|A\|_2 = \sigma_1$（最大奇异值）
- Frobenius 范数：$\|A\|_F = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$

#### 3. 低秩近似（截断 SVD）
取前 $k$ 个最大的奇异值，可以近似原矩阵：
$$A \approx U_k \Sigma_k V_k^T$$

其中 $U_k$ 取 $U$ 的前 $k$ 列，$\Sigma_k$ 取前 $k \times k$ 个元素，$V_k$ 取 $V$ 的前 $k$ 列。

这是数据压缩、主成分分析（PCA）、推荐系统的核心原理！

---

#### 应用场景

1. **主成分分析（PCA）**：右奇异向量就是主成分方向
2. **推荐系统**：SVD 矩阵分解预测用户评分
3. **图像处理**：低秩近似压缩图像
4. **自然语言处理**：潜在语义分析（LSA）
5. **最小二乘问题**：SVD 求解病态方程组
6. **信号处理**：去噪（保留大奇异值，去掉小奇异值）

---

# 第二部分：概率论与数理统计

## 第 3 章 概率基础

### 3.1 样本空间、事件与概率公理 (#)

**随机试验**：可以重复进行，结果不唯一，事前不知道结果的试验。

**样本空间 $\Omega$**：随机试验所有可能结果的集合。每个结果称为一个**样本点**。

**例题**：
- 掷一枚硬币：$\Omega = \{\text{正面}, \text{反面}\}$
- 掷一枚骰子：$\Omega = \{1, 2, 3, 4, 5, 6\}$
- 测量一个人的身高：$\Omega = [0, 300]$（单位：厘米）

**事件**：样本空间的子集（某些样本点组成的集合）。

**常用记号**：
- $A$：事件 $A$ 发生
- $A \subset B$：如果 $A$ 发生则 $B$ 一定发生
- $A \cup B$：$A$ 或 $B$ 至少一个发生
- $A \cap B$（或 $A \cap B$）：$A$ 和 $B$ 都发生
- $A^c$（补集）：$A$ 不发生
- $\emptyset$：空事件（不可能事件）

**例题**：掷骰子
- 事件 $A$ = "掷出偶数" = $\{2, 4, 6\}$
- 事件 $B$ = "掷出大于 4" = $\{5, 6\}$
- $A \cap B = \{6\}$
- $A \cup B = \{2, 4, 5, 6\}$
- $A^c = \{1, 3, 5\}$

---

#### 概率公理（Kolmogorov）

概率 $P$ 是定义在事件上的函数，满足三条公理：

1. **非负性**：对任意事件 $A$，$P(A) \geq 0$

2. **规范性**：整个样本空间的概率为 1
   $$P(\Omega) = 1$$

3. **可列可加性**：对可数个两两互斥的事件 $A_1, A_2, \dots$（即 $i \neq j \Rightarrow A_i \cap A_j = \emptyset$），有：
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

---

#### 重要性质

**性质 1：对立事件的概率**
$$P(A^c) = 1 - P(A)$$

**证明**：$A$ 和 $A^c$ 互斥，且 $A \cup A^c = \Omega$，所以：
$$P(A) + P(A^c) = P(\Omega) = 1 \Rightarrow P(A^c) = 1 - P(A)$$

**性质 2：加法公式**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**理解**：直接相加会把交集 $A \cap B$ 重复计算一次，所以减去一次。

**韦恩图表示**：
```
  ┌───────────────┐
  │  A     ┌──────┼──────┐  B
  │       │  A∩B │      │
  └───────┼──────┘      │
          └──────────────┘

面积：A + B - A∩B
```

**性质 3：单调性**
如果 $A \subset B$，则 $P(A) \leq P(B)$。

---

**例题**：掷骰子

- $P(\text{偶数}) = \frac{3}{6} = \frac{1}{2}$
- $P(\text{大于 4}) = \frac{2}{6} = \frac{1}{3}$
- $P(\text{偶数和大于 4}) = P(\{6\}) = \frac{1}{6}$
- $P(\text{偶数或大于 4}) = P(A) + P(B) - P(A \cap B) = \frac{1}{2} + \frac{1}{3} - \frac{1}{6} = \frac{4}{6} = \frac{2}{3}$

结果就是 $P(\{2, 4, 5, 6\}) = \frac{4}{6} = \frac{2}{3}$，和直接计算一致。

---

### 3.2 条件概率、全概率公式、贝叶斯定理 (#)

#### 条件概率

**问题**：已知事件 $B$ 已经发生，求事件 $A$ 发生的概率，记作 $P(A|B)$。

**定义**：
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**直观理解**：在已知 $B$ 发生的条件下，样本空间缩小到 $B$，我们重新归一化概率。

**韦恩图**：
```
  ┌───────────────────────────┐
  │  Ω                      │
  │    ┌───────┐             │
  │    │   B  │             │
  │    │  ┌──┼───┐          │
  │    │  │A∩B│   │          │
  │    │  └──┼───┘          │
  │    └───────┘             │
  └───────────────────────────┘

P(A|B) = 面积(A∩B) / 面积(B)
```

**例题**：已知掷骰子结果是偶数，求是 2 的概率

- $A =$ "结果是 2"，$B =$ "结果是偶数"
- $A \cap B = \{2\}$，$B = \{2, 4, 6\}$

$$P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{1/6}{3/6} = \frac{1}{3}$$

---

#### 乘法公式

从条件概率定义变形得到：
$$P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$$

**推广到三个事件**：
$$P(A \cap B \cap C) = P(A)P(B|A)P(C|A \cap B)$$

---

#### 划分

**定义**：$B_1, B_2, \dots, B_n$ 称为样本空间 $\Omega$ 的一个**划分**，如果：
1. $B_1, B_2, \dots, B_n$ 两两互斥：$B_i \cap B_j = \emptyset \ (i \neq j)$
2. 它们的并是整个样本空间：$B_1 \cup B_2 \cup \dots \cup B_n = \Omega$

**图示**：
```
  ┌───────┬───────┬───────┐
  │  B₁  │  B₂  │  B₃  │
  ├──────┼──────┼──────┤
  │  B₄  │  B₅  │  B₆  │
  └──────┴──────┴──────┘
  整个空间被分成互不重叠的几块
```

---

#### 全概率公式

如果 $B_1, \dots, B_n$ 是样本空间的一个划分，则对任意事件 $A$：
$$P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$

**推导**：
$$A = A \cap \Omega = A \cap \left(\bigcup_{i=1}^{n} B_i\right) = \bigcup_{i=1}^{n} (A \cap B_i)$$
因为 $B_i$ 两两互斥，所以 $A \cap B_i$ 也两两互斥，由可列可加性：
$$P(A) = \sum_{i=1}^{n} P(A \cap B_i) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$

---

**典型例题**：某疾病患病率 1%，检测准确率为：患者 99% 阳性，健康人 5% 阳性。求随机一人检测为阳性的概率。

设：
- $B_1$ = 患病，$P(B_1) = 0.01$
- $B_2$ = 健康，$P(B_2) = 0.99$
- $A$ = 检测阳性

已知：
- $P(A|B_1) = 0.99$（患者检测出阳性）
- $P(A|B_2) = 0.05$（健康人误检为阳性）

全概率公式：
$$P(A) = P(A|B_1)P(B_1) + P(A|B_2)P(B_2) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

结果：随机一个人检测出阳性的概率约 6%。

---

#### 贝叶斯定理（Bayes' Theorem）

已知先验概率 $P(B_i)$，观测到 $A$ 发生了，求后验概率 $P(B_i|A)$。

$$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{P(A)} = \frac{P(A|B_i)P(B_i)}{\sum_{j=1}^{n} P(A|B_j)P(B_j)}$$

**贝叶斯定理本质**：
- **先验概率** $P(B_i)$：在观测到 $A$ 之前，我们对 $B_i$ 的概率估计
- **后验概率** $P(B_i|A)$：在观测到 $A$ 之后，我们更新后的概率估计

---

**例题**（续上题）：已知检测为阳性，求确实患病的概率

$$P(B_1|A) = \frac{P(A|B_1)P(B_1)}{P(A)} = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167$$

**惊人结论**：即使检测准确率 99%，检测阳性的人真的患病概率只有约 17%！这是因为患病率很低（1%），大部分阳性结果都是误检。

---

#### 贝叶斯定理在机器学习中的应用：
- **朴素贝叶斯分类器**：计算后验概率 $P(y|x)$ 进行分类
- **贝叶斯参数估计**：最大后验估计（MAP）就是贝叶斯思想的应用

---

### 3.3 独立性 (#)

#### 事件的独立性

**直观理解**：两个事件独立，意味着一个事件的发生不影响另一个事件发生的概率。

**定义**：事件 $A$ 和 $B$ 独立，如果：
$$P(A \cap B) = P(A)P(B)$$

**等价定义**（当 $P(B) > 0$ 时）：
$$P(A|B) = P(A)$$

**证明**：
$$P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A)P(B)}{P(B)} = P(A)$$

---

**例题 1**：掷两枚硬币

- $A$ = "第一枚正面"，$P(A) = 0.5$
- $B$ = "第二枚正面"，$P(B) = 0.5$
- $A \cap B$ = "两枚都是正面"，$P(A \cap B) = 0.25$

验证：$P(A)P(B) = 0.5 \times 0.5 = 0.25 = P(A \cap B)$ ✓

故 $A$ 和 $B$ 独立。

---

**例题 2**：掷一枚骰子

- $A$ = "掷出偶数" = $\{2, 4, 6\}$，$P(A) = 1/2$
- $B$ = "掷出大于 3" = $\{4, 5, 6\}$，$P(B) = 1/2$
- $A \cap B$ = $\{4, 6\}$，$P(A \cap B) = 1/3$

验证：$P(A)P(B) = 1/2 \times 1/2 = 1/4 \neq 1/3 = P(A \cap B)$

故 $A$ 和 $B$ **不独立**。（知道掷出大于 3 后，是偶数的概率变了）

---

#### 多个事件的独立性

**两两独立**：$A, B, C$ 两两独立，如果：
- $P(A \cap B) = P(A)P(B)$
- $P(A \cap C) = P(A)P(C)$
- $P(B \cap C) = P(B)P(C)$

**相互独立**（更强）：除了两两独立，还要满足：
$$P(A \cap B \cap C) = P(A)P(B)P(C)$$

**注意**：两两独立 ≠ 相互独立！

---

#### 条件独立性

**定义**：给定事件 $C$，$A$ 和 $B$ 条件独立，如果：
$$P(A \cap B|C) = P(A|C)P(B|C)$$

**直观理解**：在已知 $C$ 的条件下，$A$ 和 $B$ 互不影响。

**例题**：
- $A$ = "明天下雨"
- $B$ = "后天下雨"
- $C$ = "现在是夏季"

在已知是夏季的条件下，明天和后天的降雨可能仍然相关（不独立）。但如果 $C$ = "明天的天气"，那么在已知明天天气的情况下，今天和后天的天气可能条件独立。

---

#### 随机变量的独立性

**定义**：随机变量 $X$ 和 $Y$ 独立，如果它们的联合分布等于边缘分布的乘积：

**离散型**：
$$P(X = x, Y = y) = P(X = x) \cdot P(Y = y), \quad \forall x, y$$

**连续型**：
$$f_{XY}(x, y) = f_X(x) \cdot f_Y(y), \quad \forall x, y$$

**性质**：如果 $X$ 和 $Y$ 独立，则：
- $E[XY] = E[X]E[Y]$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
- $\text{Cov}(X, Y) = 0$

---

**例题**：掷两枚硬币

令 $X$ = 第一枚的结果（1=正面，0=反面），$Y$ = 第二枚的结果。

联合分布：
| X\Y | 0 | 1 |
|-----|---|---|
| 0 | 0.25 | 0.25 |
| 1 | 0.25 | 0.25 |

边缘分布：
- $P(X=0) = 0.25 + 0.25 = 0.5$
- $P(X=1) = 0.25 + 0.25 = 0.5$
- $P(Y=0) = 0.25 + 0.25 = 0.5$
- $P(Y=1) = 0.25 + 0.25 = 0.5$

验证：$P(X=0, Y=0) = 0.25 = P(X=0)P(Y=0) = 0.5 \times 0.5 = 0.25$ ✓

所有组合都满足，故 $X$ 和 $Y$ 独立。

---

## 第 4 章 随机变量与常用分布

### 4.1 随机变量 (#)

#### 什么是随机变量？

**定义**：从样本空间 $\Omega$ 到实数 $\mathbb{R}$ 的映射 $X: \Omega \to \mathbb{R}$。

**直观理解**：随机变量是把随机试验的结果"数字化"，用数值来表示结果。

**例题**：
1. 掷两枚硬币，令 $X$ = 正面次数
   - 样本空间：$\Omega = \{HH, HT, TH, TT\}$
   - $X(HH) = 2, X(HT) = 1, X(TH) = 1, X(TT) = 0$
   - $X$ 可取 0, 1, 2

2. 测量一天的降雨量，$X$ = 降雨毫米数
   - $X$ 可取 $[0, \infty)$ 中的任意实数

---

#### 随机变量的分类

**离散型随机变量**：取值为有限个或可数无限个离散值。

**连续型随机变量**：取值为某个区间内的任意实数。

---

#### 分布函数

**累积分布函数 (CDF)**：描述随机变量取值不超过某个数的概率。
$$F_X(x) = P(X \leq x)$$

**性质**：
1. 单调不减
2. $F_X(-\infty) = 0$，$F_X(+\infty) = 1$
3. 右连续

---

#### 离散型：概率质量函数 (PMF)

**定义**：$p_X(x) = P(X = x)$

**性质**：
1. $p_X(x) \geq 0$
2. $\sum_x p_X(x) = 1$

**例题**：掷一枚公平骰子，$X$ = 点数
$$p_X(k) = P(X = k) = \frac{1}{6}, \quad k = 1, 2, 3, 4, 5, 6$$

CDF 为阶梯函数：
$$F_X(x) = \begin{cases}
0 & x < 1 \\
1/6 & 1 \leq x < 2 \\
2/6 & 2 \leq x < 3 \\
\vdots \\
1 & x \geq 6
\end{cases}$$

---

#### 连续型：概率密度函数 (PDF)

**定义**：如果 $F_X(x)$ 可导，则：
$$f_X(x) = \frac{d}{dx}F_X(x)$$

反之：
$$F_X(x) = \int_{-\infty}^{x} f_X(t) dt$$

**性质**：
1. $f_X(x) \geq 0$
2. $\int_{-\infty}^{+\infty} f_X(x) dx = 1$

**重要区别**：连续型随机变量在某一点的概率为 0！
$$P(X = a) = 0$$

我们只能计算区间概率：
$$P(a \leq X \leq b) = \int_{a}^{b} f_X(x) dx = F_X(b) - F_X(a)$$

---

**例题**：均匀分布 $X \sim U(0, 1)$

PDF：
$$f_X(x) = \begin{cases}
1 & 0 \leq x \leq 1 \\
0 & \text{其他}
\end{cases}$$

CDF：
$$F_X(x) = \begin{cases}
0 & x < 0 \\
x & 0 \leq x \leq 1 \\
1 & x > 1
\end{cases}$$

$P(0.3 \leq X \leq 0.7) = 0.7 - 0.3 = 0.4$

---

### 4.2 离散分布 (#)

#### 常见离散分布一览表

| 分布 | 参数 | 概率质量函数 | 期望 | 方差 | 应用场景 |
|------|------|-------------|------|------|----------|
| 伯努利 Bern(φ) | $0 < \phi < 1$ | $P(X=1)=\phi, P(X=0)=1-\phi$ | $\phi$ | $\phi(1-\phi)$ | 单次成功/失败 |
| 二项 B(n,φ) | $n \in \mathbb{N}, 0 < \phi < 1$ | $\binom{n}{k}\phi^k(1-\phi)^{n-k}$ | $n\phi$ | $n\phi(1-\phi)$ | $n$ 次独立试验成功次数 |
| 泊松 Poi(λ) | $\lambda > 0$ | $\frac{e^{-\lambda}\lambda^k}{k!}$ | $\lambda$ | $\lambda$ | 单位时间内事件发生次数 |
| 几何 Geo(p) | $0 < p < 1$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | 首次成功所需的试验次数 |

---

#### 1. 伯努利分布 Bernoulli(φ)

**定义**：单次试验，只有两种结果（成功/失败）。

**概率质量函数**：
$$P(X = k) = \begin{cases}
\phi & k = 1 \text{（成功）} \\
1-\phi & k = 0 \text{（失败）}
\end{cases}$$

**期望**：$E[X] = 1 \cdot \phi + 0 \cdot (1-\phi) = \phi$

**方差**：
方差衡量随机变量偏离其期望的平均程度，定义为：
$$\text{Var}(X) = E[(X - E[X])^2]$$

令 $\mu = E[X]$，则：
$$\text{Var}(X) = E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

所以常用计算公式是：
$$\text{Var}(X) = E[X^2] - (E[X])^2$$

对伯努利分布，$X$ 只可能取 0 或 1，因此 $X^2 = X$，所以：
$$E[X^2] = E[X] = \phi$$

代入方差公式：
$$\text{Var}(X) = E[X^2] - (E[X])^2 = \phi - \phi^2 = \phi(1-\phi)$$

---

**例题 1**：抛一枚公平硬币

$\phi = 0.5$

- $E[X] = 0.5$
- $\text{Var}(X) = 0.5 \times 0.5 = 0.25$

---

#### 2. 二项分布 Binomial(n, φ)

**定义**：$n$ 次独立的伯努利试验，成功次数 $X$ 服从二项分布。

**记作**：$X \sim B(n, \phi)$

**概率质量函数**：
$$P(X = k) = \binom{n}{k}\phi^k(1-\phi)^{n-k}, \quad k = 0, 1, \dots, n$$

其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 是组合数。

**期望**：$E[X] = n\phi$

**方差**：$\text{Var}(X) = n\phi(1-\phi)$

**推导期望**：

二项分布可以拆成 $n$ 个**独立同分布的伯努利试验**之和。

令 $X_i$ 为第 $i$ 次试验的结果：
$$X_i = \begin{cases} 1 & \text{成功（概率 } \phi) \\ 0 & \text{失败（概率 } 1-\phi) \end{cases}$$

则 $X_i \sim \text{Bernoulli}(\phi)$，且 $X = X_1 + X_2 + \dots + X_n$。

**单次伯努利试验的期望**：
$$E[X_i] = 1 \cdot \phi + 0 \cdot (1-\phi) = \phi$$

**由期望的线性性质**（和的期望 = 期望的和，不需要独立）：
$$E[X] = E[X_1 + \dots + X_n] = E[X_1] + \dots + E[X_n] = \phi + \phi + \dots + \phi = n\phi$$

---

**推导方差**：

先求单次伯努利试验的方差。因为 $X_i$ 只取 0 或 1，所以 $X_i^2 = X_i$，从而 $E[X_i^2] = E[X_i] = \phi$。

$$\begin{aligned}
\text{Var}(X_i) &= E[X_i^2] - (E[X_i])^2 \\
&= \phi - \phi^2 \\
&= \phi(1-\phi)
\end{aligned}$$

**利用方差的可加性**（注意：要求各变量相互独立，伯努利试验恰好满足）：
$$\begin{aligned}
\text{Var}(X) &= \text{Var}(X_1 + \dots + X_n) \\
&= \text{Var}(X_1) + \dots + \text{Var}(X_n) \\
&= n\phi(1-\phi)
\end{aligned}$$

**关键点总结**：
- 期望的线性性质**不需要**变量独立，任何时候都成立：$E[\sum X_i] = \sum E[X_i]$
- 方差的可加性**需要**变量相互独立：$\text{Var}(\sum X_i) = \sum \text{Var}(X_i)$（仅当 $X_i$ 两两独立时成立）

---

**例题 2**：抛 10 次公平硬币，正面次数 $X \sim B(10, 0.5)$

**求恰好 5 次正面的概率**：
$$P(X = 5) = \binom{10}{5} (0.5)^5 (0.5)^5 = \frac{10!}{5!5!} (0.5)^{10} = 252 \times \frac{1}{1024} \approx 0.246$$

**期望和方差**：
$$\begin{aligned}
E[X] &= 10 \times 0.5 = 5 \quad (\text{抛 10 次平均得到 5 次正面}) \\
\text{Var}(X) &= 10 \times 0.5 \times 0.5 = 2.5 \\
\sigma &= \sqrt{\text{Var}(X)} = \sqrt{2.5} \approx 1.58 \quad (\text{标准差，约 68\% 的数据落在 } 5 \pm 1.58 \text{ 内})
\end{aligned}$$

---

**二项分布的图像**（n=10, φ=0.5）：
```
P(X=k)
  │
  │         ●
  │       ●   ●
  │     ●       ●
  │   ●           ●
  │ ●               ●
  └─────────────────────→ k
    0 1 2 3 4 5 6 7 8 9 10

对称的钟形，峰值在 k=5
```

---

#### 3. 泊松分布 Poisson(λ)

**定义**：描述单位时间（或空间）内稀有事件发生的次数。

**记作**：$X \sim \text{Poi}(\lambda)$

**概率质量函数**：
$$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0, 1, 2, \dots$$

其中 $\lambda > 0$ 是单位时间内的平均发生次数（率 × 时间长度）。

---

**泊松分布是如何推导出来的？——从二项分布取极限**

泊松分布实际上是二项分布在 $n \to \infty$、$\phi \to 0$ 且 $n\phi = \lambda$ 保持常数时的极限情况。

从二项分布的 PMF 出发：
$$P(X = k) = \binom{n}{k}\phi^k(1-\phi)^{n-k}$$

令 $\phi = \frac{\lambda}{n}$（即 $n\phi = \lambda$ 固定），取 $n \to \infty$：

$$\begin{aligned}
P(X = k) &= \binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1-\frac{\lambda}{n}\right)^{n-k} \\[4pt]
&= \frac{n(n-1)\cdots(n-k+1)}{k!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1-\frac{\lambda}{n}\right)^n \cdot \left(1-\frac{\lambda}{n}\right)^{-k}
\end{aligned}$$

当 $n \to \infty$ 时：

1. $\frac{n(n-1)\cdots(n-k+1)}{n^k} \to 1$ （分子分母最高次项都是 $n^k$，系数趋近于 1）
2. $\left(1-\frac{\lambda}{n}\right)^n \to e^{-\lambda}$ （重要极限 $\lim_{n\to\infty}(1+\frac{x}{n})^n = e^x$，这里 $x = -\lambda$）
3. $\left(1-\frac{\lambda}{n}\right)^{-k} \to 1$ （$\frac{\lambda}{n} \to 0$，该项趋近于 1）

因此：
$$P(X = k) \to \frac{\lambda^k}{k!} \cdot e^{-\lambda}$$

这就是泊松分布的 PMF。

**直观理解**：当某事件发生的概率很小（$\phi \to 0$），但试验次数非常大（$n \to \infty$）时，事件发生的次数 $k$ 不会太大（因为 $\phi$ 很小），且可以用泊松分布近似二项分布。例如：一小时内大量独立访客访问网站，每人访问的概率都很小，总访问次数近似服从泊松分布。

---

**期望和方差的推导**

**方法一：利用泊松分布与二项分布的关系**

既然泊松分布是二项分布 $B(n, \phi)$ 在 $n \to \infty, \phi \to 0, n\phi = \lambda$ 时的极限：
- $E[X] = \lim\, n\phi = \lambda$
- $\text{Var}(X) = \lim\, n\phi(1-\phi) = \lim\, \lambda(1-\phi) = \lambda$（因为 $\phi \to 0$，所以 $1-\phi \to 1$）

**方法二：直接从泊松 PMF 推导期望**

$$\begin{aligned}
E[X] &= \sum_{k=0}^\infty k \cdot \frac{e^{-\lambda}\lambda^k}{k!} \\
&= \sum_{k=1}^\infty \frac{e^{-\lambda}\lambda^k}{(k-1)!} \quad (\text{去掉 } k=0 \text{ 的零项，约去 } k) \\
&= \lambda \sum_{k=1}^\infty \frac{e^{-\lambda}\lambda^{k-1}}{(k-1)!} \\
&= \lambda \sum_{j=0}^\infty \frac{e^{-\lambda}\lambda^j}{j!} \quad (\text{令 } j = k-1) \\
&= \lambda \cdot 1 = \lambda
\end{aligned}$$

最后一步 $\sum_{j=0}^\infty \frac{e^{-\lambda}\lambda^j}{j!} = 1$ 是因为所有概率之和为 1。

**推导方差**：

先求 $E[X^2]$。技巧是将其改写为 $E[X(X-1)] + E[X]$ 的形式，因为 $X(X-1)$ 更容易求和。

$$\begin{aligned}
E[X(X-1)] &= \sum_{k=0}^\infty k(k-1) \cdot \frac{e^{-\lambda}\lambda^k}{k!} \\
&= \sum_{k=2}^\infty \frac{e^{-\lambda}\lambda^k}{(k-2)!} \quad (\text{约去 } k(k-1)/k! = 1/(k-2)!) \\
&= \lambda^2 \sum_{k=2}^\infty \frac{e^{-\lambda}\lambda^{k-2}}{(k-2)!} \\
&= \lambda^2 \sum_{j=0}^\infty \frac{e^{-\lambda}\lambda^j}{j!} \quad (\text{令 } j = k-2) \\
&= \lambda^2
\end{aligned}$$

代入方差公式：
$$\begin{aligned}
\text{Var}(X) &= E[X^2] - (E[X])^2 \\
&= \big(E[X(X-1)] + E[X]\big) - \lambda^2 \\
&= (\lambda^2 + \lambda) - \lambda^2 = \lambda
\end{aligned}$$

**关键点总结**：
- 泊松分布的**期望 = 方差 = $\lambda$**，这是它的独特性质（大多数分布的期望和方差不相等）
- $\lambda$ 既是均值也是离散度的度量——$\lambda$ 越大，分布越分散
- 当 $\lambda$ 较大时（如 $\lambda \geq 20$），泊松分布近似正态分布
- 二项分布 → 泊松分布 → 正态分布是一条从离散到连续的"逼近链"

---

**例题 3**：某网站平均每小时有 5 次访问，$\lambda = 5$

**求某小时恰好有 3 次访问的概率**：
$$P(X = 3) = \frac{e^{-5}5^3}{3!} = \frac{0.00674 \times 125}{6} \approx 0.140$$

**求某小时没有访问的概率**：
$$P(X = 0) = \frac{e^{-5}5^0}{0!} = e^{-5} \approx 0.0067$$

---

**泊松分布的图像**（λ=5）：
```
P(X=k)
  │
  │           ●
  │         ●   ●
  │       ●       ●
  │     ●           ●
  │   ●               ● ●
  └───────────────────────→ k
    0 1 2 3 4 5 6 7 8 9 10

右偏分布，峰值在 k=λ=5
```

---

#### 泊松分布与二项分布的关系

当 $n$ 很大、$\phi$ 很小，且 $n\phi = \lambda$ 适中时，二项分布近似于泊松分布：
$$B(n, \phi) \approx \text{Poi}(\lambda), \quad \text{其中 } \lambda = n\phi$$

**经验法则**：当 $n \geq 20$ 且 $\phi \leq 0.05$ 时，可以用泊松分布近似。

---

**例题**：某工厂生产 1000 个零件，每个零件有 0.003 的概率是次品。求次品数的分布。

- 精确分布：$X \sim B(1000, 0.003)$
- 近似分布：$\lambda = 1000 \times 0.003 = 3$，$X \approx \text{Poi}(3)$

$$P(X \leq 2) \approx e^{-3} + 3e^{-3} + \frac{3^2 e^{-3}}{2} = e^{-3}(1 + 3 + 4.5) \approx 0.423$$

---

### 4.3 连续分布 (#)

#### 常见连续分布一览表

| 分布 | 参数 | 概率密度函数 | 期望 | 方差 | 应用场景 |
|------|------|-------------|------|------|----------|
| 均匀 U(a,b) | $a < b$ | $f(x)=\frac{1}{b-a}, a \leq x \leq b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 等可能取值 |
| 正态 N(μ,σ²) | $\mu \in \mathbb{R}, \sigma > 0$ | $f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | 自然界最常见 |
| 指数 Exp(λ) | $\lambda > 0$ | $f(x)=\lambda e^{-\lambda x}, x \geq 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | 等待时间 |

---

#### 1. 均匀分布 Uniform(a, b)

**定义**：在区间 $[a, b]$ 上等可能取值。

**记作**：$X \sim U(a, b)$

**概率密度函数**：
$$f_X(x) = \begin{cases}
\frac{1}{b-a} & a \leq x \leq b \\
0 & \text{其他}
\end{cases}$$

**累积分布函数**：
$$F_X(x) = \begin{cases}
0 & x < a \\
\frac{x-a}{b-a} & a \leq x \leq b \\
1 & x > b
\end{cases}$$

**期望**：$E[X] = \frac{a+b}{2}$（区间中点）

**方差**：$\text{Var}(X) = \frac{(b-a)^2}{12}$

---

**例题 1**：$X \sim U(0, 1)$（标准均匀分布）

- $E[X] = \frac{0+1}{2} = 0.5$
- $\text{Var}(X) = \frac{(1-0)^2}{12} = \frac{1}{12} \approx 0.0833$

**求概率**：
$$P(0.3 \leq X \leq 0.7) = \int_{0.3}^{0.7} 1 dx = 0.7 - 0.3 = 0.4$$

或用 CDF：
$$P(0.3 \leq X \leq 0.7) = F_X(0.7) - F_X(0.3) = 0.7 - 0.3 = 0.4$$

---

**均匀分布的图像**：
```
f(x)
  │
  │──────────────
  │              │
  │              │
  └──────────────┴────→ x
  a              b

在[a,b]上高度恒定 = 1/(b-a)
```

---

#### 2. 正态分布 Normal(μ, σ²)

**定义**：最重要的连续分布，也称为高斯分布。

**记作**：$X \sim N(\mu, \sigma^2)$

**概率密度函数**：
$$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**参数含义**：
- $\mu$：均值（分布的中心位置）
- $\sigma$：标准差（分布的"宽度"）

**期望**：$E[X] = \mu$

**方差**：$\text{Var}(X) = \sigma^2$

---

**标准正态分布**：$\mu = 0, \sigma = 1$

$$Z \sim N(0, 1), \quad \phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$$

**标准化**：任何正态分布都可以转化为标准正态：
$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

---

**正态分布的图像**（钟形曲线）：
```
f(x)
  │        μ=0
  │        ●
  │      ●   ●
  │    ●       ●
  │  ●           ●
  │●               ●
  └───────────────────→ x
   -3σ  0   +3σ

对称、单峰、钟形
```

---

#### 68-95-99.7 规则（经验法则）

对于任何正态分布 $N(\mu, \sigma^2)$：

- $P(\mu-\sigma \leq X \leq \mu+\sigma) \approx 0.6827$（约 68%）
- $P(\mu-2\sigma \leq X \leq \mu+2\sigma) \approx 0.9545$（约 95%）
- $P(\mu-3\sigma \leq X \leq \mu+3\sigma) \approx 0.9973$（约 99.7%）

**应用**：快速估算概率，识别异常值（超过 3σ 的值很罕见）。

---

**例题 2**：某班数学成绩服从 $N(75, 100)$（均值 75，方差 100，标准差 10）

**求成绩在 65 到 85 之间的概率**：
$$P(65 \leq X \leq 85) = P(75-10 \leq X \leq 75+10) \approx 0.68$$

**求成绩在 55 到 95 之间的概率**：
$$P(55 \leq X \leq 95) = P(75-20 \leq X \leq 75+20) \approx 0.95$$

---

#### 3. 指数分布 Exponential(λ)

**定义**：描述独立随机事件发生的时间间隔。

**记作**：$X \sim \text{Exp}(\lambda)$

**概率密度函数**：
$$f_X(x) = \begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}$$

**累积分布函数**：
$$F_X(x) = \begin{cases}
1 - e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}$$

**期望**：$E[X] = \frac{1}{\lambda}$

**方差**：$\text{Var}(X) = \frac{1}{\lambda^2}$

---

**无记忆性**：指数分布最重要的性质

$$P(X > s + t \mid X > s) = P(X > t)$$

**解释**：已经等待了 $s$ 时间，再等待 $t$ 时间的概率，和从一开始就等待 $t$ 时间的概率相同。

**应用**： radioactive decay、排队论、可靠性分析。

---

**例题 3**：某设备寿命服从 $\lambda = 0.1$ 的指数分布（单位：年）

**期望寿命**：
$$E[X] = \frac{1}{0.1} = 10 \text{年}$$

**求设备寿命超过 10 年的概率**：
$$P(X > 10) = 1 - F_X(10) = e^{-0.1 \times 10} = e^{-1} \approx 0.368$$

**求设备寿命在 5 到 15 年之间的概率**：
$$P(5 \leq X \leq 15) = F_X(15) - F_X(5) = (1-e^{-1.5}) - (1-e^{-0.5}) = e^{-0.5} - e^{-1.5} \approx 0.607 - 0.223 = 0.384$$

---

**指数分布的图像**（λ=1）：
```
f(x)
  │
  │●
  │ ●
  │  ●
  │   ●
  │    ●
  │     ●●●●●●●●
  └───────────────→ x
  0

从λ开始指数衰减
```

---

#### 正态分布与中心极限定理

**中心极限定理（Central Limit Theorem, CLT）**（预告，后面 4.7 节详述）：

> 大量独立随机变量的**和**（或**均值**）近似服从正态分布，无论这些随机变量本身服从什么分布。

**用例子理解**：

> 抛一枚公平硬币，单次结果是伯努利分布（两点分布）——不是 0 就是 1，离正态分布差远了。但抛 100 次硬币，正面**总次数** $\sum X_i$ 近似服从正态分布！每次抛硬币的结果 $X_i$ 本身完全不服从正态分布（只有两个值），"和"却近似是正态的。

**为什么这很重要？**
- 自然界中许多现象是大量微小独立因素叠加的结果（如人的身高受无数基因和环境因素影响），因此近似服从正态分布
- 统计推断中，即使不知道总体的分布，只要样本量够大，样本均值就近似正态分布——这是几乎所有统计检验的基础

---

### 4.4 多维随机变量与联合分布 (#)

#### 联合分布函数

**定义**：对于两个随机变量 $X$ 和 $Y$，联合分布函数定义为：
$$F_{XY}(x, y) = P(X \leq x, Y \leq y)$$

**解释**：$X$ 不超过 $x$ **且** $Y$ 不超过 $y$ 的概率。

---

#### 离散型：联合概率质量函数

**定义**：$p_{XY}(x, y) = P(X = x, Y = y)$

**性质**：
1. $p_{XY}(x, y) \geq 0$
2. $\sum_x \sum_y p_{XY}(x, y) = 1$

---

**例题**：$(X, Y)$ 的联合分布如下：

| X\Y | 0 | 1 | 边缘 P(X) |
|-----|---|---|-----------|
| 0 | 0.2 | 0.3 | **0.5** |
| 1 | 0.1 | 0.4 | **0.5** |
| **边缘 P(Y)** | **0.3** | **0.7** | 1.0 |

---

#### 边缘分布

从联合分布中"边缘化"掉一个变量，得到另一个变量的分布。

**离散型**：
$$p_X(x) = \sum_y p_{XY}(x, y)$$
$$p_Y(y) = \sum_x p_{XY}(x, y)$$

**连续型**：
$$f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y) dy$$
$$f_Y(y) = \int_{-\infty}^{\infty} f_{XY}(x, y) dx$$

---

**例题（续）**：求边缘分布

**X 的边缘分布**：
- $P(X=0) = 0.2 + 0.3 = 0.5$
- $P(X=1) = 0.1 + 0.4 = 0.5$

**Y 的边缘分布**：
- $P(Y=0) = 0.2 + 0.1 = 0.3$
- $P(Y=1) = 0.3 + 0.4 = 0.7$

---

#### 条件分布

**定义**：在已知 $X = x$ 的条件下，$Y$ 的分布。

**离散型**：
$$P(Y = y | X = x) = \frac{P(X = x, Y = y)}{P(X = x)}$$

**连续型**：
$$f_{Y|X}(y|x) = \frac{f_{XY}(x, y)}{f_X(x)}$$

---

**例题（续）**：求条件分布

**已知 X=0 时，Y=1 的概率**：
$$P(Y=1|X=0) = \frac{P(X=0, Y=1)}{P(X=0)} = \frac{0.3}{0.5} = 0.6$$

**已知 X=1 时，Y=0 的概率**：
$$P(Y=0|X=1) = \frac{P(X=1, Y=0)}{P(X=1)} = \frac{0.1}{0.5} = 0.2$$

---

#### 独立性

**定义**：$X$ 和 $Y$ 独立，如果：

**离散型**：
$$P(X = x, Y = y) = P(X = x) \cdot P(Y = y), \quad \forall x, y$$

**连续型**：
$$f_{XY}(x, y) = f_X(x) \cdot f_Y(y), \quad \forall x, y$$

**等价条件**：
- $F_{XY}(x, y) = F_X(x) F_Y(y)$
- 条件分布等于边缘分布：$f_{Y|X}(y|x) = f_Y(y)$

---

**例题**：判断上面的 $(X, Y)$ 是否独立

检查 $P(X=0, Y=0) = 0.2$

而 $P(X=0) \cdot P(Y=0) = 0.5 \times 0.3 = 0.15 \neq 0.2$

故 $X$ 和 $Y$ **不独立**。

---

#### 连续型联合分布的例子

**二维均匀分布**：在单位正方形 $[0,1] \times [0,1]$ 上均匀分布

$$f_{XY}(x, y) = \begin{cases} 1 & 0 \leq x \leq 1, 0 \leq y \leq 1 \\ 0 & \text{其他} \end{cases}$$

**边缘分布**：
$$f_X(x) = \int_0^1 1 \cdot dy = 1, \quad 0 \leq x \leq 1$$
$$f_Y(y) = \int_0^1 1 \cdot dx = 1, \quad 0 \leq y \leq 1$$

都是 $U(0,1)$ 分布，且 $X, Y$ 独立。

---

### 4.5 期望、方差与协方差矩阵 (#)

#### 期望（Expected Value）

**定义**：随机变量的"平均值"或"中心位置"。

**离散型**：
$$E[X] = \sum_x x \cdot p(x)$$

**连续型**：
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

**别名**：均值、数学期望，记作 $\mu_X$ 或 $E[X]$。

---

#### 期望的性质

1. **线性性**（最重要！）：
   $$E[aX + b] = aE[X] + b$$
   $$E[X + Y] = E[X] + E[Y]$$

2. **独立时乘积的期望**：
   如果 $X, Y$ 独立，则 $E[XY] = E[X]E[Y]$

3. **函数的期望**：
   $$E[g(X)] = \sum_x g(x)p(x) \quad \text{或} \quad \int g(x)f(x)dx$$

---

**例题 1**：掷公平骰子，$X$ 为点数

$$E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} = \frac{21}{6} = 3.5$$

**注意**：3.5 不是可能的取值，而是长期平均！

---

**例题 2**：伯努利分布 $X \sim \text{Bern}(\phi)$

$$E[X] = 1 \cdot \phi + 0 \cdot (1-\phi) = \phi$$

---

#### 方差（Variance）

**定义**：衡量随机变量取值偏离期望的程度。
$$\text{Var}(X) = E[(X - E[X])^2]$$

**计算公式**（更实用）：
$$\text{Var}(X) = E[X^2] - (E[X])^2$$

**推导**：
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

**标准差**：$\sigma_X = \sqrt{\text{Var}(X)}$，与 $X$ 同量纲。

---

#### 方差的性质

1. $\text{Var}(aX + b) = a^2 \text{Var}(X)$（常数 $b$ 不影响方差）

2. 如果 $X, Y$ 独立：
   $$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

3. $\text{Var}(X) \geq 0$，且 $\text{Var}(X) = 0 \iff X$ 是常数

---

**例题 3**：掷骰子，$X$ 为点数

**已算得**：$E[X] = 3.5$

**计算 $E[X^2]$**：
$$E[X^2] = 1^2 \cdot \frac{1}{6} + 2^2 \cdot \frac{1}{6} + 3^2 \cdot \frac{1}{6} + 4^2 \cdot \frac{1}{6} + 5^2 \cdot \frac{1}{6} + 6^2 \cdot \frac{1}{6}$$
$$= \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.17$$

**方差**：
$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - (3.5)^2 = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.92$$

**标准差**：$\sigma_X \approx \sqrt{2.92} \approx 1.71$

---

#### 协方差（Covariance）

**定义**：衡量两个随机变量的"协同变化"程度。
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

**计算公式**：
$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$$

**性质**：
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, Y) = a \text{Cov}(X, Y)$

---

#### 相关系数（Correlation Coefficient）

**定义**：标准化的协方差，消除了量纲影响。
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$$

**性质**：
- $-1 \leq \rho_{XY} \leq 1$
- $\rho_{XY} = 0$：不相关（但未必独立！）
- $\rho_{XY} = 1$：完全正线性相关
- $\rho_{XY} = -1$：完全负线性相关
- 如果 $X, Y$ 独立，则 $\rho_{XY} = 0$（反之不成立！）

---

**例题 4**：二维随机向量 $(X_1, X_2)$

已知：$\text{Var}(X_1) = 4$，$\text{Var}(X_2) = 9$，$\text{Cov}(X_1, X_2) = 3$

**相关系数**：
$$\rho_{12} = \frac{\text{Cov}(X_1, X_2)}{\sqrt{\text{Var}(X_1)\text{Var}(X_2)}} = \frac{3}{\sqrt{4 \times 9}} = \frac{3}{6} = 0.5$$

解释：中等程度的正相关。

---

#### 协方差矩阵（Covariance Matrix）

**定义**：$n$ 维随机向量 $X = [X_1, X_2, \dots, X_n]^T$ 的协方差矩阵是 $n \times n$ 对称矩阵：

$$\Sigma = \begin{pmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{pmatrix}$$

**性质**：
- 对称：$\Sigma = \Sigma^T$
- 半正定：对任意向量 $a$，$a^T \Sigma a \geq 0$

---

**例题 5**：二维随机向量 $(X_1, X_2)$，$\text{Var}(X_1) = 4$，$\text{Var}(X_2) = 9$，$\text{Cov}(X_1, X_2) = 3$

**协方差矩阵**：
$$\Sigma = \begin{pmatrix} 4 & 3 \\ 3 & 9 \end{pmatrix}$$

**验证半正定**：
- 顺序主子式：$4 > 0$，$\det(\Sigma) = 36 - 9 = 27 > 0$
- 故 $\Sigma$ 正定

---

**独立随机向量的协方差矩阵**：

如果 $X_1, X_2, \dots, X_n$ 两两独立，则协方差矩阵是对角矩阵：
$$\Sigma = \begin{pmatrix}
\text{Var}(X_1) & 0 & \cdots \\
0 & \text{Var}(X_2) & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}$$

---

#### 期望和方差的用途

| 场景 | 期望 | 方差 |
|------|------|------|
| 投资 | 预期收益率 | 风险（波动性） |
| 质量控制 | 目标尺寸 | 加工精度 |
| 机器学习 | 模型预测 | 预测不确定性 |

---

### 4.6 条件期望与全期望公式

#### 条件期望（Conditional Expectation）

**定义**：在已知 $Y = y$ 的条件下，$X$ 的期望值。

**离散型**：
$$E[X|Y=y] = \sum_x x \cdot P(X=x|Y=y)$$

**连续型**：
$$E[X|Y=y] = \int_{-\infty}^{\infty} x \cdot f_{X|Y}(x|y) dx$$

**直观理解**：给定某个条件后，随机变量的"新的平均值"。

---

#### 全期望公式（Law of Iterated Expectations）

**公式**：
$$E[X] = E[E[X|Y]] = \sum_y E[X|Y=y]P(Y=y)$$

**连续型**：
$$E[X] = \int_{-\infty}^{\infty} E[X|Y=y]f_Y(y)dy$$

**直观理解**：总体平均 = 各组平均的加权平均。

---

**例题 1**：有两个骰子，一个公平（$E=3.5$），一个偏向 6（$E=5$）。随机选一个掷一次，求期望。

设 $D$ = 选择的骰子（$D=0$ 公平，$D=1$ 偏向）

已知：
- $E[X|D=0] = 3.5$
- $E[X|D=1] = 5$
- $P(D=0) = P(D=1) = 0.5$

全期望公式：
$$E[X] = E[E[X|D]] = E[X|D=0]P(D=0) + E[X|D=1]P(D=1)$$
$$= 3.5 \times 0.5 + 5 \times 0.5 = 4.25$$

---

**例题 2**：某工厂有两台机器 A 和 B，A 生产 60% 的产品，B 生产 40%。A 的次品率 2%，B 的次品率 5%。求随机抽取一件产品的次品期望。

设 $X$ = 是否次品（1=次品，0=合格），$M$ = 机器（A 或 B）

- $E[X|M=A] = 0.02$
- $E[X|M=B] = 0.05$
- $P(M=A) = 0.6$，$P(M=B) = 0.4$

$$E[X] = 0.02 \times 0.6 + 0.05 \times 0.4 = 0.012 + 0.02 = 0.032$$

次品率约 3.2%。

---

#### 条件期望的性质

1. **线性性**：
   $$E[aX + bY|Z] = aE[X|Z] + bE[Y|Z]$$

2. **"塔"性质**：
   $$E[E[X|Y,Z]|Y] = E[X|Y]$$

3. **已知函数的期望**：
   如果 $X$ 是 $Y$ 的函数，则 $E[X|Y] = X$

4. **独立性**：
   如果 $X$ 和 $Y$ 独立，则 $E[X|Y] = E[X]$

---

#### 条件方差

**定义**：
$$\text{Var}(X|Y) = E[(X - E[X|Y])^2 | Y]$$

**方差分解公式**：
$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$

**解释**：
- 第一项：组内方差的平均
- 第二项：组间方差

---

### 4.7 大数定律与中心极限定理 (#)

#### 大数定律（Law of Large Numbers）

**问题**：为什么样本均值可以用来估计总体均值？

---

#### 弱大数定律

**定理**：设 $X_1, X_2, \dots, X_n$ 是独立同分布的随机变量，期望 $E[X_i] = \mu$，则样本均值 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ 依概率收敛于 $\mu$：

$$\bar{X}_n \xrightarrow{P} \mu$$

这里的 $\xrightarrow{P}$ 读作**依概率收敛**。它不是说每一次实验的样本均值都一定等于 $\mu$，而是说：当 $n$ 很大时，样本均值明显偏离 $\mu$ 的概率会变得很小。

即：对任意 $\epsilon > 0$，
$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| < \epsilon) = 1$$

---

**直观理解**：

当样本量 $n$ 足够大时，样本均值 $\bar{X}_n$ 与总体均值 $\mu$ 的差距可以任意小（概率趋近于 1）。

**例子**：
- 抛硬币：正面比例随抛掷次数增加趋近于 0.5
- 测量身高：样本平均身高随样本量增大趋近于总体平均

---

#### 强大数定律

**定理**：在相同条件下，样本均值几乎必然收敛于总体均值：
$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

这里的重点是**几乎必然**：把一次无限长的抽样过程看成一条路径，那么除了概率为 0 的异常路径外，这条路径上的样本均值最终会收敛到 $\mu$。

**强弱区别**：
- 弱大数：差距小的概率趋近于 1
- 强大数：收敛到真值的概率是 1（更强的结论）

---

#### 中心极限定理（Central Limit Theorem, CLT）

**问题**：样本均值的分布是什么？

---

#### 林德伯格 - 列维中心极限定理

**定理**：设 $X_1, X_2, \dots, X_n$ 是独立同分布的随机变量，期望 $E[X_i] = \mu$，方差 $\text{Var}(X_i) = \sigma^2 > 0$，则：

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

其中 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$。

这里的 $\xrightarrow{d}$ 读作**依分布收敛**，意思是：左边这个随机变量的分布形状，在 $n$ 变大时越来越接近标准正态分布 $N(0,1)$。它不是说左边的数值一定等于某个固定值，而是说它的概率分布越来越像标准正态分布。

这个公式可以拆成两层理解：
- $\bar{X}_n$ 是样本均值，它的中心在 $\mu$ 附近。
- $\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}}$ 是标准化：先减去均值 $\mu$，再除以样本均值的标准差 $\sigma/\sqrt{n}$，于是变成均值 0、方差 1 的尺度。

因此中心极限定理常用的近似写法是：
$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right)$$

也就是说，样本均值本身近似服从均值为 $\mu$、方差为 $\frac{\sigma^2}{n}$ 的正态分布。

---

**推导**：

$\bar{X}$ 的期望：
$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{n\mu}{n} = \mu$$

$\bar{X}$ 的方差：
$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$$

标准化：
$$Z_n = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

严格说，若原始 $X_i$ 本身不是正态分布，则有限样本下 $Z_n$ 通常并不**恰好**服从 $N(0,1)$，只是当 $n$ 越来越大时越来越接近标准正态分布。

---

**意义**：

1. **无论原始分布是什么**，样本均值近似服从正态分布（$n$ 足够大时）
2. 样本量越大，分布越集中（标准差 $\sigma/\sqrt{n}$ 越小）
3. 这是统计推断的理论基础！

---

**经验法则**：通常 $n \geq 30$ 时，正态近似就比较好。

---

**例题**：某班成绩 $\mu = 75$，$\sigma^2 = 100$（标准差 $\sigma = 10$），随机抽取 $n = 100$ 人，求样本均值大于 77 的概率。

**解**：

样本均值 $\bar{X}_n$ 近似服从：
$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right) = N\left(75, \frac{100}{100}\right) = N(75, 1)$$

标准化：
$$Z = \frac{77 - 75}{10/\sqrt{100}} = \frac{2}{1} = 2$$

查标准正态分布表：
$$P(\bar{X}_n > 77) = P(Z > 2) \approx 0.0228$$

结论：样本均值大于 77 的概率约 2.3%，是小概率事件。

---

#### 大数定律 vs 中心极限定理

| | 大数定律 | 中心极限定理 |
|---|---|---|
| **结论** | 样本均值收敛于总体均值 | 样本均值的分布近似正态 |
| **用途** | 说明估计的合理性 | 给出估计的精度（置信区间） |
| **直观** | "越平均越准" | "分布形状是钟形" |

---

### 附：极大似然估计 (MLE)

> 本节不属于 2026 新版“人工智能数学基础”模块的第 1-9 章正式编号，保留为旧版衔接与开卷补充材料。

#### 统计推断的基本问题

**已知**：从某个分布中抽取的样本 $x_1, x_2, \dots, x_n$

**未知**：分布的参数 $\theta$

**目标**：根据样本估计 $\theta$

---

#### 极大似然估计的思想

**核心想法**：选择使观测数据出现概率最大的参数值。

**直观**：既然这些观测值出现了，说明它们应该是"最可能"出现的。选择让这些数据最可能的参数。

---

#### 似然函数

**定义**：
$$L(\theta; x_1, \dots, x_n) = \prod_{i=1}^{n} f(x_i; \theta)$$

其中 $f(x; \theta)$ 是概率密度（或质量）函数。

**解释**：$L(\theta)$ 表示在参数 $\theta$ 下，观测到这组数据的概率（密度）。

---

#### 对数似然函数

取对数简化计算（乘积变求和，不影响最大值位置）：
$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta)$$

---

#### MLE 估计量

**定义**：
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta)$$

这里的 $\arg\max$ 表示“让后面那个函数取得最大值的参数”。所以这句话不是在求最大的似然值本身，而是在求哪个 $\theta$ 能让对数似然 $\ell(\theta)$ 最大。

**求解方法**：求导并令其为 0
$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

---

#### 例题 1：伯努利分布的 MLE

**问题**：抛 $n$ 次硬币，观测到有 $k$ 次正面，估计正面概率 $\phi$。

**模型**：$X_1, \dots, X_n \sim \text{Bern}(\phi)$，观测到 $k$ 个 1，$n-k$ 个 0。

**似然函数**：
$$L(\phi) = \prod_{i=1}^{n} \phi^{X_i}(1-\phi)^{1-X_i} = \phi^k(1-\phi)^{n-k}$$

**对数似然**：
$$\ell(\phi) = k\ln\phi + (n-k)\ln(1-\phi)$$

**求导**：
$$\frac{d\ell}{d\phi} = \frac{k}{\phi} - \frac{n-k}{1-\phi}$$

**令导数为 0**：
$$\frac{k}{\phi} - \frac{n-k}{1-\phi} = 0 \Rightarrow k(1-\phi) = (n-k)\phi \Rightarrow k = n\phi$$

**解得**：
$$\hat{\phi}_{MLE} = \frac{k}{n}$$

即**样本比例**！

---

#### 例题 2：高斯分布的 MLE

**问题**：$X_1, \dots, X_n \sim N(\mu, \sigma^2)$，估计 $\mu$ 和 $\sigma^2$。

**似然函数**：
$$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

**对数似然**：
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$$

---

**对 $\mu$ 求导**：
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) = 0$$

解得：
$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

即**样本均值**！

---

**对 $\sigma^2$ 求导**：
$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i-\mu)^2 = 0$$

解得：
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2$$

即**样本方差**（但分母是 $n$ 不是 $n-1$）！

---

**重要说明**：

- $\hat{\sigma}^2_{MLE}$ 是**有偏估计**：$E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$
- 无偏估计：$S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$
- 但 MLE 在大样本下是渐近无偏的，且具有最优渐近性质

---

#### MLE 的性质

1. **一致性**：$\hat{\theta}_{MLE} \xrightarrow{P} \theta$（样本量增大时收敛于真值）

   读法：估计值 $\hat{\theta}_{MLE}$ 依概率收敛到真实参数 $\theta$。直观上，样本越多，MLE 估计越不容易偏离真值。

2. **渐近正态性**：$\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} N(0, I^{-1}(\theta))$

   读法：把估计误差 $\hat{\theta}_{MLE}-\theta$ 乘上 $\sqrt{n}$ 后，它的分布会趋近于正态分布。这里的 $\sqrt{n}$ 暗示估计误差通常是 $\frac{1}{\sqrt{n}}$ 量级；$I(\theta)$ 是 Fisher 信息，$I^{-1}(\theta)$ 表示渐近方差。

3. **渐近有效性**：MLE 是渐近最优的无偏估计

4. **不变性**：如果 $\hat{\theta}$ 是 $\theta$ 的 MLE，则 $g(\hat{\theta})$ 是 $g(\theta)$ 的 MLE

---

### 附：最大后验估计 (MAP)

#### 贝叶斯观点

**频率学派**：参数 $\theta$ 是固定的未知常数

**贝叶斯学派**：参数 $\theta$ 是随机变量，有先验分布 $P(\theta)$

---

#### 贝叶斯定理

**后验分布**：
$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} = \frac{L(\theta)P(\theta)}{P(D)}$$

其中：
- $P(\theta)$：先验分布（观测数据前的信念）
- $P(D|\theta)$：似然函数
- $P(\theta|D)$：后验分布（观测数据后的更新信念）
- $P(D)$：证据（归一化常数，与 $\theta$ 无关）

---

#### 最大后验估计（MAP）

**思想**：选择使后验概率最大的参数值。

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} L(\theta)P(\theta)$$

这里第二个等号用到了 $P(D)$ 与 $\theta$ 无关：最大化后验 $P(\theta|D)$ 时，分母 $P(D)$ 只是常数，可以忽略。因此 MAP 等价于最大化“似然 × 先验”。

**对数形式**：
$$\hat{\theta}_{MAP} = \arg\max_{\theta} [\ell(\theta) + \ln P(\theta)]$$

---

#### MLE vs MAP

| | MLE | MAP |
|---|---|---|
| **优化目标** | $\arg\max \ell(\theta)$ | $\arg\max [\ell(\theta) + \ln P(\theta)]$ |
| **参数观点** | 固定常数 | 随机变量 |
| **额外信息** | 无 | 先验分布 |
| **解释** | 纯数据驱动 | 数据 + 先验知识 |
| **正则化** | 无 | 先验相当于正则项 |

---

**关键洞察**：MAP 中的 $\ln P(\theta)$ 相当于正则化项！

---

#### 例题：高斯分布均值的高斯先验 MAP 估计

**问题**：$X_i \sim N(\mu, \sigma^2)$，先验 $\mu \sim N(0, \tau^2)$，求 $\mu$ 的 MAP 估计。

**似然的对数**：
$$\ell(\mu) = -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2 + C_1$$

**先验的对数**：
$$\ln P(\mu) = -\frac{\mu^2}{2\tau^2} + C_2$$

**目标函数**：
$$\hat{\mu}_{MAP} = \arg\max_{\mu} \left[-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2 - \frac{\mu^2}{2\tau^2}\right]$$

---

**求导**：
$$\frac{d}{d\mu}\left[-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2 - \frac{\mu^2}{2\tau^2}\right] = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) - \frac{\mu}{\tau^2} = 0$$

**整理**：
$$\frac{n\bar{x}}{\sigma^2} - \frac{n\mu}{\sigma^2} - \frac{\mu}{\tau^2} = 0$$

$$\mu\left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right) = \frac{n\bar{x}}{\sigma^2}$$

**解得**：
$$\hat{\mu}_{MAP} = \frac{\frac{n}{\sigma^2}\bar{x}}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}} = \frac{n\tau^2\bar{x} + \sigma^2 \cdot 0}{n\tau^2 + \sigma^2}$$

---

**解释**：

- 分子：$n\tau^2\bar{x} + \sigma^2 \cdot 0$
- 分母：$n\tau^2 + \sigma^2$

这是**样本均值 $\bar{x}$ 和先验均值 0 的加权平均**！

权重由精度（方差的倒数）决定：
- 样本信息多（$n$ 大或 $\sigma^2$ 小）→ 更接近 $\bar{x}$
- 先验信息强（$\tau^2$ 小）→ 更接近先验均值 0

---

**极限情况**：

1. **先验无信息**（$\tau^2 \to \infty$）：
   $$\hat{\mu}_{MAP} \to \bar{x} = \hat{\mu}_{MLE}$$

2. **先验非常强**（$\tau^2 \to 0$）：
   $$\hat{\mu}_{MAP} \to 0$$
   即趋向先验均值。

3. **样本量很大**（$n \to \infty$）：
   $$\hat{\mu}_{MAP} \to \bar{x} = \hat{\mu}_{MLE}$$
   即由数据主导。

---

#### MAP 与正则化的关系

**例子**：线性回归 $y = w^T x + \epsilon$

- **MLE**：最小化平方误差 $\sum(y_i - w^T x_i)^2$
- **MAP（高斯先验 $w \sim N(0, \lambda^{-1}I)$）**：
  $$\arg\min_w \sum(y_i - w^T x_i)^2 + \lambda\|w\|^2$$

这就是**岭回归（Ridge Regression）**！

**结论**：$L_2$ 正则化等价于高斯先验下的 MAP 估计。

类似地，$L_1$ 正则化等价于拉普拉斯先验下的 MAP 估计（LASSO）。

---

# 第三部分：最优化理论与方法

## 第 5 章 多元微积分基础

### 5.1 多元函数的偏导数与梯度 (#)

#### 为什么需要多元微积分？

在机器学习中，损失函数通常依赖于多个参数：
$$L(w_1, w_2, \dots, w_n) = \sum_{i=1}^{n} (y_i - f(x_i; w))^2$$

要最小化损失函数，需要知道每个参数对函数的影响——这就是偏导数的作用。

---

#### 偏导数（Partial Derivative）

**定义**：多元函数 $f(x_1, x_2, \dots, x_n)$ 对第 $i$ 个变量的偏导数，是固定其他变量不变，只对 $x_i$ 求导：

$$\frac{\partial f}{\partial x_i}(x) = \lim_{h \to 0} \frac{f(x_1, \dots, x_i+h, \dots, x_n) - f(x)}{h}$$

**记号**：$\frac{\partial f}{\partial x_i}$、$f_{x_i}$、$\partial_i f$

---

**例题 1**：$f(x, y) = x^2 + 2xy + y^2$

**对 $x$ 求偏导**（把 $y$ 当作常数）：
$$\frac{\partial f}{\partial x} = 2x + 2y$$

**对 $y$ 求偏导**（把 $x$ 当作常数）：
$$\frac{\partial f}{\partial y} = 2x + 2y$$

**在点 $(1, 2)$ 处**：
$$\frac{\partial f}{\partial x}(1,2) = 2(1) + 2(2) = 6$$
$$\frac{\partial f}{\partial y}(1,2) = 2(1) + 2(2) = 6$$

---

**例题 2**：$f(x, y, z) = x^2y + yz^2 + xyz$

$$\frac{\partial f}{\partial x} = 2xy + yz$$
$$\frac{\partial f}{\partial y} = x^2 + z^2$$
$$\frac{\partial f}{\partial z} = 2yz + xy$$

---

#### 梯度（Gradient）

**定义**：梯度是所有一阶偏导数组成的向量：

$$\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

**本笔记约定**：把变量 $x$ 看成列向量，因此梯度 $\nabla f(x)$ 也写成列向量。

如果 $x \in \mathbb{R}^n$，则：
$$x \in \mathbb{R}^{n \times 1}, \quad \nabla f(x) \in \mathbb{R}^{n \times 1}$$

这样梯度下降的维度是匹配的：
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

其中 $x_k$ 和 $\nabla f(x_k)$ 都是 $n \times 1$ 列向量。

有些教材把梯度写成行向量，这是另一种约定。只要全书保持一致即可；本笔记统一采用“梯度为列向量”的优化 / 机器学习常用写法。

**记号**：$\nabla f$、$\text{grad} f$

---

**例题**：$f(x, y) = x^2 + 2xy + y^2$

$$\nabla f(x, y) = \begin{pmatrix} 2x + 2y \\ 2x + 2y \end{pmatrix}$$

在点 $(1, 2)$ 处：$\nabla f(1, 2) = \begin{pmatrix} 6 \\ 6 \end{pmatrix}$

---

#### 梯度的几何意义

**核心性质**：
1. **方向**：梯度指向函数增长最快的方向
2. **大小**：梯度的模长是最大增长率

**数学表达**：
$$\max_{\|v\|=1} D_v f(x) = \|\nabla f(x)\|$$

其中 $D_v f(x)$ 是方向导数。

---

**直观理解**：

想象你站在一座山上（函数图像），梯度指向"最陡的上坡方向"。

```
         山顶
          ●
         / \
        /   \
       /  ↑  \   梯度方向
      /   │   \  (最陡上坡)
     /    ●────\───→ 等高线
    │    你在这里
    │
```

**梯度下降法**：沿着负梯度方向走，函数值下降最快！

---

#### 梯度的性质

1. **线性性**：
   $$\nabla (af + bg) = a\nabla f + b\nabla g$$

2. **乘积法则**：
   $$\nabla (f \cdot g) = f\nabla g + g\nabla f$$

3. **链式法则**（向量形式）：
   如果 $h(x) = f(g(x))$，则
   $$\nabla h(x) = J_g(x)^T \nabla f(g(x))$$

   这里的 $J_g(x)$ 是内层向量函数 $g$ 的雅可比矩阵。因为本笔记把梯度写成列向量，所以公式里需要 $J_g(x)^T$ 来保证维度匹配；详细解释见 5.3 节“向量形式的链式法则”。

---

#### 方向导数

**定义**：函数沿单位向量 $v$ 方向的变化率：

$$D_v f(x) = \lim_{h \to 0} \frac{f(x + hv) - f(x)}{h}$$

**计算公式**：
$$D_v f(x) = \nabla f(x) \cdot v$$

**最大方向**：当 $v$ 与 $\nabla f$ 同向时，方向导数最大，最大值为 $\|\nabla f\|$。

---

### 5.2 雅可比矩阵与海森矩阵 (#)

#### 雅可比矩阵（Jacobian Matrix）

**适用对象**：向量值函数 $f: \mathbb{R}^n \to \mathbb{R}^m$

$$f(x) = \begin{pmatrix} f_1(x_1, \dots, x_n) \\ f_2(x_1, \dots, x_n) \\ \vdots \\ f_m(x_1, \dots, x_n) \end{pmatrix}$$

**雅可比矩阵**是 $m \times n$ 矩阵，第 $i$ 行是 $f_i$ 的梯度：

$$J_f(x) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}_{m \times n}$$

---

**例题 1**：$f(x, y) = \begin{pmatrix} x^2 + y \\ 2xy \end{pmatrix}$

$$J_f(x, y) = \begin{pmatrix}
\frac{\partial}{\partial x}(x^2 + y) & \frac{\partial}{\partial y}(x^2 + y) \\
\frac{\partial}{\partial x}(2xy) & \frac{\partial}{\partial y}(2xy)
\end{pmatrix} = \begin{pmatrix} 2x & 1 \\ 2y & 2x \end{pmatrix}$$

在点 $(1, 1)$ 处：$J_f(1, 1) = \begin{pmatrix} 2 & 1 \\ 2 & 2 \end{pmatrix}$

---

**例题 2**：神经网络一层 $f(x) = Wx + b$

- $W \in \mathbb{R}^{m \times n}$，$x \in \mathbb{R}^n$，$b \in \mathbb{R}^m$
- $f: \mathbb{R}^n \to \mathbb{R}^m$

$$J_f(x) = W$$

这是常数矩阵，与 $x$ 无关。

---

#### 海森矩阵（Hessian Matrix）

**适用对象**：标量函数 $f: \mathbb{R}^n \to \mathbb{R}$

**定义**：海森矩阵是二阶偏导数组成的 $n \times n$ 矩阵：

$$H_f(x) = \nabla^2 f(x) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

**简记**：$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

---

**对称性**：如果二阶偏导数连续，则海森矩阵对称：
$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

---

**例题 3**：$f(x, y) = x^3 + 2xy + y^2$

**一阶偏导**：
$$\frac{\partial f}{\partial x} = 3x^2 + 2y$$
$$\frac{\partial f}{\partial y} = 2x + 2y$$

**二阶偏导**：
$$\frac{\partial^2 f}{\partial x^2} = 6x, \quad \frac{\partial^2 f}{\partial x \partial y} = 2$$
$$\frac{\partial^2 f}{\partial y \partial x} = 2, \quad \frac{\partial^2 f}{\partial y^2} = 2$$

**海森矩阵**：
$$H_f(x, y) = \begin{pmatrix} 6x & 2 \\ 2 & 2 \end{pmatrix}$$

在点 $(1, 1)$ 处：$H_f(1, 1) = \begin{pmatrix} 6 & 2 \\ 2 & 2 \end{pmatrix}$

---

#### 海森矩阵的几何意义

海森矩阵描述函数的"曲率"：

**一元函数**：
- $f''(x) > 0$：向上弯曲（凸）
- $f''(x) < 0$：向下弯曲（凹）

**多元函数**：
- $H \succ 0$（正定）：局部凸（碗状）
- $H \prec 0$（负定）：局部凹（帽状）
- $H$ 有正有负：鞍点

---

#### 梯度、雅可比、海森的关系

| 对象 | 函数类型 | 结果 |
|------|----------|------|
| 偏导数 $\frac{\partial f}{\partial x_i}$ | $f: \mathbb{R}^n \to \mathbb{R}$ | 标量 |
| 梯度 $\nabla f$ | $f: \mathbb{R}^n \to \mathbb{R}$ | $n$ 维向量 |
| 雅可比 $J_f$ | $f: \mathbb{R}^n \to \mathbb{R}^m$ | $m \times n$ 矩阵 |
| 海森 $H_f$ | $f: \mathbb{R}^n \to \mathbb{R}$ | $n \times n$ 矩阵 |

**关系**：
- 梯度是雅可比的特例（$m=1$）
- 海森是梯度的雅可比：$H_f = J_{\nabla f}$

---

### 5.3 链式法则 (#)

#### 单变量链式法则（复习）

$z = f(y), y = g(x)$

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

**例题**：$z = e^{x^2}$

令 $y = x^2$，则 $z = e^y$

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} = e^y \cdot 2x = 2xe^{x^2}$$

---

#### 多变量链式法则（情况 1）

$z = f(x_1, \dots, x_n)$，每个 $x_i$ 都是 $t$ 的函数：$x_i = x_i(t)$

$$\frac{dz}{dt} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{dx_i}{dt}$$

---

**例题**：$z = x^2 + y^2$，$x = \cos t$，$y = \sin t$

**方法 1（链式法则）**：
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt} = (2x)(-\sin t) + (2y)(\cos t)$$
$$= 2\cos t(-\sin t) + 2\sin t(\cos t) = 0$$

**方法 2（直接代入）**：
$$z = \cos^2 t + \sin^2 t = 1 \Rightarrow \frac{dz}{dt} = 0$$

结果一致！

---

#### 多变量链式法则（情况 2）

$z = f(x, y)$，$x = x(u, v)$，$y = y(u, v)$

$$\frac{\partial z}{\partial u} = \frac{\partial z}{\partial x}\frac{\partial x}{\partial u} + \frac{\partial z}{\partial y}\frac{\partial y}{\partial u}$$

$$\frac{\partial z}{\partial v} = \frac{\partial z}{\partial x}\frac{\partial x}{\partial v} + \frac{\partial z}{\partial y}\frac{\partial y}{\partial v}$$

**树图记忆法**：
```
      z
     / \
    x   y
   / \ / \
  u  v u  v

∂z/∂u = (∂z/∂x)(∂x/∂u) + (∂z/∂y)(∂y/∂u)
```

---

#### 向量形式的链式法则

$f: \mathbb{R}^n \to \mathbb{R}$，$g: \mathbb{R}^m \to \mathbb{R}^n$，$h(x) = f(g(x))$

$$\nabla h(x) = J_g(x)^T \nabla f(g(x))$$

这里的 $J_g(x)$ 是 **Jacobian（雅可比矩阵）**，表示内层向量函数 $g$ 对输入 $x$ 的导数。

令：
$$y = g(x), \quad h(x) = f(y)$$

则一元链式法则的直觉仍然是：
$$\frac{dh}{dx} = \frac{df}{dy}\frac{dy}{dx}$$

只是现在 $y$ 是向量，$x$ 也是向量，$\frac{dy}{dx}$ 不再是一个数，而是一个矩阵。把所有偏导数排起来，就是：
$$J_g(x) =
\begin{pmatrix}
\frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & \cdots & \frac{\partial g_1}{\partial x_m} \\
\frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & \cdots & \frac{\partial g_2}{\partial x_m} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial g_n}{\partial x_1} & \frac{\partial g_n}{\partial x_2} & \cdots & \frac{\partial g_n}{\partial x_m}
\end{pmatrix} \in \mathbb{R}^{n \times m}$$

**为什么公式里有转置？**

因为本笔记采用“梯度是列向量”的约定：
$$\nabla f(g(x)) \in \mathbb{R}^{n \times 1}, \quad J_g(x) \in \mathbb{R}^{n \times m}$$

所以必须写成：
$$J_g(x)^T \nabla f(g(x)) \in \mathbb{R}^{m \times 1}$$

而 $\nabla h(x)$ 是关于 $x$ 的梯度，本来就应该是 $m \times 1$ 列向量。

**直观理解**：
- $\nabla f(g(x))$：外层函数 $f$ 对中间变量 $y=g(x)$ 的敏感度
- $J_g(x)$：中间变量 $y$ 对原变量 $x$ 的敏感度
- 两者合起来：复合函数 $h(x)=f(g(x))$ 对 $x$ 的敏感度

---

**例题**：$f(y) = \|y\|^2 = y^T y$，$y = g(x) = Ax + b$

$h(x) = \|Ax + b\|^2$，求 $\nabla h(x)$

**计算**：
- $\nabla f(y) = 2y$
- $J_g(x) = A$

因为 $g(x)=Ax+b$ 是线性函数，$y$ 对 $x$ 的导数就是矩阵 $A$，所以这里的雅可比矩阵为 $A$。

$$\nabla h(x) = A^T \cdot 2(Ax + b) = 2A^T(Ax + b)$$

---

#### 链式法则在神经网络中的应用

**反向传播算法的核心就是链式法则！**

考虑简单神经网络：
$$\hat{y} = \sigma(Wx + b)$$
$$L = (\hat{y} - y)^2$$

求 $\frac{\partial L}{\partial W}$：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}$$

其中 $z = Wx + b$

这就是链式法则！

---

## 第 6 章 最优化基础

优化问题的核心是找到使目标函数最小化（或最大化）的变量值。在机器学习中，这对应于找到使损失函数最小的模型参数。

### 6.1 凸集与凸函数 (#)

#### 一、凸集（Convex Set）

**定义**：集合 $C \subseteq \mathbb{R}^n$ 是凸集，如果对于任意两点 $x, y \in C$ 和任意 $\theta \in [0, 1]$，有：
$$\theta x + (1-\theta)y \in C$$

**解释**：
- $\theta x + (1-\theta)y$ 是连接 $x$ 和 $y$ 的线段上的点
- 当 $\theta = 0$ 时，得到点 $x$；当 $\theta = 1$ 时，得到点 $y$
- 凸集意味着"集合内任意两点的连线整个都在集合内"

---

**例题 1**：验证单位圆盘 $C = \{x \in \mathbb{R}^2 \mid \|x\|_2 \leq 1\}$ 是凸集。

**证明**：

对任意 $x, y \in C$，有 $\|x\|_2 \leq 1$，$\|y\|_2 \leq 1$。

考虑线段上任意点 $z = \theta x + (1-\theta)y$，其中 $\theta \in [0, 1]$。

由三角不等式：
$$\|z\|_2 = \|\theta x + (1-\theta)y\|_2 \leq \theta\|x\|_2 + (1-\theta)\|y\|_2$$

因为 $\|x\|_2 \leq 1$，$\|y\|_2 \leq 1$，$\theta \geq 0$，$(1-\theta) \geq 0$：
$$\|z\|_2 \leq \theta \cdot 1 + (1-\theta) \cdot 1 = 1$$

故 $z \in C$，单位圆盘是凸集。

---

**例题 2**：验证半空间 $H = \{x \in \mathbb{R}^n \mid a^T x \leq b\}$ 是凸集。

**证明**：

对任意 $x, y \in H$，有 $a^T x \leq b$，$a^T y \leq b$。

考虑 $z = \theta x + (1-\theta)y$：
$$a^T z = a^T(\theta x + (1-\theta)y) = \theta a^T x + (1-\theta)a^T y$$

因为 $a^T x \leq b$，$a^T y \leq b$，$\theta \geq 0$，$(1-\theta) \geq 0$：
$$a^T z \leq \theta b + (1-\theta)b = b$$

故 $z \in H$，半空间是凸集。

---

**常见凸集**：

| 凸集 | 定义 | 图像 |
|------|------|------|
| 超平面 | $\{x \mid a^T x = b\}$ | 直线/平面 |
| 半空间 | $\{x \mid a^T x \leq b\}$ | 直线一侧 |
| 球/椭球 | $\{x \mid \|x-c\|_2 \leq r\}$ | 圆形/椭圆 |
| 多面体 | $\{x \mid Ax \leq b\}$ | 多边形/多面体 |
| 正半定锥 | $\{X \in \mathbb{S}^n_+ \mid X \succeq 0\}$ | 矩阵锥 |

---

**非凸集例子**：

1. **环形区域**：$\{x \in \mathbb{R}^2 \mid 1 \leq \|x\|_2 \leq 2\}$
   - 取内圈一点和外圈一点，连线穿过中间空洞

2. **月牙形**：两个圆相交后去掉重叠部分
   - 取两端点，连线在集合外

3. **离散点集**：$\{(0,0), (1,1)\}$
   - 两点之间的点不在集合中

---

**凸集的运算性质**：

以下运算保持凸性（如果 $C_1, C_2$ 是凸集，则结果也是凸集）：

1. **交集**：$C_1 \cap C_2$
2. **并集**：$C_1 \cup C_2$ **不保持**凸性！
3. **笛卡尔积**：$C_1 \times C_2$
4. **仿射变换**：$f(C) = \{f(x) \mid x \in C\}$，其中 $f$ 是仿射函数
5. **透视变换**：$P(z, t) = z/t$（$t > 0$）

---

#### 二、凸函数（Convex Function）

**定义**：函数 $f: C \to \mathbb{R}$（$C$ 是凸集）是凸函数，如果对于任意 $x, y \in C$ 和任意 $\theta \in [0, 1]$，有：
$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

**几何解释**：

```
f(x)
 │        ● B
 │       /│\
 │      / │ \    割线（在函数上方）
 │     /  │  \
 │    /   │   \
 │   ●────┼────●
 │  A     │     C
 │        │
 └─────────────────→ x

A、C 在函数图像上，B 在割线上
f(θx+(1-θ)y) ≤ θf(x)+(1-θ)f(y)
即：函数图像在割线下方
```

**直观理解**：函数图像上任意两点的连线（割线）始终在函数图像的上方。

---

**严格凸函数**：如果上述不等式对 $x \neq y$ 和 $\theta \in (0, 1)$ 严格成立（$<$）。

**凹函数**：$-f$ 是凸函数，或者说：
$$f(\theta x + (1-\theta)y) \geq \theta f(x) + (1-\theta)f(y)$$

---

#### 三、凸函数的判定方法

#### 方法 1：一阶条件（可微函数）

**定理**：可微函数 $f$ 是凸函数，当且仅当：
$$f(y) \geq f(x) + \nabla f(x)^T(y - x), \quad \forall x, y \in C$$

读法：右边是函数在 $x$ 点的一阶线性近似，也就是切线 / 切平面；凸函数要求任意点 $y$ 的真实函数值都不低于这条切线 / 切平面给出的值。

**几何解释**：

```
f(x)
 │         函数图像
 │        ╱
 │       ╱
 │      ╱│
 │     ╱ │  切线（一阶近似）
 │    ╱  │  在函数下方
 │   ●───┼──────
 │   x   │
 └───────┴─────→ x

切线始终在函数图像下方
```

凸函数的切线（或切平面）始终在函数图像的下方。

---

**例题 3**：用一阶条件验证 $f(x) = x^2$ 是凸函数。

**解**：

一阶导数：$f'(x) = 2x$

一阶条件要求：
$$f(y) \geq f(x) + f'(x)(y-x)$$

代入 $f(x) = x^2$，$f'(x) = 2x$：
$$y^2 \geq x^2 + 2x(y-x) = x^2 + 2xy - 2x^2 = 2xy - x^2$$

移项：
$$y^2 - 2xy + x^2 \geq 0 \iff (y-x)^2 \geq 0$$

这显然成立！故 $f(x) = x^2$ 是凸函数。

---

#### 方法 2：二阶条件（二阶可微函数）

**定理**：二阶可微函数 $f$ 是凸函数，当且仅当其海森矩阵半正定：
$$\nabla^2 f(x) \succeq 0, \quad \forall x \in C$$

这里的 $\succeq 0$ 表示半正定。直观上，海森矩阵描述函数的弯曲方向；半正定表示函数在任何方向上都不会向下弯，所以函数是凸的。

**一元函数**：$f''(x) \geq 0$

**多元函数**：海森矩阵的所有特征值 $\geq 0$

---

**例题 4**：判断 $f(x, y) = x^4 + y^4$ 是否为凸函数。

**解**：

一阶偏导：
$$\frac{\partial f}{\partial x} = 4x^3, \quad \frac{\partial f}{\partial y} = 4y^3$$

二阶偏导：
$$\frac{\partial^2 f}{\partial x^2} = 12x^2, \quad \frac{\partial^2 f}{\partial y^2} = 12y^2, \quad \frac{\partial^2 f}{\partial x \partial y} = 0$$

海森矩阵：
$$H_f(x, y) = \begin{pmatrix} 12x^2 & 0 \\ 0 & 12y^2 \end{pmatrix}$$

特征值为 $12x^2 \geq 0$ 和 $12y^2 \geq 0$，始终非负。

故 $f(x, y) = x^4 + y^4$ 是凸函数。

---

**例题 5**：判断 $f(x, y) = x^2 - y^2$ 是否为凸函数。

**解**：

海森矩阵：
$$H_f(x, y) = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$$

特征值为 $2$ 和 $-2$，有负特征值。

故 $f(x, y) = x^2 - y^2$ **不是**凸函数（是鞍函数）。

---

#### 方法 3：保凸运算

以下运算保持凸性：

1. **非负加权和**：$f = w_1 f_1 + w_2 f_2$（$w_1, w_2 \geq 0$）
2. **仿射复合**：$f(Ax + b)$ 是凸的，如果 $f$ 是凸的
3. **逐点上确界**：$\sup_{\alpha} f_\alpha(x)$ 是凸的，如果每个 $f_\alpha$ 是凸的
4. **极小化**：$g(x) = \inf_y f(x, y)$ 是凸的，如果 $f$ 是凸的

---

#### 四、常见凸函数一览表

| 函数 | 定义域 | 凸性 | 验证方法 |
|------|--------|------|----------|
| 线性/仿射 | $\mathbb{R}^n$ | 凸且凹 | $f''(x) = 0$ |
| 指数函数 $e^{ax}$ | $\mathbb{R}$ | 严格凸 | $f''(x) = a^2 e^{ax} > 0$ |
| 幂函数 $x^p$ | $\mathbb{R}_{++}$ | $p \geq 1$ 或 $p \leq 0$ 时凸 | $f''(x) = p(p-1)x^{p-2}$ |
| 负对数 $-\log x$ | $\mathbb{R}_{++}$ | 严格凸 | $f''(x) = 1/x^2 > 0$ |
| 二次函数 $\frac{1}{2}x^T Ax + b^T x + c$ | $\mathbb{R}^n$ | $A \succeq 0$ 时凸 | 海森矩阵 $= A$ |
| 范数 $\|x\|$ | $\mathbb{R}^n$ | 凸 | 三角不等式 |
| 最大值函数 $\max\{x_1, \dots, x_n\}$ | $\mathbb{R}^n$ | 凸 | 逐点上确界 |

---

#### 五、凸函数的重要性质

1. **局部最小值 = 全局最小值**
   - 凸函数的任何局部最小值都是全局最小值
   - 这是凸优化如此重要的原因！

2. **可微凸函数的最优性条件**
   - $x^*$ 是全局最小值 $\iff \nabla f(x^*) = 0$

3. **Jensen 不等式**
   - 对于凸函数 $f$ 和随机变量 $X$：
   $$f(E[X]) \leq E[f(X)]$$

---

**Jensen 不等式的应用**：

设 $X$ 是随机变量，$f(x) = x^2$ 是凸函数：
$$(E[X])^2 \leq E[X^2]$$

这解释了为什么 $\text{Var}(X) = E[X^2] - (E[X])^2 \geq 0$。

---

### 6.2 L-光滑性与强凸性

#### 一、L-光滑性（L-Smoothness）

**定义**：可微函数 $f$ 是 $L$-光滑的，如果其梯度是 $L$-Lipschitz 连续的：
$$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|, \quad \forall x, y$$

其中 $L > 0$ 称为光滑常数。

**直观理解**：梯度的变化速度有上界。函数图像不会"太弯曲"。

读法：当 $x$ 和 $y$ 很接近时，两个点的梯度也不能突然差很多。$L$ 越大，允许函数弯得越厉害；$L$ 越小，函数越平滑，梯度下降的步长通常也更容易选。

---

**等价条件**：

对于 $L$-光滑函数，以下等价：

1. **梯度 Lipschitz**：$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$

2. **二次上界**：
   $$f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$$

   读法：函数值不会超过“切线 / 切平面 + 一个开口由 $L$ 控制的二次项”。这就是“弯曲程度有上界”的公式化表达。

3. **海森矩阵有界**（二阶可微时）：
   $$\|\nabla^2 f(x)\|_2 \leq L$$

   读法：二阶导数 / 曲率的大小最多是 $L$。对二次函数来说，这个 $L$ 就对应最大特征值。

   如果 $f$ 已知为凸函数，也常写成：
   $$0 \preceq \nabla^2 f(x) \preceq LI$$

---

**例题 1**：验证 $f(x) = \frac{L}{2}x^2$ 是 $L$-光滑的。

**解**：

梯度：$\nabla f(x) = Lx$

验证 Lipschitz 条件：
$$|\nabla f(x) - \nabla f(y)| = |Lx - Ly| = L|x - y|$$

恰好满足 Lipschitz 条件，光滑常数为 $L$。

---

**例题 2**：验证 $f(x) = \frac{1}{2}x^T Ax$ 是 $L$-光滑的，其中 $A$ 为对称半正定矩阵，$L = \lambda_{\max}(A)$（$A$ 的最大特征值）。

**解**：

梯度：$\nabla f(x) = Ax$

$$\|\nabla f(x) - \nabla f(y)\| = \|Ax - Ay\| = \|A(x-y)\|$$

由谱范数定义：
$$\|A(x-y)\| \leq \|A\|_2 \|x-y\| = \lambda_{\max}(A)\|x-y\|$$

故 $f$ 是 $L$-光滑的，$L = \lambda_{\max}(A)$。

---

**几何解释**：

```
f(x)
 │          ___
 │        ╱   │   二次上界：f(y) ≤ 抛物线
 │       ╱    │
 │      ╱     │
 │     ╱      ● 函数图像
 │    ╱      ╱
 │   ●──────╱──  切线
 │          │
 └──────────┴────→ x

L-光滑函数的图像始终在"切线 + 抛物线"下方
```

---

#### 二、μ-强凸性（μ-Strong Convexity）

**定义**：函数 $f$ 是 $\mu$-强凸的，如果：
$$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2, \quad \forall x, y$$

其中 $\mu > 0$。

**直观理解**：函数"足够弯曲"，不是平坦的。

读法：普通凸函数只要求函数图像在切线 / 切平面上方；强凸函数还要求它至少高出一个由 $\mu$ 控制的二次弯曲量。因此强凸函数不只是“不向下弯”，还必须“向上弯得足够多”。

---

**等价条件**：

1. **定义式**：$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$

2. **梯度强单调**：
   $$(\nabla f(x) - \nabla f(y))^T(x-y) \geq \mu\|x-y\|^2$$

3. **海森矩阵下界**（二阶可微时）：
   $$\nabla^2 f(x) \succeq \mu I$$

---

**例题 3**：验证 $f(x) = \frac{\mu}{2}x^2$ 是 $\mu$-强凸的。

**解**：

二阶导数：$f''(x) = \mu$

由二阶条件，$f''(x) \geq \mu$，故 $f$ 是 $\mu$-强凸的。

---

#### 三、L-光滑且 μ-强凸函数

**定义**：函数 $f$ 同时是 $L$-光滑和 $\mu$-强凸的，如果：
$$\mu I \preceq \nabla^2 f(x) \preceq LI$$

即海森矩阵的特征值在 $[\mu, L]$ 范围内。

这里的矩阵不等式可以按特征值理解：所有曲率都至少是 $\mu$，所以不会太平；所有曲率都最多是 $L$，所以不会太陡。优化算法的难度主要取决于最大曲率和最小曲率差得有多远。

**条件数**：$\kappa = \frac{L}{\mu}$ 称为条件数。

- $\kappa$ 小：函数"形状好"，优化容易
- $\kappa$ 大：函数"扁长"，优化困难

---

**几何解释**：

```
等高线图：

μ=L（条件数=1）：       μ<<L（条件数大）：

      ●                    ___
   ●     ●              ╱   │   ╲
  ●   x*   ●           │    │    │
   ●     ●              ╲___│___╱
      ●

圆形等高线              椭圆形等高线
容易优化                难优化（梯度方向震荡）
```

---

#### 四、在优化中的应用

#### 梯度下降收敛速率

**定理**：如果 $f$ 是 $L$-光滑的凸函数，梯度下降使用步长 $\eta = \frac{1}{L}$，则：
$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$$

即收敛速率为 $O(1/k)$。

读法：左边是第 $k$ 次迭代后距离最优函数值还差多少；右边随着 $k$ 增大按 $\frac{1}{k}$ 下降。因此迭代次数翻倍，误差上界大约减半。

**定理**：如果 $f$ 是 $L$-光滑且 $\mu$-强凸的，梯度下降使用步长 $\eta = \frac{1}{L}$，则：
$$\|x_k - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|^2$$

即**线性收敛**（几何收敛）！

读法：每迭代一次，误差都会乘上一个小于 1 的固定比例 $\left(1-\frac{\mu}{L}\right)$。这比 $O(1/k)$ 快得多；$\frac{\mu}{L}$ 越大，收敛越快。

---

**证明思路**（强凸情况）：

由 $L$-光滑性：
$$f(x_{k+1}) \leq f(x_k) + \nabla f(x_k)^T(x_{k+1} - x_k) + \frac{L}{2}\|x_{k+1} - x_k\|^2$$

代入 $x_{k+1} = x_k - \eta\nabla f(x_k)$：
$$f(x_{k+1}) \leq f(x_k) - \eta\|\nabla f(x_k)\|^2 + \frac{L\eta^2}{2}\|\nabla f(x_k)\|^2$$

取 $\eta = \frac{1}{L}$：
$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

再结合强凸性可得线性收敛。

---

#### 收敛速率对比表

| 函数类型 | 收敛速率 | 说明 |
|----------|----------|------|
| 一般凸 + L-光滑 | $O(1/k)$ | 次线性收敛 |
| μ-强凸 + L-光滑 | $O((1-\mu/L)^k)$ | 线性收敛 |
| 二阶方法（牛顿） | 误差近似平方下降 | 二次收敛（局部） |

---

### 6.3 典型优化问题模型 (#)

#### 一、线性回归（Linear Regression）

**问题**：给定样本 $(x_i, y_i)$，用线性模型预测连续值：
$$\hat{y}_i = w^T x_i + b$$

**最小二乘优化问题**：
$$\min_{w,b} \frac{1}{2}\sum_{i=1}^{n}(w^T x_i + b - y_i)^2$$

读法：让所有样本的预测误差平方和尽量小。平方损失会惩罚较大的预测误差，因此线性回归本质上是一个最小化二次函数的问题。

---

**矩阵形式**：

把偏置 $b$ 合并进参数，记设计矩阵为 $X$、标签向量为 $y$，则：
$$\min_w \frac{1}{2}\|Xw-y\|^2$$

梯度为：
$$\nabla f(w) = X^T(Xw-y)$$

令梯度为 0，得到正规方程：
$$X^TXw = X^Ty$$

如果 $X^TX$ 可逆，则闭式解为：
$$w^* = (X^TX)^{-1}X^Ty$$

**特点**：
- 目标函数是凸二次函数
- 有闭式解，也可以用梯度下降求解
- 岭回归是在最小二乘后面加 $L_2$ 正则项

---

#### 二、逻辑回归（Logistic Regression）

**问题**：二分类问题，标签 $y_i \in \{0, 1\}$

**模型**：
$$P(y=1|x; w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

**优化问题**（极大似然估计）：
$$\min_w -\sum_{i=1}^{n} [y_i \log \sigma(w^T x_i) + (1-y_i)\log(1-\sigma(w^T x_i))]$$

**性质**：
- 目标函数是凸的（可以验证海森矩阵半正定）
- 没有闭式解，需用迭代方法（梯度下降、牛顿法）

---

**梯度推导**：

令 $\hat{y}_i = \sigma(w^T x_i)$

损失函数对 $w$ 的梯度：
$$\nabla_w \mathcal{L}(w) = \sum_{i=1}^{n} (\hat{y}_i - y_i)x_i$$

**直观理解**：
- $\hat{y}_i - y_i$ 是预测误差
- 梯度是误差的加权和

---

#### 三、支持向量机（Support Vector Machine, SVM）

**问题**：二分类，寻找最大间隔超平面

**原始问题**：
$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \quad \forall i$$

**带软间隔**（允许一些样本在间隔内）：
$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n} \xi_i$$
$$\text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

---

**对偶问题**（更常用）：

$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
$$\text{s.t.} \quad \sum_{i=1}^{n} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

**核技巧**：将 $x_i^T x_j$ 替换为核函数 $K(x_i, x_j)$，可以处理非线性分类。

---

**几何解释**：

```
      ● 支持向量
     ╱│╲
    ╱ │ ╲  间隔边界
   ╱  │  ╲
  ●───┼───●  最优超平面 wᵀx+b=0
 ╱│   │   │╲
╱ │   │   │ ╲
●  │   │   │  ●
   │       │
   ← 间隔 = 2/‖w‖ →

最大化间隔 = 最小化 ‖w‖
```

---

#### 四、主成分分析（Principal Component Analysis, PCA）

**问题**：寻找数据的"主方向"（方差最大的方向）

**形式 1（方差最大化）**：
$$\max_{w: \|w\|=1} w^T \Sigma w$$

其中 $\Sigma = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T$ 是协方差矩阵。

**解**：$\Sigma$ 的最大特征值对应的单位特征向量。

---

**形式 2（重构误差最小化）**：
$$\min_{U: U^T U = I} \sum_{i=1}^{n} \|x_i - UU^T x_i\|^2$$

其中 $U \in \mathbb{R}^{d \times k}$ 是投影矩阵。

**解**：$U$ 的列是 $\Sigma$ 的前 $k$ 个特征向量。

---

**推导（形式 1）**：

目标：$\max_{w: \|w\|=1} w^T \Sigma w$

拉格朗日函数：$L(w, \lambda) = w^T \Sigma w - \lambda(w^T w - 1)$

KKT 条件：
$$\nabla_w L = 2\Sigma w - 2\lambda w = 0 \Rightarrow \Sigma w = \lambda w$$

故 $w$ 是 $\Sigma$ 的特征向量，$\lambda$ 是特征值。

目标函数值：$w^T \Sigma w = w^T (\lambda w) = \lambda$

要最大化，选择最大特征值对应的特征向量。

---

#### 五、典型优化问题对比

| 问题 | 目标函数 | 约束 | 解法 |
|------|----------|------|------|
| 线性回归 | $\frac{1}{2}\|Xw-y\|^2$ | 无 | 正规方程、梯度下降 |
| 逻辑回归 | 交叉熵 | 无 | 梯度下降、牛顿法 |
| SVM | $\frac{1}{2}\|w\|^2$ | $y_i(w^T x_i+b) \geq 1$ | SMO、对偶方法 |
| PCA | $w^T \Sigma w$ | $\|w\|=1$ | 特征分解 |
| 岭回归 | $\|Xw-y\|^2 + \lambda\|w\|^2$ | 无 | 闭式解：$(X^T X+\lambda I)^{-1}X^T y$ |
| LASSO | $\|Xw-y\|^2 + \lambda\|w\|_1$ | 无 | 坐标下降、近端梯度 |

---

## 第 7 章 最优性理论

最优性理论研究的是：一个点成为最优解（最小值点）需要满足什么条件。

### 7.1 无约束可微问题的最优性理论 (#)

#### 一、一阶必要条件

**定理**：如果 $x^*$ 是 $f$ 的局部最小值点，且 $f$ 在 $x^*$ 处可微，则：
$$\nabla f(x^*) = 0$$

**解释**：局部最小值点处的梯度必须为零（驻点）。

**直观理解**：如果梯度不为零，沿负梯度方向移动可以使函数值更小，与 $x^*$ 是局部最小值矛盾。

---

**例题 1**：求 $f(x) = x^3 - 3x$ 的驻点。

**解**：

一阶导数：$f'(x) = 3x^2 - 3$

令 $f'(x) = 0$：
$$3x^2 - 3 = 0 \Rightarrow x^2 = 1 \Rightarrow x = \pm 1$$

驻点为 $x = 1$ 和 $x = -1$。

---

#### 二、二阶必要条件

**定理**：如果 $x^*$ 是 $f$ 的局部最小值点，且 $f$ 在 $x^*$ 处二阶可微，则：
$$\nabla f(x^*) = 0 \quad \text{且} \quad \nabla^2 f(x^*) \succeq 0$$

**解释**：
- 一阶条件：梯度为零
- 二阶条件：海森矩阵半正定（函数在该点"向上弯曲"）

---

#### 三、二阶充分条件

**定理**：如果 $f$ 在 $x^*$ 处二阶可微，且：
$$\nabla f(x^*) = 0 \quad \text{且} \quad \nabla^2 f(x^*) \succ 0$$

则 $x^*$ 是严格局部最小值点。

**注意**：海森矩阵正定（不仅仅是半正定）保证了是严格局部最小值。

---

**例题 2**（续）：判断 $x = 1$ 和 $x = -1$ 的性质。

**解**：

二阶导数：$f''(x) = 6x$

**在 $x = 1$ 处**：
$$f''(1) = 6 > 0$$
海森矩阵正定，故 $x = 1$ 是严格局部最小值点。

**在 $x = -1$ 处**：
$$f''(-1) = -6 < 0$$
海森矩阵负定，故 $x = -1$ 是严格局部最大值点。

---

**函数图像**：

```
f(x)
 │
 │  ● x=-1（局部最大）
 │ ╱│╲
 │╱ │ ╲
─┼──┼──┼──→ x
 │╲ │  -1
 │ ╲│╲
 │  ● x=1（局部最小）
 │
```

---

#### 四、凸问题的充要条件

**定理**：如果 $f$ 是凸函数且可微，则：
$$x^* \text{是全局最小值点} \iff \nabla f(x^*) = 0$$

**意义**：对于凸优化问题，任何驻点都是全局最小值点！

**证明思路**：

($\Leftarrow$) 由凸函数的一阶条件：
$$f(y) \geq f(x^*) + \nabla f(x^*)^T(y - x^*) = f(x^*)$$
故 $x^*$ 是全局最小值。

($\Rightarrow$) 由一阶必要条件。

---

**例题 3**：证明 $f(x) = x^2 + 2x + 1$ 的最小值在 $x = -1$ 处。

**解**：

$f$ 是二次函数，海森矩阵 $f''(x) = 2 > 0$，故为凸函数。

梯度：$f'(x) = 2x + 2$

令 $f'(x) = 0$：$x = -1$

由凸问题的充要条件，$x = -1$ 是全局最小值点。

最小值：$f(-1) = (-1)^2 + 2(-1) + 1 = 0$

---

### 7.2 拉格朗日函数与对偶问题

#### 一、等式约束问题

**问题**：
$$\min_x f(x) \quad \text{s.t.} \quad h_i(x) = 0, \quad i = 1, \dots, m$$

**拉格朗日函数**：
$$L(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i h_i(x)$$

其中 $\lambda = (\lambda_1, \dots, \lambda_m)$ 是拉格朗日乘子。

这一步的作用是把“带约束的问题”改写成一个包含乘子的函数。$\lambda_i$ 可以理解为约束 $h_i(x)=0$ 对最优解的影响强度；在最优点处，目标函数的梯度必须和约束梯度达到平衡。

---

**KKT 条件**（等式约束）：

$$\nabla_x L(x^*, \lambda^*) = 0 \quad \text{（平稳性）}$$
$$h_i(x^*) = 0, \quad i = 1, \dots, m \quad \text{（原始可行）}$$

---

**例题 1**：$\min x^2 + y^2$ s.t. $x + y = 1$

**解**：

拉格朗日函数：$L(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 1)$

KKT 条件：
$$\frac{\partial L}{\partial x} = 2x + \lambda = 0 \quad \Rightarrow \quad x = -\frac{\lambda}{2}$$
$$\frac{\partial L}{\partial y} = 2y + \lambda = 0 \quad \Rightarrow \quad y = -\frac{\lambda}{2}$$
$$x + y = 1$$

代入约束：
$$-\frac{\lambda}{2} - \frac{\lambda}{2} = 1 \Rightarrow -\lambda = 1 \Rightarrow \lambda = -1$$

解得：$x = y = \frac{1}{2}$

最优值：$f\left(\frac{1}{2}, \frac{1}{2}\right) = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$

---

**几何解释**：

```
y
 │     x+y=1（约束直线）
 │    ╱
 │   ╱  ● (1/2,1/2) 最优解
 │  ╱  ╱│
 │ ╱  ╱ │
 │╱  ╱  │ 等高线 x²+y²=c
─┼─╱───┼──→ x
 │╱    │
 │     │

最优解处，目标函数等高线与约束相切
```

---

#### 二、不等式约束问题

**问题**：
$$\min_x f(x) \quad \text{s.t.} \quad g_j(x) \leq 0, \quad j = 1, \dots, p$$

**拉格朗日函数**：
$$L(x, \mu) = f(x) + \sum_{j=1}^{p} \mu_j g_j(x)$$

其中 $\mu_j \geq 0$ 是拉格朗日乘子。

不等式约束写成 $g_j(x)\leq 0$ 后，乘子要求 $\mu_j\geq 0$。直观上，只有被“顶住”的约束才会产生正的乘子；如果某个约束还有余量，它对当前最优解没有实际作用。

---

**KKT 条件**：

| 条件 | 公式 | 含义 |
|------|------|------|
| 平稳性 | $\nabla f(x^*) + \sum_j \mu_j^* \nabla g_j(x^*) = 0$ | 梯度平衡 |
| 原始可行 | $g_j(x^*) \leq 0$ | 满足约束 |
| 对偶可行 | $\mu_j^* \geq 0$ | 乘子非负 |
| 互补松弛 | $\mu_j^* g_j(x^*) = 0$ | 活跃约束才有乘子 |

---

**互补松弛条件的解释**：

- 如果 $g_j(x^*) < 0$（约束不活跃），则 $\mu_j^* = 0$
- 如果 $\mu_j^* > 0$（乘子非零），则 $g_j(x^*) = 0$（约束活跃）

**直观理解**：只有"绑紧"的约束才对最优解有影响。

---

#### 三、对偶函数与对偶问题

**对偶函数**：
$$d(\mu) = \inf_x L(x, \mu) = \inf_x \left[f(x) + \sum_{j=1}^{p} \mu_j g_j(x)\right]$$

读法：先固定乘子 $\mu$，再对原变量 $x$ 把拉格朗日函数最小化。这样得到的 $d(\mu)$ 是原问题最优值的一个下界。

**性质**：对偶函数 $d(\mu)$ 总是凹函数（即使原问题不是凸的）。

---

**对偶问题**：
$$\max_{\mu \geq 0} d(\mu)$$

读法：对偶问题是在所有合法乘子 $\mu$ 中，寻找最大的下界。弱对偶说这个下界永远不会超过原问题最优值；强对偶说在条件好的凸问题里，这个下界可以刚好顶到原问题最优值。

**弱对偶**：$d^* \leq p^*$（对偶最优值 ≤ 原问题最优值）

**强对偶**：$d^* = p^*$（对凸问题通常成立）

---

**例题 2**：求 $\min x^2$ s.t. $x \geq 1$（即 $1-x \leq 0$）的对偶问题。

**解**：

拉格朗日函数：$L(x, \mu) = x^2 + \mu(1-x)$

对偶函数：
$$d(\mu) = \inf_x L(x, \mu) = \inf_x [x^2 - \mu x + \mu]$$

对 $x$ 求导并令为零：
$$2x - \mu = 0 \Rightarrow x = \frac{\mu}{2}$$

代入：
$$d(\mu) = \left(\frac{\mu}{2}\right)^2 - \mu\left(\frac{\mu}{2}\right) + \mu = \mu - \frac{\mu^2}{4}$$

对偶问题：
$$\max_{\mu \geq 0} \mu - \frac{\mu^2}{4}$$

求导：$1 - \frac{\mu}{2} = 0 \Rightarrow \mu^* = 2$

对偶最优值：$d^* = 2 - \frac{4}{4} = 1$

原问题最优解显然是 $x^* = 1$，最优值 $p^* = 1^2 = 1$

故 $d^* = p^*$，强对偶成立！

---

### 7.3 约束优化问题的最优性理论

#### KKT 条件（一般约束问题）

**问题**：
$$\min_x f(x) \quad \text{s.t.} \quad h_i(x) = 0 \ (i=1,\dots,m), \quad g_j(x) \leq 0 \ (j=1,\dots,p)$$

**拉格朗日函数**：
$$L(x, \lambda, \mu) = f(x) + \sum_{i=1}^{m} \lambda_i h_i(x) + \sum_{j=1}^{p} \mu_j g_j(x)$$

它把目标函数、等式约束和不等式约束统一放进一个函数里。KKT 条件就是在这个拉格朗日函数上写出“最优点应该满足什么平衡关系”。

---

**KKT 条件**：

| 条件 | 公式 |
|------|------|
| 原始可行 | $h_i(x^*) = 0, \quad g_j(x^*) \leq 0$ |
| 对偶可行 | $\mu_j^* \geq 0$ |
| 互补松弛 | $\mu_j^* g_j(x^*) = 0$ |
| 平稳性 | $\nabla f(x^*) + \sum_i \lambda_i^* \nabla h_i(x^*) + \sum_j \mu_j^* \nabla g_j(x^*) = 0$ |

---

读法：KKT 条件可以理解成四件事同时成立：解本身满足原约束；不等式乘子非负；没卡住的约束乘子为 0；目标函数梯度和约束梯度在最优点处互相抵消。

**定理**：
- **必要性**：如果 $x^*$ 是局部最优解，且满足某些正则性条件（约束规范），则存在 KKT 乘子。
- **充分性**：如果问题是凸的（$f$ 凸，$h_i$ 线性，$g_j$ 凸），且满足 KKT 条件，则 $x^*$ 是全局最优解。

---

**例题 3**：$\min x^2 + y^2$ s.t. $x + y \geq 1$（即 $1 - x - y \leq 0$）

**解**：

拉格朗日函数：$L(x, y, \mu) = x^2 + y^2 + \mu(1 - x - y)$

KKT 条件：

1. 原始可行：$1 - x - y \leq 0$
2. 对偶可行：$\mu \geq 0$
3. 互补松弛：$\mu(1 - x - y) = 0$
4. 平稳性：$\frac{\partial L}{\partial x} = 2x - \mu = 0$，$\frac{\partial L}{\partial y} = 2y - \mu = 0$

---

**求解**：

由平稳性：$x = y = \frac{\mu}{2}$

**情况 1**：$\mu = 0$

则 $x = y = 0$，但 $1 - 0 - 0 = 1 \not\leq 0$，不满足原始可行。

**情况 2**：$\mu > 0$

由互补松弛：$1 - x - y = 0$，即 $x + y = 1$

代入 $x = y = \frac{\mu}{2}$：
$$\frac{\mu}{2} + \frac{\mu}{2} = 1 \Rightarrow \mu = 1$$

故 $x = y = \frac{1}{2}$

**验证**：
- 原始可行：$1 - \frac{1}{2} - \frac{1}{2} = 0 \leq 0$ ✓
- 对偶可行：$\mu = 1 \geq 0$ ✓
- 互补松弛：$1 \cdot 0 = 0$ ✓
- 平稳性：$2 \cdot \frac{1}{2} - 1 = 0$ ✓

最优值：$f\left(\frac{1}{2}, \frac{1}{2}\right) = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$

---

#### KKT 条件的几何解释

```
可行域边界 g(x)=0
      ╱│
     ╱ │
    ╱  │  ∇g(x*)（约束法向，指向可行域外）
   ╱   ● x*（最优解）
  ╱   ╱│
 ╱   ╱ │
╱   ╱  │
│  ∇f(x*)（目标梯度）
│
可行域 g(x)≤0

最优解处：∇f(x*) 与 ∇g(x*) 反向（相差一个正的乘子）
即：∇f(x*) + μ*∇g(x*) = 0
```

---

**意义**：对于凸问题，KKT 条件是充分必要条件。这意味着：
1. 如果找到满足 KKT 条件的点，它就是全局最优解
2. 全局最优解一定满足 KKT 条件（在约束规范下）

---

## 第 8 章 优化算法

### 8.1 梯度下降法 (#)

#### 一、梯度下降法的核心思想

**基本思想**：沿负梯度方向移动，函数值下降最快。

**更新公式**：
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

其中 $\eta > 0$ 是学习率（步长）。

---

**几何解释**：

```
f(x)
 │
 │  ● x₀
 │ ╱│
 │╱ │  沿负梯度方向移动
 │  ▼
 │  ● x₁
 │ ╱│
 │╱ │  继续移动
 │  ▼
 │  ● x₂ → 收敛到最小值
 └───────────→ x
```

---

#### 二、学习率的选择

**学习率的影响**：

| 学习率 $\eta$ | 结果 | 图像 |
|--------------|------|------|
| 太小 | 收敛太慢 | 小步慢走 |
| 适中 | 稳定收敛 | 稳步前进 |
| 太大 | 震荡甚至发散 | 来回跳跃 |
| 过大 | 直接越过最小值 | 越跑越远 |

---

**例题 1**：用梯度下降法求 $f(x) = x^2$ 的最小值。

**解**：

梯度：$\nabla f(x) = 2x$

更新公式：$x_{k+1} = x_k - \eta \cdot 2x_k = (1 - 2\eta)x_k$

**不同学习率的表现**：

| $\eta$ | $x_0$ | $x_1$ | $x_2$ | $x_3$ | 收敛性 |
|--------|-------|-------|-------|-------|--------|
| 0.1 | 1 | 0.8 | 0.64 | 0.512 | 收敛 |
| 0.3 | 1 | 0.4 | 0.16 | 0.064 | 快速收敛 |
| 0.5 | 1 | 0 | 0 | 0 | 一步到位（最优）|
| 0.8 | 1 | -0.6 | 0.36 | -0.216 | 震荡收敛 |
| 1.0 | 1 | -1 | 1 | -1 | 永久震荡 |
| 1.2 | 1 | -1.4 | 1.96 | -2.744 | 发散 |

**分析**：
- 对于 $f(x) = x^2$，最优学习率是 $\eta = 0.5 = \frac{1}{L}$（$L=2$ 是光滑常数）
- 当 $\eta < 1$ 时收敛，当 $\eta \geq 1$ 时震荡或发散

---

#### 三、收敛性定理

**定理 1**（凸函数 + L-光滑）：

如果 $f$ 是凸函数且 $L$-光滑，梯度下降使用步长 $\eta \leq \frac{2}{L}$，则：
$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k} = O\left(\frac{1}{k}\right)$$

**定理 2**（强凸 + L-光滑）：

如果 $f$ 是 $\mu$-强凸且 $L$-光滑，梯度下降使用步长 $\eta = \frac{1}{L}$，则：
$$\|x_k - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|^2$$

收敛速率是**线性**的（几何级数）！

---

#### 四、梯度下降的变体

| 变体 | 梯度计算 | 优点 | 缺点 |
|------|----------|------|------|
| 批量梯度下降 (BGD) | $\nabla f(x) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(x)$ | 稳定、可并行 | 慢、内存需求大 |
| 随机梯度下降 (SGD) | $\nabla f_i(x)$（随机一个样本） | 快、能跳出局部最优 | 噪声大、震荡 |
| 小批量梯度下降 | $\frac{1}{|B|}\sum_{i \in B} \nabla f_i(x)$ | 平衡稳定性和速度 | 需要选择 batch size |

---

### 8.2 面向神经网络的前反向传播算法 (#)

#### 一、神经网络结构

考虑一个简单的两层神经网络：

```
输入层 (x) → 隐藏层 (ReLU) → 输出层 (Sigmoid) → 预测 (ŷ)
```

**符号说明**：
- $x \in \mathbb{R}^d$：输入特征
- $W^{[1]} \in \mathbb{R}^{h \times d}$：第一层权重
- $b^{[1]} \in \mathbb{R}^h$：第一层偏置
- $W^{[2]} \in \mathbb{R}^{1 \times h}$：第二层权重
- $b^{[2]} \in \mathbb{R}$：第二层偏置

---

#### 二、前向传播（Forward Propagation）

**Step 1**：计算隐藏层的线性部分
$$z^{[1]} = W^{[1]}x + b^{[1]}$$

**Step 2**：应用激活函数（ReLU）
$$a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$

**Step 3**：计算输出层的线性部分
$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$

**Step 4**：应用激活函数（Sigmoid）
$$\hat{y} = a^{[2]} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-z^{[2]}}}$$

---

**损失函数**（二元交叉熵）：
$$\mathcal{L}(\hat{y}, y) = -[y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$$

---

#### 三、反向传播（Backward Propagation）

反向传播的核心是**链式法则**！

**Step 1**：计算输出层误差
$$\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}}$$

其中：
- $\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$
- $\frac{\partial \hat{y}}{\partial z^{[2]}} = \hat{y}(1-\hat{y})$（Sigmoid 的导数）

相乘得：
$$\delta^{[2]} = \hat{y} - y$$

---

**Step 2**：计算隐藏层误差
$$\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial z^{[1]}} = (W^{[2]})^T \delta^{[2]} \odot \text{ReLU}'(z^{[1]})$$

其中 $\odot$ 是逐元素乘法（Hadamard 积）。

$\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$

---

**Step 3**：计算梯度
$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]}(a^{[1]})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]}$$
$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]}x^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \delta^{[1]}$$

---

#### 四、完整数值例子

**设置**：
- $x = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$，$y = 1$
- $W^{[1]} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$，$b^{[1]} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$
- $W^{[2]} = \begin{pmatrix} 1 & 1 \end{pmatrix}$，$b^{[2]} = 0$

---

**前向传播**：

1. $z^{[1]} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$

2. $a^{[1]} = \text{ReLU}\begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$

3. $z^{[2]} = \begin{pmatrix} 1 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ 2 \end{pmatrix} + 0 = 3$

4. $\hat{y} = \sigma(3) = \frac{1}{1+e^{-3}} \approx 0.9526$

5. $\mathcal{L} = -[1 \times \ln(0.9526) + 0] \approx 0.0486$

---

**反向传播**：

1. $\delta^{[2]} = \hat{y} - y = 0.9526 - 1 = -0.0474$

2. $\text{ReLU}'(z^{[1]}) = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$（因为 $z^{[1]} > 0$）

3. $\delta^{[1]} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \times (-0.0474) \odot \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} -0.0474 \\ -0.0474 \end{pmatrix}$

4. 梯度：
   - $\frac{\partial \mathcal{L}}{\partial W^{[2]}} = (-0.0474) \times \begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} -0.0474 & -0.0948 \end{pmatrix}$
   - $\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \begin{pmatrix} -0.0474 \\ -0.0474 \end{pmatrix} \begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} -0.0474 & -0.0948 \\ -0.0474 & -0.0948 \end{pmatrix}$

---

### 8.3 随机梯度下降法 (SGD) (#)

#### 一、SGD 的核心思想

**问题**：批量梯度下降每次迭代需要计算所有样本的梯度，当 $n$ 很大时非常慢。

**SGD 思想**：每次只使用一个随机样本来估计梯度。

---

**更新公式**：

批量梯度下降：
$$w_{k+1} = w_k - \eta \sum_{i=1}^{n} \nabla f_i(w_k)$$

SGD：
$$w_{k+1} = w_k - \eta \nabla f_i(w_k)$$
其中 $i$ 是随机选择的样本索引。

---

#### 二、SGD 与 BGD 的对比

| 特性 | BGD | SGD |
|------|-----|-----|
| 每次迭代计算量 | $O(n)$ | $O(1)$ |
| 收敛稳定性 | 稳定 | 震荡 |
| 跳出局部最优 | 难 | 能 |
| 收敛速率 | $O(1/k)$ | $O(1/\sqrt{k})$ |
| 内存需求 | 高 | 低 |

---

**收敛速率差异的原因**：

SGD 的梯度是真实梯度的**无偏估计**：
$$E[\nabla f_i(w)] = \frac{1}{n}\sum_{i=1}^n \nabla f_i(w) = \nabla f(w)$$

但方差较大，导致收敛路径震荡。

---

#### 三、Mini-batch SGD

**折中方案**：每次使用一小批样本（batch size 通常为 32, 64, 128, ...）

**更新公式**：
$$w_{k+1} = w_k - \eta \frac{1}{|B|}\sum_{i \in B} \nabla f_i(w_k)$$

**优点**：
- 比 SGD 稳定（方差小）
- 比 BGD 快
- 可以利用 GPU 并行计算

---

**例题**：线性回归 $f_i(w) = \frac{1}{2}(w^T x_i - y_i)^2$

梯度：$\nabla f_i(w) = (w^T x_i - y_i)x_i$

| 方法 | 更新公式 |
|------|----------|
| BGD | $w_{k+1} = w_k - \eta \sum_{i=1}^{n} (w^T x_i - y_i)x_i$ |
| SGD | $w_{k+1} = w_k - \eta (w^T x_i - y_i)x_i$（随机选 $i$） |
| Mini-batch | $w_{k+1} = w_k - \eta \frac{1}{32}\sum_{i \in B} (w^T x_i - y_i)x_i$ |

---

### 8.4 动量法

#### 一、Polyak 动量

**思想**：模拟物理中的动量概念，累积之前的更新方向。

**更新公式**：
$$v_{k+1} = \gamma v_k + \eta \nabla f(x_k)$$
$$x_{k+1} = x_k - v_{k+1}$$

其中 $\gamma \in [0, 1)$ 是动量系数（通常取 0.9）。

---

**物理直观**：

想象一个小球从山上滚下：
- 重力（梯度）使小球加速
- 摩擦力（$\gamma$）减缓速度
- 小球会累积速度，越滚越快

---

**例题 1**：$f(x) = x^2$，$\nabla f(x) = 2x$

设 $x_0 = 1$，$\eta = 0.1$，$\gamma = 0.9$，$v_0 = 0$

| k | $x_k$ | $\nabla f(x_k)$ | $v_{k+1}$ |
|---|-------|-----------------|-----------|
| 0 | 1.0 | 2.0 | $0.9 \times 0 + 0.1 \times 2.0 = 0.2$ |
| 1 | 0.8 | 1.6 | $0.9 \times 0.2 + 0.1 \times 1.6 = 0.34$ |
| 2 | 0.46 | 0.92 | $0.9 \times 0.34 + 0.1 \times 0.92 = 0.398$ |
| 3 | 0.062 | 0.124 | $0.9 \times 0.398 + 0.1 \times 0.124 = 0.371$ |
| 4 | -0.309 | ... | ... |

对比普通梯度下降（$\eta=0.1$）：
- $x_0 = 1$，$x_1 = 0.8$，$x_2 = 0.64$，$x_3 = 0.512$，$x_4 = 0.4096$

动量法收敛更快！

---

#### 二、Nesterov 动量

**思想**：提前查看"前方"的梯度，更准确地更新。

**更新公式**：
$$v_{k+1} = \gamma v_k + \eta \nabla f(x_k - \gamma v_k)$$
$$x_{k+1} = x_k - v_{k+1}$$

---

**对比**：

```
普通动量：          Nesterov 动量：
   x_k                  x_k
   ││                   │\
   ││ 在当前位置计算梯度   │ \ 提前看前方
   ▼│                   ▼  \▼
  更新                  更新点
```

**效果**：Nesterov 动量通常比普通动量收敛更快、更稳定。

---

### 8.5 自适应梯度算法

#### 一、AdaGrad（Adaptive Gradient）

**思想**：为每个参数自适应调整学习率。

**更新公式**：
$$G_k = \sum_{i=1}^{k} (\nabla f(x_i))^2 \quad \text{（逐元素平方和）}$$
$$x_{k+1} = x_k - \frac{\eta}{\sqrt{G_k + \epsilon}} \odot \nabla f(x_k)$$

其中 $\epsilon \approx 10^{-8}$ 防止除零。

---

**直观解释**：

- 频繁更新的参数：$G_k$ 大 → 有效学习率小 → 更新慢
- 稀疏更新的参数：$G_k$ 小 → 有效学习率大 → 更新快

**适用场景**：稀疏特征（如 NLP 中的词向量）

---

**例题**：一维情况，$\nabla f(x_1) = 2$，$\nabla f(x_2) = 1$，$\eta = 1$

| k | $\nabla f(x_k)$ | $G_k$ | 有效学习率 $\frac{\eta}{\sqrt{G_k}}$ |
|---|-----------------|-------|--------------------------------------|
| 1 | 2 | 4 | $1/2 = 0.5$ |
| 2 | 1 | 4+1=5 | $1/\sqrt{5} \approx 0.447$ |
| 3 | 0.5 | 5+0.25=5.25 | $1/\sqrt{5.25} \approx 0.436$ |

学习率逐渐减小！

---

**缺点**：学习率单调递减，可能过早停止更新。

---

#### 二、RMSProp（Root Mean Square Propagation）

**思想**：使用指数加权平均，避免学习率过快衰减。

**更新公式**：
$$E[g^2]_k = \beta E[g^2]_{k-1} + (1-\beta)(\nabla f(x_k))^2$$
$$x_{k+1} = x_k - \frac{\eta}{\sqrt{E[g^2]_k + \epsilon}} \odot \nabla f(x_k)$$

其中 $\beta \approx 0.9$ 是衰减率。

---

**对比 AdaGrad**：

| 特性 | AdaGrad | RMSProp |
|------|---------|---------|
| 历史梯度 | 累加 | 指数加权平均 |
| 学习率 | 单调递减 | 动态调整 |
| 适用场景 | 稀疏问题 | 非平稳问题 |

---

#### 三、Adam（Adaptive Moment Estimation）

**思想**：结合动量法和 RMSProp 的优点。

**更新公式**：

1. 一阶矩（类似动量）：
   $$m_k = \beta_1 m_{k-1} + (1-\beta_1)\nabla f(x_k)$$

2. 二阶矩（类似 RMSProp）：
   $$v_k = \beta_2 v_{k-1} + (1-\beta_2)(\nabla f(x_k))^2$$

3. 偏差修正：
   $$\hat{m}_k = \frac{m_k}{1-\beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1-\beta_2^k}$$

4. 更新：
   $$x_{k+1} = x_k - \frac{\eta}{\sqrt{\hat{v}_k} + \epsilon} \hat{m}_k$$

---

**常用参数**：
- $\beta_1 = 0.9$（一阶矩衰减）
- $\beta_2 = 0.999$（二阶矩衰减）
- $\epsilon = 10^{-8}$（数值稳定）
- $\eta = 0.001$（学习率）

---

**具体计算例子**：

设 $\nabla f(x_1) = 2$，$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\eta = 0.001$，$m_0 = v_0 = 0$

**第一步**：
- $m_1 = 0.9 \times 0 + 0.1 \times 2 = 0.2$
- $v_1 = 0.999 \times 0 + 0.001 \times 4 = 0.004$
- $\hat{m}_1 = \frac{0.2}{1-0.9} = 2$
- $\hat{v}_1 = \frac{0.004}{1-0.999} = 4$
- $x_2 = x_1 - \frac{0.001}{\sqrt{4} + 10^{-8}} \times 2 \approx x_1 - 0.001$

---

#### 四、优化算法对比总结

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| SGD | 简单、能跳出局部最优 | 收敛慢、震荡 | 凸问题、小规模 |
| SGD + 动量 | 收敛快 | 需要调动量参数 | 深度学习 |
| AdaGrad | 自适应、适合稀疏 | 学习率衰减快 | 稀疏特征 |
| RMSProp | 自适应、适合非平稳 | 需要调 $\beta$ | RNN、LSTM |
| Adam | 综合性能好 | 可能不收敛到最优 | 大多数深度学习任务 |

---

### 8.6 增广拉格朗日算法

**问题**：
$$\min_x f(x) \quad \text{s.t.} \quad h(x) = 0$$

**增广拉格朗日函数**：
$$L_\rho(x, \lambda) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2$$

其中 $\rho > 0$ 是惩罚参数。

---

**更新公式**：
$$x_{k+1} = \arg\min_x L_\rho(x, \lambda_k)$$
$$\lambda_{k+1} = \lambda_k + \rho h(x_{k+1})$$

---

**例题**：$\min x^2$ s.t. $x = 1$

增广拉格朗日：$L_\rho(x, \lambda) = x^2 + \lambda(x-1) + \frac{\rho}{2}(x-1)^2$

**迭代**（设 $\rho = 2$，$\lambda_0 = 0$）：

1. $x_1 = \arg\min_x [x^2 + 0 + (x-1)^2] = \arg\min_x [2x^2 - 2x + 1]$
   - 导数：$4x - 2 = 0 \Rightarrow x_1 = 0.5$
   - $\lambda_1 = 0 + 2(0.5 - 1) = -1$

2. $x_2 = \arg\min_x [x^2 - (x-1) + (x-1)^2] = \arg\min_x [2x^2 - 4x + 2]$
   - 导数：$4x - 4 = 0 \Rightarrow x_2 = 1$
   - $\lambda_2 = -1 + 2(1 - 1) = -1$

收敛到最优解 $x = 1$！

---

**优点**：
- 比标准拉格朗日方法收敛更快
- 可以处理等式和不等式约束
- 是 ADMM 等算法的基础

---

# 第四部分：大模型数学基础

> 新版考纲第 9 章的官方重点只有五项：`循环神经网络`、`自注意力机制`、`Transformer 模型`、`Decoder-only 大模型参数量 / 计算量 / 内存量`、`LoRA`。本节前面的 `Softmax / 交叉熵 / ERM` 作为配套背景保留，不单独视为新版条目。

## 第 9 章 Transformer 与注意力机制

### 附：Softmax 与多分类交叉熵（背景）

#### 一、Softmax 函数

**定义**：将向量 $z \in \mathbb{R}^K$ 映射为概率分布：
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, \dots, K$$

读法：第 $i$ 类的分数 $z_i$ 先取指数变成正数，再除以所有类别指数分数之和，因此输出可以当成各类别的概率。

**性质**：
1. 输出非负：$\text{softmax}(z)_i > 0$
2. 和为 1：$\sum_{i=1}^K \text{softmax}(z)_i = 1$
3. 平移不变性：$\text{softmax}(z + c) = \text{softmax}(z)$（$c$ 是常数）

---

**例题 1**：$z = [2, 1, 0]^T$

**计算**：
- $e^2 \approx 7.389$
- $e^1 \approx 2.718$
- $e^0 = 1$
- 分母：$7.389 + 2.718 + 1 = 11.107$

$$\text{softmax}(z) = \begin{pmatrix} \frac{7.389}{11.107} \\ \frac{2.718}{11.107} \\ \frac{1}{11.107} \end{pmatrix} \approx \begin{pmatrix} 0.665 \\ 0.245 \\ 0.090 \end{pmatrix}$$

**验证**：$0.665 + 0.245 + 0.090 = 1.000$ ✓

---

**数值稳定性技巧**：

由于 $e^z$ 可能溢出，实际实现时减去最大值：
$$\text{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}$$

上例中，$z - \max(z) = [2-2, 1-2, 0-2]^T = [0, -1, -2]^T$

$$\text{softmax}(z) = \begin{pmatrix} \frac{e^0}{e^0 + e^{-1} + e^{-2}} \\ \frac{e^{-1}}{e^0 + e^{-1} + e^{-2}} \\ \frac{e^{-2}}{e^0 + e^{-1} + e^{-2}} \end{pmatrix} = \begin{pmatrix} \frac{1}{1 + 0.368 + 0.135} \\ \frac{0.368}{1.503} \\ \frac{0.135}{1.503} \end{pmatrix} \approx \begin{pmatrix} 0.665 \\ 0.245 \\ 0.090 \end{pmatrix}$$

结果相同！

---

#### 二、多分类交叉熵损失

**定义**：
$$\mathcal{L}(\hat{y}, y) = -\sum_{i=1}^{K} y_i \log \hat{y}_i$$

其中：
- $y$ 是 one-hot 标签向量（只有一个元素为 1，其余为 0）
- $\hat{y} = \text{softmax}(z)$ 是预测概率分布

读法：真实类别对应的预测概率越大，负对数越小，损失越小；如果真实类别的预测概率接近 0，损失会非常大。

---

**简化形式**（one-hot 标签）：

如果真实类别是 $c$（即 $y_c = 1$，其余为 0），则：
$$\mathcal{L} = -\log \hat{y}_c$$

---

**例题 2**：真实标签 $y = [1, 0, 0]^T$（类别 1），预测 $\hat{y} = [0.665, 0.245, 0.090]^T$

**损失**：
$$\mathcal{L} = -[1 \times \ln(0.665) + 0 \times \ln(0.245) + 0 \times \ln(0.090)] = -\ln(0.665) \approx 0.408$$

**解释**：预测概率 0.665 对应类别 1，损失是它的负对数。

---

**损失的意义**：

| 预测概率 $\hat{y}_c$ | 损失 $-\log(\hat{y}_c)$ | 解释 |
|---------------------|------------------------|------|
| 0.99 | 0.01 | 非常确定，损失小 |
| 0.5 | 0.693 | 不确定，中等损失 |
| 0.1 | 2.30 | 很不确信，损失大 |
| 0.01 | 4.60 | 完全错了，损失很大 |

---

### 附：经验风险最小化（背景）

#### 经验风险最小化（Empirical Risk Minimization, ERM）

**核心思想**：在训练集上最小化平均损失。

**公式**：
$$\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f(x_i; \theta), y_i)$$

其中：
- $\theta$：模型参数
- $f(x; \theta)$：模型预测函数
- $\mathcal{L}$：损失函数
- $n$：训练样本数

读法：遍历训练集中的每个样本，计算预测值和真实标签之间的损失，再取平均；训练模型就是调整 $\theta$，让这个平均损失尽量小。

---

**带正则化的 ERM**：

$$\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f(x_i; \theta), y_i) + \lambda R(\theta)$$

**正则化项的作用**：前半部分要求模型拟合训练数据，后半部分 $\lambda R(\theta)$ 惩罚过于复杂的参数。$\lambda$ 越大，越强调简单模型；$\lambda$ 越小，越强调训练集拟合。

| 正则化 | 公式 | 效果 |
|--------|------|------|
| $L_2$（岭回归） | $\|\theta\|_2^2 = \sum_j \theta_j^2$ | 参数趋向小的值 |
| $L_1$（LASSO） | $\|\theta\|_1 = \sum_j |\theta_j|$ | 参数趋向稀疏 |
| Dropout | 随机丢弃神经元 | 减少共适应 |

---

### 9.1 循环神经网络与序列建模

#### 一、循环神经网络（RNN）

**核心递推公式**：
$$h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h), \quad y_t = W_{hy}h_t + b_y$$

其中：
- $x_t$：第 $t$ 个时刻输入
- $h_t$：隐藏状态，汇总到当前时刻为止的历史信息
- $\phi(\cdot)$：非线性激活函数（如 `tanh` 或 `ReLU`）

读法：当前隐藏状态 $h_t$ 同时看当前输入 $x_t$ 和上一个隐藏状态 $h_{t-1}$。所以 RNN 不是每个位置独立计算，而是把历史信息一步步传下去。

**RNN 的考点抓手**：
1. 会写出递推关系，说明"当前状态依赖当前输入和上一时刻隐藏状态"。
2. 能解释为什么它适合序列建模：参数共享，天然按时间展开。
3. 能说明它的局限：难并行、长距离依赖弱、容易梯度消失 / 爆炸。

**RNN 与 Transformer 对比**：

| 模型 | 信息传递方式 | 并行性 | 长距离依赖 |
|------|--------------|--------|------------|
| RNN | 逐时刻递推 | 差 | 容易退化 |
| Transformer | 全局注意力 | 强 | 更容易建模 |

---

#### 二、语言模型的目标

**目标**：建模序列的概率分布
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, \dots, w_{t-1})$$

**链式法则**：任何序列都可以分解为条件概率的乘积。

读法：一句话的整体概率，可以拆成“第 1 个词的概率 × 在前面词已知时第 2 个词的概率 × ……”。语言模型真正学习的是每一步的下一个词概率。

---

**例题**：句子 "I love NLP"

$$P(\text{"I love NLP"}) = P(\text{"I"}) \times P(\text{"love"}|\text{"I"}) \times P(\text{"NLP"}|\text{"I love"})$$

**语言模型的任务**：学习这些条件概率。

---

#### 三、自回归模型（Autoregressive Model）

**核心思想**：用前面的词预测下一个词。

$$P(w_t | w_{<t}) = \text{softmax}(W h_t + b)$$

其中：
- $h_t$ 是时刻 $t$ 的隐藏状态
- $W$ 是权重矩阵
- 输出维度等于词表大小

读法：模型把当前隐藏状态 $h_t$ 变成词表上每个词的分数，再用 softmax 变成概率分布，最后选概率高的词作为下一个词的候选。

---

**N-gram 模型**（传统方法）：

$$P(w_t | w_{t-1}, \dots, w_{t-N+1})$$

- 只依赖前 $N-1$ 个词
- 需要平滑处理未见过的 n-gram

**神经网络语言模型**：

使用 RNN、LSTM 或 Transformer 编码整个历史。

---

### 9.2 自注意力机制

#### 一、注意力机制的核心思想

**直观理解**：给定一个查询（Query），在一组键值对（Key-Value）中检索相关信息。

**类比**：数据库查询
- Query：你的搜索关键词
- Key：数据库中的索引
- Value：实际的数据

---

#### 二、Scaled Dot-Product Attention

**公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵（$n$ 个查询，每个 $d_k$ 维）
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵
- $\sqrt{d_k}$：缩放因子

读法：先用 $QK^T$ 计算“每个 Query 和每个 Key 有多相关”，再用 softmax 把相关性变成权重，最后用这些权重对 $V$ 做加权平均。输出的每一行都是根据注意力权重混合出来的新表示。

---

**为什么需要缩放因子 $\sqrt{d_k}$**？

当 $d_k$ 较大时，点积 $QK^T$ 的值会很大，导致 softmax 进入梯度消失区域。

**例子**：
- 如果 $d_k = 64$，点积可能达到 $64 \times 1 \times 1 = 64$
- $e^{64}$ 会溢出！
- 除以 $\sqrt{64} = 8$ 后，数值范围缩小 8 倍

---

#### 三、详细计算例子

**设置**：
- $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \in \mathbb{R}^{2 \times 2}$（2 个查询）
- $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \in \mathbb{R}^{2 \times 2}$（2 个键）
- $V = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \in \mathbb{R}^{2 \times 2}$（2 个值）
- $d_k = 2$

---

**Step 1**：计算注意力分数（点积）
$$QK^T = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

解释：
- 第 1 个查询与第 1 个键的相似度 = 1
- 第 1 个查询与第 2 个键的相似度 = 0
- 第 2 个查询与第 1 个键的相似度 = 0
- 第 2 个查询与第 2 个键的相似度 = 1

---

**Step 2**：缩放
$$\frac{QK^T}{\sqrt{d_k}} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \approx \begin{pmatrix} 0.707 & 0 \\ 0 & 0.707 \end{pmatrix}$$

---

**Step 3**：Softmax（按行）

第 1 行：$\text{softmax}([0.707, 0]) = \begin{pmatrix} \frac{e^{0.707}}{e^{0.707}+e^0} & \frac{e^0}{e^{0.707}+e^0} \end{pmatrix} \approx \begin{pmatrix} 0.67 & 0.33 \end{pmatrix}$

第 2 行：$\text{softmax}([0, 0.707]) = \begin{pmatrix} \frac{e^0}{e^0+e^{0.707}} & \frac{e^{0.707}}{e^0+e^{0.707}} \end{pmatrix} \approx \begin{pmatrix} 0.33 & 0.67 \end{pmatrix}$

$$\text{Attention Scores} \approx \begin{pmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{pmatrix}$$

---

**Step 4**：加权求和（乘以 $V$）

$$\text{Output} = \begin{pmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 0.67 \times 1 + 0.33 \times 3 & 0.67 \times 2 + 0.33 \times 4 \\ 0.33 \times 1 + 0.67 \times 3 & 0.33 \times 2 + 0.67 \times 4 \end{pmatrix}$$

$$= \begin{pmatrix} 0.67 + 0.99 & 1.34 + 1.32 \\ 0.33 + 2.01 & 0.66 + 2.68 \end{pmatrix} \approx \begin{pmatrix} 1.66 & 2.66 \\ 2.34 & 3.34 \end{pmatrix}$$

---

**解释**：
- 第 1 个输出：主要是第 1 个值（权重 0.67）和第 2 个值（权重 0.33）的加权平均
- 第 2 个输出：主要是第 2 个值（权重 0.67）和第 1 个值（权重 0.33）的加权平均

---

#### 四、自注意力（Self-Attention）

**自注意力**：$Q, K, V$ 都来自同一个输入序列。

**计算过程**：
1. 输入 $X \in \mathbb{R}^{n \times d}$（$n$ 个词，每个 $d$ 维）
2. 线性投影：$Q = XW_Q$，$K = XW_K$，$V = XW_V$
3. 注意力：$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

---

**自注意力的意义**：
- 每个词可以"注意"到序列中的所有其他词
- 注意力权重表示词与词之间的相关性
- 可以捕捉长距离依赖

---

**样题对应：按新版样题手算一遍自注意力**

设输入矩阵
$$X = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}, \quad W_Q = W_K = W_V = I_2$$

则
$$Q = XW_Q = X, \quad K = XW_K = X, \quad V = XW_V = X$$

先算分数矩阵：
$$QK^T = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$$

记
$$S = \frac{QK^T}{\sqrt{2}} = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \sqrt{2} \end{pmatrix}$$

按行做 softmax，得到权重矩阵
$$P = \text{softmax}(S) = \begin{pmatrix}
\frac{1}{2} & \frac{1}{2} \\
\frac{1}{1+e^{1/\sqrt{2}}} & \frac{e^{1/\sqrt{2}}}{1+e^{1/\sqrt{2}}}
\end{pmatrix}$$

因此输出矩阵
$$O = PV = \begin{pmatrix}
1 & \frac{1}{2} \\
1 & \frac{e^{1/\sqrt{2}}}{1+e^{1/\sqrt{2}}}
\end{pmatrix}$$

**长序列时的主要瓶颈**：
- 计算瓶颈来自 $QK^T$，复杂度约为 $O(n^2 d_k)$。
- 存储瓶颈来自注意力分数矩阵和 softmax 权重矩阵，空间复杂度约为 $O(n^2)$。

---

### 9.3 Transformer 模型

#### 一、多头注意力（Multi-Head Attention）

**思想**：多个注意力头并行工作，捕获不同的特征。

**公式**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中：
- $h$：头的数量（如 8 或 16）
- $W_i^Q \in \mathbb{R}^{d \times d_k}$，$W_i^K \in \mathbb{R}^{d \times d_k}$，$W_i^V \in \mathbb{R}^{d \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d}$：输出投影矩阵

读法：每个头先把输入投影到一套自己的 $Q,K,V$ 空间里，单独做一次注意力；多个头的结果拼接后，再乘 $W^O$ 融合回模型维度。

---

**为什么用多头？**

- 不同头可以关注不同的信息
  - 一些头关注语法关系
  - 一些头关注语义关系
  - 一些头关注长距离依赖
- 增强模型的表达能力

---

#### 二、位置编码（Positional Encoding）

**问题**：Transformer 没有递归和卷积，无法捕捉序列顺序。

**解决**：添加位置编码到输入嵌入。

**正弦位置编码**：
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中：
- $pos$：位置（0, 1, 2, ...）
- $i$：维度索引
- $d$：隐藏层维度

---

**例题 2**：$d = 4$，计算位置编码

**位置 $pos = 0$**：
- $PE_{(0, 0)} = \sin\left(\frac{0}{10000^0}\right) = \sin(0) = 0$
- $PE_{(0, 1)} = \cos\left(\frac{0}{10000^0}\right) = \cos(0) = 1$
- $PE_{(0, 2)} = \sin\left(\frac{0}{10000^{2/4}}\right) = \sin(0) = 0$
- $PE_{(0, 3)} = \cos\left(\frac{0}{10000^{2/4}}\right) = \cos(0) = 1$

$$PE_0 = [0, 1, 0, 1]$$

---

**位置 $pos = 1$**：
- $PE_{(1, 0)} = \sin\left(\frac{1}{10000^0}\right) = \sin(1) \approx 0.8415$
- $PE_{(1, 1)} = \cos\left(\frac{1}{10000^0}\right) = \cos(1) \approx 0.5403$
- $PE_{(1, 2)} = \sin\left(\frac{1}{10000^{1/2}}\right) = \sin(0.01) \approx 0.01$
- $PE_{(1, 3)} = \cos\left(\frac{1}{10000^{1/2}}\right) = \cos(0.01) \approx 0.99995$

$$PE_1 \approx [0.84, 0.54, 0.01, 1.0]$$

---

**位置编码的性质**：
1. 每个位置有唯一的编码
2. 相近位置的编码相近
3. 可以学习相对位置关系（因为 $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$）

---

#### 三、Encoder 层结构

**Encoder 层**包含：

1. **多头自注意力**
   - 输入：$X$
   - 输出：每个词的上下文表示

2. **残差连接 + LayerNorm**
   - $\text{LayerNorm}(X + \text{MultiHead}(X))$
   - 残差：帮助梯度流动
   - 归一化：稳定训练

3. **前馈网络 (FFN)**
   - $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
   - 通常是 ReLU 激活的两层 MLP

4. **残差连接 + LayerNorm**
   - $\text{LayerNorm}(x + \text{FFN}(x))$

---

**完整 Encoder 架构图**（简化）：

```
输入 X
  │
  ▼
┌─────────────────────┐
│ 多头自注意力        │
└─────────────────────┘
  │
  ▼
残差连接 + LayerNorm
  │
  ▼
┌─────────────────────┐
│ 前馈网络 (FFN)      │
└─────────────────────┘
  │
  ▼
残差连接 + LayerNorm
  │
  ▼
输出
```

---

### 9.4 Decoder-only 大模型参数量、计算量、内存量的计算与分析

#### 一、参数量计算

**Decoder-only Transformer 的参数组成**：

1. **词嵌入层**：$V \times d$
   - $V$：词表大小（如 50257）
   - $d$：隐藏层维度

2. **位置嵌入**：$L_{max} \times d$（可学习）或 0（固定正弦编码）

3. **每个 Transformer 层**：
   - 注意力：$4d^2$（Q、K、V、O 四个投影）
   - 前馈网络：$8d^2$（两层：$d \to 4d \to d$）
   - LayerNorm：$4d$（两个 LayerNorm，每个 $2d$）

4. **最终 LayerNorm**：$2d$

**总参数量**（$L$ 层，忽略 LayerNorm）：
$$P \approx V \times d + L \times (4d^2 + 8d^2) = V \times d + 12Ld^2$$

---

**具体例子**：GPT-2 Small 配置

| 参数 | 值 |
|------|-----|
| 词表大小 $V$ | 50257 |
| 隐藏层维度 $d$ | 768 |
| 层数 $L$ | 12 |
| 注意力头数 | 12 |

**计算**：
- 词嵌入：$50257 \times 768 \approx 38.6$M
- 每层注意力：$4 \times 768^2 = 4 \times 589824 \approx 2.36$M
- 每层 FFN：$8 \times 768^2 = 8 \times 589824 \approx 4.72$M
- 12 层总计：$12 \times (2.36 + 4.72) \approx 84.96$M
- **总参数量**：$38.6 + 84.96 \approx 123.56$M ≈ 124M

---

#### 二、FLOPs 计算

**矩阵乘法 FLOPs**：
$$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p} \Rightarrow AB \text{需要} 2mnp \text{次 FLOPs}$$

（$mnp$ 次乘法 + $mnp$ 次加法）

---

**Transformer 前向传播**：

每个 token 的 FLOPs ≈ $2 \times \text{参数量}$

**原因**：每个参数参与一次乘法（乘输入）和一次加法（累加）。

---

**训练总计算量**：

$$\text{FLOPs} \approx 6 \times \text{tokens} \times \text{params}$$

**解释**：
- 前向传播：$2 \times \text{tokens} \times \text{params}$
- 反向传播：$4 \times \text{tokens} \times \text{params}$（需要计算梯度）
- 总计：$6 \times \text{tokens} \times \text{params}$

---

**例题**：训练 300B tokens 的 124M 模型

$$\text{FLOPs} \approx 6 \times 3 \times 10^{11} \times 1.24 \times 10^8 \approx 2.23 \times 10^{20}$$

如果用 100 TFLOPS 的 GPU（每秒 $10^{14}$ FLOPs）：
$$\text{时间} \approx \frac{2.23 \times 10^{20}}{10^{14}} \approx 2.23 \times 10^6 \text{秒} \approx 26 \text{天}$$

---

#### 三、显存需求

**训练时显存组成**：

1. **模型参数**：$P \times 4$ 字节（FP32）或 $P \times 2$ 字节（FP16）

2. **梯度**：与参数量相同

3. **优化器状态**（Adam）：
   - 一阶矩 $m$：$P \times 4$ 字节
   - 二阶矩 $v$：$P \times 4$ 字节
   - 共 $8P$ 字节

4. **激活值**：与 batch size、序列长度、层数成正比
   - 粗略估计：$\text{batch} \times \text{seq\_len} \times d \times L \times 4$ 字节

---

**经验公式**（FP16 训练）：

$$\text{Memory} \approx 16P \text{ bytes}$$

包括：参数（2P）+ 梯度（2P）+ 优化器（8P）+ 激活值（~4P）

---

**例题**：124M 参数模型 FP16 训练

$$\text{Memory} \approx 16 \times 124 \times 10^6 \text{ bytes} \approx 1984 \text{ MB} \approx 2 \text{ GB}$$

实际需要考虑 batch size 和序列长度，可能需要 3-4 GB。

---

**推理显存**（仅加载模型）：

$$\text{Memory} \approx 2P \text{ bytes（FP16）或} 4P \text{ bytes（FP32）}$$

124M 参数 FP16 推理：约 250 MB

---

### 9.5 低秩微调方法 (LoRA)

#### 一、核心思想

**问题**：大模型参数量巨大，全量微调成本高。

**LoRA 思想**：冻结预训练权重，只训练低秩分解的增量矩阵。

---

**原始权重更新**：
$$W = W_0 + \Delta W$$

其中 $W_0$ 是预训练权重（冻结），$\Delta W$ 是待学习的增量。

---

**LoRA 参数化**：
$$\Delta W = BA$$

其中：
- $B \in \mathbb{R}^{d \times r}$（下行矩阵）
- $A \in \mathbb{R}^{r \times d}$（上行矩阵）
- $r \ll d$ 是低秩维度（如 8 或 16）

**维度说明**：有些论文定义 $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$，两种都可以，只要 $BA$ 的维度与 $W_0$ 匹配。

---

**前向传播**：
$$h = Wx = (W_0 + \alpha BA)x = W_0 x + \alpha BAx$$

其中 $\alpha$ 是缩放系数。

**$\alpha$ 的选择**：
- 原论文：$\alpha = \frac{d}{r}$（自动缩放）
- 有些实现：$\alpha$ 是超参数（如 1, 2, 4, ...）

---

#### 二、具体数值例子

**设置**：
- $W_0 \in \mathbb{R}^{4 \times 4}$，$r = 2$

$$W_0 = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{pmatrix}$$

---

**LoRA 矩阵**（随机初始化）：
$$A = \begin{pmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \end{pmatrix} \in \mathbb{R}^{2 \times 4}$$
$$B = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix} \in \mathbb{R}^{4 \times 2}$$

---

**计算 $\Delta W = BA$**：

$$\Delta W = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \end{pmatrix}$$

$$= \begin{pmatrix}
1 \times 0.1 + 0 \times 0.5 & 1 \times 0.2 + 0 \times 0.6 & 1 \times 0.3 + 0 \times 0.7 & 1 \times 0.4 + 0 \times 0.8 \\
0 \times 0.1 + 1 \times 0.5 & 0 \times 0.2 + 1 \times 0.6 & 0 \times 0.3 + 1 \times 0.7 & 0 \times 0.4 + 1 \times 0.8 \\
1 \times 0.1 + 1 \times 0.5 & 1 \times 0.2 + 1 \times 0.6 & 1 \times 0.3 + 1 \times 0.7 & 1 \times 0.4 + 1 \times 0.8 \\
0 \times 0.1 + 0 \times 0.5 & 0 \times 0.2 + 0 \times 0.6 & 0 \times 0.3 + 0 \times 0.7 & 0 \times 0.4 + 0 \times 0.8
\end{pmatrix}$$

$$= \begin{pmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.5 & 0.6 & 0.7 & 0.8 \\
0.6 & 0.8 & 1.0 & 1.2 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

---

**更新后的权重**（设 $\alpha = 1$）：
$$W = W_0 + \Delta W = \begin{pmatrix} 1.1 & 2.2 & 3.3 & 4.4 \\ 5.5 & 6.6 & 7.7 & 8.8 \\ 9.6 & 10.8 & 12.0 & 13.2 \\ 13 & 14 & 15 & 16 \end{pmatrix}$$

---

#### 三、参数量对比

| 微调方法 | 参数量 | 例子（$d=4096, r=8$） |
|----------|--------|----------------------|
| 全量微调 | $d^2$ | $4096^2 = 16,777,216$ |
| LoRA | $2dr$ | $2 \times 4096 \times 8 = 65,536$ |
| **减少倍数** | $\frac{d}{2r}$ | $\frac{4096}{16} = 256$ 倍 |

---

#### 四、样题对应：LoRA 两层网络的前反向传播

设
$$h = (W_1 + A_1B_1)x, \quad z = \sigma(h), \quad y = (W_2 + A_2B_2)z, \quad L = f(y)$$

记输出端梯度
$$g_y = \frac{\partial L}{\partial y} = \nabla_y f(y)$$

则有
$$g_h = \frac{\partial L}{\partial h} = \left(W_2 + A_2B_2\right)^T g_y \odot \sigma'(h)$$

于是四个 LoRA 矩阵的梯度可以直接写成：

$$\frac{\partial L}{\partial A_2} = g_y(B_2z)^T$$
$$\frac{\partial L}{\partial B_2} = (A_2^T g_y) z^T$$
$$\frac{\partial L}{\partial A_1} = g_h(B_1x)^T$$
$$\frac{\partial L}{\partial B_1} = (A_1^T g_h) x^T$$

**参数量对比**：
- 预训练阶段需要学习的参数量：$mn + pm$
- 使用 LoRA 微调时需要学习的参数量：$mr + rn + ps + sm = r(m+n) + s(p+m)$
- 当 $r \ll \min\{m,n\}$ 且 $s \ll \min\{p,m\}$ 时，LoRA 会显著减少可训练参数量

这正是新版样题里最容易直接命中的推导模板。

---

#### 五、LoRA 的优势

1. **参数高效**：只训练 0.1%~1% 的参数

2. **推理无开销**：训练完成后可以合并权重
   $$W' = W_0 + \alpha BA$$
   推理时直接使用 $W'$，无额外计算

3. **多任务共享**：多个任务可以共享 $W_0$，只需存储不同的 $(A, B)$

4. **即插即用**：可以应用到任何线性层

---

#### 六、LoRA 应用到 Transformer

**通常应用位置**：
- 注意力层的 $W_Q, W_K, W_V, W_O$
- 前馈网络层

**常见配置**：
- $r \in \{8, 16, 32, 64\}$
- $\alpha \in \{1, 2, 4, 8, 16\}$
- dropout: 0.1~0.5

---

## 复习建议

1. **掌握基础部分 (#)**：标注 `#` 的仍然是闭卷和开卷都绕不开的核心内容。
2. **闭卷优先级**：先确保线代基本计算、概率与随机变量、凸性判定、梯度 / Hessian、自注意力手算足够熟练。
3. **新版不必过度投入收敛性证明**：梯度下降、动量法、Adam 等更重要的是会写迭代格式、会解释每一项的作用。
4. **MLE / MAP 作为衔接阅读**：新版数学考纲里它们不再是最核心的概率考点，但仍适合开卷整理到索引里。
5. **大模型部分至少要会三类题**：`Q / K / V / S / P / O` 计算，`Decoder-only` 参数 / FLOPs / 显存估算，以及 `LoRA` 参数量与梯度。

---

*祝复习顺利！*
