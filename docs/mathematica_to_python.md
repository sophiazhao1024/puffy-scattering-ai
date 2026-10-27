# Mathematica → Python：功能迁移说明

本项目是独立重建的物理参考与 AI 实验，不是自动翻译器，也没有执行原 `new(1).nb`。原 notebook 的位置索引见 notebook_index.csv；用 tools/inspect_notebook.py 可在本地重新提取输入单元。

| Mathematica 工作环节 | Python 中对应思路 | 实际边界 |
|---|---|---|
| 设定 α、μ、mφ、v、R 等参数 | `dimensional()` 显式换算 b、κ、y | 未导入任何原全局变量 |
| 分段均匀球有限尺寸势 | `overlap()` + `ball_u()` | 独立卷积算法；不复制长解析表达式 |
| `NDSolve` / 径向方程 | `solve_ivp(method='DOP853')` | 使用标准径向方程与新的正则化变量 |
| 贝塞尔函数与相移匹配 | `spherical_jn`、`spherical_yn` 与交叉匹配 | 区分球贝塞尔与普通贝塞尔 |
| `Table`、循环求分波 | `range(lmax+2)`、NumPy相位差 | 需要额外 δ_{lmax+1} 才能求最后一项 |
| 扫描参数与输出列表 | CSV，参数每行显式保存 | 本次只扫一个固定切片 |
| 作图和数据分析 | CSV可导入Matplotlib/Excel | 未声称论文图5–9被重绘 |
| 无对应原模块 | GP、主动选点、留出测试 | 完全新增研究原型 |

## 真正对齐原 notebook 所需步骤

1. 在具有匹配版本和依赖的 Wolfram 环境中，从空内核按原输入顺序执行，不只查看缓存输出。
2. 导出参数表：质量、半径、单位、耦合、密度归一化、势正负号、起点、匹配半径、每一阶相移与截面定义。
3. 消除符号冲突和隐藏全局状态，确认普通 Bessel 与 SphericalBessel 的使用约定。
4. 在外区、中心、两球接触和包容边界附近比较势；不等半径的几何边界应核对 |Rχ−Rn| 与 Rχ+Rn。
5. 用同一组物理参数分别计算相移，使用 sin(δPython−δMathematica) 而非直接减角度，避免模π分支误报。
6. 固定传递截面与总截面的定义，核对 k=μv、面积单位和核相干增强，再比较绝对截面。

建议导出CSV列：`b,kappa,ychi,yn,l,delta,potential_sign,normalization,method,rtol,match_radius`；原始参数另存JSON。程序目前没有自动导入 notebook 的接口。解析 notebook 文本不等于执行 Mathematica。

## 论文印刷式的审计标记

论文式 (2.17) 打印的能量/势组合及角动量项需要核实；式 (2.21) 的系数与一般约化质量约定也需核对。本项目采用标准式 u''(t)+[κ²+bU−l(l+1)/t²]u=0，并在 README 中明示。使用一般不等半径的精确球体重叠几何；该选择需要与论文式 (2.6)、附录 A 及 notebook 中任何使用 2Rχ 的分段单独对照，不能把差异默认为程序等价。
