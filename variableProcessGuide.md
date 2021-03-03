在处理变量时参考如下表格进行：

|变量类型|变量之间是否有序|变量数目|处理方法|
|---|---|---|---|
|类别变量|有|少&多|映射为0,1,2..|
|类别变量|无|少|OneHot|
|类别变量|无|多|聚类后OneHot<br>WOE编码<br>Embedding<br>Hash|
|数值变量|-|少|保持不变<br>OneHot<br>数值变换,e.g,函数变换，标准化，归一化|
|数值变量|-|多|保持不变<br>数值变换,e.g.,函数变换，标准化，归一化<br>分箱后OneHot,方法有：等宽，等深，卡方（针对有监督学习）|

具体代码案例参考：
类别变量参考：`categoricalColumnProcess.ipynb`
数值变量参考：`numericalColumnProcess.ipynb`
