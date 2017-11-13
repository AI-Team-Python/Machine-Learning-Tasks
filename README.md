# machine-learning-tasks

Machine Learning Tasks. 该仓库将逐步添加各种常用机器学习基础算法及例子

【请熟悉 markdown 语法】


## 更新记录

1. 2017.10.25-【BinKes】

    初始化框架、约定文档书写规范、变量命名规范

2. xxx

    ccccc

## 创建python环境

例如： 创建一个版本是 3.6 的 python 环境，同时安装该环境下需要用到的工具 ipython 和 tensorflow。

  `conda create -n ml python=3.6 ipython tensorflow`

  >> 注意：使用conda命令需要先安装 miniconda3（安装方法请网上搜）

## python项目结构（规范）

### 维护文档、文字表达规范

  - 文档一例用 markdown 书写，每个项目都需要有相应的 README.md 文档进行说明，包括项目的标题、功能、使用方法、算法详情、接口规范。算法比较复杂的可以另外加 .md 文件加以说明。
  - 使用 `#` 建立文档层级关系，一级内容与二级内容间换两行，其余换一行
  - 所有描述性语言汉语与数字、英文之间加空格

### 命名规范

  - 仓库名称： 用简短的项目功能相关单词加 `-` 组成，小写
  - 类名使用驼峰式命名规则，如："MyClass"
  - 全局常量用大写字符串表示，如：PARAM
  - 普通变量、函数名称、对象名称等使用下划线命名法，如：my_object

文档结果举例：
 
```
|-- machine-learning-tasks/         // 仓库名称
    |-- .gitignore                  // github 提交时候忽视的文件
    |-- README.md                   // 项目说明文档
    |-- machine_learning_tasks      // 项目名称
        |-- k_nn/                   // k 近邻
            |-- __init__.py 
            |-- k_nn.py             // k 近邻算法
            |-- datas/              // 存放数据的文件夹
            |-- k_nn.md             // knn算法相关说明
    |-- requrements.txt             // 项目依赖的包及其版本号
```
### 算法概要


### 接口说明


### 使用方法




## install

pip install keras -i https://pypi.tuna.tsinghua.edu.cn