

# LLM-Project1

模型：chatgpt4o

Code availability：https://github.com/grassdream/LLM-Project1

## ARC任务-3ee1011a.json

抽象与推理语料库（**A**bstraction and **R**easoning **C**orpus，简称ARC）是由Keras神经网络库的创建者François Chollet提出的一个基准测试，旨在评估人工智能系统在未知任务中学习新技能的能力。ARC由一系列任务组成，每个任务包含3到5对训练输入和输出示例，以及一个测试输入，要求系统根据训练示例中学习的模式预测相应的输出。

我选择的是3ee1011a.json

下面是问题的可视化，可以看到，这个题就是把输入的线条变成正方形的边，外围正方形的宽度为1，内部的则自适应。

![](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119154121050.png)

接下来问chatgpt4o。

> https://chatgpt.com/share/673c3f33-5020-8012-8d7f-2e787adbdde4

### Prompt

*请你读取这个json文件，这是一个题目，你需要写一个python程序从输入推断出输出，题目的原理是把输入的一个个线条按照长度变成正方形的边长，谢谢你！ 图片里面是arc任务的可视化页面；图形形状以json的形式实现二维矩阵+颜色显示 任务目标： 使得模型可以接受输入的json并输出正确的json 生成的输出应该是环形嵌套结构，但每条线条对应的“正方形”宽度为 1，数字仅表示颜色。 解决方案逻辑： 按长度排序：线条按长度从大到小排序，决定环形嵌套顺序。 环形嵌套：每条线条的颜色表示对应环的位置，且环的宽度固定为 1。 外层环的边宽固定为1，最内层应填充一个完整的正方形，其大小取决于剩余区域 正方形的边长和线条的长度相同 外层到内层绘制：从外到内逐层构建环形结构。 注意：请你根据train的output和你的输出比较，必须保持完成一致 此外你要对输入和输出都可视化出来，并保证颜色字典一致*

![image-20241119154441243](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119154441243.png)

![](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119154521870.png)

经过gpt自己的几轮否定之否定，终于成功了，我认为prompt的关键点在于告诉gpt该如何检查自己做的对不对。也就是我的这句话起了作用：“注意：请你根据train的output和你的输出比较，必须保持完成一致”，即使有错别字也没事。

接着让gpt生成一下注释：

*请你提供完整代码（加上可视化标准的输出的代码，还有测试的例子）图的标题用英文，并且加上友好的中文注释和代码思路*

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载
# 加载任务数据文件
file_path = '/mnt/data/3ee1011a.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# 2. 可视化函数
def parse_and_draw(input_grid, output_grid, expected_output=None, title=""):
    """
    输入和输出矩阵的可视化，支持可选的标准输出对比
    """
    fig, axes = plt.subplots(1, 3 if expected_output is not None else 2, figsize=(18, 6))
    axes[0].imshow(input_grid, cmap="tab20", vmin=0, vmax=20)
    axes[0].set_title(f"{title} - Input")
    axes[1].imshow(output_grid, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title(f"{title} - Generated Output")
    if expected_output is not None:
        axes[2].imshow(expected_output, cmap="tab20", vmin=0, vmax=20)
        axes[2].set_title(f"{title} - Expected Output")
    plt.show()

# 3. 线条提取
def find_lines_detailed(grid):
    """
    从矩阵中提取所有线条（水平和垂直），记录线条颜色和长度
    """
    lines = []
    rows, cols = grid.shape

    # 提取水平线条
    for r in range(rows):
        start, length = None, 0
        for c in range(cols):
            if grid[r][c] != 0:
                if start is None:
                    start = c
                    length = 1
                else:
                    length += 1
            elif start is not None:
                lines.append((grid[r][start], length))
                start, length = None, 0
        if start is not None:
            lines.append((grid[r][start], length))

    # 提取垂直线条
    for c in range(cols):
        start, length = None, 0
        for r in range(rows):
            if grid[r][c] != 0:
                if start is None:
                    start = r
                    length = 1
                else:
                    length += 1
            elif start is not None:
                lines.append((grid[start][c], length))
                start, length = None, 0
        if start is not None:
            lines.append((grid[start][c], length))

    return lines

# 4. 环形嵌套结构生成
def build_ring_structure_precise(lines, target_size):
    """
    根据提取的线条生成环形嵌套矩阵，确保符合目标尺寸
    """
    if not lines:
        return np.zeros(target_size, dtype=int)  # 如果没有线条，返回全零矩阵

    lines.sort(key=lambda x: -x[1])  # 按长度从大到小排序
    n = target_size[0]  # 使用目标矩阵的大小
    grid = np.zeros((n, n), dtype=int)

    # 逐层构建环形结构
    for idx, (color, length) in enumerate(lines):
        start = idx
        end = n - idx - 1
        for i in range(start, end + 1):
            grid[start][i] = color
            grid[end][i] = color
            grid[i][start] = color
            grid[i][end] = color

    return grid

# 5. 验证和测试
results_precise = []
for i, item in enumerate(data["train"]):
    input_grid = np.array(item["input"])
    expected_output = np.array(item["output"])
    target_size = expected_output.shape

    # 提取线条并生成输出
    lines = find_lines_detailed(input_grid)
    generated_output_precise = build_ring_structure_precise(lines, target_size)

    # 保存验证结果
    results_precise.append((i, np.array_equal(generated_output_precise, expected_output), generated_output_precise, expected_output))

    # 可视化比较
    parse_and_draw(input_grid, generated_output_precise, expected_output, f"Train Example {i} - Precise")

# 总结验证结果
all_correct_precise = all(result[1] for result in results_precise)
print(f"All train examples correct: {all_correct_precise}")

# 测试数据处理
for i, item in enumerate(data["test"]):
    input_grid = np.array(item["input"])
    target_size = (7, 7)  # 假设目标尺寸为7x7，可以根据需要动态调整

    # 提取线条并生成输出
    lines = find_lines_detailed(input_grid)
    generated_output_test = build_ring_structure_precise(lines, target_size)

    # 可视化测试数据输出
    parse_and_draw(input_grid, generated_output_test, title=f"Test Example {i}")
```

本地运行代码的结果，做对了：
![](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119154937380.png)

### 心得

4o可以实现大模型自己运行代码，并且能够自己纠错，这就省去了我把它生成的代码复制运行，然后把bug再给回去的过程，非常好！

## Crypto-Math（字母对应数字）

我是自己找的题目，觉得比较有意思。简单来说就是有个由字母组成的等式，每个字母代表一个数字，我们要通过这个等式把字母和数字的对应关系找到。

![](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119155500053.png)

> https://www.braingle.com/puzzlepedia/6-672/crypto-math.html

### Prompt

https://chatgpt.com/c/673c45cd-5a3c-8012-9850-d47f0147ca0b

*请根据以下描述为我编写一段 Python 代码，用于解决字母算术问题。字母算术问题涉及将一个等式中的字母映射为唯一的数字，使等式成立。以下是具体需求： ### 输入描述： 1. 输入是一个包含字母的等式字符串，例如：   - `BASE + BALL = GAMES`   - `MOSES + MEETS = SALOME` 2. 等式左侧可以包含多个加法项，中间用 `+` 连接，等式右侧为结果，用 `=` 分隔。 ### 功能要求： 1. 提取等式中的所有唯一字母，并为每个字母分配唯一的数字（从 0 到 9）。 2. 如果字母的数量超过 10（因为数字范围是 0-9），直接返回提示信息，例如“字母数量超过限制，无法求解”。 3. 确保映射的过程中，任何单词的首字母不能映射为数字 0，因为整数不允许以 0 开头。 4. 对每个可能的字母到数字映射，替换等式中的字母为对应数字，并检查等式是否成立。 5. 如果找到解，返回以下内容：   - 字母到数字的映射。   - 替换字母后的完整数字等式。 6. 如果找不到解，返回提示信息“无解”。 ### 输出要求： 1. 输出的结果应包含字母到数字的映射，例如：*   

```
python
   {'B': 3, 'A': 5, 'S': 9, 'E': 0, 'L': 2, 'G': 1, 'M': 8}
```

2. *同时，输出将字母替换为数字后的等式，例如：*   

```
python
   "3059 + 3522 = 15890"
```

3. *如果没有找到解，输出类似：*   

```
python
   "No solution found for 'BASE + BALL = GAMES'"
```

*\### 代码实现的具体逻辑： 1. 提取等式中的字母并构建一个字母集合。 2. 遍历所有可能的数字排列（用 `itertools.permutations`）。 3. 检查当前映射是否满足以下条件：   - 每个字母对应一个唯一的数字。   - 单词的首字母不能映射为 0。   - 替换后等式成立。 4. 如果找到符合条件的映射，返回结果；否则继续检查。 5. 如果所有映射都不满足条件，返回“无解”。 ### 示例验证： 代码需附带以下测试用例验证： 1. 输入 `BASE + BALL = GAMES`，应输出解决方案。 2. 输入 `MOSES + MEETS = SALOME`，应输出解决方案。 3. 验证极端情况：如果字母总数超过 10，例如 `ABCDEFGHIJKLMNOPQRSTUVWXYZ = ZYXWVUTSRQPONMLKJIHGFEDCBA`，需直接返回提示“字母数量超过限制，无法求解”。 4. 验证没有解的情况，例如 `AAA + BBB = CCC`，输出“无解”。 请根据以上详细描述编写代码，并附带完整的测试用例和输出示例。*

![](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119161244797.png)

成功了，再让gpt4o写个注释，并用dataframe呈现。

*帮我加入详细的中文注释，并用df展示结果*

```python
def solve_alphametics_with_comments(equation):
    """
    解决字母算术问题，并返回字母到数字的映射和替换后的等式。
    
    参数:
    - equation: 字母算术等式的字符串，例如 "BASE + BALL = GAMES"
    
    返回:
    - 如果有解，返回包含字母到数字映射和替换后的等式的字典
    - 如果无解或格式错误，返回对应的错误信息
    """
    try:
        # 分割等式为左边和右边两部分
        left, right = equation.replace(" ", "").split("=")
        # 左边的多项通过 "+" 分割
        words = left.split("+") + [right]
    except ValueError:
        # 如果格式错误，返回提示信息
        return f"无效的等式格式: {equation}"
    
    # 提取等式中所有唯一字母
    letters = set("".join(words))
    if len(letters) > 10:
        # 如果字母数量超过 10，直接返回无法求解的信息
        return "字母数量超过限制（超过10个），无法求解。"

    # 获取每个单词的首字母（这些不能映射为 0）
    first_letters = {word[0] for word in words}

    # 使用 itertools.permutations 生成所有可能的数字排列
    for perm in itertools.permutations(range(10), len(letters)):
        # 构建字母到数字的映射
        letter_to_digit = dict(zip(letters, perm))
        
        # 确保首字母不映射为 0
        if any(letter_to_digit[fl] == 0 for fl in first_letters):
            continue

        # 将单词中的字母替换为对应数字，并转为整数
        def word_to_number(word):
            return int("".join(str(letter_to_digit[ch]) for ch in word))

        # 计算左边的值和右边的值
        left_values = [word_to_number(word) for word in left.split("+")]
        right_value = word_to_number(right)

        # 检查等式是否成立
        if sum(left_values) == right_value:
            # 如果成立，返回结果
            substituted_equation = f"{' + '.join(map(str, left_values))} = {right_value}"
            return {
                "mapping": letter_to_digit,
                "equation": substituted_equation,
            }

    # 如果所有排列都不满足条件，返回无解信息
    return f"未找到 {equation} 的解"


# 测试用例
test_cases = [
    "BASE + BALL = GAMES",  # 有解
    "MOSES + MEETS = SALOME",  # 有解
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ = ZYXWVUTSRQPONMLKJIHGFEDCBA",  # 超过字母限制
    "AAA + BBB = CCC"  # 无解
]

# 格式化测试结果以便展示
formatted_results = []
for test in test_cases:
    result = solve_alphametics_with_comments(test)
    if isinstance(result, dict):
        # 有解时保存映射和替换后的等式
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": result["mapping"],
            "替换后的等式": result["equation"]
        })
    else:
        # 无解时保存错误信息
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": "N/A",
            "替换后的等式": result
        })

# 转为 DataFrame 方便展示
results_df = pd.DataFrame(formatted_results)

# 显示结果
tools.display_dataframe_to_user(name="字母算术测试结果", dataframe=results_df)
```

本地运行一下，并且再加几个例子，看看能不能运行出来。

```python
# Test cases
test_cases = [
    'FIFTY + STATES = AMERICA',
'SQUARE +  DANCE = DANCER',
'ELF + ELF = FOOL',
'CROSS + ROADS = DANGER',
'GREEN + ORANGE = COLORS'
]
```

和答案比对了一下，都算出来了。

![image-20241119161441023](https://hero-bin-1316238890.cos.ap-shanghai.myqcloud.com/image-20241119161441023.png)

### 心得

把任务明确了，模型就能给出能跑的代码！

## Code availability

https://github.com/grassdream/LLM-Project1

```python
# %% [markdown]
# ## ARC任务-3ee1011a.json

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载
# 加载任务数据文件
file_path = '3ee1011a.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# 2. 可视化函数
def parse_and_draw(input_grid, output_grid, expected_output=None, title=""):
    """
    输入和输出矩阵的可视化，支持可选的标准输出对比
    """
    fig, axes = plt.subplots(1, 3 if expected_output is not None else 2, figsize=(18, 6))
    axes[0].imshow(input_grid, cmap="tab20", vmin=0, vmax=20)
    axes[0].set_title(f"{title} - Input")
    axes[1].imshow(output_grid, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title(f"{title} - Generated Output")
    if expected_output is not None:
        axes[2].imshow(expected_output, cmap="tab20", vmin=0, vmax=20)
        axes[2].set_title(f"{title} - Expected Output")
    plt.show()

# 3. 线条提取
def find_lines_detailed(grid):
    """
    从矩阵中提取所有线条（水平和垂直），记录线条颜色和长度
    """
    lines = []
    rows, cols = grid.shape

    # 提取水平线条
    for r in range(rows):
        start, length = None, 0
        for c in range(cols):
            if grid[r][c] != 0:
                if start is None:
                    start = c
                    length = 1
                else:
                    length += 1
            elif start is not None:
                lines.append((grid[r][start], length))
                start, length = None, 0
        if start is not None:
            lines.append((grid[r][start], length))

    # 提取垂直线条
    for c in range(cols):
        start, length = None, 0
        for r in range(rows):
            if grid[r][c] != 0:
                if start is None:
                    start = r
                    length = 1
                else:
                    length += 1
            elif start is not None:
                lines.append((grid[start][c], length))
                start, length = None, 0
        if start is not None:
            lines.append((grid[start][c], length))

    return lines

# 4. 环形嵌套结构生成
def build_ring_structure_precise(lines, target_size):
    """
    根据提取的线条生成环形嵌套矩阵，确保符合目标尺寸
    """
    if not lines:
        return np.zeros(target_size, dtype=int)  # 如果没有线条，返回全零矩阵

    lines.sort(key=lambda x: -x[1])  # 按长度从大到小排序
    n = target_size[0]  # 使用目标矩阵的大小
    grid = np.zeros((n, n), dtype=int)

    # 逐层构建环形结构
    for idx, (color, length) in enumerate(lines):
        start = idx
        end = n - idx - 1
        for i in range(start, end + 1):
            grid[start][i] = color
            grid[end][i] = color
            grid[i][start] = color
            grid[i][end] = color

    return grid

# 5. 验证和测试
results_precise = []
for i, item in enumerate(data["train"]):
    input_grid = np.array(item["input"])
    expected_output = np.array(item["output"])
    target_size = expected_output.shape

    # 提取线条并生成输出
    lines = find_lines_detailed(input_grid)
    generated_output_precise = build_ring_structure_precise(lines, target_size)

    # 保存验证结果
    results_precise.append((i, np.array_equal(generated_output_precise, expected_output), generated_output_precise, expected_output))

    # 可视化比较
    parse_and_draw(input_grid, generated_output_precise, expected_output, f"Train Example {i} - Precise")

# 总结验证结果
all_correct_precise = all(result[1] for result in results_precise)
print(f"All train examples correct: {all_correct_precise}")

# 测试数据处理
for i, item in enumerate(data["test"]):
    input_grid = np.array(item["input"])
    target_size = (7, 7)  # 假设目标尺寸为7x7，可以根据需要动态调整

    # 提取线条并生成输出
    lines = find_lines_detailed(input_grid)
    generated_output_test = build_ring_structure_precise(lines, target_size)

    # 可视化测试数据输出
    parse_and_draw(input_grid, generated_output_test, title=f"Test Example {i}")


# %% [markdown]
# ## 字母算算术问题

# %%
import pandas as pd
import itertools
def solve_alphametics_with_comments(equation):
    """
    解决字母算术问题，并返回字母到数字的映射和替换后的等式。
    
    参数:
    - equation: 字母算术等式的字符串，例如 "BASE + BALL = GAMES"
    
    返回:
    - 如果有解，返回包含字母到数字映射和替换后的等式的字典
    - 如果无解或格式错误，返回对应的错误信息
    """
    try:
        # 分割等式为左边和右边两部分
        left, right = equation.replace(" ", "").split("=")
        # 左边的多项通过 "+" 分割
        words = left.split("+") + [right]
    except ValueError:
        # 如果格式错误，返回提示信息
        return f"无效的等式格式: {equation}"
    
    # 提取等式中所有唯一字母
    letters = set("".join(words))
    if len(letters) > 10:
        # 如果字母数量超过 10，直接返回无法求解的信息
        return "字母数量超过限制（超过10个），无法求解。"

    # 获取每个单词的首字母（这些不能映射为 0）
    first_letters = {word[0] for word in words}

    # 使用 itertools.permutations 生成所有可能的数字排列
    for perm in itertools.permutations(range(10), len(letters)):
        # 构建字母到数字的映射
        letter_to_digit = dict(zip(letters, perm))
        
        # 确保首字母不映射为 0
        if any(letter_to_digit[fl] == 0 for fl in first_letters):
            continue

        # 将单词中的字母替换为对应数字，并转为整数
        def word_to_number(word):
            return int("".join(str(letter_to_digit[ch]) for ch in word))

        # 计算左边的值和右边的值
        left_values = [word_to_number(word) for word in left.split("+")]
        right_value = word_to_number(right)

        # 检查等式是否成立
        if sum(left_values) == right_value:
            # 如果成立，返回结果
            substituted_equation = f"{' + '.join(map(str, left_values))} = {right_value}"
            return {
                "mapping": letter_to_digit,
                "equation": substituted_equation,
            }

    # 如果所有排列都不满足条件，返回无解信息
    return f"未找到 {equation} 的解"


# 测试用例
test_cases = [
    "BASE + BALL = GAMES",  # 有解
    "MOSES + MEETS = SALOME",  # 有解
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ = ZYXWVUTSRQPONMLKJIHGFEDCBA",  # 超过字母限制
    "AAA + BBB = CCC"  # 无解
]

# 格式化测试结果以便展示
formatted_results = []
for test in test_cases:
    result = solve_alphametics_with_comments(test)
    if isinstance(result, dict):
        # 有解时保存映射和替换后的等式
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": result["mapping"],
            "替换后的等式": result["equation"]
        })
    else:
        # 无解时保存错误信息
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": "N/A",
            "替换后的等式": result
        })

# 转为 DataFrame 方便展示
results_df = pd.DataFrame(formatted_results)
results_df

# %%
# Test cases
test_cases = [
    'FIFTY + STATES = AMERICA',
'SQUARE +  DANCE = DANCER',
'ELF + ELF = FOOL',
'CROSS + ROADS = DANGER',
'GREEN + ORANGE = COLORS'
]
# 格式化测试结果以便展示
formatted_results = []
for test in test_cases:
    result = solve_alphametics_with_comments(test)
    if isinstance(result, dict):
        # 有解时保存映射和替换后的等式
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": result["mapping"],
            "替换后的等式": result["equation"]
        })
    else:
        # 无解时保存错误信息
        formatted_results.append({
            "等式": test,
            "字母到数字的映射": "N/A",
            "替换后的等式": result
        })

# 转为 DataFrame 方便展示
results_df = pd.DataFrame(formatted_results)
results_df
```

