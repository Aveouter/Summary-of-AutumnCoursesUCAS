def calculate(a, b, operator):
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        if b != 0 and isinstance(a / b, int):
            return a / b
        else:
            return None


def init_backtrack(numbers):
    new_numbers = []
    expressions = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            remaining_numbers = [numbers[k] for k in range(len(numbers)) if k != i and k != j]

            for operator in ['+', '-', '*', '/']:
                if operator == '/' and numbers[j] == 0:
                    continue

                new_number = calculate(numbers[i], numbers[j], operator)
                if new_number is not None:
                    new_numbers.append([new_number] + remaining_numbers)
                    expressions.append(f"({numbers[i]} {operator} {numbers[j]})")
    return new_numbers, expressions


def finall_backtrack(new_numbers_2, expressions_2, target):
    new_numbers_3 = []
    expressions_3 = []
    for i, value in enumerate(new_numbers_2):
        for j in range(1, len(value)):
            remaining_numbers = [value[k] for k in range(1, len(value)) if k != j]
            for operator in ['+', '-', '*', '/']:
                if operator == '/' and value[j] == 0:
                    continue

                new_number = calculate(value[0], value[j], operator)
                if new_number is not None:
                    if new_number == target and remaining_numbers == []:
                        print([expressions_2[i], f"({value[0]} {operator} {value[j]})"])
                        singal  = 1
                    new_numbers_3.append([new_number] + remaining_numbers)
                    expressions_3.append([expressions_2[i], f"({value[0]} {operator} {value[j]})"])
    return singal

def backtrack2(numbers, target=24):
    new_numbers_2 = []
    numbers, expressions_1 = init_backtrack(numbers)
    expressions_2 = []
    for i, value in enumerate(numbers):
        for j in range(1, len(value)):
            remaining_numbers = [value[k] for k in range(1, len(value)) if k != j]
            for operator in ['+', '-', '*', '/']:
                if operator == '/' and value[j] == 0:
                    continue

                new_number = calculate(value[0], value[j], operator)
                if new_number is not None:
                    new_numbers_2.append([new_number] + remaining_numbers)
                    expressions_2.append([expressions_1[i], f"({value[0]} {operator} {value[j]})"])
    x = finall_backtrack(new_numbers_2, expressions_2, target)

    return x


def solve_24_point(numbers):
    target = 24
    expressions = []
    x = backtrack2(numbers, target=24)

    if not x:
        return "未找到解。"


# 示例用法：
numbers = [1, 2, 3, 7]
solution = solve_24_point(numbers)

