
import tiktoken
import random

def calc_token_num(s: str):
    enc = tiktoken.get_encoding('cl100k_base')
    return len(enc.encode(s))


def gen_base_content(filename: str, token_limit: int):
    data = open(filename, 'r', encoding='utf8').readlines()

    total_token = 0
    result = []
    for line in data:
        current_line_token = calc_token_num(line)
        if total_token + current_line_token > token_limit:
            break
        total_token += current_line_token
        result.append(line)
    return result


def gen_test_case_v1(base_content: list):
    test_chunks = [
        "Defined variable var_num_1=1200. ",
        "Defined variable var_num_2=var_num_1 + 100. ",
        "Defined variable var_num_3=var_num_2 + 50. ",
    ]

    test_questions = """

---------
Based on the content of the documents above, calculate the value of the variable var_num_3 step by step.
"""

    inserted_test_datas = [
        f"""

'''
{line}
'''

""" for line in test_chunks]

    insert_pos = []
    for idx in range(len(inserted_test_datas)):
        block_size = (len(base_content) - 1) // len(inserted_test_datas)
        insert_pos.append(random.randint(block_size * idx, block_size * (idx + 1)))

    print(f'total_lines: {len(base_content)}')
    print(f'test_insert_pos: {insert_pos}, {[p/len(base_content) for p in insert_pos]}')

    output_lines = base_content[::]
    for idx, pos in reversed(list(enumerate(insert_pos))):
        output_lines.insert(pos, inserted_test_datas[idx])

    output_lines.append(test_questions)

    return ''.join(output_lines)


def gen_test_case_v2(base_content: list):
    test_chunks = [
        "Defined var_earth = 1200. ",
        "Defined var_mars = var_earth + 100. ",
        "Defined var_jupiter = var_mars + 50. ",
    ]

    test_questions = """

---------
Based on the content of the documents above, calculate the value of the var_jupiter step by step.
"""

    inserted_test_datas = [
        f"""

{line}

""" for line in test_chunks]

    insert_pos = []
    for idx in range(len(inserted_test_datas)):
        block_size = (len(base_content) - 1) // len(inserted_test_datas)
        insert_pos.append(random.randint(block_size * idx, block_size * (idx + 1)))

    print(f'total_lines: {len(base_content)}')
    print(f'test_insert_pos: {insert_pos}, {[p/len(base_content) for p in insert_pos]}')

    output_lines = base_content[::]
    for idx, pos in reversed(list(enumerate(insert_pos))):
        output_lines.insert(pos, inserted_test_datas[idx])

    output_lines.append(test_questions)

    return ''.join(output_lines)


base_content = gen_base_content('HarryPotter3.txt', 1024 * 100)
output = gen_test_case_v2(base_content)
open('test_case3v2_128k_sample3.txt', 'w', encoding='utf8').write(output)
