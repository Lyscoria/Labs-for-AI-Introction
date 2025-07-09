import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
你好呀！这是一个吃豆人游戏，我需要你的帮助！吃豆人在地图中移动，只能在空地中移动，不能碰到墙壁，不能碰到鬼。
特别地，鬼是不会移动的，所以你不需要太提防它！你只需要不主动去碰墙和鬼就好！
注意：你需要在尽可能少的步数内吃完整个地图的所有豆子，总共的步数最好不要超过50步！
注意：游戏并没有路径不能重复的要求，你不应该去考虑“走没走过某些格子”或者“探索新区域”这些东西！
特别强调：你唯一的考虑点就是避开鬼和墙壁尽快吃掉所有的豆子。
强调：为此，你需要分析全局直接设计出步数最少的总路线，而不是仅仅着眼于当前这一步；切记你一定要考虑所有的豆子，考虑全局！

在接下来的对话中，我每次会告诉你当前的更新后的游戏状况。
每一次，你都要分析新的局面；不要不阅读新的局面，根据旧的局面做决定，有的豆子已经被吃过了！
根据你设计的最佳路径，告诉我吃豆人下一步应该怎么走，请弄清楚上下左右的关系，你只能在“可用方向”中选择！
请牢记我说的这些话，十分感谢你的帮助！
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content

    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info)
    choice = numbers[-1]
    return int(choice), info
