import numpy as np


def chess_board(tr, tc, dr, dc, size):
    global tile
    if size == 1:
        return
    s = size // 2
    t = tile
    tile += 1

    if dr < tr + s and dc < tc + s:
        chess_board(tr, tc, dr, dc, s)
    else:
        board[tr + s - 1][tc + s - 1] = t
        chess_board(tr, tc, tr + s - 1, tc + s - 1, s)

    if dr < tr + s and dc >= tc + s:
        chess_board(tr, tc + s, dr, dc, s)
    else:
        board[tr + s - 1][tc + s] = t
        chess_board(tr, tc + s, tr + s - 1, tc + s, s)

    if dr >= tr + s and dc < tc + s:
        chess_board(tr + s, tc, dr, dc, s)
    else:
        board[tr + s][tc + s - 1] = t
        chess_board(tr + s, tc, tr + s, tc + s - 1, s)

    if dr >= tr + s and dc >= tc + s:
        chess_board(tr + s, tc + s, dr, dc, s)
    else:
        board[tr + s][tc + s] = t
        chess_board(tr + s, tc + s, tr + s, tc + s, s)


board_size = 8
board = [[0 for _ in range(board_size)] for _ in range(board_size)]
tile = 1
chess_board(0, 0, 3, 3, board_size)

# 打印棋盘
for row in board:
    print(row)
