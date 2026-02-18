import cv2
import numpy as np

# Inner corners
rows = 6
cols = 9

# Square size in mm for printing reference
square_size_mm = 25

# Squares count (inner corners + 1)
board_rows = rows + 1
board_cols = cols + 1

square_px = 100  # resolution per square

board = np.zeros((board_rows * square_px,
                  board_cols * square_px), dtype=np.uint8)

for i in range(board_rows):
    for j in range(board_cols):
        if (i + j) % 2 == 0:
            board[i*square_px:(i+1)*square_px,
                  j*square_px:(j+1)*square_px] = 255

cv2.imwrite("chessboard_9x6_25mm.png", board)
