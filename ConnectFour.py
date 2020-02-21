import numpy as np
import pygame

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board


    
def drop_piece(board,row,columnSelection,piece):
    board[row][columnSelection] = piece

def is_valid_location(board,columnSelection):
    return board[5][columnSelection] == 0

def get_next_open_row(board,columnSelection):
    for row in range(ROW_COUNT):
        if board[row][columnSelection] == 0:
            return row
            
def print_board(board):
    print(np.flip(board, 0))

def winning_move(board,piece):
    # Check horizontals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check verticals
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(3,ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def draw_board(board):
    pass

board = create_board()
print_board(board)
game_over = False
turn = 0

# Game Start
print("Welcome to Connect Four")
print("  0  1  2  3  4  5  6  ")
print(create_board())

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = ROW_COUNT+1 * SQUARESIZE

size = (width,height)

screen = pygame.display.set_mode(size)

# Main Loop
while not game_over:

    if turn == 0:
        columnSelection = int(input("Player 1 Make your selection (0-6) \n"))

        if is_valid_location(board,columnSelection):
            row = get_next_open_row(board,columnSelection)
            drop_piece(board,row,columnSelection,piece = 1)
            
            if winning_move(board,1):
                print("player 1 wins!!!")
                game_over = True
                break
                
        

    else:
        columnSelection = int(input("Player 2 Make your selection (0-6) \n"))
        if is_valid_location(board,columnSelection):
            row = get_next_open_row(board,columnSelection)
            drop_piece(board,row,columnSelection,piece = 2)
        
            if winning_move(board,2):
                print("player 2 wins!!!")
                game_over = True
                break

    print_board(board)  
    
    turn += 1
    turn = turn%2
    
