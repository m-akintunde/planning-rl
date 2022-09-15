NAMES = {90: 'initial state',
         73: 'grocery store',
         46: 'school',
         14: 'construction site',
         9:  'hospital'}
# OBJ = [int_to_pair(i) for i, j in enumerate(cm) if j == 3]

MILESTONES = [90, 73, 46, 14, 9]

BOARD_ROWS = 10
BOARD_COLS = 10

CM = [
     1, 1, 1, 3, 3, 3, 3, 1, 1, 1,
     1, 1, 1, 3, 1, 3, 3, 1, 1, 1,
     1, 1, 1, 3, 1, 3, 3, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 3, 3, 1, 3, 1, 1,
     1, 1, 1, 3, 3, 3, 3, 3, 1, 1,
     3, 3, 3, 1, 1, 1, 1, 1, 1, 1,
     3, 3, 3, 1, 1, 1, 1, 1, 1, 1,
     3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]


# Converts a pair to an integer.
def pair_to_int(i, j):
    return i * 10 + j


# Convert an int to a coordinate pair.
def int_to_pair(i):
    return (i // 10, i % 10)