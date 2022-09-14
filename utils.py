# Converts a pair to an integer.
def pair_to_int(i, j):
    return i * 10 + j

# Convert an int to a coordinate pair.
def int_to_pair(i):
    return (i // 10, i % 10)
