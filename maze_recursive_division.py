# https://blog.csdn.net/juzihongle1/article/details/73135920

import random
import numpy as np

# 这个函数将当前区域划分为四个小区域，并随机的在三个区域挖洞，
# 让四个区域彼此联通，分隔与挖洞点都是随机生成的。


def Recursive_division(r1, r2, c1, c2, M, image):
    if r1 < r2 and c1 < c2:
        rm = random.randint(r1, r2-1)
        cm = random.randint(c1, c2-1)
        cd1 = random.randint(c1, cm)
        cd2 = random.randint(cm+1, c2)
        rd1 = random.randint(r1, rm)
        rd2 = random.randint(rm+1, r2)
        d = random.randint(1, 4)
        if d == 1:
            M[rd2, cm, 2] = 1
            M[rd2, cm+1, 0] = 1
            M[rm, cd1, 3] = 1
            M[rm+1, cd1, 1] = 1
            M[rm, cd2, 3] = 1
            M[rm+1, cd2, 1] = 1
        elif d == 2:
            M[rd1, cm, 2] = 1
            M[rd1, cm+1, 0] = 1
            M[rm, cd1, 3] = 1
            M[rm+1, cd1, 1] = 1
            M[rm, cd2, 3] = 1
            M[rm+1, cd2, 1] = 1
        elif d == 3:
            M[rd1, cm, 2] = 1
            M[rd1, cm+1, 0] = 1
            M[rd2, cm, 2] = 1
            M[rd2, cm+1, 0] = 1
            M[rm, cd2, 3] = 1
            M[rm+1, cd2, 1] = 1
        elif d == 4:
            M[rd1, cm, 2] = 1
            M[rd1, cm+1, 0] = 1
            M[rd2, cm, 2] = 1
            M[rd2, cm+1, 0] = 1
            M[rm, cd1, 3] = 1
            M[rm+1, cd1, 1] = 1

        Recursive_division(r1, rm, c1, cm, M, image)
        Recursive_division(r1, rm, cm+1, c2, M, image)
        Recursive_division(rm+1, r2, cm+1, c2, M, image)
        Recursive_division(rm+1, r2, c1, cm, M, image)

    elif r1 < r2:
        rm = random.randint(r1, r2-1)
        M[rm, c1, 3] = 1
        M[rm+1, c1, 1] = 1
        Recursive_division(r1, rm, c1, c1, M, image)
        Recursive_division(rm+1, r2, c1, c1, M, image)
    elif c1 < c2:
        cm = random.randint(c1, c2-1)
        M[r1, cm, 2] = 1
        M[r1, cm+1, 0] = 1
        Recursive_division(r1, r1, c1, cm, M, image)
        Recursive_division(r1, r1, cm+1, c2, M, image)


def generateMaze(nrow, ncol, seed):
    random.seed(seed)
    # num_rows = int(input("Rows: "))                 # number of rows
    # num_cols = int(input("Columns: "))              # number of columns
    num_rows = nrow
    num_cols = ncol
    r1 = 0
    r2 = num_rows-1
    c1 = 0
    c2 = num_cols-1

    pscl = 3
    # The array M is going to hold the array information for each cell.
    # The first four coordinates tell if walls exist on those sides
    # and the fifth indicates if the cell has been visited in the search.
    # M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
    M = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)

    # The array image is going to be the output image to display
    image = np.zeros((num_rows*pscl, num_cols*pscl), dtype=np.uint8)

    Recursive_division(r1, r2, c1, c2, M, image)

    # Open the walls at the start and finish
    # Open the walls at the start and finish
    M[0, 0, 0] = 1
    M[num_rows-1, num_cols-1, 2] = 1
    # Generate the image for display
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            cell_data = M[row, col]
            for i in range(pscl*row+1, pscl*row+2):
                image[i, range(pscl*col+1, pscl*col+2)] = 3
            if cell_data[0] == 1:
                image[range(pscl*row+1, pscl*row+2), range(pscl*col, pscl*col+1)] = 3
            if cell_data[1] == 1:
                image[range(pscl*row, pscl*row+1), range(pscl*col+1, pscl*col+2)] = 3
            if cell_data[2] == 1:
                image[range(pscl*row+1, pscl*row+2), range(pscl*col+2, pscl*col+3)] = 3
            if cell_data[3] == 1:
                image[range(pscl*row+2, pscl*row+3), range(pscl*col+1, pscl*col+2)] = 3

    start = 1, 0
    end   = -2, -1
    image[start[0], start[1]] = 1
    image[end[0], end[1]] = 2

    return image


if __name__ == "__main__":
    image = generateMaze(nrow=20, ncol=20, seed=1234)
    image[1,1] = 4
    # Display the image
    from matplotlib import pyplot as plt
    plt.imshow(image, interpolation='none')
    plt.show()
    # print(image[:10, :10])
