def isEmpty(val):

    '''
    Returns if a Grid is empty
    
    Parameters:
        val (Int) : Value of the Grid
    
    Returns:
        Boolean : True if val==0 else False
    
    '''

    return True if val==0 else False


def rowCheck(val, row, sudoku):

    '''
    Checks if the val is already present in row of sudoku
    
    Parameters:
        val (Int) : Value to be checked
        row (Int) : Row number between 0-9 which is to be checked
        sudoku (2D List) : Sudoku Grid
    
    Returns:
        Boolean : False if value is present in sudoku[row][0:9] else True
    '''

    for col in range(9):
        if sudoku[row][col]==val:
            return False
    return True


def colCheck(val, col, sudoku):

    '''
    Checks if the val is already present in col of sudoku
    
    Parameters:
        val (Int) : Value to be checked
        col (Int) : Column number between 0-9 which is to be checked
        sudoku (2D List) : Sudoku Grid
    
    Returns:
        Boolean : False if value is present in sudoku[0:9][col] else True

    '''

    for row in range(9):
        if sudoku[row][col]==val:
            return False
    return True


def gridCheck(val, row, col, sudoku):

    '''
    Checks if the val is already present in 3*3 grid of sudoku
    
    Parameters:
        val (Int) : Value to be checked
        row (Int) : Row number between 0-9 of element
        col (Int) : Column number between 0-9 element
        sudoku (2D List) : Sudoku Grid
    
    Returns:
        Boolean : False if value is present in sudoku[startRow:endRow][startCol:endCol] else True

    '''

    startRow=(row//3)*3
    endRow=startRow+3
    startCol=(col//3)*3
    endCol=startCol+3
    for r in range(startRow, endRow):
        for c in range(startCol, endCol):
            if sudoku[r][c]==val:
                return False
    return True


def isValid(val, row, col, sudoku):

    '''
    Checks if the sudoku[row][col]=val is Valid or Not

    Parameters:
        val (Int) : Value of Cell
        row (Int) : Row Number
        col (Int) : Column Number
        sudoku (2D List) : Sudoku Grid

    Returns:
        Boolean : True if sudoku[row][col] is valid else False

    '''

    if rowCheck(val, row, sudoku) and colCheck(val, col, sudoku) and gridCheck(val, row, col, sudoku):
        return True
    return False
    

def backtrack(row, col, sudoku):

    '''
    Applies a Backtracking paradigm to solve sudoku
    
    Parameters:
        row (Int) : Current Row number
        col (Int) : Current Column number
        sudoku (2D List) : Sudoku Grid

    Returns:
        Boolean : True if solution was found else False
    
    '''

    if row==8 and col>8:
        return True
    if col>8:
        col=0
        row+=1
    if not(isEmpty(sudoku[row][col])):
        return backtrack(row, col+1, sudoku)
    for i in range(1,10):
        if not(isValid(i, row, col, sudoku)):
            continue
        sudoku[row][col]=i
        if backtrack(row, col+1, sudoku):
            return True
        sudoku[row][col]=0
    return False


def solveSudoku(sudoku):

    '''
    Solves sudoku
    
    Parameters:
        sudoku (2D List) : Sudoku Grid

    Returns:
        Boolean : True if solution was found else False
    
    '''

    return backtrack(0, 0, sudoku)
    