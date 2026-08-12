# ============================================================
# day01_linalg.py
#
# Day 01: Linear Algebra with Pure Python
#
# Goal:
#   Learn how basic matrix operations work before using NumPy.
#
# We will implement:
#   1. dot()       -> vector × vector
#   2. matvec()    -> matrix × vector
#   3. matmul()    -> matrix × matrix
#   4. transpose() -> rows become columns
#
# Then we will:
#   5. Test our functions
#   6. Compare them with NumPy
#   7. Test them with random inputs
#   8. Measure speed on 200 × 200 matrices
#
# Important:
#   The four functions below use only Python.
#   They do NOT use NumPy.
# ============================================================


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# random:
#   Creates random numbers for our tests.
import random

# time:
#   Measures how long an operation takes.
import time

# NumPy:
#   We use this only for comparison.
import numpy as np


# ============================================================
# PART 1 — PURE PYTHON LINEAR ALGEBRA
# ============================================================


def dot(a, b):
    """
    Calculate the dot product of two vectors.

    Example:

        a = [1, 2, 3]
        b = [4, 5, 6]

    Calculation:

        1*4 + 2*5 + 3*6

        = 4 + 10 + 18

        = 32
    """

    # This stores the running total.
    total = 0

    # Go through each position in the vectors.
    for i in range(len(a)):

        # Multiply matching values.
        product = a[i] * b[i]

        # Add the product to the total.
        total = total + product

    # Give the final answer back to the caller.
    return total


def matvec(M, v):
    """
    Calculate matrix × vector.

    Example:

        M = [
            [1, 2, 3],
            [4, 5, 6]
        ]

        v = [10, 20, 30]

    First row:

        1*10 + 2*20 + 3*30
        = 140

    Second row:

        4*10 + 5*20 + 6*30
        = 320

    Result:

        [140, 320]

    Key idea:

        Each row of M makes one dot product with v.
    """

    # This list will store the output.
    output = []

    # Go through every row in the matrix.
    for row in M:

        # Calculate the dot product of this row and the vector.
        value = dot(row, v)

        # Add the value to the output.
        output.append(value)

    # Return the complete vector.
    return output


def transpose(M):
    """
    Transpose a matrix.

    Transpose means:

        rows become columns
        columns become rows

    Example:

        [
            [1, 2, 3],
            [4, 5, 6]
        ]

    becomes:

        [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
    """

    # We need one output row for each input column.
    output = []

    # Get each column position.
    for column_index in range(len(M[0])):

        # This will become one row in the new matrix.
        new_row = []

        # Go through every row in the original matrix.
        for row in M:

            # Take the value at this column position.
            value = row[column_index]

            # Add it to the new row.
            new_row.append(value)

        # Add the new row to the output matrix.
        output.append(new_row)

    return output


def matmul(A, B):
    """
    Calculate matrix × matrix.

    Important idea:

        Every value in the result is a dot product.

    Example:

        A = [
            [1, 2],
            [3, 4]
        ]

        B = [
            [5, 6],
            [7, 8]
        ]

    First result value:

        [1, 2] · [5, 7]

        = 1*5 + 2*7

        = 19

    Result:

        [
            [19, 22],
            [43, 50]
        ]

    We transpose B first.

    This changes B:

        [5, 6]
        [7, 8]

    into:

        [5, 7]
        [6, 8]

    Now every column of B is easy to access as a row.
    """

    # Turn B's columns into rows.
    B_transposed = transpose(B)

    # This will store the final matrix.
    output = []

    # Go through every row in A.
    for row in A:

        # This stores one row of the result.
        result_row = []

        # Go through every column of B.
        #
        # After transpose(), each column is now a row.
        for column in B_transposed:

            # One result value is one dot product.
            value = dot(row, column)

            # Add that value to this result row.
            result_row.append(value)

        # Add the completed row to the output.
        output.append(result_row)

    return output


# ============================================================
# PART 2 — BASIC TESTS
# ============================================================

print("=" * 60)
print("PART 1: BASIC TESTS")
print("=" * 60)


# ------------------------------------------------------------
# Test dot()
# ------------------------------------------------------------

a = [1, 2, 3]
b = [4, 5, 6]

result = dot(a, b)

print("dot:", result)

# The correct answer is:
#
# 1*4 + 2*5 + 3*6 = 32
#
# assert stops the program if the result is wrong.
assert result == 32


# ------------------------------------------------------------
# Test matvec()
# ------------------------------------------------------------

M = [
    [1, 2, 3],
    [4, 5, 6],
]

v = [10, 20, 30]

result = matvec(M, v)

print("matvec:", result)

assert result == [140, 320]


# ------------------------------------------------------------
# Test transpose()
# ------------------------------------------------------------

result = transpose(M)

print("transpose:", result)

assert result == [
    [1, 4],
    [2, 5],
    [3, 6],
]


# ------------------------------------------------------------
# Test matmul()
# ------------------------------------------------------------

A = [
    [1, 2],
    [3, 4],
]

B = [
    [5, 6],
    [7, 8],
]

result = matmul(A, B)

print("matmul:", result)

assert result == [
    [19, 22],
    [43, 50],
]


print("Basic tests: PASS")


# ============================================================
# PART 3 — NUMPY EQUIVALENTS
# ============================================================

print()
print("=" * 60)
print("PART 2: NUMPY EQUIVALENTS")
print("=" * 60)


# Convert Python lists into NumPy arrays.
np_a = np.array(a)
np_b = np.array(b)

np_M = np.array(M)
np_v = np.array(v)

np_A = np.array(A)
np_B = np.array(B)


# ------------------------------------------------------------
# NumPy dot product
# ------------------------------------------------------------

numpy_dot = np.dot(np_a, np_b)

print("NumPy dot:", numpy_dot)


# ------------------------------------------------------------
# NumPy matrix × vector
# ------------------------------------------------------------

numpy_matvec = np_M @ np_v

print("NumPy matvec:", numpy_matvec)


# ------------------------------------------------------------
# NumPy transpose
# ------------------------------------------------------------

numpy_transpose = np_M.T

print("NumPy transpose:")
print(numpy_transpose)


# ------------------------------------------------------------
# NumPy matrix × matrix
# ------------------------------------------------------------

numpy_matmul = np_A @ np_B

print("NumPy matmul:")
print(numpy_matmul)


# ============================================================
# PART 4 — COMPARE OUR CODE WITH NUMPY
# ============================================================

print()
print("=" * 60)
print("PART 3: COMPARE WITH NUMPY")
print("=" * 60)


# np.allclose() checks if two numerical results
# are equal within a small floating-point tolerance.

assert np.allclose(
    dot(a, b),
    np.dot(np_a, np_b),
)

assert np.allclose(
    matvec(M, v),
    np_M @ np_v,
)

assert np.allclose(
    transpose(M),
    np_M.T,
)

assert np.allclose(
    matmul(A, B),
    np_A @ np_B,
)


print("All basic NumPy comparisons: PASS")


# ============================================================
# PART 5 — RANDOM TESTS
# ============================================================

print()
print("=" * 60)
print("PART 4: RANDOM TESTS")
print("=" * 60)


def random_matrix(rows, columns):
    """
    Create a matrix containing random numbers.

    The numbers are between -10 and 10.
    """

    matrix = []

    for _ in range(rows):

        row = []

        for _ in range(columns):

            value = random.uniform(-10, 10)

            row.append(value)

        matrix.append(row)

    return matrix


def random_vector(size):
    """
    Create a vector containing random numbers.
    """

    vector = []

    for _ in range(size):

        value = random.uniform(-10, 10)

        vector.append(value)

    return vector


# ------------------------------------------------------------
# Random dot-product tests
# ------------------------------------------------------------

for _ in range(10):

    # Create two random vectors.
    a_random = random_vector(10)
    b_random = random_vector(10)

    # Run our version.
    python_result = dot(a_random, b_random)

    # Run NumPy's version.
    numpy_result = np.dot(
        np.array(a_random),
        np.array(b_random),
    )

    # Compare the results.
    assert np.allclose(
        python_result,
        numpy_result,
    )


# ------------------------------------------------------------
# Random matrix × vector tests
# ------------------------------------------------------------

for _ in range(10):

    # Create a 5 × 5 matrix.
    M_random = random_matrix(5, 5)

    # Create a vector with 5 values.
    v_random = random_vector(5)

    # Run our version.
    python_result = matvec(M_random, v_random)

    # Run NumPy's version.
    numpy_result = np.array(M_random) @ np.array(v_random)

    # Compare the results.
    assert np.allclose(
        python_result,
        numpy_result,
    )


# ------------------------------------------------------------
# Random matrix × matrix tests
# ------------------------------------------------------------

for _ in range(10):

    # Create two 5 × 5 matrices.
    A_random = random_matrix(5, 5)
    B_random = random_matrix(5, 5)

    # Run our version.
    python_result = matmul(A_random, B_random)

    # Run NumPy's version.
    numpy_result = np.array(A_random) @ np.array(B_random)

    # Compare the results.
    assert np.allclose(
        python_result,
        numpy_result,
    )


# ------------------------------------------------------------
# Random transpose tests
# ------------------------------------------------------------

for _ in range(10):

    # Use a non-square matrix too.
    M_random = random_matrix(4, 7)

    # Run our version.
    python_result = transpose(M_random)

    # Run NumPy's version.
    numpy_result = np.array(M_random).T

    # Compare the results.
    assert np.allclose(
        python_result,
        numpy_result,
    )


print("Random tests: PASS")


# ============================================================
# PART 6 — 200 × 200 BENCHMARK
# ============================================================

print()
print("=" * 60)
print("PART 5: 200 × 200 MATRIX MULTIPLICATION")
print("=" * 60)


# Matrix size.
SIZE = 200


# Create two random 200 × 200 matrices.
A_big = random_matrix(SIZE, SIZE)
B_big = random_matrix(SIZE, SIZE)


# ------------------------------------------------------------
# Time pure Python
# ------------------------------------------------------------

print("Running pure Python matmul...")

start = time.perf_counter()

python_result = matmul(A_big, B_big)

end = time.perf_counter()

python_time = end - start


# ------------------------------------------------------------
# Create NumPy matrices
# ------------------------------------------------------------

np_A_big = np.array(A_big)
np_B_big = np.array(B_big)


# ------------------------------------------------------------
# Time NumPy
# ------------------------------------------------------------

print("Running NumPy matmul...")

start = time.perf_counter()

numpy_result = np_A_big @ np_B_big

end = time.perf_counter()

numpy_time = end - start


# ------------------------------------------------------------
# Check that the benchmark results are correct.
# ------------------------------------------------------------

assert np.allclose(
    python_result,
    numpy_result,
)


# ------------------------------------------------------------
# Calculate speed ratio.
# ------------------------------------------------------------

speed_ratio = python_time / numpy_time


# ------------------------------------------------------------
# Print benchmark results.
# ------------------------------------------------------------

print()
print("Pure Python time:", python_time, "seconds")
print("NumPy time:      ", numpy_time, "seconds")
print("Speed ratio:     ", speed_ratio, "x")


# ============================================================
# PART 7 — FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("DAY 01 COMPLETE")
print("=" * 60)

print("All functions work.")
print("All NumPy comparisons passed.")
print("All random tests passed.")
print("The 200 × 200 results match.")
print()
print("Pure Python is approximately", speed_ratio, "x slower")
print("than NumPy for this benchmark on this computer.")