from variables import ProteusVariable

cumulative_triangle = [[20, 30, 40], [10, 20], [5]]
row_sums = [sum(row) for row in cumulative_triangle]
column_sums = [
    sum(row[i] for row in cumulative_triangle)
    for i in range(len(cumulative_triangle[0]))
]
development_factors = [
    [
        row_sums[j + 1] / (row_sums[j] - cumulative_triangle[j, j])
        for j in range(len(cumulative_triangle) - 1)
    ]
]
