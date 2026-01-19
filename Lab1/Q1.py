# Implement ATA  -  A = [[1 2 3] ,[4 5 6]]
#list of lists define matrix
A = [[1,2,3],
     [4,5,6]]     #2 rows,3 columns

AT = [[1,4],
      [2,5],
      [3,6]]      #3 rows,2 columns

#empty list to store result of A^T A
ATA = []

for i in range(len(AT)):        # rows of AT
    row = []
    for j in range(len(A[0])):  # columns of A
        sum = 0
        for k in range(len(A)): # summation loop
            sum += AT[i][k] * A[k][j]
        row.append(sum)
    ATA.append(row)

print("A^T A =", ATA)

# Output is
# A^T A = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
