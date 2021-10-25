print("R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0)")

for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                if i == 0 and j == 0 and k == 0 and l == 0:
                    continue
                print(" + R.get(%d, %d, %d, %d) * S.get(%d, %d, %d, %d)" %
                      (i, j, k, l, l, k, j, i))
