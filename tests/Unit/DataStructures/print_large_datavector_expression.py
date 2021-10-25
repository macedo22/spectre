print("R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0)")

for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                if a == 0 and b == 0 and c == 0 and d == 0:
                    continue
                print(" + R.get(%d, %d, %d, %d) * S.get(%d, %d, %d, %d)" %
                      (a, b, c, d, d, c, b, a))
