def myfunction(n):
    return lambda a : a * n

mydouble = myfunction(2)

print(mydouble(10))