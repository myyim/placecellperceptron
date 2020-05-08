# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.
# See https://github.com/myyim/number_theory for more details

import math

def lcm(a,b):
    """ lcm(a,b) returns the LCM of a and b"""
    import fractions
    return abs(a*b)/fractions.gcd(a,b) if a and b else 0

def GCD(l):
    """ GCD(l) returns the GCD of entries in l"""
    import fractions
    gcd = l[0]
    for j in range(1,len(l)):
        gcd = fractions.gcd(gcd,l[j])
    return gcd

def nCr(n,r):
    """nCr(n,r) returns n choose r"""
    if n>=r:
        return math.factorial(n)/math.factorial(r)/math.factorial(n-r)
    else:
        return 0

def partitions(n, I=1):
    """partitions(n,I=1) is a generator that lists all partitions of a positive integer n with partition size >= I.
        Example Usage:
        > partition = partitions(4,2)
        > for p in partition:
        >    print p
        """
    yield (n,)
    for j in range(I, n//2 + 1):
        for p in partitions(n-j, j):
            yield p + (j,)

def Stirling2(n,k):
    """Stirling numbers of the second kind S(n,k)"""
    num = 0
    for i in range(k+1):
        num += (-1)**i*nCr(k,i)*(k-i)**n
    return num/math.factorial(k)

def PolyBernoulli(n,k):
    """Poly-Bernoulli number B^(-k)_n"""
    num = 0
    for m in range(n+1):
        num += (-1)**(m+n)*math.factorial(m)*Stirling2(n,m)*(m+1)**k
    return num
