import numpy as np
import math
def mindelayt(M,s,h):
    M1 = np.where(M == 1)[0]
    M0 = np.where(M == 0)[0]
    if (len(M1) == 0):
        beita1 = 0
        t = []
        for i in range(0, 10000):
            beita1 = beita1 + 0.0001
            listtao = maxlocalt(beita1, M, h, s)
            t.append(beita1 + listtao[0])
        return min(t)
    beita1 = minRF(M, s, h)
    t = findb(M, s, h, beita1)
    return t
def minRF(M,s,h):
    M1 = np.where(M == 1)[0]
    hi = np.array([h[i] for i in M1])
    # gi=np.array([g[i] for i in M1])
    si = np.array([s[i] for i in M1],np.dtype(float))
    n = 0
    p = 2
    w = 1 * 10 ** 6
    u= 0.7
    omiga = 10**-6
    mt=[]
    for i in range(len(M1)):
        mt.append((si[i]*math.log(2))/(w*(u*p*hi[i]*hi[i])/(omiga*omiga)))
    tmin = [max(mt)]
    tmax = [10**6]
    while tmax[n] - tmin[n] > 10**-7:
        t1sum = 0
        t2sum = 0
        lamuda = tmin[n] + 0.382 * (tmax[n] - tmin[n])
        miu = tmin[n] + 0.618 * (tmax[n] - tmin[n])
        for i in range(len(M1)):
            j = 1
            m = 1
            T = [8 * 10]
            T1 = [8 * 10]
            T.append(si[i] / (w * math.log2(1 + (u* p * hi[i] ** 2 * lamuda) / (T[j - 1] * omiga ** 2))))
            while abs(T[j] - T[j-1]) > 10**-8:
                beita = lamuda
                T.append(si[i] / (w * math.log2(1 + (u* p * hi[i] ** 2 * beita) / (T[j] * omiga ** 2))))
                j=j+1
            t1sum = t1sum + T[j-1]
            T1.append(si[i] / (w * math.log2(1 + (u* p * hi[i] ** 2 * miu) / (T1[m - 1] * omiga ** 2))))
            while abs(T1[m] - T1[m-1]) > 10**-8:
                beita = miu
                T1.append(si[i] /(w * math.log2(1 + (u* p * hi[i] ** 2 * beita) / (T1[m] * omiga ** 2))))
                m=m+1
            t2sum = t2sum + T1[m-1]
        t1sum=t1sum+lamuda
        t2sum=t2sum+miu
        if t1sum > t2sum:
            tmin.append(lamuda)
            tmax.append(tmax[n])
        else:
            tmin.append(tmin[n])
            tmax.append(miu)
        n = n + 1
    return tmin[n]
def minsumtranst(M,s,h,beita):
    M1 = np.where(M == 1)[0]
    hi = np.array([h[i] for i in M1],dtype=np.float64)
    # gi=np.array([g[i] for i in M1])
    si = np.array([s[i] for i in M1],dtype=np.float64)
    p = 2
    u = 0.7
    w = 1 * 10 ** 6
    omiga = 10**-6
    tsum=0
    for i in range(len(M1)):
        j = 1
        T = [8]
        T.append(si[i] / (w * math.log2(1 + (u* p * hi[i] ** 2 * beita) / (T[j-1] * omiga ** 2))))
        while abs(T[j] - T[j - 1]) > 10 ** -7:
            T.append(si[i] / (w * math.log2(1 + (u* p * hi[i] ** 2 * beita) / (T[j] * omiga ** 2))))
            j = j + 1
        tsum = tsum + T[j-1]
    return tsum
def maxbeita(M,h,s):
    f = 1.5 * (10 ** 7)
    O = 100
    ki = 10 ** -26
    u = 0.7
    p = 2
    M0 = np.where(M == 0)[0]
    hi = np.array([h[i] for i in M0])
    si = np.array([s[i] for i in M0],np.dtype(float))
    beita=[]
    for i in  range(len(M0)):
        beita.append((f * f * O * ki*si[i]) /(u * p* hi[i]))
    return max(beita)
def maxlocalt(beita1,M,h,s):
    f=1.5*(10**7)
    O=100
    ki = 10 ** -26
    u = 0.7
    p = 2
    M0 = np.where(M == 0)[0]
    hi = np.array([h[i] for i in M0])
    si = np.array([s[i] for i in M0],np.dtype(float))
    tao=[]
    beita=[]
    k = []
    for i in range(len(M0)):
        beita.append((f * f * O * ki * si[i]) / (u * p * hi[i]))
        if beita1 < beita[i]:
            tao.append(math.sqrt((ki * si[i] ** 3 * O ** 3) / (u * p * hi[i] * beita1)))
            k.append(((ki * si[i] ** 3 * O ** 3) / (4 * u * p * hi[i]))** (1 / 3))
        else:
            tao.append((si[i] * O) / f)
            k.append(((ki * si[i] ** 3 * O ** 3) / (4 * u * p * hi[i])) ** (1 / 3))
    return [max(tao), beita[tao.index(max(tao))], k[tao.index(max(tao))],si[tao.index(max(tao))],hi[tao.index(max(tao))]]
def tao(b,s,h):
    O = 100
    ki = 10 ** -26
    u = 0.7
    p = 2
    t=math.sqrt((ki * s ** 3 * O ** 3) / (u * p * h * b))
    return t
#t1>t2二分
def bisearch1(M,s,h,s0,h0,b1,b2,t1,t2):
    mid=b1
    while b2-b1>10**-6:
            mid = (b1 + b2) / 2
            t1 = mid + minsumtranst(M, s, h, mid)
            t2 = mid + tao(mid, s0, h0)
            if t1 > t2:
                b1 = mid
            else:
                b2 = mid
    return mid
#t1<t2二分
def bisearch2(M,s,h,s0,h0,b1,b2,t1,t2):
    mid = b1
    while b2-b1>10**-6:
            mid = (b1 + b2) / 2
            t1 = mid + minsumtranst(M, s, h, mid)
            t2 = mid + tao(mid, s0, h0)
            if t1 > t2:
                b2 = mid
            else:
                b1 = mid
    return mid
def finminb(beita1,h0,s0,s,h):
    f = 1.5 * (10 ** 7)
    O = 100
    ki = 10 ** -26
    u = 0.7
    p = 2
    b=[]
    beita11=[]
    for i in range(len(h0)):
        if h!=h0[i] and s!=s0[i]:
            b.append((ki*O*f*f*s**3)/(u*p*h*s0[i]*s0[i]))
    for i in range(len(b)):
        if b[i]>beita1:
            beita11.append(b[i])
    if len(beita11)!=0:
        beita111=min(beita11)
    else:
        beita111=1000
    return beita111
def finmaxb1(m,beita1,h0,s0,s,h):
    f = 1.5 * (10 ** 7)
    O = 100
    ki = 10 ** -26
    u = 0.7
    p = 2
    b = []
    beita11 = []
    for i in range(len(h0)):
        if h!=h0[i] and s!=s0[i]:
            b.append((ki*O*f*f*s**3)/(u*p*h*s0[i]*s0[i]))
    for i in range(len(b)):
        if b[i] < beita1 and b[i]>=m:
            beita11.append(b[i])
    if len(beita11)!=0:
        beita111=max(beita11)
    else:
        beita111=0
    return beita111
def finmaxb2(m,beita1,h0,s0,s,h):
    f = 1.5 * (10 ** 7)
    O = 100
    ki = 10 ** -26
    u = 0.7
    p = 2
    b = []
    beita11 = []
    for i in range(len(h0)):
        if h!=h0[i] and s!=s0[i]:
            b.append((ki*O*f*f*s0[i]**3)/(u*p*h*s*s))
    for i in range(len(b)):
        if b[i] < beita1 and b[i]>=m:
            beita11.append(b[i])
    if len(beita11)!=0:
        beita111=max(beita11)
    else:
        beita111=0
    return beita111
def findb(M,s,h,beita1):
    t = minsumtranst(M, s, h, beita1)
    M0 = np.where(M == 0)[0]
    h0 = np.array([h[i] for i in M0])
    s0 = np.array([s[i] for i in M0], np.dtype(float))
    M1 = np.where(M == 1)[0]
    h1 = np.array([h[i] for i in M1])
    s1 = np.array([s[i] for i in M1], np.dtype(float))
    u=0.7
    p = 2
    w = 1 * 10 ** 6
    omiga = 10 ** -6
    m = []
    for i in range(len(M1)):
        m.append((s1[i] * math.log(2)) / (w * (u* p * h1[i] * h1[i]) / (omiga * omiga)))
    if len(M0) != 0:
        listtao = maxlocalt(beita1, M, h, s)
        T1 = beita1 + t
        T2 = beita1 + listtao[0]
        if (T1 >= T2):
            return T1
        else:
            if listtao[2] < listtao[1]:
                if beita1 < listtao[2]:
                    beita11 = finminb(beita1, h0, s0, listtao[3], listtao[4])
                    if beita11 > listtao[2]:
                        T2 = listtao[2] + tao(listtao[2], listtao[3], listtao[4])
                        T1 = listtao[2] + minsumtranst(M, s, h, listtao[2])
                        if T2 >= T1:
                            return T2
                        else:
                            beita = bisearch2(M, s, h, listtao[3], listtao[4], beita1, listtao[2], T1, T2)
                            return beita+minsumtranst(M,s,h,beita)
                    else:
                        T2 = beita11 + tao(beita11, listtao[3], listtao[4])
                        T1 = beita11 + minsumtranst(M, s, h, beita11)
                        if T2 >= T1:
                            return findb(M, s, h, beita11)
                        else:
                            beita = bisearch2(M, s, h, listtao[3], listtao[4], beita1, beita11, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                elif beita1 > listtao[2] and beita1 < listtao[1]:
                    beita11 = finmaxb1(max(m), beita1, h0, s0, listtao[3], listtao[4])
                    if beita11 < listtao[2]:
                        if max(m)<=listtao[2]:
                            T2 = listtao[2] + tao(listtao[2], listtao[3], listtao[4])
                            T1 = listtao[2] + minsumtranst(M, s, h, listtao[2])
                            if T2 >= T1:
                                return T2
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], listtao[2], beita1, T1,T2)
                                return beita + minsumtranst(M, s, h, beita)
                        else:
                            beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                    else:
                        T2 = beita11 + tao(beita11, listtao[3], listtao[4])
                        T1 = beita11 + minsumtranst(M, s, h, beita11)
                        if T2 >= T1:
                            return findb(M, s, h, beita11)
                        else:
                            beita = bisearch1(M, s, h, listtao[3], listtao[4], beita11, beita1, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                else:
                    beita11 = finmaxb2(max(m), beita1, h0, s0, listtao[3], listtao[4])
                    if beita11 < listtao[1]:
                        beita111 = finmaxb1(max(m), listtao[1], h0, s0, listtao[3], listtao[4])
                        if beita111 < listtao[2]:
                            if max(m)<=listtao[2]:
                                T2 = listtao[2] + tao(listtao[2], listtao[3], listtao[4])
                                T1 = listtao[2] + minsumtranst(M, s, h, listtao[2])
                                if T2 >= T1:
                                    return T2
                                else:
                                    beita = bisearch1(M, s, h, listtao[3], listtao[4], listtao[2], beita1, T1, T2)
                                    return beita + minsumtranst(M, s, h, beita)
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                                return beita + minsumtranst(M, s, h, beita)
                        else:
                            if max(m)<=beita111:
                                T2 = beita111 + tao(beita111, listtao[3], listtao[4])
                                T1 = beita111 + minsumtranst(M, s, h, beita111)
                                if T2 >= T1:
                                    return findb(M, s, h, beita111)
                                else:
                                    beita = bisearch1(M, s, h, listtao[3], listtao[4], beita111, beita1, T1, T2)
                                    return beita + minsumtranst(M, s, h, beita)
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                                return beita + minsumtranst(M, s, h, beita)
                    else:
                        if max(m) <= beita11:
                            T2 = beita11 + tao(beita11, listtao[3], listtao[4])
                            T1 = beita11 + minsumtranst(M, s, h, beita11)
                            if T2 >= T1:
                                return findb(M, s, h, beita11)
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], beita11, beita1, T1, T2)
                                return beita + minsumtranst(M, s, h, beita)
                        else:
                            beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
            else:
                if beita1 < listtao[1]:
                    beita11 = finminb(beita1, h0, s0, listtao[3], listtao[4])
                    if beita11 > listtao[1]:
                        T2 = listtao[1] + tao(listtao[1], listtao[3], listtao[4])
                        T1 = listtao[1] + minsumtranst(M, s, h, listtao[1])
                        if T2 >= T1:
                            return T2
                        else:
                            beita = bisearch2(M, s, h, listtao[3], listtao[4], beita1, listtao[1], T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                    else:
                        T2 = beita11 + tao(beita11, listtao[3], listtao[4])
                        T1 = beita11 + minsumtranst(M, s, h, beita11)
                        if T2 >= T1:
                            return findb(M, s, h, beita11)
                        else:
                            beita = bisearch2(M, s, h, listtao[3], listtao[4], beita1, beita11, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                else:
                    beita11 = finmaxb2(max(m), beita1, h0, s0, listtao[3], listtao[4])
                    if beita11 < listtao[1]:
                        if max(m)<=listtao[1]:
                            T2 = listtao[1] + tao(listtao[1], listtao[3], listtao[4])
                            T1 = listtao[1] + minsumtranst(M, s, h, listtao[1])
                            if T2 >= T1:
                                return T2
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], listtao[1], beita1, T1, T2)
                                return beita + minsumtranst(M, s, h, beita)
                        else:
                            beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
                    else:
                        if max(m)<=beita11:
                            T2 = beita11 + tao(beita11, listtao[3], listtao[4])
                            T1 = beita11 + minsumtranst(M, s, h, listtao[1])
                            if T2 >= T1:
                                return findb(M, s, h, beita11)
                            else:
                                beita = bisearch1(M, s, h, listtao[3], listtao[4], beita11, beita1, T1, T2)
                                return beita + minsumtranst(M, s, h, beita)
                        else:
                            beita = bisearch1(M, s, h, listtao[3], listtao[4], max(m), beita1, T1, T2)
                            return beita + minsumtranst(M, s, h, beita)
    else:
        return beita1+t
def addm(h):
    m = []
    h1 = max(h)
    for i in range(len(h)):
        if h[i] < max(h):
            m.append(0)
        else:
            m.append(1)
    return np.array(m)
def addm2(s,h):
    m = []
    h1=np.median(h)
    s1=np.median(s)
    for i in range(len(s)):
        if h[i]>=h1 and s[i]<=s1:
            m.append(1)
        else:
            m.append(0)
    return np.array(m)
def addm3(s,h):
    m = []
    h1=np.median(h)
    s1=np.median(s)
    for i in range(len(s)):
        if h[i]<h1 and s[i]>s1:
            m.append(0)
        else:
            m.append(1)
    return np.array(m)
def addm4(h):
    m = []
    h1 = np.median(h)
    if h[0] < 0.1:
        m.append(0)
    else:
        m.append(1)
    m.append(1)
    m.append(1)
    m.append(1)
    m.append(1)
    m.append(1)
    if h[6] < 0.1:
        m.append(0)
    else:
        m.append(1)
    m.append(0)
    if h[8] < 0.1:
        m.append(0)
    else:
        m.append(1)
    m.append(1)
    return np.array(m)
def addm5(h):
    m = []
    h1=np.median(h)
    for i in range(len(h)):
        if h[i]<h1:
            m.append(0)
        else:
            m.append(1)
    return np.array(m)
