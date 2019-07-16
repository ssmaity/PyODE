##
## This contain IVP functions.
##


import numpy as np


##
## Euler method
##
def euler(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		x[i] = x[i-1] + h * f(t[i-1], x[i-1])
	return t, x


##
## Modified Euler method
##
def meuler(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		K1 = f(t[i-1], x[i-1])
		K2 = f(t[i], x[i-1] + h*K1)
		x[i] = x[i-1] + h*(K1 + K2)/2.0
	return t, x


##
## Heun method
##	
def heun(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		K1 = f(t[i-1], x[i-1])
		K2 = f(t[i-1] + (2.0/3.0)*h, x[i-1] + (2.0/3.0)*h*K1)
		x[i] = x[i-1] + h*(K1 + 3*K2)/4.0
	return t, x


##
## Midpoint method
##
def midpt(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		x[i] = x[i-1] + h*f(t[i-1] + h/2.0, x[i-1] + h*f(t[i-1], x[i-1])/2.0)
	return t, x


##
## Runge Kutta 2nd order method
##	
def rk2(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		K1 = h*f(t[i-1], x[i-1])
		K2 = h*f(t[i-1] + h/2.0, x[i-1] + K1/2.0)
		x[i] = x[i-1] + K2
	return t, x


##
## Runge Kutta 4th order method
##	
def rk4(f, a, b, N, x0):
	h = (b-a)/N
	t = np.arange(a, b+h, h)
	x = np.zeros(len(t))
	x[0] = x0
	for i in range(1, len(t)):
		K1 = h*f(t[i-1], x[i-1])
		K2 = h*f(t[i-1] + h/2.0, x[i-1] + K1/2.0)
		K3 = h*f(t[i-1] + h/2.0, x[i-1] + K2/2.0)
		K4 = h*f(t[i], x[i-1] + K3)
		x[i] = x[i-1] + (K1 + 2*K2 + 2*K3 + K4)/6.0
	return t, x


##
## Runge Kutta Fehlberg method
##
def rkf(f, a, b, x0, tol, hmin, hmax):
    a2  =   1/4
    a3  =   3/8
    a4  =   12/13
    a5  =   1
    a6  =   1/2

    b21 =   1/4
    b31 =   3/32
    b32 =   9/32
    b41 =   1932/2197
    b42 =  -7200/2197
    b43 =   7296/2197
    b51 =   439/216
    b52 =  -8
    b53 =   3680/513
    b54 =  -845/4104
    b61 =  -8/27
    b62 =   2
    b63 =  -3544/2565
    b64 =   1859/4104
    b65 =  -11/40

    r1  =   1/360
    r3  =  -128/4275
    r4  =  -2197/75240
    r5  =   1/50
    r6  =   2/55

    c1  =   25/216
    c3  =   1408/2565
    c4  =   2197/4104
    c5  =  -1/5

    t = a
    x = x0
    h = hmax

    T = np.array(t)
    X = np.array(x)

    while t < b:

        K1 = h * f(t, x)
        K2 = h * f(t + a2 * h, x + b21 * K1)
        K3 = h * f(t + a3 * h, x + b31 * K1 + b32 * K2)
        K4 = h * f(t + a4 * h, x + b41 * K1 + b42 * K2 + b43 * K3)
        K5 = h * f(t + a5 * h, x + b51 * K1 + b52 * K2 + b53 * K3 + b54 * K4)
        K6 = h * f(t + a6 * h, x + b61 * K1 + b62 * K2 + b63 * K3 + b64 * K4 + b65 * K5)

        err = abs(r1 * K1 + r3 * K3 + r4 * K4 + r5 * K5 + r6 * K6)/h

        if err <= tol:
            t +=  h
            x += c1 * K1 + c3 * K3 + c4 * K4 + c5 * K5
            T = np.append(T, t)
            X = np.append(X, x)

        h = h * min(max(0.84 * (tol/err)**0.25, 0.1), 4.0)

        if h > hmax:
            h = hmax
        
        if h < hmin:
            print( "Error! Stepsize crossed the lower limit.")
            break
 
        if t + h > b:
            h = b - t

    return T, X
 
 
##
## Adams Bashforth Moulton 4th order predictor corrector method
##   
def abm(f, a, b, x0, N):
	h = (b-a)/N
	t = np.linspace(a, b, N + 1)
	x = np.zeros(len(t))
	x[0] = x0
	
	for i in [1, 2, 3]:
		K1 = h * f(t[i-1], x[i-1])
		K2 = h* f(t[i-1] + h/2, x[i-1] + K1/2)
		K3 = h * f(t[i-1] + h/2, x[i-1] + K2/2)
		K4 = h * f(t[i-1] + h, x[i-1] + K3)
		x[i] = x[i-1] + (K1 + 2*K2 + 2*K3 + K4)/6
		
	for i in range(4, len(t)):
		x[i] = x[i-1] + h*(55*f(t[i-1], x[i-1]) - 59*f(t[i-2], x[i-2]) + 37*f(t[i-3], x[i-3]) - 9*f(t[i-4], x[i-4]))/24
		x[i] = x[i-1] + h*(9*f(t[i], x[i]) + 19*f(t[i-1], x[i-1]) - 5*f(t[i-2], x[i-2]) + f(t[i-3], x[i-3]))/24
	
	return t, x
