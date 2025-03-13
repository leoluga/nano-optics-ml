import numpy as np
import matplotlib.pyplot as plt
import math

def plotspectrum(f,Sn):
    plt.figure(0)
    plt.plot(f,np.abs(Sn),linewidth=3,color='b')
    plt.xlabel('Frequency ($cm^{-1}$)', fontsize = 20)
    plt.ylabel('Normalized $S_n$ amplitude', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.figure(1)
    plt.plot(f,np.angle(Sn),linewidth=3,color='r')
    plt.xlabel('Frequency ($cm^{-1}$)', fontsize = 20)
    plt.ylabel('Normalized $S_n$ phase', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def readmaterial(f, filename):
    material = np.genfromtxt(filename, delimiter=',')
    material = np.flipud(material)
    fp = material[:,0]
    eps = np.interp(f, fp, material[:,1]) + 1j*np.interp(f, fp, material[:,2])
    return eps

def readmaterial2(f, filename1, filename2):
    material1 = np.genfromtxt(filename1, delimiter=',')
    f1 = material1[:,0]
    eps1 = np.interp(f,f1, material1[:,1])
    material2 = np.genfromtxt(filename2, delimiter=',')
    f2 = material2[:,0]
    eps2 = np.interp(f,f2,material2[:,1])
    eps = eps1+1j*eps2
    return eps

def finitedipole(w,z,eps,a,L,g):
    beta=(eps-1)/(eps+1)
    alpha_eff=np.zeros((len(w),len(z)),dtype=complex)
    for i in range(len(z)):
        alpha_eff[:,i]=beta*(g-(a+z[i]/L)*np.log(4*L/(4*z[i]+3*a)))/(np.log(4*L/a)-beta*(g-(3*a+4*z[i])/(4*L))*np.log(2*L/(2*z[i]+a)))
    alpha_eff=np.transpose(alpha_eff)
    return alpha_eff

def finitedipole_2(w,z,eps,a,L,g):
    beta=(eps-1)/(eps+1)
    alpha_eff=np.zeros((len(w),len(z)),dtype=complex)
    W_0 = z + 1.31*a
    W_1 = z + a/2
    for j in range(len(w)):
        for i in range(len(z)):
            f_0 = (g-(a+z[i]+W_0[i])/(2*L))*np.log(4*L/(a+2*z[i]+2*W_0[i]))/np.log(4*L/a)
            f_1 = (g-(a+z[i]+W_1[i])/(2*L))*np.log(4*L/(a+2*z[i]+2*W_1[i]))/np.log(4*L/a)
            alpha_eff[j,i] = beta[j]*f_0/(1-beta[j]*f_1)
    alpha_eff=np.transpose(alpha_eff)
    return alpha_eff

def pointdipole(w,z,eps,a):
    z = z+a
    beta=(eps-1)/(eps+1)
    alpha_eff=np.zeros((len(w),len(z)),dtype=complex)
    alpha = 4*math.pi*a**3
    for j in range(len(w)):
        for i in range(len(z)):
            alpha_eff[j,i] = alpha/(1-alpha*beta[j]/(16*math.pi*z[i]**3))
    alpha_eff=np.transpose(alpha_eff)
    return alpha_eff
    
def demodulate(alpha_eff,t,n):
    fourier=np.cos(n*t)
    fourier=np.repeat(fourier[:,np.newaxis],len(alpha_eff[1,:]),1)
    Sn=np.trapz(alpha_eff*fourier,axis=0)
    return Sn

def lorentzeps(w,w0,s,gamma,eps_inf):
    eps=eps_inf+s/(w0**2-w**2-1j*w*gamma)
    return eps

def Sn(w,eps,z,a,L,g,t,n):
    s_samp=demodulate(finitedipole(w,z,eps,a,L,g),t,n)
    s_ref=demodulate(finitedipole(w,z,-10000+10000j,a,L,g),t,n)
    sn=s_samp/s_ref
    return sn

def Sn_2(w,eps,z,a,L,g,t,n):
    s_samp=demodulate(finitedipole_2(w,z,eps,a,L,g),t,n)
    eps_ref = np.ones(1)*(-10000+10000*1j)
    s_ref=demodulate(finitedipole_2([1],z,eps_ref,a,L,g),t,n)
    sn=s_samp/s_ref
    return sn

def Sn_3(w,eps,z,a,t,n):
    s_samp=demodulate(pointdipole(w,z,eps,a),t,n)
    eps_ref = np.ones(1)*(11.7)
    s_ref=demodulate(pointdipole([1],z,eps_ref,a),t,n)
    sn=s_samp/s_ref
    return sn

def farfieldfactor(f,eps):
    import math
    n0 = np.ones(len(f))
    #n1 = np.sqrt(eps)
    n1 = np.sqrt((0.5*np.sqrt(eps*np.conj(eps))+np.real(eps)/2)) + np.sqrt((0.5*np.sqrt(eps*np.conj(eps))-np.real(eps)/2))*1j
    theta = math.pi/3
    cos_theta_t = np.sqrt(1-(n0*np.sin(theta)/n1)**2)
    r = (-n0*cos_theta_t + n1*np.cos(theta))/(n0*cos_theta_t + n1*np.cos(theta))
    FFF = (1+r)*(1+r)
    return FFF

def multilayer(f,eps_2,eps_3,eps_ref,d,n):
    import math
    import nearfield as nf
    eps_1 = np.ones(len(f))
    beta_12 = (eps_1 - eps_2)/(eps_1+eps_2)
    beta_23 = (eps_2 - eps_3)/(eps_2+eps_3)
    beta_21 = -beta_12 
    k = np.logspace(-7, 1, 500)  #######attention
    dk = np.diff(k)
    dk = np.append(dk, np.array([0]))
    dz = 0.0001
    z = np.array([0, dz])
    phi_0 = np.zeros(len(z), dtype=complex)
    phi_1 = np.zeros(len(z), dtype=complex)
    a = 25
    L = 60
    g = 0.7*np.exp(0.1*1j)
    A = 50
    h0 = 2
    t = np.linspace(0, math.pi,30)
    H = A - A*np.cos(t) + h0
    W_0 = H + 1.31*a
    W_1 = H + a/2
    alpha_eff = np.zeros((len(f), len(H)), dtype=complex)
    
    for i in range(len(f)):
        for j in range(len(H)):
            A_0 = (np.exp(-2*k*W_0[j]))*(beta_12[i]+beta_23[i]*np.exp(-2*k*d))/(1-beta_21[i]*beta_23[i]*np.exp(-2*k*d))
            A_1 = (np.exp(-2*k*W_1[j]))*(beta_12[i]+beta_23[i]*np.exp(-2*k*d))/(1-beta_21[i]*beta_23[i]*np.exp(-2*k*d))
            for ii in range(len(z)):
                phi_0[ii] = np.trapz(np.real(A_0)*np.exp(k*z[ii])*dk, axis=0) + 1j*np.trapz(np.imag(A_0)*np.exp(k*z[ii])*dk, axis=0)
                phi_1[ii] = np.trapz(np.real(A_1)*np.exp(k*z[ii])*dk, axis=0) + 1j*np.trapz(np.imag(A_1)*np.exp(k*z[ii])*dk, axis=0)
            phi_prime_0 = np.diff(phi_0)/dz
            phi_prime_1 = np.diff(phi_1)/dz
            beta_x_0 = -(phi_0[0]**2)/phi_prime_0
            X_0 = phi_0[0]/phi_prime_0-W_0[j]
            beta_x_1 = -(phi_1[0]**2)/phi_prime_1
            X_1 = phi_1[0]/phi_prime_1-W_1[j]
            f_0 = (g-(a+H[j]+X_0)/(2*L))*np.log(4*L/(a+2*H[j]+2*X_0))/np.log(4*L/a)
            f_1 = (g-(a+H[j]+X_1)/(2*L))*np.log(4*L/(a+2*H[j]+2*X_1))/np.log(4*L/a)
            alpha_eff[i,j] = (beta_x_0*f_0)/(1-beta_x_1*f_1)
    alpha_eff = np.transpose(alpha_eff)
    sn = nf.demodulate(alpha_eff, t, n)

    beta_ref = (eps_ref-1)/(eps_ref+1)
    alpha_eff_ref = np.zeros((len(f), len(H)), dtype=complex)
    for j in range(len(f)):
        for i in range(len(H)):
            f_0 = (g-(a+H[i]+W_0[i])/(2*L))*np.log(4*L/(a+2*H[i]+2*W_0[i]))/np.log(4*L/a)
            f_1 = (g-(a+H[i]+W_1[i])/(2*L))*np.log(4*L/(a+2*H[i]+2*W_1[i]))/np.log(4*L/a)
            alpha_eff_ref[j,i] = beta_ref*f_0/(1-beta_ref*f_1)
    alpha_eff_ref = np.transpose(alpha_eff_ref)
    sn_ref = nf.demodulate(alpha_eff_ref, t, n)
    Sn = sn/sn_ref
    return Sn

def thinfilm(f,eps_2,eps_3,eps_ref,d,n):
    from NearFieldOptics import Materials as M
    from common.baseclasses import AWA
    eps_1 = 1
    len_w = len(f)
    len_q = 200
    len_z = 30
    q = np.logspace(0,8,len_q) 
    t = np.linspace(0,math.pi,len_z)
    a = 30e-7
    dz = 50e-7
    b = 31e-7
    z = b + dz*(1-np.cos(t))
    G = np.zeros((len_w,len_z), dtype=complex)
    s = np.zeros((len_w,len_z), dtype=complex)
    
    eps_2=AWA(eps_2 ,axes=[f],axis_names=['Frequency (cm-1)'])
    eps_2.cslice[1600]
    eps_3=AWA(eps_3 ,axes=[f],axis_names=['Frequency (cm-1)'])
    eps_3.cslice[1600]
    m1 = M.TabulatedMaterial(eps_2)
    m2 = M.TabulatedMaterial(eps_3)
    
    layers=M.LayeredMediaTM((m1,d),exit=m2)
    rp=layers.reflection_p(f,q)
    for i in range(len_z):
        G[:,i] = np.trapz(q**2*np.exp(-2*q*z[i])*rp,q)
        s[:,i] = 1/(1-G[:,i]*a**3)
        
    Sn = demodulate(np.transpose(s),t,n)
    G = np.zeros((len_w,len_z), dtype=complex)
    s = np.zeros((len_w,len_z), dtype=complex)

    rp_ref = (eps_ref-eps_1)/(eps_ref+eps_1)
    for i in range(len_z):
        G[:,i] = np.trapz(q**2*np.exp(-2*q*z[i])*rp_ref,q)
        s[:,i] = 1/(1-G[:,i]*a**3)       
        Sn_ref= demodulate(np.transpose(s),t,n)
    sn = Sn/Sn_ref
    return rp, sn
    