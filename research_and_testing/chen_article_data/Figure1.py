import numpy as np
import math
import nearfield as nf

def load_materials():
    def nk_to_eps(n,k):
        eps1 = n**2-k**2
        eps2 = 2*n*k
        return eps1, eps2
    data = np.genfromtxt(r'C:\nano_optics_ml_data\raw\Au_nk.csv', delimiter = ',')
    f = data[:,0]
    f_Au = np.linspace(1000, 1200,500)
    n = np.interp(f_Au, f, data[:,1])
    k = np.interp(f_Au, f, data[:,2])
    eps1, eps2 = nk_to_eps(n,k)
    eps_Au = eps1 + 1j*eps2
    Sn_Au = np.ones(len(eps_Au))
    Sn_Au_cal_pd = np.ones(len(eps_Au))
    Sn_Au_cal_fd = np.ones(len(eps_Au))   
    
    t = np.linspace(0, math.pi,30)
    A = 68
    h0 = 0
    a = 50
    L = 500
    z = A - A*np.cos(t) + h0
    n = 2
    g = 0.7*np.exp(0.07*1j)
    SiO2 = np.genfromtxt(r'C:\nano_optics_ml_data\raw\SiO2_eff.csv',delimiter=',')
    eps_SiO2 = SiO2[:,0] + 1j*SiO2[:,1]
    Sn_SiO2 = SiO2[:,2] + 1j*SiO2[:,3]
    f_SiO2 = SiO2[:,4]
    Sn_SiO2_cal_pd = 0.25*nf.farfieldfactor(f_SiO2,eps_SiO2)*nf.Sn_3(f_SiO2,eps_SiO2,z,a,t,n)
    Sn_SiO2_cal_fd = 0.25*nf.farfieldfactor(f_SiO2,eps_SiO2)*nf.Sn_2(f_SiO2,eps_SiO2,z,a,L,g,t,n)
    
    STO = np.genfromtxt(r'C:\nano_optics_ml_data\raw\STO_eff.csv',delimiter=',')
    eps_STO = STO[:,0] + 1j*STO[:,1]
    Sn_STO = STO[:,2] + 1j*STO[:,3]
    f_STO = STO[:,4]
    Sn_STO_cal_pd = 0.25*nf.farfieldfactor(f_STO,eps_STO)*nf.Sn_3(f_STO,eps_STO,z,a,t,n)
    Sn_STO_cal_fd = 0.25*nf.farfieldfactor(f_STO,eps_STO)*nf.Sn_2(f_STO,eps_STO,z,a,L,g,t,n)

    GaAs = np.genfromtxt(r'C:\nano_optics_ml_data\raw\GaAs_eff.csv',delimiter=',')
    eps_GaAs = GaAs[:,0] + 1j*GaAs[:,1]
    Sn_GaAs = GaAs[:,2] + 1j*GaAs[:,3]
    f_GaAs = GaAs[:,4]
    Sn_GaAs_cal_pd = 0.25*nf.farfieldfactor(f_GaAs,eps_GaAs)*nf.Sn_3(f_GaAs,eps_GaAs,z,a,t,n)
    Sn_GaAs_cal_fd = 0.25*nf.farfieldfactor(f_GaAs,eps_GaAs)*nf.Sn_2(f_GaAs,eps_GaAs,z,a,L,g,t,n)
    
    LSAT = np.genfromtxt(r'C:\nano_optics_ml_data\raw\LSAT_eff.csv',delimiter=',')
    eps_LSAT = LSAT[:,0] + 1j*LSAT[:,1]
    Sn_LSAT = LSAT[:,2] + 1j*LSAT[:,3]
    f_LSAT = LSAT[:,4]
    Sn_LSAT_cal_pd = 0.25*nf.farfieldfactor(f_LSAT,eps_LSAT)*nf.Sn_3(f_LSAT,eps_LSAT,z,a,t,n)
    Sn_LSAT_cal_fd = 0.25*nf.farfieldfactor(f_LSAT,eps_LSAT)*nf.Sn_2(f_LSAT,eps_LSAT,z,a,L,g,t,n)
    
    NGO = np.genfromtxt(r'C:\nano_optics_ml_data\raw\NGO_eff.csv',delimiter=',')
    eps_NGO = NGO[:,0] + 1j*NGO[:,1]
    Sn_NGO = NGO[:,2] + 1j*NGO[:,3]
    f_NGO = NGO[:,4]
    Sn_NGO_cal_pd = 0.25*nf.farfieldfactor(f_NGO,eps_NGO)*nf.Sn_3(f_NGO,eps_NGO,z,a,t,n)
    Sn_NGO_cal_fd = 0.25*nf.farfieldfactor(f_NGO,eps_NGO)*nf.Sn_2(f_NGO,eps_NGO,z,a,L,g,t,n)
    
    CaF2 = np.genfromtxt(r'C:\nano_optics_ml_data\raw\CaF2_eff.csv',delimiter=',')
    data = np.genfromtxt(r'C:\nano_optics_ml_data\raw\CaF2_nk.csv', delimiter = ',')
    f_CaF2 = CaF2[:,4]
    f = data[:,0]
    n_CaF2 = np.interp(f_CaF2, f, data[:,1])
    k_CaF2 = np.interp(f_CaF2, f, data[:,2])
    eps1, eps2 = nk_to_eps(n_CaF2,k_CaF2)
    eps_CaF2 = eps1 + 1j*eps2
    Sn_CaF2 = CaF2[:,2] + 1j*CaF2[:,3]
    Sn_CaF2_cal_pd = 0.25*nf.farfieldfactor(f_CaF2,eps_CaF2)*nf.Sn_3(f_CaF2,eps_CaF2,z,a,t,n)
    Sn_CaF2_cal_fd = 0.25*nf.farfieldfactor(f_CaF2,eps_CaF2)*nf.Sn_2(f_CaF2,eps_CaF2,z,a,L,g,t,n)
    
load_materials()


a=np.zeros((len(f_SiO2),7))
a[:,0] = f_SiO2
a[:,1] = np.real(Sn_SiO2_cal_fd)
a[:,2] = np.imag(Sn_SiO2_cal_fd)
a[:,3] = np.real(eps_SiO2)
a[:,4] = np.imag(eps_SiO2)
a[:,5] = np.real(Sn_SiO2)
a[:,6] = np.imag(Sn_SiO2)

a=np.zeros((len(f_STO),7))
a[:,0] = f_STO
a[:,1] = np.real(Sn_STO_cal_fd)
a[:,2] = np.imag(Sn_STO_cal_fd)
a[:,3] = np.real(eps_STO)
a[:,4] = np.imag(eps_STO)
a[:,5] = np.real(Sn_STO)
a[:,6] = np.imag(Sn_STO)

a=np.zeros((len(f_NGO),7))
a[:,0] = f_NGO
a[:,1] = np.real(Sn_NGO_cal_fd)
a[:,2] = np.imag(Sn_NGO_cal_fd)
a[:,3] = np.real(eps_NGO)
a[:,4] = np.imag(eps_NGO)
a[:,5] = np.real(Sn_NGO)
a[:,6] = np.imag(Sn_NGO)

a=np.zeros((len(f_CaF2),7))
a[:,0] = f_CaF2
a[:,1] = np.real(Sn_CaF2_cal_fd)
a[:,2] = np.imag(Sn_CaF2_cal_fd)
a[:,3] = np.real(eps_CaF2)
a[:,4] = np.imag(eps_CaF2)
a[:,5] = np.real(Sn_CaF2)
a[:,6] = np.imag(Sn_CaF2)

a=np.zeros((len(f_LSAT),7))
a[:,0] = f_LSAT
a[:,1] = np.real(Sn_LSAT_cal_fd)
a[:,2] = np.imag(Sn_LSAT_cal_fd)
a[:,3] = np.real(eps_LSAT)
a[:,4] = np.imag(eps_LSAT)
a[:,5] = np.real(Sn_LSAT)
a[:,6] = np.imag(Sn_LSAT)