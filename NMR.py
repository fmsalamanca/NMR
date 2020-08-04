import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import os.path
import time as t
import shutil
import subprocess


def deffects2(x,A,tau,b,A1,tau1,b1):
    return A*np.exp(-np.power(x/tau,b))+A1*np.exp(-np.power(x/tau1,b1))

def deffects(x,A,tau,b):
     return deffects2(x,A,tau,b,A1=0,tau1=1,b1=1)   
ID       = "DS_SIGNUS_DQ_rd067_bis_80C.txt"
path     = "/home/fernando/Papers/Devulcanization/"
#path     = "/home/fernando/Papers/Fillers/BIRLACARBON2019/"
thispath = "/home/fernando/Papers/NMRsoftware/"
name     = "_DQ_rd06__80C"
files    = str(ID)+name+".txt"
filename = path+files
filename = path+ID
data     = np.loadtxt(filename)

time = data[:,0]
Iref = data[:,1]
IDQ  = data[:,2]
del data
test = True
if test:
    fig1 = plt.figure()
    plt.plot(time,Iref,'ks',label='Iref')
    plt.plot(time,IDQ,'rs',label='IDQ')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Time (ms)')
    plt.ylabel('I (a.u.)')
    plt.title('Not normalized NMR intensity signal')
    plt.legend()
    
    plt.show()

IDQ  = IDQ/Iref[0]
Iref = Iref/Iref[0]

if test:
    fig2 = plt.figure()
    plt.plot(time,Iref,'ks',label='Iref')
    plt.plot(time,IDQ,'rs',label='IDQ')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Time (ms)')
    plt.ylabel('I')
    plt.title('Normalized NMR intensity signal')
    plt.legend()
    
    plt.show()

IsumDQ = Iref + IDQ
IsubDQ = Iref - IDQ

if test:
    fig3 = plt.figure()
    ax = fig3.add_subplot(1,1,1)
    ax.set_yscale('log')
    plt.plot(time,IsubDQ,'ks',label='IsubDQ')
    plt.xlim(left=0.01,right=60)
    plt.ylim(bottom=0.01,top=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('IsubDQ')
    plt.title('Normalized Iref - IDQ NMR intensity signal')
    plt.legend()
    
    plt.show()
if test:
    deffectsdouble = bool(input('Press Enter to use Single Defect Model or type anything to use Double Defect Model. '))
if not test:
    deffectsdouble = False
if test:
    primetime = float(input('Choose the first point where the regression will take place. Time = '))
if not test:
    primetime = 8.5

if deffectsdouble:
    print('You chose double!')
    params, covar = curve_fit(deffects2, time[time>primetime], IsubDQ[time>primetime])
else:
    params, covar = curve_fit(deffects, time[time>primetime], IsubDQ[time>primetime])
    print('You chose single!')
A       = params[0]
tau     = params[1]
b       = params[2]
Aerr    = covar[0,0]
tauerr  = covar[1,1]
berr    = covar[2,2]

if deffectsdouble:
    A1      = params[3]
    tau1    = params[4]
    b1      = params[5]
    A1err   = covar[3,3]
    tau1err = covar[4,4]
    b1err   = covar[5,5]
else:
    A1      = 0
    A1err   = 0

if test:
    fig4 = plt.figure()
    ax = fig4.add_subplot(1,1,1)
    ax.set_yscale('log')
    plt.plot(time[time>primetime],IsubDQ[time>primetime],'ks',label='IsubDQ')
    if deffectsdouble:
        plt.plot(time[time>primetime],deffects2(time[time>primetime],A,tau,b,A1,tau1,b1),'r',label='Model')
    else:
        plt.plot(time[time>primetime],deffects(time[time>primetime],A,tau,b),'r',label='Model')
    plt.xlim(right=60)
    plt.xlabel('Time (ms)')
    plt.ylabel('IsubDQ')
    plt.title('Normalized deffects NMR intensity signal')
    if deffectsdouble:
        plt.text(10,0.00011,r"$A = %f , %f \pm %f , %f$" % (A,A1,Aerr,A1err),{'fontsize': 15})
        plt.text(10,0.00008,r"$\tau = %f , %f \pm %f , %f$" % (tau,tau1,tauerr,tau1err),{'fontsize': 15})
        plt.text(10,0.00006,r"$b = %f , %f \pm %f , %f$" % (b,b1,berr,b1err),{'fontsize': 15})
        plt.text(10,0.0002,r"$I_{deff} = A\cdot e^{-\left(\frac{t}{\tau}\right)^b} + A_1\cdot e^{-\left(\frac{t}{\tau_1}\right)^{b_1}}$",{'fontsize': 25})
    else:
        plt.text(10,0.00011,r"$A = %f \pm %f$" % (A,Aerr),{'fontsize': 15})
        plt.text(10,0.00008,r"$\tau = %f \pm %f$" % (tau,tauerr),{'fontsize': 15})
        plt.text(10,0.00006,r"$b = %f \pm %f$" % (b,berr),{'fontsize': 15})
        plt.text(10,0.0002,r'$I_{deff} = A\cdot e^{-\left(\frac{t}{\tau}\right)^b}$',{'fontsize': 25})
    plt.legend()
    
    plt.show()

if deffectsdouble:
    IsDQ = IsumDQ - deffects2(time,A,tau,b,A1,tau1,b1)
else:
    IsDQ = IsumDQ - deffects(time,A,tau,b)
InDQ = IDQ/IsDQ

maxindex = np.where(InDQ == max(InDQ[(time<10) & (InDQ<0.6) & (InDQ>0)]))[0]
cutpoint = InDQ[maxindex-1]
cutpoint = InDQ[42]
print('Cut point for Fast Tikhonov regularization: %f' % cutpoint)
#print('Covariance: %f' % (1-np.linalg.det(covar)))
if test:
    fig5 = plt.figure()
    plt.plot(time,InDQ,'ks',label='InDQ')
    plt.axhline(0.5,c='red')
    plt.xlim(left=0,right=20)
    plt.ylim(bottom=0,top=0.6)
    plt.xlabel('Time (ms)')
    plt.ylabel('InDQ')
    plt.title('Normalized Multiple Quantum NMR intensity signal')
    plt.legend(loc='upper right')
    
    plt.show()

np.savetxt('ftikreg.dat',np.transpose([time,InDQ]))
print('New ftikreg.dat file was created succesfully.')

pars = open('ftikreg.par').read().splitlines()
pars[0] = files + '     FILENAME'
print('ftikreg.par FILENAME was modified succesfully.')
pars[6] = str(cutpoint) + '       CUT  (0.45 is recommended for Gauss Kernel) '
print('ftikreg.par CUTPOINT was modified succesfully.')
open('ftikreg.par','w').write('\n'.join(pars))

if test:
    print('Now running ftikreg.exe: ')
    #shutil.move("/home/fernando/Papers/NMRsoftware/ftikreg.dat","/home/fernando/")
    #shutil.move("/home/fernando/Papers/NMRsoftware/ftikreg.par","/home/fernando/")
    subprocess.Popen("wine /home/fernando/Papers/NMRsoftware/ftikreg_2.00.exe",shell=True)

    

time_to_wait = 8
time_counter = 0
while not os.path.exists(str(ID)+name+".txt_avg"):
        t.sleep(1)
        time_counter += 1
        if time_counter > time_to_wait:break
print('Done!')



avg     = np.loadtxt(str(ID)+name+'.txt'+'_avg'+'.dat')
bupc    = np.loadtxt(str(ID)+name+'.txt'+'_bupc'+'.dat')
chi     = np.loadtxt(str(ID)+name+'.txt'+'_chi'+'.dat')
distrib = np.loadtxt(str(ID)+name+'.txt'+'_distrib'+'.dat')
var     = np.loadtxt(str(ID)+name+'.txt'+'_var'+'.dat')

if test:
    fig6, ax6 = plt.subplots()
    ax6.loglog(chi[:,0],chi[:,1],'ks',label=r'$\chi^2 distributions$')
    vec = np.linspace(1,len(chi[:,0]),num=len(chi[:,0]))
    for i, txt in enumerate(vec):
        ax6.annotate(int(txt), (chi[i,0],chi[i,1]+0.00001))
    plt.ylabel(r'$\chi^2$')
    plt.title('Best fit')
    plt.legend(loc='upper right')
    
    plt.show()
if test:
    chivalue = int(input('Which distribution do you want to work with? It is recommended not to take one living in the plateau. '))
if not test:
    chivalue = 10
avgvalue = avg[len(avg[:,chivalue])-1,chivalue]*1000 #Dres in Hz
varvalue = var[len(var[:,chivalue])-1,chivalue]*1000 #err in Hz
relerr   = varvalue/avgvalue #relative error in %

np.savetxt('bestdistribution_'+str(ID)+'.dat',np.transpose([distrib[:,0],distrib[:,chivalue]]))

if test:
    fig7 = plt.figure()
    plt.plot(distrib[:,0],distrib[:,chivalue],'k')
    plt.xlim(left=0,right=1)
    plt.ylim(bottom=0)
    plt.xlabel(r'$D_{res} (kHz)$')
    plt.ylabel('Frequency')
    plt.title(r'$D_{res}$'+' best distribution')
    plt.text(0.55,7,r'$D_{res} = %.3f \pm %.3f\,\,(Hz)$' % (avgvalue,varvalue))
    plt.text(0.55,6,r'$\delta = $'+str(np.round(relerr,3)))
    plt.text(0.55,5,r'$\Phi_{deff} =$'+r'$ %f \pm %f$' % (np.round((A+A1)*100,3),np.round((Aerr+A1err)*100,3)) + ' %')
    plt.show()

if test:
    fig8 = plt.figure()
    plt.plot(time,InDQ,'ks',label='InDQ')
    plt.plot(time,bupc[:,chivalue],'bo',label='InDQ bupc')
    plt.axhline(0.5,c='red')
    plt.xlim(left=0,right=20)
    plt.ylim(bottom=0,top=0.6)
    plt.xlabel('Time (ms)')
    plt.ylabel('InDQ')
    plt.title('Normalized Multiple Quantum NMR intensity signal')
    plt.legend(loc='upper right')
    
    plt.show()

dirdest = path+str(ID)+name
if not os.path.exists(dirdest):
        os.makedirs(dirdest)
else:
    raise('You should change the directory name for this analysis')


if test:
    shutil.move(filename,dirdest)
    shutil.move(str(ID)+name+'.txt'+'_avg'+'.dat',dirdest)
    shutil.move(str(ID)+name+'.txt'+'_bupc'+'.dat',dirdest)
    shutil.move(str(ID)+name+'.txt'+'_chi'+'.dat',dirdest)
    shutil.move(str(ID)+name+'.txt'+'_distrib'+'.dat',dirdest)
    shutil.move(str(ID)+name+'.txt'+'_var'+'.dat',dirdest)
    shutil.move('bestdistribution_'+str(ID)+'.dat',dirdest)
print('Files moved.')
if test:
    fig1.savefig(dirdest+'/INOTNOR_'+str(ID)+'.pdf')
    fig2.savefig(dirdest+'/INOR_'+str(ID)+'.pdf')
    fig3.savefig(dirdest+'/Isub_'+str(ID)+'.pdf')
    fig4.savefig(dirdest+'/Ireg_'+str(ID)+'.pdf')
    fig5.savefig(dirdest+'/InDQ_'+str(ID)+'.pdf')
    fig6.savefig(dirdest+'/Chisq_'+str(ID)+'.pdf')
    fig7.savefig(dirdest+'/Dist_'+str(ID)+'.pdf')
    fig8.savefig(dirdest+'/InDQtotal_'+str(ID)+'.pdf')
print('Figures saved.')