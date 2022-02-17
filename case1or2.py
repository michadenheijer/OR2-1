# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:24:04 2022

@author: Danib
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def appointment_process(patients,arrival,consulting,duetime):
    waiting = np.zeros(patients)
    
    
    interarrival = np.zeros(patients-1)
    for i in range(0,patients-1):     
        interarrival[i] = arrival[i+1]-arrival[i]
    
    waiting = np.zeros(patients)
    waiting[0] = -1*min(arrival[0],0)
    
    for i in range(1,patients):
        waiting[i] = max((waiting[i-1]+consulting[i-1]-interarrival[i-1]),0)
              
    
    if(arrival[patients-1] + waiting[patients-1] + consulting[patients-1] <= 60):
        overtime = 0
    else:
        overtime = arrival[patients-1] + waiting[patients-1] + consulting[patients-1] - duetime

    return np.sum(waiting) + overtime

def case1b(consult,arrivals):
    paramconsult = stats.gamma.fit(consult)
    
    x = np.linspace(0,np.amax(consult))
    pdf_fit = stats.gamma.pdf(x,*paramconsult)
    #plt.plot(x, pdf_fit, color='r')
    #plt.hist(consult,density=True, bins=30)
    #plt.show()
    
    
    paramarrivals = stats.triang.fit(arrivals)
    x = np.linspace(0,np.amax(arrivals))
    pdf_fit = stats.triang.pdf(x,*paramarrivals)
    plt.plot(x, pdf_fit, color='r')
    plt.hist(consult,density=True,bins=30)
    plt.show()



def case1c(patients,theta):
    arrivalsample = stats.triang.rvs(size=patients)
    print(arrivalsample)
    consultsample = stats.gamma.rvs(size=patients)
    print(consultsample)
    meanconsult = np.mean(consultsample)
    schedule = np.zeros(patients.size)
    
    for i in range(1,schedule.size - 1):
        schedule[i] = i*theta*meanconsult
    

def main():     
    patients = 5
    arrival = np.array([-1,13,34,42,65])
    consulting = np.array([17,12,25,11,22])
    duetime = 90
    output = appointment_process(patients,arrival,consulting,duetime)
    
    consult = np.genfromtxt("consult_times.txt")
    arrivals = np.genfromtxt("delta_arrivals.txt")
    
    theta = 1
    patients = 18
    schedule = case1c(patients,theta)
    
    
    
    
if __name__ == "__main__":
    main()
