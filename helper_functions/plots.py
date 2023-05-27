import numpy as np
import matplotlib.pyplot as plt
import torch
#plt.rcParams.update({'font.size': 12})
def plot_results(results):
    cost_avg = results['cost_avg']
    UVW_mse_avg = results['UVW_mse_avg']
    p_mre_avg = results['p_mre_avg']
    lambda_mre_avg = results['lambda_mre_avg']
    timestamps_avg = results['timestamps_avg']
    plot_dir = results['plot_dir']

    # Plot
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.plot(timestamps_avg,cost_avg,'go--', linewidth=2, markersize=5,markevery=5)
    ax.set_xlabel('Time (seconds)')
    ax.legend(['Proposed'],loc='best')
    ax.set_ylabel('Cost')
    ax.grid(True)  
    plt.savefig(plot_dir+'cost.png')

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.semilogy(timestamps_avg,UVW_mse_avg,'bo--', linewidth=2, markersize=5,markevery=5)
    ax.set_xlabel('Time (seconds)')
    ax.legend(['Proposed'],loc='best')
    ax.set_ylabel('MSE')
    ax.grid(True)  
    plt.savefig(plot_dir+'UVW_mse.png')

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.semilogy(timestamps_avg,p_mre_avg,'ro--', linewidth=2, markersize=5,markevery=5)
    ax.set_xlabel('Time (seconds)')
    ax.legend(['Proposed'],loc='best')
    ax.set_ylabel("MRE of $p_{ijk}$s")
    ax.grid(True)  
    plt.savefig(plot_dir+'p_mre.png')

    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.semilogy(timestamps_avg,lambda_mre_avg,'mo--', linewidth=2, markersize=5,markevery=5)
    ax.set_xlabel('Time (seconds)')
    ax.legend(['Proposed'],loc='best')
    ax.set_ylabel('MRE of $\lambda_{ijk}$s')
    ax.grid(True)  
    plt.savefig(plot_dir+'lambda_mre.png')