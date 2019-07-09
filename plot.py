import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

#== parser start
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--job-id-max', type=int, default=1)
parser.add_argument('--result-path', type=str, default='./results/')
# experiment setting
parser.add_argument('--dataset', type=str, default='mnist') 
parser.add_argument('--data-aug', type=int, default=0) 
parser.add_argument('--model', type=str, default='LeNet') 
# method setting
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--ssize', type=int, default=64)
args = parser.parse_args()
#== parser end

#===============================================================
#=== preparation
#=============================================================== 

result_path = args.result_path
ID_name = args.dataset + '_' + str(args.data_aug) + '_' + args.model
load_path = result_path + ID_name
plot_path = result_path + 'plot/'
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)
plot_path += ID_name

load_path_0 = load_path + '_' + str(0) + '_' + str(args.batch_size) 
load_path_1 = load_path + '_' + str(1) + '_' + str(args.batch_size) + '_' + str(args.ssize) 

# dummy load to take shape
dumm = np.load(load_path_0 + '_' + str(1) + '_' + 'pl' + '.npy')
pl_result_0 = np.zeros((dumm.shape[0], dumm.shape[1], dumm.shape[2], args.job_id_max))
pl_result_1 = np.zeros((dumm.shape[0], dumm.shape[1], dumm.shape[2], args.job_id_max))
for i in range(args.job_id_max):
    pl_result_0[:,:,:,i]  = np.load(load_path_0 + '_' + str(i+1) + '_' + 'pl' + '.npy')
    pl_result_1[:,:,:,i]  = np.load(load_path_1 + '_' + str(i+1) + '_' + 'pl' + '.npy')
end_epoch = pl_result_0.shape[0]

test_err_0 = 100 - pl_result_0[:,1,1,:]
test_err_1 = 100 - pl_result_1[:,1,1,:]
train_loss_0 = pl_result_0[:,0,0,:]
train_loss_1 = pl_result_1[:,0,0,:]

# #===============================================================
# #=== test (mean, std), and time (mean, std), train (mean, std)
# #=============================================================== 

test_err_0_mean, test_err_0_std = np.mean(test_err_0, axis=1), np.std(test_err_0, axis=1)
test_err_1_mean, test_err_1_std = np.mean(test_err_1, axis=1), np.std(test_err_1, axis=1)

train_loss_0_mean, train_loss_0_std = np.mean(train_loss_0, axis=1), np.std(train_loss_0, axis=1)
train_loss_1_mean, train_loss_1_std = np.mean(train_loss_1, axis=1), np.std(train_loss_1, axis=1)

# #===============================================================
# #=== plot: test error and train loss
# #=============================================================== 
# Visualize the decision boundaries
# font = {'size':46}
# font = {'size':38}
# font = {'size':15}
# mpl.rc('font', **font)
    
start_epoch = 0
xstep = 10
if end_epoch > 100:
    xstep = 50
label_0 = r"mini-batch SGD"
label_1 = r"$q$-SGD" 
epoch = range(start_epoch, end_epoch)

xfmt_0 = mpl.ticker.LogFormatterMathtext()
xfmt_minor_0 = mpl.ticker.NullFormatter()  
xfmt_1 = mpl.ticker.LogFormatterMathtext()
xfmt_minor_1 = mpl.ticker.NullFormatter()
ystep_0 = 0.4
ystep_1 = 0.1 
locmaj_0 = mpl.ticker.LogLocator(base=10,numticks=12)
locmaj_1 = mpl.ticker.LogLocator(base=10,numticks=12)
locmin_0 = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.5,0.7),numticks=12)
locmin_1 = mpl.ticker.LogLocator(base=10.0,subs=np.arange(0, 1.0, step=ystep_1),numticks=12)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,10))
fig.subplots_adjust(hspace=0.05)

axs[0].plot(epoch, test_err_0_mean[start_epoch:end_epoch], 'c', label=label_0, linewidth=3)
axs[0].plot(epoch, test_err_1_mean[start_epoch:end_epoch], 'k--', label=label_1, linewidth=3)
axs[0].set_yscale('log')
axs[0].tick_params(axis='y', which='major', labelsize=46)
axs[0].tick_params(axis='y', which='minor', labelsize=28)
axs[0].yaxis.set_major_formatter(xfmt_0), axs[0].yaxis.set_minor_formatter(xfmt_minor_0)
axs[0].yaxis.set_major_locator(locmaj_0), axs[0].yaxis.set_minor_locator(locmin_0)
axs[0].legend()

axs[1].plot(epoch, train_loss_0_mean[start_epoch:end_epoch], 'c', label=label_0, linewidth=3)
axs[1].plot(epoch, train_loss_1_mean[start_epoch:end_epoch], 'k--', label=label_1, linewidth=3)
axs[1].set_xlabel('epoch')
axs[1].set_xlim(0, end_epoch)
axs[1].set_xticks(np.arange(0, end_epoch, step=xstep))
axs[1].set_yscale('log')
axs[1].tick_params(axis='y', which='major', labelsize=46)
axs[1].tick_params(axis='y', which='minor', labelsize=28)
axs[1].yaxis.set_major_formatter(xfmt_1), axs[1].yaxis.set_minor_formatter(xfmt_minor_1)
axs[1].yaxis.set_major_locator(locmaj_1), axs[1].yaxis.set_minor_locator(locmin_1)

fig.savefig(plot_path + '_ep' + '_plt.png', bbox_inches=mpl.transforms.Bbox([[-0.66, -0.3], [7.8, 9.1]]), pad_inches = 0.1)      
 
 

