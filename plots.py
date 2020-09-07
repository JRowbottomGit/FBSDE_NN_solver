import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
#https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/

path = "./output_colab_Tanh20kdecstep/"
# path = ""
#
# fig_id = "Torch_Tanh_20k_0_001Step_FC"
fig_id = "Jax_Tanh_20k_DecStep_FC"
# fig_id = "Jax_Tanh_20k_DecStep_FC_T10x"
# fig_id = "Jax_Tanh_20k_DecStep_FC_NoTDG"
# fig_id = "Jax_Tanh_20k_DecStep_FC_params_YtildeFree"
# fig_id = "Jax_Tanh_20k_DecStep_FC_shallow"
# fig_id = "Jax_Tanh_20k_DecStep_FC_N200"
# fig_id = "Jax_Tanh_20k_DecStep_FC_Trainingsnaps"
# fig_id = "Jax_Tanh_20k_0_001Step_FC"
# fig_id = "Jax_ReLU_20k_DecStep_FC_Params"
# fig_id = "Jax_Tanh_20k_DecStep_FC_params_pertX0"
# fig_id = "Jax_Euler_Tanh_DecStep"
# fig_id = "Jax_Euler_Tanh_DecStep_50"
# fig_id = "Jax_Ito_Tanh_DecStep_50"
# fig_id = "Jax_Ito_Tanh_DecStep_50_HVP"
# fig_id = "Jax_RNNGRU_Tanh_h50_H0"
# fig_id = "Jax_RNNGRU_Tanh_h50"
Library = "Jax"
Activation = "Sigmoid"
LR = "Dec Step"
Walltime = "2138.10"

t_test = np.load(path + 't_test.npy')    ###drop for pytorch
W_test = np.load(path + 'W_test.npy')
# t_plot = np.load(path + 't_test.npy')     ###show for pytorch
t_plot = np.load(path + 't_plot.npy')    ###drop for pytorch
X_pred = np.load(path + 'X_pred.npy')
Y_pred = np.load(path + 'Y_pred.npy')
Y_tilde_pred = np.load(path + 'Y_tilde_pred.npy')    ###drop for pytorch
Y_test = np.load(path + 'Y_test.npy')
graph = np.load(path + 'graph.npy')
# DYDT = np.load(path + 'DYDT_test.npy')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
JR_SIZE = 30
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def update_xlabels(ax):
    xlabels = [format(label, ',.0f') for label in ax.get_xticks()]
    ax.set_xticklabels(xlabels)

#Training Loss
fig = plt.figure()
# ax = plt.subplot()
plt.plot(graph[0], graph[1])
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.yscale("log")
plt.title('Evolution of the Training Loss')
plt.savefig(path + fig_id+"TrainLoss.pdf", bbox_inches='tight')
axx = fig.axes[0]
update_xlabels(axx)
# axx.get_xaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()

plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

# plot X path
# plt.figure()
# plt.plot(t_plot[0,:,0], X_pred[2,:,0]) #1st dimension of X
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('X path sample')
# plt.legend()
# plt.savefig("Jax_X_path_sample")
# plt.show()

#Y predicted and exact - samples
samples = 4
plt.figure()
for i in range(samples):
    if i == 0:
        mylabel = 'Learned $u(t,X_t)$'
    else:
        mylabel = "_Hidden label"
    plt.plot(t_plot[i, :, 0], Y_pred[i, :, 0], 'b', label=mylabel)

    if i == 0:
        mylabel = 'Exact $u(t,X_t)$'
    else:
        mylabel = "_Hidden label"
    plt.plot(t_plot[i, :, 0], Y_test[i, :, 0], 'r--', label=mylabel)

    # plt.plot(t_plot[i, :, 0], Y_pred[i, :, 0], 'b', label='Learned $u(t,X_t)$')
    # plt.plot(t_plot[i, :, 0], Y_test[i, :, 0], 'r--', label='Exact $u(t,X_t)$')

plt.xlabel('Time')
plt.ylabel('Y Value')
plt.title('Y Paths')
plt.legend()
plt.savefig(path + fig_id + "Y_pred_Y_test_paths.pdf", bbox_inches='tight')
plt.show()

# plot DYDT path
# samples = 10
# plt.figure()
# # plt.plot(t_plot[0,:,0], DYDT[0,:,0])
# for i in range(samples):
#     plt.plot(t_plot[i, :, 0], DYDT[i,:,0])#, 'b', label='Learned $u(t,X_t)$')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('DYDT path sample')
# plt.legend()
# plt.savefig("DYDT")
# plt.show()

#Average over test batch
plt.figure()
Y_pred_mean = np.mean(Y_pred, 0)
Y_test_mean = np.mean(Y_test, 0)
plt.plot(t_plot[0, :, 0], Y_pred_mean[:, 0], 'b', label='Learned $u(t,X_t)$')
plt.plot(t_plot[0, :, 0], Y_test_mean[:, 0], 'r--', label='Exact $u(t,X_t)$')
plt.title('Predicted Y against Analytical Solution')
plt.legend()
plt.savefig(path + fig_id + "_average_Y_pred_Y_test.pdf", bbox_inches='tight')
plt.show()

#Test the Loss error versus the training error
plt.figure()
Y_pred_mean = np.mean(Y_pred, 0)
Y_test_mean = np.mean(Y_test, 0)
Y_tilde_mean = np.mean(Y_tilde_pred, 0)    ###drop for pytorch
plt.plot(t_plot[0, 1:, 0], Y_pred_mean[1:, 0], 'b', label='Learned $u(t,X_t)$')
plt.plot(t_plot[0, 1:, 0], Y_test_mean[1:, 0], 'r--', label='Exact $u(t,X_t)$')
plt.plot(t_plot[0, 1:, 0], Y_tilde_mean[1:, 0], 'g--', label='Tilde $u(t,X_t)$')    ###drop for pytorch
plt.legend()
plt.title('Predicted Y and Y_tilde against Analytical Solution')
plt.savefig(path + fig_id + "_average_Y_pred_Y_test_Y_tilde.pdf", bbox_inches='tight')
plt.show()

#mean error
errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
# errors = np.sqrt((Y_test_mean - Y_pred_mean) ** 2 / Y_test ** 2)

mean_errors = np.mean(errors, 0)
std_errors = np.std(errors, 0)
plt.figure()
plt.plot(t_plot[0, :, 0], mean_errors, 'b', label='Test Mean Error')
plt.plot(t_plot[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='Test Mean Error + 2 s.d.')
D=100
plt.xlabel('$t$')
plt.ylabel('Relative Error')
# plt.title('Jax' + str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
plt.legend()
plt.title("Relative Test Error against Analytical Solution")
# plt.savefig('Jax' + str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
plt.savefig(path + fig_id + "_relative_error.pdf", bbox_inches='tight')
plt.show()

# cwd = os.getcwd()
# print(cwd)
# text_file = open("JRJaxVec_Output.txt", "w")
# text_file.write(f"where is this file\nhere: {cwd}")
# text_file.close()

Title = fig_id
LN = graph[1,-1]
Y0 = Y_pred[0,0,0]
Y0error = mean_errors[0,0]
Y05error = mean_errors[26,0] #[101,0]#[6,0]#[26,0]
Y1error = mean_errors[-1,0]

table_headers = ["Library","Activation","LR","Final Loss","Y0","Training Time","Y0_error", "Y0.5_error", "Y1_error"]
table_list = [Library,Activation,LR, LN, Y0, Walltime, Y0error, Y05error, Y1error]
# table_headers = ["Experiment","Final Loss","Y0","Training Time","Y0_error", "Y0.5_error", "Y1_error"]
# table_list = [Title, LN, Y0, Walltime, Y0error, Y05error, Y1error]

import pandas as pd
output_df = pd.DataFrame([table_list], columns = table_headers)

save_path = path + "results_table.xlsx"
with pd.ExcelWriter(save_path) as writer:
    output_df.to_excel(writer, sheet_name='results_table', index=False)
#
#
# mean_errors_df = pd.DataFrame([mean_errors], columns = table_headers)
Y_pred_df = pd.DataFrame(Y_pred[:,:,0])
Y_test_df = pd.DataFrame(Y_test[:,:,0])
#
save_path = path + "mean_errors.xlsx"
with pd.ExcelWriter(save_path) as writer:
    Y_pred_df.to_excel(writer, sheet_name='Y_pred', index=False)
    Y_test_df.to_excel(writer, sheet_name='Y_test', index=False)
