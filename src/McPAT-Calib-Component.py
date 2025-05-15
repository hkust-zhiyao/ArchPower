import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from DataLoader import *
import os

def draw_figure(gt,pd,name):
    plt.clf()
        
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(6, 5))
    min_value = 10000
    max_value = 0
    for i in range(gt.shape[0]):
        pred_value = pd[i]
        label_value = gt[i]
        if pred_value>label_value:
            min_sample=label_value
            max_sample=pred_value
        else:
            max_sample=label_value
            min_sample=pred_value
        if max_value<max_sample:
            max_value=max_sample
        if min_value>min_sample:
            min_value=min_sample
    min_value = 0
    plt.plot([min_value,max_value],[min_value,max_value],color='silver')
        
    #plt.plot([0.4,1.6],[0.4,1.6],color='silver')
    color_set = ['b','g','r','c','m','y','k','skyblue','olive','gray','coral','gold','peru','pink','cyan','']
    # for i in range(gt.shape[0]//8):
    #     x = gt[i*8:(i+1)*8]
    #     y = pd[i*8:(i+1)*8]
    #     plt.scatter(x,y,marker='.',color=color_set[i],label="C{}".format(i),alpha=0.5,s=160)

    for i in range(gt.shape[0]//8):
        # ac_index = [j * 8 + i for j in range(gt.shape[0]//8)]
        # x = gt[ac_index]
        # y = pd[ac_index]
        x = gt[i*8:(i+1)*8]
        y = pd[i*8:(i+1)*8]
        plt.scatter(x,y,marker='.',color=color_set[i],label="C{}".format(i),alpha=0.5,s=160)


    # legend = plt.legend(fontsize=19, ncol=1, columnspacing=0.5, 
    #            labelspacing=0.4, borderaxespad=0.2, loc='upper left')
    # legend.get_frame().set_linewidth(5)

    plt.xlabel('Ground Truth (W)', fontsize=22)
    plt.ylabel('Prediction (W)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    r_report = np.corrcoef(gt,pd)[1][0]
    #r_report = r2_score(gt,pd)
    mape_report = mean_absolute_percentage_error(gt,pd)
    print(name)
    print("R = {}".format(r_report))
    print("MAPE = {}%".format(mape_report * 100))  
    #if 'train' not in name:
    #    result_store.append([name.split('/')[-1], r_report, mape_report * 100])
    
    plt.text(min_value, max_value - (max_value - min_value) / 7,"MAPE={:.1f}%\n".format(mape_report*100)+'R'+"={:.2f}".format(r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.savefig("../figure/{}_comp.png".format(name),dpi=200)
    plt.close()
    return

def load_data(uarch,power_group):
    feat_list = []
    for idx in range(len(comp)):
        comp_name = comp[idx]
        loaded_feat = np.load('../dataset/component_feature/{}.npy'.format(comp_name))
        feat_list.append(loaded_feat)
    feature = np.hstack(feat_list)

    label = np.load('../dataset/label.npy'.format(uarch))
    label = label.reshape((label.shape[0], 12, 5))
    label = label[:, :, power_group]

    if uarch == 'BOOM':
        feature = feature[0:120]
        label = label[0:120]
    else:
        feature = feature[120:]
        label = label[120:]

    return feature, label

def generate_training_test(feature, label_total, training_index, testing_index):
    training_set = []
    for idx in training_index:
        for i in range(8):
            training_set.append(idx * 8 + i)

    testing_set = []
    for idx in testing_index:
        for i in range(8):
            testing_set.append(idx * 8 + i)

    return feature[training_set], label_total[training_set], feature[testing_set], label_total[testing_set]

def build_logic_model(feature,label):
    model_list = []
    iter = 0
    for component in feature_of_components.keys():
        
        # get feature and label
        start_event = feature_of_components[component][0]
        end_event = feature_of_components[component][1]
        feature_index = [item for item in range(start_event,end_event)]
        label_index = iter + 1

        this_model = xgb.XGBRegressor()
        this_model.fit(feature[:,feature_index],label[:,label_index])
        model_list.append(this_model)
        iter = iter + 1
    return model_list

def test_logic_model(feature,label,model_list):
    pred_list = []
    iter = 0
    for component in feature_of_components.keys():
            
        # get feature and label
        start_event = feature_of_components[component][0]
        end_event = feature_of_components[component][1]
        feature_index = [item for item in range(start_event,end_event)]
        label_index = iter + 1
            
        # build model
        this_model = model_list[iter]
        logic_perd = this_model.predict(feature[:,feature_index])

        pred_list.append(logic_perd)
        iter = iter + 1
    
    power_value = np.zeros(feature.shape[0])
    for power in pred_list:
        power_value = power_value + power      
    return np.vstack([power_value] + pred_list).T


def known_n_config(training_index, testing_index, uarch, fname):
    feature, label_total = load_data(uarch, 0)

    train_feature, train_label_total, test_feature, test_label_total = generate_training_test(feature, label_total, training_index, testing_index)

    # print(train_feature.shape,train_label_total.shape, test_feature.shape, test_label_total.shape)
    # exit()

    logic_model = build_logic_model(train_feature, train_label_total)
    logic_pred = test_logic_model(test_feature, test_label_total, logic_model)

    for i in range(12):
        draw_figure(test_label_total[:,i],logic_pred[:,i],"McPAT_Calib_Component/{}_{}_".format(uarch,fname)+figure_name[i])
    
    return

known_n_config([0,7,14], [1,2,3,4,5,6,8,9,10,11,12,13], "BOOM", "evenly")
known_n_config([0,1,2], [3,4,5,6,7,8,9,10,11,12,13,14], "BOOM", "small")
known_n_config([12,13,14], [0,1,2,3,4,5,6,7,8,9,10,11], "BOOM", "large")

known_n_config([0,5,9], [1,2,3,4,6,7,8], "XS", "evenly")
known_n_config([0,1,2], [3,4,5,6,7,8,9], "XS", "small")
known_n_config([7,8,9], [0,1,2,3,4,5,6], "XS", "large")