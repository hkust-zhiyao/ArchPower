import numpy as np

comp = ["BP", "ICache", "IFU", "RNU", "LSU", "DCache", "Regfile", "ISU", "ROB", "FU-Pool", "Others"]
figure_name = ["Total", "BP", "ICache", "IFU", "RNU", "LSU", "DCache", "Regfile", "ISU", "ROB", "FU-Pool", "Others"]
feature_of_components = {}
encode_table = {
            "BP":[0],
            "ICache":[3,4],
            "DCache":[0,1],
            "ISU":[0],
            "Others":[1],
            "IFU":[1],
            "ROB":[1],
            "Regfile":[1,2],
            "RNU":[1],
            "D-TLB":[0],
            "LSU":[0]
        }

feat_total = np.load('../dataset/feature.npy')
cur_feat = feat_total
acc = 0
for idx in range(len(comp)):
    comp_name = comp[idx]
    loaded_feat = np.load('../dataset/component_feature/{}.npy'.format(comp_name))
    feature_of_components[comp_name] = [acc, acc + loaded_feat.shape[1]]
    acc = acc + loaded_feat.shape[1]

# print(feature_of_components)