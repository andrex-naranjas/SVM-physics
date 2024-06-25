"""
-----------------------------------------------------
 Authors: A. Ramirez-Morales
-----------------------------------------------------
"""
import os
import sys
import pandas as pd
import datetime
# framework includes
from common import model_maker as mm
from common import stats_summary as ss


process = int(sys.argv[1])       # batch process
name = str(sys.argv[2])          # sample name
balance_name = str(sys.argv[2])  # balance name
path = str(sys.argv[3])          # path where code lives
boot_kfold = str(sys.argv[4])    # use bootstrap or kfold
exotic_single = str(sys.argv[5]) # use exotic or standard classifiers

model_auc = mm.model_loader_batch(process, exotic_single=exotic_single)[1]
model_auc_names = mm.model_loader_batch(process, exotic_single=exotic_single)[0]
n_cycles = 10
k_folds  = 5
n_reps   = 10
roc_area = "deci"

if model_auc[2] == "absv":
    roc_area = "absv"

print('sample:', name, 'model name:', model_auc[0], '  validation', boot_kfold)
start = datetime.datetime.now()

# auc, prc, f1, rec, acc, gmn, time, n_class, n_train
auc, prec_pos, prec_neg, acc, time, n_class, n_train = ss.cross_validation(sample_name="DY",
                                                                          balance_name=balance_name,
                                                                          model=model_auc[1],
                                                                          is_precom=model_auc[2],
                                                                          kernel_fcn=model_auc[3],
                                                                          roc_area=roc_area,
                                                                          selection=model_auc[3+1],
                                                                          GA_mut=model_auc[4+1],
                                                                          GA_score=model_auc[5+1],
                                                                          GA_selec=model_auc[6+1],
                                                                          GA_coef=model_auc[7+1],
                                                                          kfolds=k_folds, n_reps=n_reps, path=path)

col_auc      = pd.DataFrame(data=auc,       columns=["auc"])
col_prec_pos = pd.DataFrame(data=prec_pos,  columns=["prec_pos"])
col_prec_neg = pd.DataFrame(data=prec_neg,  columns=["prec_neg"])
col_acc      = pd.DataFrame(data=acc,       columns=["acc"])
col_time     = pd.DataFrame(data=time,      columns=["time"])
col_base     = pd.DataFrame(data=n_class,   columns=["n_base"])
col_size     = pd.DataFrame(data=n_train,   columns=["n_train"])

df = pd.concat([col_auc["auc"], col_prec_pos["prec_pos"], col_prec_neg["prec_neg"], col_acc["acc"], col_time["time"], col_base["n_base"], col_size["n_train"]],
               axis=1, keys=["auc", "prec_pos", "prec_neg", "acc", "time", "n_base", "n_train"])

dir_name_csv = path + "/results/stats_results/"+name+"/"+boot_kfold+"/"

if not os.path.exists(dir_name_csv):
    os.makedirs(dir_name_csv)
    
name_csv = dir_name_csv + model_auc[0]+"_"+boot_kfold+".csv" 
df.to_csv(str(name_csv), index=False)

end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed time = " + str(elapsed_time))
print(model_auc[0], name)
print(df)
print('All names')
for i in range(len(model_auc_names)):
    print(model_auc_names[i][0])
