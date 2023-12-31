"""
-----------------------------------------------------------------------------------
 Authors: A. Ramirez-Morales
 htcondor job submitter
 usage: python3 ./scripts/submit_stats_svm.py sample_name boot/kfold exotic/single
-----------------------------------------------------------------------------------
"""
from sys import argv
from os import system,getenv,getuid,getcwd,popen
from common import model_maker as mm

workpath = getcwd()

if(len(argv)<=3):
    sample_name = 'titanic'
    boot_kfold = 'boot'
elif(len(argv)==4):
    sample_name  = argv[1]
    boot_kfold   = argv[2]
    exotic_single = argv[3]

if exotic_single=='exotic':
    n_flavors = len(mm.model_flavors_exotic())
elif exotic_single=='single':
    n_flavors = len(mm.model_flavors_single())

py3_path = popen('which python3').read().strip()

classad='''
universe = vanilla
executable = {0}
getenv = True
arguments = {1}/scripts/batch_stats_summary.py $(Process) {2} {1} {3} {4}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
request_memory = 2048
output = {1}/output_batch/{2}/$(Process).out
error = {1}/output_batch/{2}/$(Process).err
log = {1}/output_batch/{2}/$(Process).log
Queue {5}
'''.format(py3_path, workpath, sample_name, boot_kfold, exotic_single, n_flavors)

logpath = '.'

with open(logpath+'/condor.jdl','w') as jdlfile:
  jdlfile.write(classad)

print("************************ Batch jobs for: ", sample_name, "and stats:", boot_kfold, "************************")
system('condor_submit %s/condor.jdl'%logpath)
