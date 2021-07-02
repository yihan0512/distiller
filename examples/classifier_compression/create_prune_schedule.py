from distiller.utils import yaml_ordered_save, yaml_ordered_load
import ipdb
import argparse
import os

def read_prune_schedule(sdict):
    sched = []
    pruners = list(sdict['pruners'].keys())
    for pruner in pruners:
        rate = sdict['pruners'][pruner]['desired_sparsity']
        sched.append(rate)
    return sched


def write_prune_schedule(sdict, sched):
    pruners = list(sdict['pruners'].keys())
    for i in range(len(sched)):
        sdict['pruners'][pruners[i]]['desired_sparsity'] = sched[i]
    return sdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate the detectors')
    parser.add_argument('-f', '--file', type=str, help='yaml file')
    args = parser.parse_args()

    with open(args.file, 'r') as stream:
        sched_dict = yaml_ordered_load(stream)

    schedule = read_prune_schedule(sched_dict)

    count = 0
    while schedule[-1] < 0.9:
        schedule1 = []
        for rate in schedule:
            if rate < 0.9:
                schedule1.append(rate+0.1)
            else:
                schedule1.append(rate)
        new_sched_dict = write_prune_schedule(sched_dict, schedule1)

        schedule_dir = '../extended_doublenet_search/' 
        os.makedirs(schedule_dir, exist_ok=True)
        yaml_ordered_save(os.path.join(schedule_dir, 'filter_rank.'+str(count)+'.yaml'), new_sched_dict)
        schedule = schedule1
        count += 1

