#!/usr/bin/python3.6

import configparser
import os
import argparse
import time
import jinja2
import easydict
import subprocess

import local_id_db_utils as db


CLUSTER_STR = '''
clusters:
    idc:
        appid: YzbVqHmyte
        appkey: LRtqPlkPhlKdPaSURFVA
        endpoint: idc-train
        hdfs: hdfs://hobot-bigdata/
    idcsmall:
        appid: rflpcUpMEJ
        appkey: rvOHShPbASQCReoMoqJC
        endpoint: idc-train
        hdfs: hdfs://hobot-bigdata/
    idcv2:
        appid: YzbVqHmyte
        appkey: LRtqPlkPhlKdPaSURFVA
        endpoint: idc-v2
        hdfs: hdfs://hobot-bigdata-ucloud/
    idcv2small:
        appid: rflpcUpMEJ
        appkey: rvOHShPbASQCReoMoqJC
        endpoint: idc-v2
        hdfs: hdfs://hobot-bigdata-ucloud/
    aliyun:
        appid: KlDOoYHgoZ
        appkey: coRQRelanvAtBkvRsSXH
        endpoint: aliyun-v2
        hdfs: hdfs://hobot-bigdata-ucloud/
'''

CLUSTER_TO_J2FILE = {
    '1': {
        'submit_scripts': ('py_submit.sh.qsub.j2',),
        'run_scripts': ('py_job.sh.qsub.j2',),
    },
    'idc': {
        'submit_scripts': ('py_submit.sh.idc.j2', 'py_job.yaml.j2'),
        'run_scripts': ('py_job.sh.idc.j2',),
    },
    'idcv2': {
        'submit_scripts': ('py_submit.sh.idc.j2', 'py_job.yaml.j2'),
        'run_scripts': ('py_job.sh.idc.j2',),
    },
    'ksyun': {
        'submit_scripts': ('py_submit.sh.idc.j2', 'py_job.yaml.j2'),
        'run_scripts': ('py_job.sh.idc.j2',),
    },
    'aliyun': {
        'submit_scripts': ('py_submit.sh.idc.j2', 'py_job.yaml.j2'),
        'run_scripts': ('py_job.sh.idc.j2',),
    },
}

BASE_J2FILE_REQIRED_VARS = {
    'upload_dir': 'FILE_FOLDER',
    'pods': 'PODS',
    'priority': 'PRIORITY',
    'walltime': 'WALLTIME',
    'docker_image': 'DOCKER_IMAGE',
    'times': 'REAP_RUN',

    # created online
    'job_name': 'JOB_NAME',
    'job_script': 'JOB_SCRIPT',
    'scripts': 'SCRIPTS',
}
J2FILE_REQIRED_VARS = {
    '1': {
    },
    'idc': {
        'worker': 'WORKER',
    },
    'idcv2': {
        'worker': 'WORKER',
    },
    'ksyun': {
        'worker': 'WORKER',
    },
    'aliyun': {
        'worker': 'WORKER',
    },
}
for k, v in J2FILE_REQIRED_VARS.items():
    J2FILE_REQIRED_VARS[k] = {**BASE_J2FILE_REQIRED_VARS, **v}

CLUSTER_SPECIFIC_PARA = {
    '1': {
        'tensorboard_log_dir': './logs/'
    },
    'idc': {
        # 'tensorboard_log_dir': '/job_tboard/'
    },
    'idcv2': {
        # 'tensorboard_log_dir': '/job_tboard/'
    },
    'ksyun': {
        # 'tensorboard_log_dir': '/job_tboard/'
    },
    'aliyun': {
        # 'tensorboard_log_dir': '/job_tboard/'
    },
}

STR_REPLACE = {
    '1': [
        lambda x: x.replace('hobot-bigdata', 'hobot-bigdata-mos'),
        lambda x: x.replace('conda_lib9_torch10', 'conda_lib8_torch10'),
    ],
    'idc': [],
    'idcv2': [],
    'ksyun': [
        lambda x: x.replace('hobot-bigdata', 'ksbigdata'),
    ],
    'aliyun': [
        # lambda x: x.replace('hobot-bigdata', 'hobot-bigdata-aliyun'),
    ],
}


def convert_multi_line(ini_file):
    start_convert = False
    all_lines = ini_file.read().split('\n')
    merge_idx = []
    for idx, line in enumerate(all_lines):
        if '\\' in line and not start_convert:
            merge_idx.append(idx)
            start_convert = True
        if '\\' not in line and start_convert:
            merge_idx.append(idx)
            start_convert = False
    for s, e in zip(merge_idx[0::2], merge_idx[1::2]):
        merge_str = ' && '.join(
            [i for i in all_lines[s+1:e+1] if not i.startswith(';')]
        )
        all_lines[s] = all_lines[s] + merge_str
        all_lines[s] = all_lines[s].replace('\\', '')
        all_lines[s+1:e+1] = ''
    return '\n'.join(all_lines)


def multiple_submit(value_configs):
    existing = 0
    for key, cfg in value_configs.items():
        if '{{ ' in cfg and ' }}' in cfg:
            existing += 1
            exist_key = key
            config_generator = eval(cfg.split('{{ ')[1].split(' }}')[0])
        if existing > 1:
            raise RuntimeError('Only tuning one param supported')
    v_dict = dict(value_configs)
    if existing == 0:
        exist_key = ''
        config_generator = ('',)
    for iter_value in config_generator:
        if type(iter_value) == float:
            v_dict[exist_key] = f'{iter_value:.1e}'
        else:
            v_dict[exist_key] = f'{iter_value}'
        yield v_dict


def submit_and_update_local_id_db(submit_command, 
                                  server, 
                                  id_db='~/.hobot/ids.db',
                                  local_log_path='./job-logs'):
    r = subprocess.run(
        submit_command, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = r.stdout + r.stderr
    output = output.decode('utf-8')
    for l in output.split('\n'):
        if 'Use jobname' in l:
            job_name = l.split('Use jobname: ')[-1]
    if '~' in id_db:
        id_db = os.path.expanduser(id_db)
    new_local_id = db.add_new_jobname_server(job_name, server, id_db)
    print(output)
    if 'ERR' not in output:
        if not os.path.isdir(local_log_path):
            os.mkdir(local_log_path)
        os.system(
            f'touch {os.path.join(local_log_path, str(new_local_id))}.txt')


def main(args):
    """Submit frameworks:
    1. create py_submit.sh
    2. create py_job.sh
    3. submit, update local id db, update local log path
    """

    with open(args.ini_path, 'r') as f:
        ini_file = convert_multi_line(f)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(ini_file)

    config_job = dict(config['JOB'])
    config_job_para = config['JOB_PARA']
    config_sri_para = config['SCRIPT_PARA']
    config_para_de = config['PARA_DESRB']

    # update wall time
    wall_time = config_job['WALLTIME'] 
    wall_time = str(int(float(wall_time) * 24 * 60))
    config_job['WALLTIME'] = wall_time

    # mapping
    cluster_id = config_job['USE_CLUSTER_ID']
    cluster_id_extra = config_job['USE_CLUSTER_EXTRA']
    for mapping_func in STR_REPLACE[cluster_id]:
        for k, v in config_job_para.items():
            config_job_para[k] = mapping_func(v)

    # add cfg param
    config_sri_para = {**config_sri_para, **CLUSTER_SPECIFIC_PARA[cluster_id]}

    # update rep time
    config_job['REAP_RUN'] = list(range(int(config_job['REAP_RUN'])))

    # write cluster config file
    server = cluster_id+cluster_id_extra
    with open('./gpucluster.yaml', 'w') as f:
        f.write(
            CLUSTER_STR +
            f'\ncurrent-cluster: {server}'
        )

    # multiple submit
    for src_para in multiple_submit(config_sri_para):
        config_job['SCRIPTS'] = config_job_para['SCRIPTS']

        # get job name
        job_name = config_job['DSCRB'].replace('.', '').lower()
        job_name += f"-{config_job['PODS']}-"
        job_name += '-'.join([
            i + src_para[config_para_de[i]]
            .replace('.', '')
            .replace('/', '-')
            .replace(',', '-')
            .replace('_', '-')
            .replace('+', '')
            .lower()
            for i in config_para_de
        ])
        config_job['JOB_NAME'] = job_name

        # get job scripts
        job_script = config_job_para['SCRIPT_PATH']
        for k, v in src_para.items():
            if len(k) == 1:
                suffix = ' -'
            elif k[:2] == '__':
                suffix = ' '
                k = k.replace('__', '')
            else:
                suffix = ' --'
            job_script += f"{suffix}{k} {v.replace('PASS', '')}" if len(
                k) > 0 else ''
        config_job['JOB_SCRIPT'] = job_script

        # for j2 variables
        variables = {}
        for k, v in J2FILE_REQIRED_VARS[cluster_id].items():
            variables[k] = config_job[v]

        # fill in j2 template
        for template in CLUSTER_TO_J2FILE[cluster_id]['submit_scripts']:
            with open(os.path.join(args.template_path, template), 'r') as f:
                t = jinja2.Template(f.read())
            file = t.render(**variables)
            file_name = '.'.join(template.split('.')[:2])
            with open(file_name, 'w') as f:
                f.write(file)

        # init run_scripts
        run_scripts = config_job.get('RUN_SCRIPTS')
        if run_scripts is not None:
            if args.run_scripts is not None:
                raise RuntimeWarning(
                    f'Found two run_scripts: {run_scripts} and {args.run_scripts}')

        if run_scripts is not None:
            run_scripts = [run_scripts]
        elif args.run_scripts is not None:
            # to enable auto-tab fill when sending commands
            run_scripts = [args.run_scripts.strip(args.template_path)]
        else:
            run_scripts = CLUSTER_TO_J2FILE[cluster_id]['run_scripts']
        for template in run_scripts:
            with open(os.path.join(args.template_path, template), 'r') as f:
                t = jinja2.Template(f.read())
            file = t.render(**variables)
            file_name = '.'.join(template.split('.')[:2])
            with open(os.path.join(config_job['FILE_FOLDER'], file_name), 'w') as f:
                f.write(file)

        os.system("chmod 700 py_submit.sh")
        os.system(f"chmod 700 ./{config_job['FILE_FOLDER']}/py_job.sh")
        if args.run_submit:
            os.system("./py_submit.sh")
            # submit_and_update_local_id_db("./py_submit.sh", server)
            print('-'*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ini_path', type=str)
    parser.add_argument('--template_path', type=str, default='./ini')
    parser.add_argument('-s', '--run_submit', action='store_true')
    parser.add_argument('-r', '--run_scripts', type=str, default=None)
    args = parser.parse_args()

    main(args)
