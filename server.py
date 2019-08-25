import argparse
import os
import subprocess
import requests

import local_id_db_utils as db
from submit import CLUSTER_STR


SERVER_IP = {
    'idcrun': '10.10.10.228:35327',
    'idcdone': '10.10.10.228:37014',
    'ksyunrun': '10.10.216.3:31132',
    'ksyundone': '10.10.216.3:32127',
    'aliyunrun': '',
    'aliyundone': '',
}


def get_id_from_name(job_name, server='idc', status='run') -> int:
    url = f'http://{SERVER_IP[server+status]}/{job_name}/log'
    start_str = '<span class="name">hobot-job-'
    end_str = '-0.log</span>'
    job_id = requests.get(url).text.split(start_str)[1].split(end_str)[0]
    return int(job_id)


def get_log(job_name, server, server_id, status):
    url = f'http://{SERVER_IP[server+status]}/{job_name}/log/hobot-job-{server_id}-0.log'
    return requests.get(url).text


def get_job_status(server) -> dict:
    """return a list of tuples, (job_name, status)
    """
    home_hobot_dir = os.path.expanduser('~/.hobot')
    with open(os.path.join(home_hobot_dir, 'gpucluster.yaml'), 'w') as f:
        f.write(
            CLUSTER_STR +
            f'\ncurrent-cluster: mycluster{server}'
        )
    c2e = {
        '排队中': 'wait', '预运行': 'wait', '运行中': 'run',
    }
    all_jobs = subprocess.run('traincli jobs', 
                              shell=True, stdout=subprocess.PIPE)
    all_jobs = all_jobs.stdout.decode('utf-8').split('\n')
    status_dict = {}
    for idx, line in enumerate(all_jobs):
        if idx%2 and idx > 1:
            print(line)
            entries = line.split('|')
            job_name = entries[1].strip()
            status = c2e[entries[3].strip()]
            status_dict[job_name] = status
    return status_dict


def is_running_job(job_name, status_dict):
    return status_dict[job_name] == 'run'


def filter_waiting_jobs(all_missing_list) -> list:
    """remove waiting jobs from all_missing_list
    """
    server_id_set = set()
    for job in all_missing_list:
        server_id_set.add(job[2])
    status_dict = {}
    for server_id in server_id_set:
        status_dict.update(get_job_status(server_id))
    is_running_job_func = lambda x: is_running_job(x, status_dict)
    return filter(is_running_job_func, all_missing_list), status_dict


def update_local_db(id_db='~/.hobot/ids.db'):
    id_db = os.path.expanduser(id_db)
    all_missing_list = db.get_all_empty_server_info(id_db)
    to_do_list, status_dict = filter_waiting_jobs(all_missing_list)
    processed_list = []
    for row in to_do_list:
        status = 'done'
        if is_running_job(row[1], status_dict):
            status = 'run'
        server_id = get_id_from_name(row[1], row[2], status)
        tmp_row = list(row)
        tmp_row[-1] = server_id
        processed_list.append(tmp_row)
    db.update_row_list(processed_list, id_db) 


def sync_running_log(local_log_path='./job-logs', id_db='~/.hobot/ids.db'):
    """
    1. look up server and server-id in db
    2. check if it is running (according to job_name)
    3. download log to local
    """
    if not os.path.isdir(local_log_path):
        os.mkdir(local_log_path)
    all_logs = os.listdir(local_log_path)
    all_logs = filter(lambda x: 'done' in x, all_logs)
    local_id_list = [int(log.split('.')[0]) for log in all_logs]
    print(f'Find {len(all_logs)} files needed sync.')
    db_row_list = db.read_row_list(local_id_list)
    db_row_ids = [row[0] for row in db_row_list]
    running_row_list, status_dict = filter_waiting_jobs(db_row_list)
    running_row_ids = [row[0] for row in running_row_list]
    show_still_waiting_jobs(db_row_ids, running_row_ids)
    for row in running_row_list:
        status = 'done'
        if is_running_job(row[1], status_dict):
            status = 'run'
        with open(f'{row[0]}.txt', 'w') as f:
            f.write(get_log(*[*row[1:], status]))
        if status == 'done':
            os.system(f'mv {row[0]}.txt {row[0]}.done.txt')
    print(f'Finished sync {len(running_row_list)} jobs.')


def show_still_waiting_jobs(db_row_ids, running_row_ids):
    waiting_list = []
    for db_id in db_row_ids:
        if db_id not in running_row_ids:
            waiting_list.append(db_id)
    if len(waiting_list) > 0:
        print(f'Waiting jobs: {' '.join(map(str, waiting_list))}.')


if __name__ == "__main__":
    """add argparse here
    add[a] : add an id into the local_log_path
    rm[r] : remove an id from local_log_path
    sync[s] : do sync
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--add', type=str, default=None)
    parser.add_argument('-r', '--rm', type=str, default=None)
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('--local_log_path', type=str, default='job-logs')
    parser.add_argument('--id_db', type=str, default='~/.hobot/ids.db')
    args = parser.parse_args()

    if args.a in not None:
        os.system(f'touch {args.a}.txt')
    
    if args.r is not None:
        os.system(f'rm {args.a}.*')

    if args.s:
        sync_running_log(args.local_log_path, args.id_db)
