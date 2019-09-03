import argparse
import os
import subprocess
import requests
import sqlite3

import local_id_db_utils as db
from submit import CLUSTER_STR


SERVER_IP = {
    # 'idcrun': '10.10.10.228:35327',
    'idcrun': '10.10.10.228:37041',
    # 'idcdone': '10.10.10.228:37014',
    'idcdone': '10.10.10.228:37041',
    'ksyunrun': '10.10.216.3:31132',
    'ksyundone': '10.10.216.3:32127',
    'aliyunrun': '10.10.201.6:41720',
    'aliyundone': '10.10.201.6:41720',
}


SERVER_MAP = {
    '1': 'idc', '2': 'idc', '3': 'ksyun', '4': 'ksyun',
}


def get_id_from_name(job_name, server='idc', status='run') -> int:
    url = f'http://{SERVER_IP[SERVER_MAP[server]+status]}/{job_name}/log/'
    start_str = '<span class="name">hobot-job-'
    end_str = '-0.log</span>'
    job_id = requests.get(url).text.split(start_str)[1].split(end_str)[0]
    return int(job_id)


def get_log(job_name, server, server_id, status):
    url = (f'http://{SERVER_IP[SERVER_MAP[server]+status]}'
           f'/{job_name}/log/hobot-job-{server_id}-0.log')
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
    all_jobs = subprocess.run('traincli jobs', shell=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    all_jobs = all_jobs.stdout.decode('utf-8').strip().split('\n')
    status_dict = {}
    if len(all_jobs) < 5:
        return {}
    for line in all_jobs[3:-1]:
        # print(line)
        entries = line.split('|')
        job_name = entries[1].strip()
        status = c2e[entries[3].strip()]
        status_dict[job_name] = status
    return status_dict


def is_running_job(job_name, status_dict):
    if job_name not in status_dict.keys():
        return False
    return status_dict[job_name] == 'run'


def is_waiting_job(job_name, status_dict):
    if job_name not in status_dict.keys():
        return False
    return status_dict[job_name] == 'wait'


def filter_waiting_jobs(all_missing_list, keep_done=False) -> list:
    """remove waiting jobs from all_missing_list
    """
    server_id_set = set()
    for job in all_missing_list:
        server_id_set.add(job[2])
    status_dict = {}
    for server_id in server_id_set:
        status_dict.update(get_job_status(server_id))
    if keep_done:
        is_waiting_job_func = lambda x: not is_waiting_job(x[1], status_dict)
        filtered_list = list(filter(is_waiting_job_func, all_missing_list))
    else:
        is_running_job_func = lambda x: is_running_job(x[1], status_dict)
        filtered_list = list(filter(is_running_job_func, all_missing_list))
    return filtered_list, status_dict


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
    all_logs = list(filter(lambda x: 'done' not in x, all_logs))
    local_id_list = [int(log.split('.')[0]) for log in all_logs]
    print(f'Find {len(all_logs)} files needed sync. '
          f"IDs: {','.join(map(str, local_id_list))}")
    db_row_list = db.read_row_list(local_id_list, id_db)
    db_row_ids = [row[0] for row in db_row_list]
    running_row_list, status_dict = filter_waiting_jobs(db_row_list, True)
    running_row_ids = [row[0] for row in running_row_list]
    show_still_waiting_jobs(status_dict)
    for row in running_row_list:
        log_file = os.path.join(local_log_path, f'{row[0]}.txt')
        status = 'run'
        if row[1] not in status_dict.keys():
            status = 'done'
        with open(log_file, 'w') as f:
            f.write(get_log(*[*row[1:], status]))
        if status == 'done':
            os.system(f"mv {log_file} {log_file.replace('.txt', '.done.txt')}")
    print(f'Finished sync {len(running_row_list)} jobs.')


def show_still_waiting_jobs(status_dict):
    print(f"Waiting jobs:")
    has_wait = False
    for k, v in status_dict.items():
        if v == 'wait':
            print('\t'+k)
            has_wait = True
    if not has_wait:
        print('\tNone')
    


if __name__ == "__main__":
    """add argparse here
    add[a] : add an id into the local_log_path
    rm[r] : remove an id from local_log_path
    sync[s] : do sync
    show : show all rows in db
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--add', type=str, default=None)
    parser.add_argument('-r', '--rm', type=str, default=None)
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--local_log_path', type=str, default='./job-logs')
    parser.add_argument('--id_db', type=str, default='~/.hobot/ids.db')
    parser.add_argument('--print', type=str, default=None)
    args = parser.parse_args()

    if args.print is not None:
        log_name = f'{args.print}.txt'
        file_name = os.path.join(args.local_log_path, log_name)
        if not os.path.isfile(file_name):
            file_name = file_name.replace('.txt', '.done.txt')
        os.system(f"tail {file_name} -n 100")

    if '~' in args.id_db:
        args.id_db = os.path.expanduser(args.id_db)

    if args.add is not None:
        os.system(f'touch {args.a}.txt')
    
    if args.rm is not None:
        os.system(f'rm {args.a}.*')

    if args.sync:
        update_local_db(args.id_db)
        sync_running_log(args.local_log_path, args.id_db)

    if args.show:
        conn = sqlite3.connect(args.id_db)
        c = conn.cursor()
        row_list = c.execute("SELECT * FROM ids").fetchall()
        for row in row_list:
            print(row)
        conn.commit()
        conn.close()

