import subprocess

sync_folders = [
    # '/user/liangchen.song/data',
    '/user/liangchen.song/models',
]

ignore_files = [
    'cls', 'seg', 'depth', 'det', 'pose', 'visda',
    'gen', 'reid', 'imagenet-patch',
]

main_cluster = 'hdfs://hobot-bigdata'
other_clusters = [
    'hdfs://hobot-bigdata-aliyun',
    'hdfs://ksbigdata',
]


def get_all_files(address):
    files = []
    all_files = subprocess.run(
        f'hdfs dfs -ls -R {address}'.split(), capture_output=True).stdout
    af = all_files.decode('utf-8').split('\n')
    print(f'Found {len(af)} files in {address}.')
    for f in af:
        if len(f) == 0:
            continue
        f_details = f.split()
        if f_details[4] == '0':
            continue
        append = True
        for igf in ignore_files:
            if igf in f:
                append = False
        if append:
            files.append(f_details[-1])
    print(f'Total {len(files)} after filtered.')
    return files


def is_file(address):
    r = subprocess.run(
        f'hdfs dfs -ls {address}'.split(), capture_output=True)
    r = r.stdout + r.stderr
    if 'No such file or directory' in r.decode('utf-8'):
        return False
    return True


def get_folder(address):
    address = address.split('/')[:-1]
    address = '/'.join(address)
    return address


def is_dir(address):
    address = get_folder(address)
    r = subprocess.run(
        f'hdfs dfs -ls {address}'.split(), capture_output=True)
    r = r.stdout + r.stderr
    if 'No such file or directory' in r.decode('utf-8'):
        return False
    return True


def is_same_file(address_a, address_b):
    sum_a = subprocess.run(f'hdfs dfs -checksum {address_a}'.split(),
                           capture_output=True).stdout.decode('utf-8').split()[-1]
    sum_b = subprocess.run(f'hdfs dfs -checksum {address_b}'.split(),
                           capture_output=True).stdout.decode('utf-8').split()[-1]
    if sum_a == sum_b:
        return True
    return False


for folder in sync_folders:
    address = main_cluster + folder
    for file in get_all_files(address):
        for o_cluster in other_clusters:
            o_file = file.replace(main_cluster, o_cluster)
            if is_file(o_file):
                if is_same_file(o_file, file):
                    continue
                subprocess.run(f'hdfs dfs -rm {o_file}'.split())
            elif not is_dir(o_file):
                folder = get_folder(file)
                o_folder = get_folder(o_file)
                print(f'Start Sync: {folder} -> {o_folder}')
                subprocess.run(f'hdfs dfs -cp -r {folder} {o_folder}'.split())
                print(f'Finish Sync: {folder} -> {o_folder}')
                continue
            print(f'Start Sync: {file} -> {o_file}')
            subprocess.run(f'hdfs dfs -cp {file} {o_file}'.split())
            print(f'Finish Sync: {file} -> {o_file}')
