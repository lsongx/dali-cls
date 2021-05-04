import os
import json


final_root = '/cluster_home/plat_gpu/'
all_jobs = os.listdir(final_root)
print(len(all_jobs))
all_saves = {}
idx = 0
for job in all_jobs:
    if '-' not in job:
        print(f'{job} ignored')
    cleaned = 0
    total = 0
    total_dict = {}
    for root, dirs, files in os.walk(os.path.join(final_root, job), topdown=False):
        for name in files:
            try:
                filesize = os.path.getsize(os.path.join(root, name))/10e6
            except:
                filesize = 0
            total += filesize
            total_dict[os.path.join(root, name)] = filesize

            if 'log' in name or '.sh' in name or 'tfevents' in name:
                filename = os.path.join(root, name)
                new_name = filename.split(final_root)[-1].replace('/', '--')
                new_name_save = f'{new_name[:20]}-{idx}'
                if os.system(f'cp {filename} /job_data/logs/{new_name_save}'):
                    import pdb; pdb.set_trace()
                all_saves[new_name_save] = new_name
                idx += 1

            if '.pth' in name or '.json' in name:
                os.remove(os.path.join(root, name))
                # print(f'{os.path.join(root, name)} removed')
                cleaned += filesize
            elif 'tfevents' in name and filesize>10:
                os.remove(os.path.join(root, name))
                cleaned += filesize
            elif filesize>5:
                print(f'{os.path.join(root, name)} with size {filesize}MB kept')

    if total-cleaned>5:
        print(f'{job} clean {cleaned}MB, {total-cleaned}MB remain')
        for k, v in total_dict.items():
            if v>10:
                print(k)
                # import pdb; pdb.set_trace()

print(f'saved {len(all_saves)}')
with open('/job_data/logs/all_saved.json', 'w') as f:
    json.dump(all_saves, f)
