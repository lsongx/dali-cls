import os
import sqlite3


def init_local_id(id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE ids 
        (local INTEGER PRIMARY KEY AUTOINCREMENT, 
         name TEXT, server TEXT, serverid INT)'''
    )
    conn.commit()
    conn.close()


def add_new_jobname_server(job_name, server, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    c.execute(f"INSERT INTO ids(name, server) VALUES ('{job_name}', '{server}')")
    new_id = c.execute(f"SELECT * FROM ids WHERE name='{job_name}'").fetchall()[0][0]
    conn.commit()
    conn.close()
    return new_id


def get_all_empty_server_info(id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    row_list = c.execute("SELECT * FROM ids WHERE serverid IS NULL").fetchall()
    conn.commit()
    conn.close()
    return row_list


def update_row_list(row_list, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    for row in row_list:
        c.execute(f"UPDATE ids SET serverid={row[3]} WHERE local={row[0]}")
    conn.commit()
    conn.close()


def read_row_list(local_id_list, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    row_list = []
    for local_id in local_id_list:
        row_list += c.execute(f"SELECT * FROM ids WHERE local={local_id}").fetchall()
    conn.commit()
    conn.close()
    return row_list


if __name__ == '__main__':
    home_root = os.path.expanduser('~/.hobot')
    if not os.path.isdir(home_root):
        os.mkdir(home_root)
    id_db = os.path.join(home_root, 'ids.db')
    if os.path.isfile(id_db):
        raise RuntimeError('Database already exists.')
    else:
        init_local_id(id_db)
