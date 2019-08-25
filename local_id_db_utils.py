import os
import sqlite3


def init_local_id(id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE ids 
        (local INT PRIMARY KEY ASC AUTOINCREMENT, 
         name TEXT, server TEXT, server-id INT)'''
    )
    conn.commit()
    conn.close()


def add_new_jobname_server(job_name, server, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    c.execute(f"INSERT INTO ids(name, server) VALUES ('{job_name}', '{server}')")
    conn.commit()
    conn.close()


def get_all_empty_server_info(id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    row_list = c.execute("SELECT ALL FROM ids WHERE server-id IS NULL")
    conn.commit()
    conn.close()
    return row_list


def update_row_list(row_list, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    for row in row_list:
        c.execute(f"UPDATE ids SET server-id={row[3]} WHERE local={row[0]}")
    conn.commit()
    conn.close()


def read_row_list(local_id_list, id_db):
    conn = sqlite3.connect(id_db)
    c = conn.cursor()
    row_list = []
    for local_id in local_id_list:
        row_list += c.execute("SELECT ALL FROM ids WHERE local={local_id}")
    conn.commit()
    conn.close()


if __name__ == '__main__':
    home_root = os.path.expanduser('~/.hobot')
    if not os.path.isdir():
        os.mkdir(home_root)
    id_db = os.path.join(home_root, 'ids.db')
    if os.path.isfile(id_db):
        raise RuntimeError('Database already exists.')
    else:
        init_local_id(id_db)
