def update_cfg_from_args(args, cfg):
    ta = vars(args)
    for k in ta.keys():
        k_list = k.split('.')
        update_key_value(k_list, ta[k], cfg)


def update_key_value(key, value, cfg):
    if len(key) == 1:
        if key[0] in cfg:
            cfg[key[0]] = value
    else:
        tk = key.pop(0)
        if tk in cfg:
            update_key_value(key, value, cfg[tk])
