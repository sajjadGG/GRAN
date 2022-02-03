def get_optimizer(conf, params):
    if conf.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=conf.lr,
            momentum=conf.momentum,
            weight_decay=conf.wd,
        )
    elif conf.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=conf.lr, weight_decay=conf.wd)
    else:
        raise ValueError("Non-supported optimizer!")
