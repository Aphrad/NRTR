import numpy

def init_metric(type="train"):
    metric_iters = dict()
    if type not in ["train", "val", "test"]:
        print("Wrong Type")
    else:
        if type == "train":
            metric_iters["loss_value"] = list()
            metric_iters["class_error"] = list()
        elif type == "val":
            metric_iters["loss_value"] = list()
            metric_iters["class_error"] = list()
        elif type == "test":
            metric_iters["patch"] = dict()
            metric_iters["patch"] = {"PEC":list(), "REC":list(), "F-1":list(), "JAC":list()}
            metric_iters["image"] = dict()
            metric_iters["image"] = {"pred":0, "tgt":0, "intersection":0}
    return metric_iters

def clear_metric(metric_iters):
    for k, v in metric_iters.items():
        metric_iters[k].clear()
    return metric_iters

def update_metric(metric_iters, metric):
    for k, v in metric.items():
        metric_iters[k].append(v)
    return metric_iters

def get_mean_metric(metric_iters):
    mean_metric = {k: numpy.mean(metric_iters[k]) for k, v in metric_iters.items()}
    return mean_metric

