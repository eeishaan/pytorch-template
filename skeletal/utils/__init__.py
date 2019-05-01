from skeletal.constants import PARAM_DIR


def get_param_file(embedding_name, cluster_name):
    return PARAM_DIR / "{}_{}.yml".format(embedding_name, cluster_name)
