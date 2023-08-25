
def str2bool(s):
    if s.lower() in ['true']:
        return True
    elif s.lower() in ['false']:
        return False
    else:
        raise ValueError()
