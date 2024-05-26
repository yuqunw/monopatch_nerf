import inspect

class ArgDictClass:
    '''
    Stores all arguments as a dictionary
    '''
    def __init__(self):
        super().__init__()
        # store hyper parameters automatically
        frame = inspect.currentframe()
        vals = dict(inspect.getargvalues(frame)[-1])
        curr_args = {k: v for k, v in vals.items() if k not in ['self', 'vals', 'frame', '__class__']}
        sig = inspect.signature(self.__init__)
        params = {
            k: v
            for k, v in
            sig.parameters.items()
            if v.kind != 4
        }
        stacks = inspect.stack()
        args = {}
        for stack in stacks:
            frame = stack.frame
            keys, _, _, vals = inspect.getargvalues(frame)
            for key, val in vals.items():
                if key not in ['self', 'frame', 'vals', 'curr_args', 'params', 'stacks', 'stack', 'args', 'kwargs', '__class__', 'sig']:
                    args[key] = val
            if all([_k in keys for _k in params.keys()]):
                break
        # hyper parameters are saved
        self.args_dict = {**args, **curr_args}