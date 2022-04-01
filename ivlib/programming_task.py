
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class State:
    def __init__(self, name):
        self.name = name
        self.ins = dict()
        self.outs = dict()
        self.outs["default"] = None
        self.properties = dict()
        self.source = str()
        self.placeholders = list()

    def code(self):
        values = list()
        for ph in self.placeholders:
            if ph.startswith("val_"):
                values.append(str(ph[4:]))
            if ph.startswith("out_"):
                out_state = self.outs[ph[4:]]
                if out_state is None:
                    values.append(None)
                else:
                    values.append(out_state.code())
        return self.source.format(*values)


class Step:
    def __init__(self):
        self.graph = DotDict()
        self.graph.states = dict()
        self.graph.out = dict()

    def code(self):
        return self.graph.states["start"].code()


step_num = 0


def next_state_name():
    global step_num
    step_num += 1
    return "state_"+str(step_num)


def first_step():
    step = Step()
    state = State("start")
    state.source = "{0}"
    state.placeholders.append("out_default")
    step.graph.states["start"] = state
    return step


def action_end_state(step, state_name, transition):
    step.graph.states["end"] = State("end")
    step.graph.states[state_name].outs[transition] = State("end")


def action_import_lib(step, state_name, transition, lib_name):
    new_state = State(next_state_name())
    new_state.properties["lib "+str(lib_name)+" imported"] = True
    new_state.properties["lib name"] = lib_name
    new_state.source = "import {0}\n{1}"

    new_state.placeholders.append('val_'+str(lib_name))
    new_state.placeholders.append('out_default')

    step.graph.states[new_state.name] = new_state
    step.graph.states[state_name].outs[transition] = new_state
    return new_state.name


def action_while_loop(step, state_name, transition, condition):
    

def contract_no_undefined_transitions(g):
    for s in g.states:
        for d in s.outs:
            if d is None:
                return False
    return True
