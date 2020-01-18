import string, sys, re, os, runpy
from functools import wraps

type_map = {}
functions = []
error_type = ''
try_wrap = ''

c_header_preamble = []
c_api_body_preamble = []


def bad_param_error(msg):
    return 'throw std::runtime_error("{}")'.format(msg)

class Template(string.Template):
    idpattern = '[_a-zA-Z0-9@]+'


class Type:
    def __init__(self, name):
        self.name = name.strip()

    def is_pointer(self):
        return self.name.endswith('*')

    def is_reference(self):
        return self.name.endswith('&')

    def is_const(self):
        return self.name.startswith('const ')

    def add_pointer(self):
        return Type(self.name + '*')

    def add_const(self):
        return Type('const ' + self.name)

    def inner_type(self):
        i = self.name.find('<')
        j = self.name.rfind('>')
        if i > 0 and j > 0:
            return Type(self.name[i + 1:j])
        else:
            return None

    def remove_generic(self):
        i = self.name.find('<')
        j = self.name.rfind('>')
        if i > 0 and j > 0:
            return Type(self.name[0:i] + self.name[j + 1:])
        else:
            return self

    def remove_pointer(self):
        if self.is_pointer():
            return Type(self.name[0:-1])
        return self

    def remove_reference(self):
        if self.is_reference():
            return Type(self.name[0:-1])
        return self

    def remove_const(self):
        if self.is_const():
            return Type(self.name[6:])
        return self

    def basic(self):
        return self.remove_pointer().remove_const().remove_reference()

    def const_compatible(self, t):
        if t.is_const():
            return self.add_const()
        return self

    def str(self):
        return self.name


header_function = Template('''
${error_type} ${name}(${params});
''')

c_api_impl = Template('''
extern "C" ${error_type} ${name}(${params})
{
    return ${try_wrap}([&] {
        ${body};
    });
}
''')


class CFunction:
    def __init__(self, name):
        self.name = name
        self.params = []
        self.body = []

    def add_param(self, type, pname):
        self.params.append('{} {}'.format(type, pname))

    def add_statement(self, stmt):
        self.body.append(stmt)

    def substitute(self, form):
        return form.substitute(error_type=error_type,
                               try_wrap=try_wrap,
                               name=self.name,
                               params=', '.join(self.params),
                               body=";\n        ".join(self.body))

    def generate_header(self):
        return self.substitute(header_function)

    def generate_body(self):
        return self.substitute(c_api_impl)


class BadParam:
    def __init__(self, cond, msg):
        self.cond = cond
        self.msg = msg


class Parameter:
    def __init__(self, name, type, optional=False, returns=False):
        self.name = name
        self.type = Type(type)
        self.optional = optional
        self.cparams = []
        self.size_cparam = -1
        self.size_name = ''
        self.read = '${name}'
        self.write = ['*${name} = ${result}']
        self.returns = returns
        self.bad_param_check = None

    def substitute(self, s, result=None):
        ctype = None
        if len(self.cparams) > 0:
            ctype = Type(self.cparams[0][0]).basic().str()
        return Template(s).safe_substitute(name=self.name,
                                                  type=self.type.str(),
                                                  ctype=ctype or '',
                                                  size=self.size_name,
                                                  result=result or '')

    def add_param(self, t, name=None):
        if not isinstance(t, str):
            t = t.str()
        self.cparams.append((t, name or self.name))

    def add_size_param(self, name=None):
        self.size_cparam = len(self.cparams)
        self.size_name = name or self.name + '_size'
        if self.returns:
            self.add_param('size_t *', self.size_name)
        else:
            self.add_param('size_t', self.size_name)

    def bad_param(self, cond, msg):
        self.bad_param_check = BadParam(cond, msg)

    def remove_size_param(self, name):
        p = None
        if self.size_cparam >= 0:
            p = self.cparams[self.size_cparam]
            del self.cparams[self.size_cparam]
            self.size_name = name
        return p

    def update(self):
        t = self.type.basic().str()
        g = self.type.remove_generic().basic().str()
        if t in type_map:
            type_map[t](self)
        elif g in type_map:
            type_map[g](self)
        else:
            if self.returns:
                self.add_param(self.type.remove_reference().add_pointer())
            else:
                self.add_param(self.type.remove_reference())
        if isinstance(self.write, str):
            raise ValueError("Error for {}: write cannot be a string".format(
                self.type.str()))

    def input(self):
        return '(' + self.substitute(self.read) + ')'

    def outputs(self, result=None):
        return [self.substitute(w, result) for w in self.write]

    def add_to_cfunction(self, cfunction):
        for t, name in self.cparams:
            cfunction.add_param(self.substitute(t), self.substitute(name))
        if self.bad_param_check:
            msg = 'Bad parameter {name}: {msg}'.format(
                name=self.name, msg=self.bad_param_check.msg)
            cfunction.add_statement('if ({cond}) {body}'.format(
                cond=self.substitute(self.bad_param_check.cond),
                body=bad_param_error(msg)))


def template_var(s):
    return '${' + s + '}'


def to_template_vars(params):
    return ', '.join([template_var(p.name) for p in params])


class Function:
    def __init__(self,
                 name,
                 params=None,
                 shared_size=False,
                 returns=None,
                 invoke=None,
                 fname=None,
                 return_name=None,
                 **kwargs):
        self.name = name
        self.params = params or []
        self.shared_size = False
        self.cfunction = None
        self.fname = fname
        self.invoke = invoke or '${__fname__}($@)'
        self.return_name = return_name or 'out'
        self.returns = Parameter(self.return_name, returns,
                                 returns=True) if returns else None

    def share_params(self):
        if self.shared_size == True:
            size_param_name = 'size'
            size_type = Type('size_t')
            for param in self.params:
                p = param.remove_size_param(size_param_name)
                if p:
                    size_type = Type(p[0])
            self.params.append(Parameter(size_param_name, size_type.str()))

    def update(self):
        self.share_params()
        for param in self.params:
            param.update()
        if self.returns:
            self.returns.update()
        self.create_cfunction()

    def inputs(self):
        return ', '.join([p.input() for p in self.params])

    def input_map(self):
        m = {}
        for p in self.params:
            m[p.name] = p.input()
        m['return'] = self.return_name
        m['@'] = self.inputs()
        m['__fname__'] = self.fname
        return m

    def get_invoke(self):
        return Template(self.invoke).safe_substitute(self.input_map())

    def write_to_tmp_var(self):
        return len(self.returns.write) > 1 or self.returns.write[0].count(
            '${result}') > 1

    def create_cfunction(self):
        self.cfunction = CFunction(self.name)
        # Add the return as a parameter
        if self.returns:
            self.returns.add_to_cfunction(self.cfunction)
        # Add the input parameters
        for param in self.params:
            param.add_to_cfunction(self.cfunction)
        f = self.get_invoke()
        # Write the assignments
        assigns = []
        if self.returns:
            result = f
            if self.write_to_tmp_var():
                f = 'auto&& api_result = ' + f
                result = 'api_result'
            else:
                f = None
            assigns = self.returns.outputs(result)
        if f:
            self.cfunction.add_statement(f)
        for assign in assigns:
            self.cfunction.add_statement(assign)


def params(virtual=None, **kwargs):
    result = []
    for name in virtual or {}:
        result.append(Parameter(name, virtual[name]))
    for name in kwargs:
        result.append(Parameter(name, kwargs[name]))
    return result


def add_function(name, *args, **kwargs):
    functions.append(Function(name, *args, **kwargs))


def once(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not decorated.has_run:
            decorated.has_run = True
            return f(*args, **kwargs)

    decorated.has_run = False
    return decorated


@once
def process_functions():
    for f in functions:
        f.update()


def generate_lines(p):
    return '\n'.join(p)


def generate_c_header():
    process_functions()
    return generate_lines(c_header_preamble +
                          [f.cfunction.generate_header() for f in functions])


def generate_c_api_body():
    process_functions()
    return generate_lines(c_api_body_preamble +
                          [f.cfunction.generate_body() for f in functions])


def cwrap(name):
    def with_cwrap(f):
        type_map[name] = f

        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    return with_cwrap


handle_typedef = Template('''
typedef struct ${ctype} * ${ctype}_t;
typedef const struct ${ctype} * const_${ctype}_t;
''')

handle_definition = Template('''
extern "C" struct ${ctype};
struct ${ctype} {
    template<class... Ts>
    ${ctype}(Ts&&... xs)
    : object(std::forward<Ts>(xs)...)
    {}
    ${cpptype} object;
};
''')

handle_preamble = '''
template<class T, class U, class Target=std::remove_pointer_t<T>>
Target* object_cast(U* x)
{
    return reinterpret_cast<Target*>(x);
}
template<class T, class U, class Target=std::remove_pointer_t<T>>
const Target* object_cast(const U* x)
{
    return reinterpret_cast<const Target*>(x);
}

template<class T, class... Ts, class Target=std::remove_pointer_t<T>>
Target* allocate(Ts&&... xs)
{
    return new Target(std::forward<Ts>(xs)...); // NOLINT
}

template<class T>
void destroy(T* x)
{
    delete x; // NOLINT
}
'''


@once
def add_handle_preamble():
    c_api_body_preamble.append(handle_preamble)


def add_handle(name, ctype, cpptype, destroy=None):
    opaque_type = ctype + '_t'

    def handle_wrap(p):
        t = Type(opaque_type)
        if p.type.is_const():
            t = Type('const_' + opaque_type)
        if p.returns:
            p.add_param(t.add_pointer())
            if p.type.is_reference():
                p.write = ['*${name} = object_cast<${ctype}>(&(${result}))']
            elif p.type.is_pointer():
                p.write = ['*${name} = object_cast<${ctype}>(${result})']
            else:
                p.write = ['*${name} = allocate<${ctype}>(${result})']
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
            p.read = '${name}->object'

    type_map[cpptype] = handle_wrap
    add_function(destroy or ctype + '_' + 'destroy',
                 params({name: opaque_type}),
                 fname='destroy')
    add_handle_preamble()
    c_header_preamble.append(handle_typedef.substitute(locals()))
    c_api_body_preamble.append(handle_definition.substitute(locals()))


@cwrap('std::vector')
def vector_c_wrap(p):
    t = p.type.inner_type().add_pointer()
    if p.returns:
        if p.type.is_reference():
            if p.type.is_const():
                t = t.add_const()
            p.add_param(t.add_pointer())
            p.add_size_param()
            p.bad_param('${name} == nullptr or ${size} == nullptr',
                        'Null pointer')
            p.write = [
                '*${name} = ${result}.data()', '*${size} = ${result}.size()'
            ]
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
            p.write = [
                'std::copy(${result}.begin(), ${result}.end(), ${name})'
            ]
    else:
        p.add_param(t)
        p.add_size_param()
        p.bad_param('${name} == nullptr', 'Null pointer')
        p.read = '${type}(${name}, ${name}+${size})'


class Handle:
    def __init__(self, name, ctype, cpptype):
        self.name = name
        self.ctype = ctype
        self.cpptype = cpptype
        add_handle(name, ctype, cpptype)

    def cname(self, name):
        return self.ctype + '_' + name

    def substitute(self, s, **kwargs):
        return Template(s).safe_substitute(name=self.name, ctype=self.ctype, cpptype=self.cpptype, **kwargs)

    def constructor(self, name, params=None, fname=None, invoke=None,
                    **kwargs):
        create = self.substitute('allocate<${cpptype}>($@)')
        if fname:
            create = self.substitute('allocate<${cpptype}>(${fname}($@))', fname=fname)

        add_function(self.cname(name),
                     params=params,
                     invoke=invoke or create,
                     returns=self.cpptype + '*',
                     return_name=self.name,
                     **kwargs)
        return self

    def method(self, name, params=None, fname=None, invoke=None, **kwargs):
        p = Parameter(self.name, self.cpptype)
        args = to_template_vars(params or [])
        add_function(
            self.cname(name),
            params=[p] + (params or []),
            invoke=invoke or self.substitute('${var}.${fname}(${args})', var=template_var(self.name), fname=fname or name, args=args),
            **kwargs)
        return self

    def function(self, name, params=None, **kwargs):
        add_function(
            self.cname(name),
            params=params,
            **kwargs)
        return self


def handle(ctype, cpptype, name=None):
    def with_handle(f):
        n = name or f.__name__
        h = Handle(n, ctype, cpptype)
        f(h)

        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    return with_handle


def template_eval(template, **kwargs):
    start = '<%'
    end = '%>'
    escaped = (re.escape(start), re.escape(end))
    mark = re.compile('%s(.*?)%s' % escaped, re.DOTALL)
    for key in kwargs:
        exec ('%s = %s' % (key, kwargs[key]))
    for item in mark.findall(template):
        e = eval(item.strip())
        template = template.replace(start + item + end, str(e))
    return template


def run():
    runpy.run_path(sys.argv[1])
    if len(sys.argv) > 2:
        f = open(sys.argv[2]).read()
        r = template_eval(f)
        sys.stdout.write(r)
    else:
        sys.stdout.write(generate_c_header())
        sys.stdout.write(generate_c_api_body())


if __name__ == "__main__":
    sys.modules['api'] = sys.modules['__main__']
    run()
