import string, sys, re, os, runpy
from functools import wraps

type_map = {}
cpp_type_map = {}
functions = []
cpp_classes = []
error_type = ''
success_type = ''
try_wrap = ''

c_header_preamble = []
c_api_body_preamble = []
cpp_header_preamble = []


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

    def decay(self):
        t = self.remove_reference()
        if t.is_pointer():
            return t
        else:
            return t.remove_const()

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
        self.cpp_read = '${name}'
        self.cpp_write = '${name}'
        self.returns = returns
        self.bad_param_check = None

    def get_name(self, prefix=None):
        if prefix:
            return prefix + self.name
        else:
            return self.name

    def get_cpp_type(self):
        if self.type.str() in cpp_type_map:
            return cpp_type_map[self.type.basic().str()]
        elif self.type.basic().str() in cpp_type_map:
            return cpp_type_map[self.type.basic().str()]
        elif self.returns:
            return self.type.decay().str()
        else:
            return self.type.str()

    def substitute(self, s, prefix=None, result=None):
        ctype = None
        if len(self.cparams) > 0:
            ctype = Type(self.cparams[0][0]).basic().str()
        return Template(s).safe_substitute(name=self.get_name(prefix),
                                           type=self.type.str(),
                                           ctype=ctype or '',
                                           cpptype=self.get_cpp_type(),
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

    def cpp_param(self, prefix=None):
        return self.substitute('${cpptype} ${name}', prefix=prefix)

    def cpp_arg(self, prefix=None):
        return self.substitute(self.cpp_read, prefix=prefix)

    def cpp_output_args(self, prefix=None):
        return [
            '&{prefix}{n}'.format(prefix=prefix, n=n) for t, n in self.cparams
        ]

    def output_declarations(self, prefix=None):
        return [
            '{type} {prefix}{n};'.format(type=Type(t).remove_pointer().str(),
                                         prefix=prefix,
                                         n=n) for t, n in self.cparams
        ]

    def output_args(self, prefix=None):
        return [
            '&{prefix}{n};'.format(prefix=prefix, n=n) for t, n in self.cparams
        ]

    def cpp_output(self, prefix=None):
        return self.substitute(self.cpp_write, prefix=prefix)


    def input(self, prefix=None):
        return '(' + self.substitute(self.read, prefix=prefix) + ')'

    def outputs(self, result=None):
        return [self.substitute(w, result=result) for w in self.write]

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


cpp_class_template = Template('''

struct ${name} : handle_base<${ctype}, decltype(&${destroy}), ${destroy}>
{
    ${name}(${ctype} p, bool own = true)
    : m_handle(nullptr)
    {
        this->set_handle(p, own);
    }
    ${constructors}

    ${methods}
};
''')

cpp_class_method_template = Template('''
    ${return_type} ${name}(${params}) const
    {
        ${outputs}
        this->call_handle(${args});
        return ${result};
    }
''')

cpp_class_void_method_template = Template('''
    void ${name}(${params}) const
    {
        this->call_handle(${args});
    }
''')

cpp_class_constructor_template = Template('''
    ${name}(${params})
    : m_handle(nullptr)
    {
        m_handle = this->make_handle(${args});
    }
''')


class CPPMember:
    def __init__(self, name, function, prefix, method=True):
        self.name = name
        self.function = function
        self.prefix = prefix
        self.method = method

    def get_function_params(self):
        if self.method:
            return self.function.params[1:]
        else:
            return self.function.params

    def get_args(self):
        output_args = []
        if self.function.returns:
            output_args = self.function.returns.cpp_output_args(self.prefix)
        return ', '.join(
            ['&{}'.format(self.function.cfunction.name)] +
            output_args +
            [p.cpp_arg(self.prefix) for p in self.get_function_params()])

    def get_params(self):
        return ', '.join(
            [p.cpp_param(self.prefix) for p in self.get_function_params()])

    def get_return_declarations(self):
        if self.function.returns:
            return '\n        '.join([
                d
                for d in self.function.returns.output_declarations(self.prefix)
            ])
        else:
            return ''

    def get_result(self):
        return self.function.returns.input(self.prefix)

    def generate_method(self):
        if self.function.returns:
            return_type = self.function.returns.get_cpp_type()
            return cpp_class_method_template.safe_substitute(
                return_type=return_type,
                name=self.name,
                cfunction=self.function.cfunction.name,
                result=self.function.returns.cpp_output(self.prefix),
                params=self.get_params(),
                outputs=self.get_return_declarations(),
                args=self.get_args(),
                success=success_type)
        else:
            return cpp_class_void_method_template.safe_substitute(
                name=self.name,
                cfunction=self.function.cfunction.name,
                params=self.get_params(),
                args=self.get_args(),
                success=success_type)

    def generate_constructor(self, name):
        return cpp_class_constructor_template.safe_substitute(
            name=name,
            cfunction=self.function.cfunction.name,
            params=self.get_params(),
            args=self.get_args(),
            success=success_type)


class CPPClass:
    def __init__(self, name, ctype):
        self.name = name
        self.ctype = ctype
        self.constructors = []
        self.methods = []
        self.prefix = 'p'

    def add_method(self, name, f):
        self.methods.append(CPPMember(name, f, self.prefix, method=True))

    def add_constructor(self, name, f):
        self.constructors.append(CPPMember(name, f, self.prefix, method=True))

    def generate_methods(self):
        return '\n    '.join([m.generate_method() for m in self.methods])

    def generate_constructors(self):
        return '\n    '.join(
            [m.generate_constructor(self.name) for m in self.constructors])

    def substitute(self, s, **kwargs):
        t = s
        if isinstance(s, str):
            t = string.Template(s)
        destroy = self.ctype + '_destroy'
        return t.safe_substitute(name=self.name,
                                 ctype=self.ctype,
                                 destroy=destroy,
                                 **kwargs)

    def generate(self):
        return self.substitute(
            cpp_class_template,
            constructors=self.substitute(self.generate_constructors()),
            methods=self.substitute(self.generate_methods()))


def params(virtual=None, **kwargs):
    result = []
    for name in virtual or {}:
        result.append(Parameter(name, virtual[name]))
    for name in kwargs:
        result.append(Parameter(name, kwargs[name]))
    return result


def add_function(name, *args, **kwargs):
    f = Function(name, *args, **kwargs)
    functions.append(f)
    return f


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


def generate_cpp_header():
    process_functions()
    return generate_lines(cpp_header_preamble +
                          [c.generate() for c in cpp_classes])


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

cpp_handle_preamble = '''
template<class T, class Deleter, Deleter deleter>
struct handle_base
{

    template<class F, class... Ts>
    void make_handle(F f, Ts&&... xs)
    {
        T* result = nullptr;
        auto e = F(&result, std::forward<Ts>(xs)...);
        if (e != ${success})
            throw std::runtime_error("Failed to call function");
        set_handle(result);
    }

    template<class F, class... Ts>
    void call_handle(F f, Ts&&... xs)
    {
        auto e = F(this->get_handle_ptr(), std::forward<Ts>(xs)...);
        if (e != ${success})
            throw std::runtime_error("Failed to call function");
    }

    const std::shared_ptr<T>& get_handle() const
    {
        return m_handle;
    }

    T* get_handle_ptr() const
    {
        assert(m_handle != nullptr);
        return get_handle().get();
    }

    void set_handle(T* ptr, bool own = true)
    {
        if (own)
            m_handle = std::shared_ptr<T>{ptr, deleter};
        else
            m_handle = std::shared_ptr<T>{ptr, [](T*) {}};
    }

protected:
    std::shared_ptr<T> m_handle;
};

'''


@once
def add_handle_preamble():
    c_api_body_preamble.append(handle_preamble)
    cpp_header_preamble.append(
        string.Template(cpp_handle_preamble).substitute(success=success_type))


def add_handle(name, ctype, cpptype, destroy=None):
    opaque_type = ctype + '_t'

    def handle_wrap(p):
        t = Type(opaque_type)
        if p.type.is_const():
            t = Type('const_' + opaque_type)
        if p.returns:
            p.add_param(t.add_pointer())
            if p.type.is_reference():
                p.cpp_write = '${cpptype}(${name}, false)'
                p.write = ['*${name} = object_cast<${ctype}>(&(${result}))']
            elif p.type.is_pointer():
                p.cpp_write = '${cpptype}(${name}, false)'
                p.write = ['*${name} = object_cast<${ctype}>(${result})']
            else:
                p.cpp_write = '${cpptype}(${name})'
                p.write = ['*${name} = allocate<${ctype}>(${result})']
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
            p.read = '${name}->object'
            p.cpp_read = '${name}.get_handle_ptr()'

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
            p.cpp_write = '${type}(${name}, ${name}+${size})'
            p.write = [
                '*${name} = ${result}.data()', '*${size} = ${result}.size()'
            ]
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
            p.cpp_write = '${type}(${name}, ${name}+${size})'
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
        self.cpp_class = CPPClass(name, ctype)
        add_handle(name, ctype, cpptype)
        cpp_type_map[cpptype] = name

    def cname(self, name):
        return self.ctype + '_' + name

    def substitute(self, s, **kwargs):
        return Template(s).safe_substitute(name=self.name,
                                           ctype=self.ctype,
                                           cpptype=self.cpptype,
                                           **kwargs)

    def constructor(self, name, params=None, fname=None, invoke=None,
                    **kwargs):
        create = self.substitute('allocate<${cpptype}>($@)')
        if fname:
            create = self.substitute('allocate<${cpptype}>(${fname}($@))',
                                     fname=fname)

        f = add_function(self.cname(name),
                         params=params,
                         invoke=invoke or create,
                         returns=self.cpptype + '*',
                         return_name=self.name,
                         **kwargs)
        self.cpp_class.add_constructor(name, f)
        return self

    def method(self, name, params=None, fname=None, invoke=None, cpp_name=None, **kwargs):
        p = Parameter(self.name, self.cpptype)
        args = to_template_vars(params or [])
        f = add_function(self.cname(name),
                         params=[p] + (params or []),
                         invoke=invoke
                         or self.substitute('${var}.${fname}(${args})',
                                            var=template_var(self.name),
                                            fname=fname or name,
                                            args=args),
                         **kwargs)
        self.cpp_class.add_method(cpp_name or name, f)
        return self

    def function(self, name, params=None, **kwargs):
        add_function(self.cname(name), params=params, **kwargs)
        return self

    def add_cpp_class(self):
        cpp_classes.append(self.cpp_class)


def handle(ctype, cpptype, name=None):
    def with_handle(f):
        n = name or f.__name__
        h = Handle(n, ctype, cpptype)
        f(h)
        h.add_cpp_class()

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
        sys.stdout.write(generate_cpp_header())


if __name__ == "__main__":
    sys.modules['api'] = sys.modules['__main__']
    run()
