import api


def bad_param_error(msg):
    return 'MIGRAPHX_THROW(migraphx_status_bad_param, "{}")'.format(msg)


api.error_type = 'migraphx_status'
api.try_wrap = 'migraphx::try_'
api.bad_param_error = bad_param_error


@api.cwrap('migraphx::shape::type_t')
def shape_type_wrap(p):
    if p.returns:
        p.add_param('migraphx_shape_datatype_t *')
        p.bad_param('${name} == nullptr', 'Null pointer')
        p.write = ['*${name} = migraphx::to_shape_type(${result})']
    else:
        p.add_param('migraphx_shape_datatype_t')
        p.read = 'migraphx::to_shape_type(${name})'


@api.cwrap('migraphx::compile_options')
def compile_options_type_wrap(p):
    if p.returns:
        p.add_param('migraphx_compile_options *')
        p.bad_param('${name} == nullptr', 'Null pointer')
        p.write = ['*${name} = migraphx::to_compile_options(${result})']
    else:
        p.add_param('migraphx_compile_options *')
        p.read = '${name} ? migraphx::to_compile_options(*${name}) : migraphx::compile_options{}'


def auto_handle(f):
    return api.handle('migraphx_' + f.__name__, 'migraphx::' + f.__name__)(f)


@auto_handle
def shape(h):
    h.constructor(
        'create',
        api.params(type='migraphx::shape::type_t',
                   lengths='std::vector<size_t>'))
    h.method('lengths', fname='lens', returns='const std::vector<size_t>&')
    h.method('strides', returns='const std::vector<size_t>&')
    h.method('type', returns='migraphx::shape::type_t')


@auto_handle
def argument(h):
    h.constructor('create', api.params(shape='migraphx::shape',
                                       buffer='void*'))
    h.method('shape', fname='get_shape', returns='const migraphx::shape&')
    h.method('buffer', fname='data', returns='char*')


@auto_handle
def target(h):
    h.constructor('create',
                  api.params(name='const char*'),
                  fname='migraphx::get_target')


@api.handle('migraphx_program_parameter_shapes',
            'std::unordered_map<std::string, migraphx::shape>')
def program_parameter_shapes(h):
    h.method('size', returns='size_t')
    h.method('get',
             api.params(name='const char*'),
             returns='const migraphx::shape&')
    h.method('names',
             invoke='migraphx::get_names(${program_parameter_shapes})',
             returns='std::vector<const char*>')


@api.handle('migraphx_program_parameters',
            'std::unordered_map<std::string, migraphx::argument>')
def program_parameters(h):
    h.constructor('create')
    h.method('add',
             api.params(name='const char*', argument='migraphx::argument'),
             invoke='${program_parameters}[${name}] = ${argument}')


@auto_handle
def program(h):
    h.method(
        'compile',
        api.params(target='migraphx::target',
                   options='migraphx::compile_options'))
    h.method('parameter_shapes', fname='get_parameter_shapes')
    h.method('run',
             api.params(
                 params='std::unordered_map<std::string, migraphx::argument>'),
             returns='migraphx::argument')
