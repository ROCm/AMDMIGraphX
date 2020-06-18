import api


def bad_param_error(msg):
    return 'MIGRAPHX_THROW(migraphx_status_bad_param, "{}")'.format(msg)


api.error_type = 'migraphx_status'
api.success_type = 'migraphx_status_success'
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
        p.read = '${name} == nullptr ? migraphx::compile_options{} : migraphx::to_compile_options(*${name})'


@api.cwrap('migraphx::onnx_options')
def onnx_options_type_wrap(p):
    if p.returns:
        p.add_param('migraphx_onnx_options *')
        p.bad_param('${name} == nullptr', 'Null pointer')
        p.write = ['*${name} = migraphx::to_onnx_options(${result})']
    else:
        p.add_param('migraphx_onnx_options *')
        p.read = '${name} == nullptr ? migraphx::onnx_options{} : migraphx::to_onnx_options(*${name})'


def auto_handle(f):
    return api.handle('migraphx_' + f.__name__, 'migraphx::' + f.__name__)(f)


@auto_handle
def shape(h):
    h.constructor(
        'create',
        api.params(type='migraphx::shape::type_t',
                   lengths='std::vector<size_t>'))
    h.constructor(
        'create_with_strides',
        api.params(type='migraphx::shape::type_t',
                   lengths='std::vector<size_t>',
                   strides='std::vector<size_t>'))
    h.constructor('create_scalar', api.params(type='migraphx::shape::type_t'))
    h.method('lengths',
             fname='lens',
             returns='const std::vector<size_t>&',
             const=True)
    h.method('strides', returns='const std::vector<size_t>&', const=True)
    h.method('type', returns='migraphx::shape::type_t', const=True)
    h.method('bytes', returns='size_t', const=True)
    h.method('equal',
             api.params(x='const migraphx::shape&'),
             invoke='migraphx::equal($@)',
             returns='bool',
             const=True)


@auto_handle
def argument(h):
    h.constructor('create',
                  api.params(shape='const migraphx::shape&', buffer='void*'))
    h.method('shape',
             fname='get_shape',
             cpp_name='get_shape',
             returns='const migraphx::shape&',
             const=True)
    h.method('buffer',
             fname='data',
             cpp_name='data',
             returns='char*',
             const=True)
    h.method('equal',
             api.params(x='const migraphx::argument&'),
             invoke='migraphx::equal($@)',
             returns='bool',
             const=True)


api.add_function('migraphx_argument_generate',
                 api.params(s='const migraphx::shape&', seed='size_t'),
                 fname='migraphx::generate_argument',
                 returns='migraphx::argument')


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
             fname='at',
             cpp_name='operator[]',
             returns='const migraphx::shape&')
    h.method('names',
             invoke='migraphx::get_names(${program_parameter_shapes})',
             returns='std::vector<const char*>')


@api.handle('migraphx_program_parameters',
            'std::unordered_map<std::string, migraphx::argument>')
def program_parameters(h):
    h.constructor('create')
    h.method('add',
             api.params(name='const char*',
                        argument='const migraphx::argument&'),
             invoke='${program_parameters}[${name}] = ${argument}')


@api.handle('migraphx_arguments', 'std::vector<migraphx::argument>')
def arguments(h):
    h.method('size', returns='size_t')
    h.method('get',
             api.params(idx='size_t'),
             fname='at',
             cpp_name='operator[]',
             returns='const migraphx::argument&')


@api.handle('migraphx_shapes', 'std::vector<migraphx::shape>')
def shapes(h):
    h.method('size', returns='size_t')
    h.method('get',
             api.params(idx='size_t'),
             fname='at',
             cpp_name='operator[]',
             returns='const migraphx::shape&')


@auto_handle
def program(h):
    h.method(
        'compile',
        api.params(target='migraphx::target',
                   options='migraphx::compile_options'))
    h.method('get_parameter_shapes',
             returns='std::unordered_map<std::string, migraphx::shape>')
    h.method('get_output_shapes',
             invoke='migraphx::get_output_shapes($@)',
             returns='std::vector<migraphx::shape>')
    h.method('print', invoke='migraphx::print($@)', const=True)
    h.method('run',
             api.params(
                 params='std::unordered_map<std::string, migraphx::argument>'),
             invoke='migraphx::run($@)',
             returns='std::vector<migraphx::argument>')
    h.method('equal',
             api.params(x='const migraphx::program&'),
             invoke='migraphx::equal($@)',
             returns='bool',
             const=True)


@auto_handle
def onnx_options(h):
    h.constructor('create')
    h.method(
        'set_input_parameter_shape',
        api.params(name='const char*', dims='std::vector<size_t>'),
        invoke='migraphx::set_input_parameter_shape($@)',
    )
    h.method(
        'set_default_dim_value',
        api.params(value='size_t'),
        invoke='migraphx::set_default_dim_value($@)',
    )


api.add_function('migraphx_parse_onnx',
                 api.params(name='const char*',
                            options='migraphx::onnx_options'),
                 fname='migraphx::parse_onnx',
                 returns='migraphx::program')

api.add_function('migraphx_parse_onnx_buffer',
                 api.params(data='const void*',
                            size='size_t',
                            options='migraphx::onnx_options'),
                 fname='migraphx::parse_onnx_buffer',
                 returns='migraphx::program')


@api.handle('migraphx_quantize_op_names', 'std::vector<std::string>')
def quantize_op_names(h):
    h.constructor('create')
    h.method('add', api.params(name='const char*'), fname='push_back')


api.add_function('migraphx_quantize_fp16',
                 api.params(prog='migraphx::program&',
                            name='std::vector<std::string>&'),
                 fname='migraphx::quantize_fp16_wrap')

api.add_function('migraphx_quantize_fp16_default',
                 api.params(prog='migraphx::program&'),
                 fname='migraphx::quantize_fp16')


@auto_handle
def quantize_options(h):
    h.constructor('create')
    h.method(
        'add_op_name',
        api.params(name='const char*'),
        invoke='migraphx::add_op_name($@)',
    )
    h.method(
        'add_calibration_data',
        api.params(data='std::unordered_map<std::string, migraphx::argument>'),
        invoke='migraphx::add_calibration_data($@)',
    )


api.add_function('migraphx_quantize_int8',
                 api.params(prog='migraphx::program&',
                            target='migraphx::target',
                            options='migraphx::quantize_options'),
                 fname='migraphx::quantize_int8_wrap')
