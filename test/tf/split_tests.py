import sys, os, string

# tf/
#   include/
#   models/
#   tests/

def collect_test_cases(files):
    test_cases = []
    for file in files:
        test_case = None
        for line in open(file).readlines():
            # print(line)
            if line.startswith('TEST_CASE'):
                test_case = line
            elif test_case:
                test_case = test_case + line
            if line.startswith('}'):
                if test_case:
                    test_cases.append(test_case)
                test_case = None
    return test_cases

def get_function_parameter(case, names):
    for name in names:
        if not name in case:
            continue
        n = len(name) + 2
        i = case.index(name) + n
        end = case.find('"', i)
        return case[i:end]


def group_by(l, select):
    result = {}
    for item in l:
        key = select(item)
        if not key in result:
            result[key] = []
        result[key].append(item)
    return result

def removesuffix(s):
    if '.' in s:
        return s.rsplit('.', 1)[0]
    return s

def write_to(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

header_guard_template = string.Template('''
#ifndef MIGRAPHX_GUARD_TEST_TF_${name}_HPP
#define MIGRAPHX_GUARD_TEST_TF_${name}_HPP

${content}

#endif
''')

conv_test_util = '''
migraphx::program create_conv()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    std::vector<float> weight_data(3 * 3 * 3 * 32);
    std::fill(weight_data.begin(), weight_data.end(), 1.0f);
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::float_type, {3, 3, 3, 32}}, weight_data);

    migraphx::op::convolution op;
    op.padding  = {1, 1, 1, 1};
    op.stride   = {1, 1};
    op.dilation = {1, 1};
    auto l2 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 0, 1}}}), l1);
    mm->add_instruction(op, l0, l2);
    return p;
}
'''

def write_header(filename, content):
    basename = os.path.basename(filename)
    name = removesuffix(basename).upper()
    write_to(filename, header_guard_template.substitute(content=content, name=name))


include_guide = {
    'conv_test': 'conv_test_utils.hpp',
    'conv_add_test': 'conv_test_utils.hpp',
    'conv_nchw_test': 'conv_test_utils.hpp',
    'conv_relu_test': 'conv_test_utils.hpp',
    'conv_relu6_test': 'conv_test_utils.hpp',
}

def create_includes(case):
    includes = ""
    for key, include in include_guide.items():
        if not key in case:
            continue
        if include in includes:
            continue
        includes = "#include <{}>\n".format(include)
    return includes

parse_template = string.Template('''
#include <tf_test.hpp>
${includes}

${test_case}

''')

main_test = '''
#include <test.hpp>

int main(int argc, const char* argv[]) { test::run(argc, argv); }
'''


def write_case(p, name, cases):
    content = '\n'.join(cases)
    includes = create_includes(content)
    content = parse_template.substitute(test_case=content, includes=includes)
    write_to(os.path.join(p, name+'.cpp'), content)


def get_pb_file(case):
    param = get_function_parameter(case, ['optimize_tf', 'parse_tf'])
    if not param:
        print("No pb file found for test case:\n", case)
    basename = removesuffix(param)
    return basename.replace('/', '_')

def main():
    args = sys.argv
    tf_dir = args[1]
    tf_test = os.path.join(tf_dir, 'tf_test.cpp')
    group_cases = group_by(collect_test_cases([tf_test]), get_pb_file)
    parse_dir = os.path.join(tf_dir, 'tests')
    for key, cases in group_cases.items():
        write_case(parse_dir, key, cases)
    write_to(os.path.join(parse_dir, 'main.cpp'), main_test)
    include_dir = os.path.join(tf_dir, 'include')
    write_header(os.path.join(include_dir, 'tf_conv_utils.hpp'), conv_test_util)

if __name__ == "__main__":
    main()