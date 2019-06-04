import string, sys, re, os


trivial = [
    'std::size_t',
    'instruction_ref'
]

headers = '''
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
'''

form = string.Template('''

/*
* Type-erased interface for:
* 
* struct ${struct_name}
* {
${comment_members}
* };
* 
*/

struct ${struct_name}
{
    // Constructors
    ${struct_name} () = default;

    template <typename PrivateDetailTypeErasedT>
    ${struct_name} (PrivateDetailTypeErasedT value) :
        private_detail_te_handle_mem_var (
            std::make_shared<
                private_detail_te_handle_type<typename std::remove_reference<PrivateDetailTypeErasedT>::type>
            >(std::forward<PrivateDetailTypeErasedT>(value))
        )
    {}

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    ${struct_name} & operator= (PrivateDetailTypeErasedT value)
    {
        if (private_detail_te_handle_mem_var.unique())
            *private_detail_te_handle_mem_var = std::forward<PrivateDetailTypeErasedT>(value);
        else if (!private_detail_te_handle_mem_var)
            private_detail_te_handle_mem_var = std::make_shared<PrivateDetailTypeErasedT>(std::forward<PrivateDetailTypeErasedT>(value));
        return *this;
    }

    // Cast
    template<typename PrivateDetailTypeErasedT>
    PrivateDetailTypeErasedT * any_cast()
    {
        return private_detail_te_get_handle().type() == typeid(PrivateDetailTypeErasedT) ?
        std::addressof(static_cast<private_detail_te_handle_type<typename std::remove_cv<PrivateDetailTypeErasedT>::type> &>(private_detail_te_get_handle()).private_detail_te_value) :
        nullptr;
    }

    template<typename PrivateDetailTypeErasedT>
    const typename std::remove_cv<PrivateDetailTypeErasedT>::type * any_cast() const
    {
        return private_detail_te_get_handle().type() == typeid(PrivateDetailTypeErasedT) ?
        std::addressof(static_cast<const private_detail_te_handle_type<typename std::remove_cv<PrivateDetailTypeErasedT>::type> &>(private_detail_te_get_handle()).private_detail_te_value) :
        nullptr;
    }

    const std::type_info& type_id() const
    {
        if(private_detail_te_handle_empty()) return typeid(std::nullptr_t);
        else return private_detail_te_get_handle().type();
    }

    ${nonvirtual_members}

    friend bool is_shared(const ${struct_name} & private_detail_x, const ${struct_name} & private_detail_y)
    {
        return private_detail_x.private_detail_te_handle_mem_var == private_detail_y.private_detail_te_handle_mem_var;
    }

private:
    struct private_detail_te_handle_base_type
    {
        virtual ~private_detail_te_handle_base_type () {}
        virtual std::shared_ptr<private_detail_te_handle_base_type> clone () const = 0;
        virtual const std::type_info& type() const = 0;

        ${pure_virtual_members}
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type :
        private_detail_te_handle_base_type
    {
        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type (PrivateDetailTypeErasedT value,
                typename std::enable_if<
                    std::is_reference<PrivateDetailTypeErasedU>::value
                >::type * = nullptr) :
            private_detail_te_value (value)
        {}

        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type (PrivateDetailTypeErasedT value,
                typename std::enable_if<
                    !std::is_reference<PrivateDetailTypeErasedU>::value,
                    int
                >::type * = nullptr) noexcept :
            private_detail_te_value (std::move(value))
        {}

        std::shared_ptr<private_detail_te_handle_base_type> clone () const override
        { return std::make_shared<private_detail_te_handle_type>(private_detail_te_value); }

        const std::type_info& type() const override
        {
            return typeid(private_detail_te_value);
        }

        ${virtual_members}

        PrivateDetailTypeErasedT private_detail_te_value;
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type<std::reference_wrapper<PrivateDetailTypeErasedT>> :
        private_detail_te_handle_type<PrivateDetailTypeErasedT &>
    {
        private_detail_te_handle_type (std::reference_wrapper<PrivateDetailTypeErasedT> ref) :
            private_detail_te_handle_type<PrivateDetailTypeErasedT &> (ref.get())
        {}
    };

    bool private_detail_te_handle_empty() const
    {
        return private_detail_te_handle_mem_var == nullptr;
    }

    const private_detail_te_handle_base_type & private_detail_te_get_handle () const
    {
        assert(private_detail_te_handle_mem_var != nullptr); 
        return *private_detail_te_handle_mem_var; 
    }

    private_detail_te_handle_base_type & private_detail_te_get_handle ()
    {
        assert(private_detail_te_handle_mem_var != nullptr); 
        if (!private_detail_te_handle_mem_var.unique())
            private_detail_te_handle_mem_var = private_detail_te_handle_mem_var->clone();
        return *private_detail_te_handle_mem_var;
    }

    std::shared_ptr<private_detail_te_handle_base_type> private_detail_te_handle_mem_var;
};

template<typename ValueType>
inline const ValueType * any_cast(const ${struct_name} * x)
{
    return x->any_cast<ValueType>();
}

template<typename ValueType>
inline ValueType * any_cast(${struct_name} * x)
{
    return x->any_cast<ValueType>();
}

template<typename ValueType>
inline ValueType & any_cast(${struct_name} & x)
{
    auto * y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if (y == nullptr) throw std::bad_cast();
    return *y;
}

template<typename ValueType>
inline const ValueType & any_cast(const ${struct_name} & x)
{
    const auto * y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if (y == nullptr) throw std::bad_cast();
    return *y;
}
''')

nonvirtual_member = string.Template('''
${friend} ${return_type} ${name}(${params}) ${const}
{
    assert(${this}.private_detail_te_handle_mem_var);
    ${return_} ${this}.private_detail_te_get_handle().${internal_name}(${member_args});
}
''')

pure_virtual_member = string.Template("virtual ${return_type} ${internal_name}(${member_params}) ${member_const} = 0;\n")

virtual_member = string.Template('''
${return_type} ${internal_name}(${member_params}) ${member_const} override
{
    ${using}
    ${return_} ${call};
}
''')

comment_member = string.Template('''*     ${friend} ${return_type} ${name}(${params}) ${const};''')

def trim_type_name(name):
    n = name.strip()
    if n.startswith('const'):
        return trim_type_name(n[5:])
    if n.endswith(('&', '*')):
        return trim_type_name(n[0:-1])
    return n

def internal_name(name):
    internal_names = {
        'operator<<': 'operator_shift_left',
        'operator>>': 'operator_shift_right',
    }
    if name in internal_names:
        return internal_names[name]
    else:
        return name

def generate_call(m, friend, indirect):
    if m['name'].startswith('operator'):
        op = m['name'][8:]
        args = m['args']
        if ',' in args:
            return args.replace(',', op)
        else:
            return string.Template('${op}${args}').substitute(op=op, args=args)
    if friend:
        return string.Template('${name}(${args})').substitute(m)
    if indirect:
        if m['args']:
            return string.Template('${default}(private_detail_te_value, ${args})').substitute(m)
        else:
            return string.Template('${default}(private_detail_te_value)').substitute(m)
    return string.Template('private_detail_te_value.${name}(${args})').substitute(m)

def convert_member(d, struct_name):
    for name in d:
        member = { 
            'name': name,
            'internal_name': internal_name(name),
            'const': '',
            'member_const': '',
            'friend': '',
            'this': '(*this)',
            'using': '',
            'brief': '',
            'return_': ''
        }
        args = []
        params = []
        member_args = []
        member_params = []
        skip = False
        friend = False
        indirect = False
        if 'friend' in d[name]:
            friend = True
            skip = True
        if 'default' in d[name]:
            indirect = True
        for x in d[name]:
            t = d[name][x]
            if x == 'return':
                member['return_type'] = t if t else 'void'
                if member['return_type'] != 'void': member['return_'] = 'return'
            elif x == 'const':
                member['const'] = 'const'
                member['member_const'] = 'const'
            elif x == 'friend':
                member['friend'] = 'friend'
            elif x == 'default':
                member['default'] = t
            elif x == 'using':
                member['using'] = 'using {};'.format(d[name]['using'])
            elif x == '__brief__':
                member['doc'] = '/// ' + t
            elif x.startswith('__') and x.endswith('__'):
                continue
            else:
                use_member = not(skip and struct_name == trim_type_name(t))
                arg_name = x
                if not use_member: 
                    arg_name = 'private_detail_te_value'
                    member['this'] = x
                    if 'const' in t:
                        member['member_const'] = 'const'
                if t.endswith(('&', '*')) or t in trivial:
                    if use_member: member_args.append(x)
                    args.append(arg_name)
                else:
                    if use_member: member_args.append('std::move({})'.format(x))
                    args.append('std::move({})'.format(arg_name))
                params.append(t+' '+x)
                if use_member: member_params.append(t+' '+x)
                else: skip = False
        member['args'] = ','.join(args)
        member['member_args'] = ','.join(member_args)
        member['params'] = ','.join(params)
        member['params'] = ','.join(params)
        member['member_params'] = ','.join(member_params)
        member['call'] = generate_call(member, friend, indirect)
        return member
    return None


def generate_form(name, members):
    nonvirtual_members = []
    pure_virtual_members = []
    virtual_members = []
    comment_members = []
    for member in members:
        m = convert_member(member, name)
        nonvirtual_members.append(nonvirtual_member.substitute(m))
        pure_virtual_members.append(pure_virtual_member.substitute(m))
        virtual_members.append(virtual_member.substitute(m))
        comment_members.append(comment_member.substitute(m))
    return form.substitute(
        nonvirtual_members=''.join(nonvirtual_members),
        pure_virtual_members=''.join(pure_virtual_members),
        virtual_members=''.join(virtual_members),
        comment_members='\n'.join(comment_members),
        struct_name=name
    )

def virtual(name, returns=None, **kwargs):
    args = kwargs
    args['return'] = returns
    return { name: args }

def friend(name, returns=None, **kwargs):
    args = kwargs
    args['return'] = returns
    args['friend'] = 'friend'
    return { name: args }


def interface(name, *members):
    return generate_form(name, members)

def template_eval(template,**kwargs):
    start = '<%'
    end = '%>'
    escaped = (re.escape(start), re.escape(end))
    mark = re.compile('%s(.*?)%s' % escaped, re.DOTALL)
    for key in kwargs:
        exec('%s = %s' % (key, kwargs[key]))
    for item in mark.findall(template):
        template = template.replace(start+item+end, str(eval(item.strip())))
    return template

f = open(sys.argv[1]).read()
r = template_eval(f)
sys.stdout.write(r)
