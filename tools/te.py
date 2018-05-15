import string, sys, re, os

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

    ${nonvirtual_members}

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

    const private_detail_te_handle_base_type & private_detail_te_get_handle () const
    { return *private_detail_te_handle_mem_var; }

    private_detail_te_handle_base_type & private_detail_te_get_handle ()
    {
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
${return_type} ${name}(${params}) ${const}
{
    assert(private_detail_te_handle_mem_var);
    return private_detail_te_get_handle().${name}(${args});
}
''')

pure_virtual_member = string.Template("virtual ${return_type} ${name}(${params}) ${const} = 0;\n")

virtual_member = string.Template('''
${return_type} ${name}(${params}) ${const} override
{
    return private_detail_te_value.${name}(${args});
}
''')

comment_member = string.Template('''*     ${return_type} ${name}(${params}) ${const};''')

def convert_member(d):
    for name in d:
        member = { 'name': name, 'const': ''}
        args = []
        params = []
        for x in d[name]:
            t = d[name][x]
            if x == 'return':
                member['return_type'] = t
            elif x == 'const':
                member['const'] = 'const'
            else:
                if t.endswith(('&', '*')):
                    args.append(x)
                else:
                    args.append('std::move({})'.format(x))
                params.append(t+' '+x)
        member['args'] = ','.join(args)
        member['params'] = ','.join(params)
        return member
    return None


def generate_form(name, members):
    nonvirtual_members = []
    pure_virtual_members = []
    virtual_members = []
    comment_members = []
    for member in members:
        m = convert_member(member)
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


def interface(name, *members):
    return generate_form(name, members)

def template_eval(template,**kwargs):
    start = '{%'
    end = '%}'
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
