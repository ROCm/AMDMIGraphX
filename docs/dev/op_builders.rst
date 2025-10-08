==================================
Op Builders
==================================

Overview
========

Op Builders simplify the creation of complex operations by composing multiple operators together. 
Op builders serve as a bridge between high-level operations (e.g. GEMM, Convolution, MatMul) and the low-level instruction graph that MIGraphX operates on.

What are Op Builders?
=====================

Op Builders encapsulate the logic for creating complex operations from individual MIGraphX primitive operators 
(a primitive operator being an operator that has its own reference and/or gpu implementation e.g. dot, multibroadcast, slice, add).

It is important to note that Op Builders do not introduce new constructs in the MIGraphX IR, they merely insert multiple instructions into the module.

The primary objective of Op Builders is to minimize code duplication across code locations which need to utilize the same complex operations. 
Take for example a TensorFlow and ONNX parser implementations for an operator like Einsum. Having the builder for Einsum enables us to simply perform the necessary 
parsing of attributes and then call the builder, instead of performing the graph building in situ in both parsers, thus duplicating code.

This example makes it clear that an op builder such as the one mentioned must be a superset of both TF and ONNX operation functionality if it is to satisfy both. 
The parsers have to configure the builder calls appropriately via the builder attributes.

Builders may use other builders as well. Taking again the example of Einsum, a quite complex operation, which might utilize a MatMul. 
Instead of inserting all the instructions that constitute a MatMul, the Einsum builder would just call the MatMul builder.


Usage 
===============

.. code-block:: cpp

    #include <migraphx/op/builder/insert.hpp>

    migraphx::module m;
    migraphx::options{};
    ...
    auto a = m.add_parameter(...)
    auto b = m.add_parameter(...)
    options.insert({"alpha", 2});
    options.insert({"transB", true});

    // Add operation at end of module
    auto result = migraphx::op::builder::add("gemm", m, {a, b}, options);

    // Insert operation at specific location
    auto result = migraphx::op::builder::insert("gemm", module, insert_location, {a, b}, options);

Builders are intended to be used via two helper functions, with two overloads each:
 * add - builder will insert instructions at the end of the module
 * insert - builder will insert instructions at the provided insertion location

.. code-block:: cpp

    std::vector<instruction_ref> insert(const std::string& name,
                                        module& m,
                                        instruction_ref ins,
                                        const std::vector<instruction_ref>& args,
                                        const value& options);
    
    std::vector<instruction_ref> insert(const std::string& name,
                                        module& m,
                                        instruction_ref ins,
                                        const std::vector<instruction_ref>& args,
                                        const std::vector<module_ref>& module_args,
                                        const value& options);
    
    std::vector<instruction_ref> add(const std::string& name,
                                     module& m,
                                     const std::vector<instruction_ref>& args,
                                     const value& options);
    
    std::vector<instruction_ref> add(const std::string& name,
                                     module& m,
                                     const std::vector<instruction_ref>& args,
                                     const std::vector<module_ref>& module_args,
                                     const value& options);


 * name - specify the name of the builder that is to be used
 * m - module into which the builder will insert instructions
 * ins - location in module at which the builder will insert instructions
 * args - inputs for the builder operation
 * module_args - submodule inputs for the builder operations
 * options - dictionary of key-value pairs with which the builder attributes will be initialized. The attribute name is the key of the pair. 

Details
============

1. Base Class
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

 template <class T>
 struct op_builder : auto_register<register_builder_action, T>
 {
    static std::string name()
    {
        static const std::string& name = get_type_name<T>();
        return name.substr(name.rfind("::") + 2);
    }
 };

The base class that all op builders must inherit from. It wraps builder registration into the builder registry and a 
default implementation of the name() method that all builders need to provide. It is nearly identical to the operator base class.

2. Builder  Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

 struct gelu_quick : op_builder<gelu_quick>
 {
    float alpha = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x         = args[0];
        auto x_type    = x->get_shape().type();
        auto alpha_lit = m.add_literal(migraphx::literal{migraphx::shape{x_type}, {alpha}});
        auto mul_alpha = insert_common_op(m, ins, make_op("mul"), {alpha_lit, x});
        auto sigmoid   = m.insert_instruction(ins, migraphx::make_op("sigmoid"), mul_alpha);
        return {insert_common_op(m, ins, make_op("mul"), {x, sigmoid})};
    }
 };

Each builder must provide a reflect, name and insert method.

A default implementation of the name method is provided by the op_builder base class. The string it returns will be the same as the name of the builder struct.

The reflect method is used for serialization. All struct members that need to be serialized must be referenced in the implementation.

The insert method is used by all builder wrapper functions. It implements the graph building that the builder performs. If the builder requires any submodules as inputs, the method signature can be:

.. code-block:: cpp

 std::vector<instruction_ref> insert(module& m, 
                                     instruction_ref ins, 
                                     const std::vector<instruction_ref>&args, 
                                     const std::vector<module_ref>& module_args) const;