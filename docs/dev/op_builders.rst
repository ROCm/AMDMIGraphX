==================================
Op Builders
==================================

Overview
========

Op Builders simplify the creation of complex operations by composing multiple operators together. 
They act as a bridge between high-level operations (e.g. GEMM, Convolution, MatMul) and the low-level instruction graph that MIGraphX operates on.

What are Op Builders?
=====================

Op Builders encapsulate the logic for creating complex operations from individual MIGraphX primitive operators 
(where a primitive operator is that which has its own reference and/or GPU implementation e.g., dot, multibroadcast, slice, add).


.. note::

   Op Builders do not introduce new constructs in the MIGraphX IR, 
   they merely insert multiple instructions into the module.

The primary goal of Op Builders is to reduce code duplication in places where the same complex operations are needed. 
For example, TensorFlow and ONNX parser implementations for an operator like Einsum can use a shared builder. 
Each parser simply parses attributes and calls the builder, instead of duplicating graph construction logic in multiple locations.

Because of this, a builder must be a superset of the functionality needed for all parsers that use it. 
Parsers configure the builder appropriately using its attributes.

Builders can also call other builders. For instance, the Einsum builder may internally use the MatMul builder rather than manually inserting all instructions for a MatMul operation.


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

* **add** – inserts instructions at the end of the module 
* **insert** – inserts instructions at a specified location

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


* **name** – the builder name to use
* **m** – module where instructions are inserted
* **ins** – insertion location within the module
* **args** – input instructions for the operation
* **module_args** – submodule inputs for the operation
* **options** – key-value pairs initializing the builder attributes, where the attribute name is the key of the pair

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

All op builders must inherit from this base class. It handles registration in the builder registry and 
provides a default ``name()`` method. This class is nearly identical to the operator base class. 

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

Each builder must implement ``reflect``, ``name``, and ``insert`` methods:

* **name** – provided by the base class; returns the builder struct name 

* **reflect** – used for serialization; all members to be serialized must be referenced here

* **insert** – implements the graph-building logic of the builder. If submodules are required, the signature can be as follows:

.. code-block:: cpp

 std::vector<instruction_ref> insert(module& m, 
                                     instruction_ref ins, 
                                     const std::vector<instruction_ref>&args, 
                                     const std::vector<module_ref>& module_args) const;