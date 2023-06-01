Matchers
========

Introduction
------------

The matchers provide a way compose several predicates together. Many of the matchers can be composed so that ``m(m1, m2)`` will first check that ``m`` matches and then it will check that ``m1`` and ``m2`` will match.

The most commonly-used matcher is the ``name`` matcher. It will match the instruction that have the operator that is equal to the name specified::

    auto match_sum = name("sum");

This will find ``sum`` operators. We can also find ``sum`` operators which the output is ``standard_shape``:

    auto match_sum = name("sum")(standard_shape());

Arguments
---------

We also want to match arguments to the instructions as well. One way, is to match each argument using the ``arg`` matcher::

    auto match_sum = name("sum")(arg(0)(name("@literal"), arg(1)(name("@literal"))));

This will match a ``sum`` operator with the two arguments that are literals. Of course, instead of writing ``arg(0)`` and ``arg(1)`` everytime, the ``args`` matcher can be used::

    auto match_sum = name("sum")(args(name("@literal"), name("@literal")));


Binding
-------

As we traverse through the instructions we may want reference some of the instructions we find along the way. We can do this by calling ``.bind``::

    auto match_sum = name("sum")(args(
                                    name("@literal").bind("one"), 
                                    name("@literal").bind("two")
                                )).bind("sum");


This will associate the instruction to a name that can be read from the ``matcher_result`` when it matches.

Finding matches
---------------

Finally, when you want to use the matchers to find instructions a callback object can be written which has the matcher and an ``apply`` function which will take the ``matcher_result`` when the match is found::

    struct match_find_sum
    {
        auto matcher() const { return name("sum"); }

        void apply(program& p, matcher_result r) const 
        { 
            // Do something with the result
        }
    };

    find_matches(prog, match_find_sum{});


Creating matchers
-----------------

There are several ways to create matchers. The macros ``MIGRAPH_BASIC_MATCHER`` and ``MIGRAPH_PRED_MATCHER`` help with creating matchers. For example, we can create a matcher for shapes that are broadcasted::

    MIGRAPH_PRED_MATCHER(broadcasted_shape, instruction_ref ins) 
    { 
        return ins->get_shape().broadcasted(); 
    }


If we want parameters to the predicate, then we will need to use the ``make_basic_pred_matcher`` to create the matcher. For example, here is how we would create a matcher to check the number of dimensions of the shape::

    inline auto number_of_dims(std::size_t n)
    {
        return make_basic_pred_matcher([=](instruction_ref ins) { 
            return ins->get_shape().lens().size() == n; 
        });
    }


