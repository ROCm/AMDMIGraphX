# Exporting Frozen Graphs in Tensorflow 2 
In order to use a trained model as input to MIGraphX, the model must be first be saved in a frozen graph format. This was accomplished in Tensorflow 1 by launching a graph in a tf.Session and then saving the session. However, Tensorflow has decided to deprecate Sessions in favor of functions and SavedModel format.  

## SavedModel format
The simplest way to save a model is through saved\_model.save()

```
MODEL_NAME = <my_model>
tf.saved_model.save(model, "./models/{}".format(MODEL_NAME))
```

This will create an equivalent tensorflow program which can later be loaded for fine-tuning or inference, although it is not compatible with MIGraphX.

## Convert to ConcreteFunction
First, we need to get the function equivalent of the model:
```
full_model = tf.function(lambda x: model(x))
```
Next, we concretize the function to avoid retracing:
```
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
```

## Freeze ConcreteFunction and Serialize
Since we are saving the graph for the purpose of inference, all variables can be made constant (i.e. "frozen"):
```
frozen_func = convert_variables_to_constants_v2(full_model)
```
Next, we obtain a serialized GraphDef representation of the graph:
```
frozen_func.graph.as_graph_def()
```
*\*Optional*: The layers of the graph can be manually verified by including the following:
```
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
```

## Save Frozen Graph as Protobuf
Finally, we save to hard drive. 
```
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="{}_frozen_graph.pb".format(MODEL_NAME),
                  as_text=False)
```
And now the frozen graph is stored as `./frozen_models/<MODEL_NAME>_frozen_graph.pb`

To quickly verify that the graph was succesfully saved in a compatible way:
```
/opt/rocm/bin/migraphx-driver read ./frozen_models/<MODEL_NAME>_frozen_graph.pb
```

