import coremltools as ct
import tensorflow as tf

class ModelOptimizer:
    def __init__(self):
        self.compute_units = ct.ComputeUnit.ALL
    
    def quantize_model(self, model):
 
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert()
    
    def convert_to_coreml(self, model, input_shape=(1, 224, 224, 3)):
     
        mlmodel = ct.convert(
            model,
            source='tensorflow',
            inputs=[ct.ImageType(
                name="input_1", 
                shape=input_shape,
                scale=1/255.0
            )],
            compute_units=self.compute_units,
            minimum_deployment_target=ct.target.iOS15,
            convert_to="mlprogram"
        )
        return mlmodel 