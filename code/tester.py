
import tensorflow as tf
from ai_benchmark import AIBenchmark

def tf_test():
 tf.test.gpu_device_name(),
 tf.config.list_physical_devices("GPU"),
 print(tf.test.is_built_with_cuda()),
 return

def benchmark():
    #Reu benchmark for tensorflow 
    benchmark = AIBenchmark(use_CPU=None, verbose_level=3)
    # results = benchmark.run(precision="high")
    result = benchmark.run()