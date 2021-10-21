
import tensorflow as tf
from ai_benchmark import AIBenchmark

# Test if tensorflow can detect hardware successfully (CUDA will fail on M1)
def tf_test():
 tf.test.gpu_device_name(),
 tf.config.list_physical_devices("GPU"),
 print(tf.test.is_built_with_cuda()),
 return

#Run benchmark for tensorflow 
def benchmark(use_CPU=None, verbose_level=3):
  benchmark = AIBenchmark(use_CPU, verbose_level)
  # results = benchmark.run(precision="high")
  result = benchmark.run()