
import tensorflow as tf


def tf_test():
 tf.test.gpu_device_name(),
 tf.config.list_physical_devices("GPU"),
 print(tf.test.is_built_with_cuda()),
 return


