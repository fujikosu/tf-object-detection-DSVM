import tensorflow as tf

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

ItemHandler = tf.contrib.slim.tfexample_decoder.ItemHandler

class BackupHandler(ItemHandler):
  """An ItemHandler that tries two ItemHandlers in order."""

  def __init__(self, handler, backup):
    """Initializes the BackupHandler handler.
    If the first Handler's tensors_to_item returns a Tensor with no elements,
    the second Handler is used.
    Args:
      handler: The primary ItemHandler.
      backup: The backup ItemHandler.
    Raises:
      ValueError: if either is not an ItemHandler.
    """
    if not isinstance(handler, ItemHandler):
      raise ValueError('Primary handler is of type %s instead of ItemHandler'
                       % type(handler))
    if not isinstance(backup, ItemHandler):
      raise ValueError('Backup handler is of type %s instead of ItemHandler'
                       % type(backup))
    self._handler = handler
    self._backup = backup
    super(BackupHandler, self).__init__(handler.keys + backup.keys)

  def tensors_to_item(self, keys_to_tensors):
    item = self._handler.tensors_to_item(keys_to_tensors)
    return control_flow_ops.cond(
        pred=math_ops.equal(math_ops.reduce_prod(array_ops.shape(item)), 0),
        true_fn=lambda: self._backup.tensors_to_item(keys_to_tensors),
        false_fn=lambda: item)

class Tensor(ItemHandler):
  """An ItemHandler that returns a parsed Tensor."""

  def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0):
    """Initializes the Tensor handler.
    Tensors are, by default, returned without any reshaping. However, there are
    two mechanisms which allow reshaping to occur at load time. If `shape_keys`
    is provided, both the `Tensor` corresponding to `tensor_key` and
    `shape_keys` is loaded and the former `Tensor` is reshaped with the values
    of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
    corresponding to `tensor_key` is loaded and reshape appropriately.
    If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
    returned without any reshaping.
    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.
    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    if shape_keys and shape is not None:
      raise ValueError('Cannot specify both shape_keys and shape parameters.')
    if shape_keys and not isinstance(shape_keys, list):
      shape_keys = [shape_keys]
    self._tensor_key = tensor_key
    self._shape_keys = shape_keys
    self._shape = shape
    self._default_value = default_value
    keys = [tensor_key]
    if shape_keys:
      keys.extend(shape_keys)
    super(Tensor, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._tensor_key]
    shape = self._shape
    if self._shape_keys:
      shape_dims = []
      for k in self._shape_keys:
        shape_dim = keys_to_tensors[k]
        if isinstance(shape_dim, sparse_tensor.SparseTensor):
          shape_dim = sparse_ops.sparse_tensor_to_dense(shape_dim)
        shape_dims.append(shape_dim)
      shape = array_ops.reshape(array_ops.stack(shape_dims), [-1])
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if shape is not None:
        tensor = sparse_ops.sparse_reshape(tensor, shape)
      tensor = sparse_ops.sparse_tensor_to_dense(tensor, self._default_value)
    else:
      if shape is not None:
        tensor = array_ops.reshape(tensor, shape)
    return tensor

class LookupTensor(Tensor):
  """An ItemHandler that returns a parsed Tensor, the result of a lookup."""

  def __init__(self,
               tensor_key,
               table,
               shape_keys=None,
               shape=None,
               default_value=''):
    """Initializes the LookupTensor handler.
    See Tensor.  Simply calls a vocabulary (most often, a label mapping) lookup.
    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      table: A tf.lookup table.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.
    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    self._table = table
    super(LookupTensor, self).__init__(tensor_key, shape_keys, shape,
                                       default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(LookupTensor, self).tensors_to_item(keys_to_tensors)
    return self._table.lookup(unmapped_tensor)