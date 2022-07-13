# coding=utf-8
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pytype: skip-file

import unittest
from io import StringIO

import mock

from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline

from . import runinference

def check_torch_keyed_model_handler(actual):
  expected = '''[START torch_keyed_model_handler]
('first_question', PredictionResult(example=tensor([105.]), inference=tensor([523.6982], grad_fn=<UnbindBackward0>)))
('second_question', PredictionResult(example=tensor([108.]), inference=tensor([538.5867], grad_fn=<UnbindBackward0>)))
('third_question', PredictionResult(example=tensor([1000.]), inference=tensor([4965.4019], grad_fn=<UnbindBackward0>)))
('fourth_question', PredictionResult(example=tensor([1013.]), inference=tensor([5029.9180], grad_fn=<UnbindBackward0>)))
[END torch_keyed_model_handler]'''.splitlines()[1:-1]
  assert_matches_stdout(actual, expected)


def check_sklearn_keyed_model_handler(actual):
  expected = '''[START sklearn_keyed_model_handler]
('first_question', PredictionResult(example=[105.0], inference=array([525.])))
('second_question', PredictionResult(example=[108.0], inference=array([540.])))
('third_question', PredictionResult(example=[1000.0], inference=array([5000.])))
('fourth_question', PredictionResult(example=[1013.0], inference=array([5065.])))
[END sklearn_keyed_model_handler] '''.splitlines()[1:-1]
  assert_matches_stdout(actual, expected)


def check_torch_unkeyed_model_handler(actual):
  expected = '''[START torch_unkeyed_model_handler]
PredictionResult(example=tensor([10.]), inference=tensor([52.2325], grad_fn=<UnbindBackward0>))
PredictionResult(example=tensor([40.]), inference=tensor([201.1165], grad_fn=<UnbindBackward0>))
PredictionResult(example=tensor([60.]), inference=tensor([300.3724], grad_fn=<UnbindBackward0>))
PredictionResult(example=tensor([90.]), inference=tensor([449.2563], grad_fn=<UnbindBackward0>))
[END torch_unkeyed_model_handler] '''.splitlines()[1:-1]
  assert_matches_stdout(actual, expected)


def check_sklearn_unkeyed_model_handler(actual):
  expected = '''[START sklearn_unkeyed_model_handler]
PredictionResult(example=array([20.], dtype=float32), inference=array([100.], dtype=float32))
PredictionResult(example=array([40.], dtype=float32), inference=array([200.], dtype=float32))
PredictionResult(example=array([60.], dtype=float32), inference=array([300.], dtype=float32))
PredictionResult(example=array([90.], dtype=float32), inference=array([450.], dtype=float32))
[END sklearn_unkeyed_model_handler]  '''.splitlines()[1:-1]
  assert_matches_stdout(actual, expected)

def check_images(actual):
  expected = '''[START images]
('img1', PredictionResult(example=tensor([[[-1.5000, -1.0000, -0.5000],
         [-0.5000,  0.0000,  0.5000],
         [ 0.5000,  1.0000,  1.5000]],

        [[-1.5000, -1.0000, -0.5000],
         [-0.5000,  0.0000,  0.5000],
         [ 0.5000,  1.0000,  1.5000]],

        [[-1.5000, -1.0000, -0.5000],
         [-0.5000,  0.0000,  0.5000],
         [ 0.5000,  1.0000,  1.5000]]]), inference=tensor([ 3.5683e-01,  9.0884e-02,  3.2032e-02, -2.7272e-02,  1.0669e-01,
         2.2701e-01, -3.1091e-03, -5.4634e-02, -4.0783e-02, -8.0977e-02,
        -5.5986e-04, -1.4602e-01, -1.5191e-01, -1.5954e-01,  1.4563e-01,
        -5.7862e-02, -1.5852e-01, -1.2437e-02, -1.4222e-01,  1.0028e-03,
        -7.9435e-02,  1.4894e-01,  2.8163e-02, -5.3855e-03, -7.9888e-02,
         4.6348e-02,  1.1197e-01, -3.7419e-02,  2.3920e-01, -2.8248e-02,
        -2.6650e-02, -8.5504e-03,  1.5207e-02,  7.9291e-02,  1.5565e-01,
        -7.1944e-02, -4.6512e-02, -1.8924e-01,  4.6965e-02, -1.5944e-03,
         1.3959e-01,  9.8568e-02,  2.0420e-02,  9.4112e-02,  3.1174e-02,
         1.0087e-02, -1.3086e-02,  1.1846e-01,  8.7540e-02, -1.2301e-01,
         3.2854e-02,  8.4917e-02,  1.2448e-01, -2.5287e-02, -7.3409e-02,
        -1.1591e-02, -1.4769e-01, -7.9151e-02,  2.5951e-03,  1.3660e-01,
         9.5405e-03,  2.4378e-02,  8.0256e-02,  2.5951e-01,  1.6304e-01,
         2.0210e-01,  1.5987e-02, -8.9112e-02,  5.8570e-02, -5.5732e-02,
        -4.7485e-02,  6.1771e-02, -2.8878e-02,  5.0312e-02, -1.9037e-01,
        -1.8129e-01, -1.9530e-01, -2.5573e-01,  2.7371e-01, -2.8674e-02,
         2.0790e-02, -6.4327e-02, -4.1551e-02, -1.2290e-01,  3.8615e-02,
        -1.3975e-01, -6.3646e-04,  9.7620e-02, -1.7438e-01,  1.6956e-01,
        -2.0965e-01, -1.3273e-01, -1.3451e-01, -3.4363e-01, -1.2360e-01,
        -3.1218e-02, -5.1001e-02, -1.3107e-01, -1.8724e-01,  2.9557e-02,
        -1.0874e-01,  1.5818e-01, -6.5634e-03,  3.1777e-01,  4.9741e-02,
        -1.9474e-01, -3.6570e-02,  2.0813e-01,  1.3213e-01,  5.6386e-02,
         6.9549e-02,  2.9412e-01, -1.3775e-01, -1.2358e-01, -1.4398e-01,
        -4.9040e-02,  2.5956e-01,  6.1848e-02,  2.1823e-02, -1.3804e-01,
        -1.0539e-02, -4.5413e-02, -7.8384e-02, -2.3806e-01, -9.2057e-02,
        -1.9528e-01,  4.4539e-02, -4.0086e-02,  9.7053e-02, -9.8453e-02,
        -2.5066e-02,  9.3256e-02,  5.9387e-02, -8.3502e-02, -1.9050e-01,
         7.7315e-02, -9.3390e-02, -2.0135e-01, -5.6411e-03, -2.5194e-01,
        -2.3343e-01, -6.5554e-02, -2.3221e-01, -1.5690e-02, -1.2314e-01,
        -1.8637e-01,  7.0443e-02,  2.1268e-01, -1.2352e-01,  1.8688e-01,
        -5.5663e-02, -6.7409e-02, -9.6119e-02,  1.4398e-01, -4.9087e-02,
         6.6166e-02, -6.7278e-02,  1.7448e-02,  6.6993e-02,  1.3325e-01,
         1.5309e-03,  8.6180e-02,  1.6454e-01,  1.8724e-01,  8.5476e-02,
         1.7512e-01,  7.0847e-02,  1.5982e-02,  2.9516e-01,  5.6259e-02,
        -1.2611e-01,  1.7295e-01, -6.4119e-03,  6.8328e-02,  5.1926e-02,
        -6.3890e-02, -2.2654e-02, -4.9559e-02,  2.1164e-01,  1.7332e-01,
         1.4771e-01,  6.9247e-02, -1.8422e-01, -3.9242e-02, -1.3469e-01,
        -4.6020e-03, -1.3514e-01,  4.0221e-02,  3.0310e-02,  1.0013e-01,
         1.1439e-01,  8.1460e-02,  7.4241e-02, -3.7900e-02,  1.1842e-02,
         1.3830e-01, -5.7671e-02, -1.3765e-01, -1.0405e-01,  5.5148e-02,
        -4.2595e-02,  3.7954e-04,  1.4831e-02,  5.6354e-02,  4.6033e-03,
        -7.1437e-02,  3.2145e-02, -8.4454e-02,  2.9462e-02,  1.2100e-01,
         1.7419e-01,  1.2659e-01,  1.6796e-02, -4.9309e-02, -3.8073e-02,
         6.1991e-03,  7.8503e-02, -4.7729e-02,  8.9091e-02, -1.4053e-01,
        -4.3045e-02, -5.9660e-02, -1.4073e-03,  5.3842e-02, -7.7723e-02,
         1.0567e-01,  2.4412e-02, -2.0184e-02,  2.3346e-02,  1.1960e-01,
        -3.8588e-02, -1.4864e-01, -1.9793e-02,  6.6383e-02,  1.0551e-01,
         7.0635e-03,  2.2988e-01,  1.4779e-01, -5.9922e-02, -6.0558e-02,
        -1.1841e-01,  6.3095e-02,  1.8479e-01,  1.6809e-01, -6.5633e-02,
         1.6512e-01,  3.0770e-02,  4.4539e-02, -6.2504e-02, -1.0756e-01,
        -8.4295e-02,  5.2747e-02,  1.4774e-01,  3.7409e-02,  8.8617e-02,
        -1.7419e-01, -6.3943e-02,  3.4330e-02, -2.2261e-02, -1.6577e-01,
         1.2981e-01, -6.2774e-02,  8.8273e-03,  3.1526e-02,  3.0241e-02,
        -3.5171e-02,  8.5848e-02, -1.1499e-01,  1.6733e-01, -9.5837e-02,
        -2.2827e-02, -1.1032e-02,  5.7182e-02, -1.5080e-02, -4.6881e-03,
        -1.3489e-01, -1.1560e-01, -2.0596e-01, -2.1369e-01, -8.7695e-02,
        -1.8341e-01,  7.5218e-02, -7.6128e-02, -1.0770e-01, -3.5446e-02,
        -3.5699e-03, -6.9266e-02, -5.2234e-02, -8.0890e-02, -1.8928e-01,
        -1.1242e-01, -1.3991e-01, -1.3631e-02, -3.9713e-02, -3.8502e-02,
         1.2384e-02, -4.7860e-02, -5.0469e-02, -4.9030e-02, -1.1937e-01,
        -4.5890e-02, -5.4148e-02,  1.0174e-01,  4.2825e-02, -1.2455e-01,
        -5.2635e-02,  1.6931e-02, -3.2642e-02,  2.5775e-02, -5.8521e-04,
         1.1131e-01, -3.2237e-01, -2.1263e-01, -1.5523e-01, -2.2633e-02,
        -1.3209e-01, -2.0707e-01, -1.4395e-01,  2.8557e-02, -1.1151e-01,
        -1.1699e-01, -1.0361e-01,  2.3665e-02,  1.1394e-01, -8.0708e-03,
        -9.5043e-02, -1.6412e-02,  5.8678e-02, -3.8916e-02, -6.9492e-02,
         1.9954e-01,  1.1880e-01,  1.3385e-02, -2.6249e-01,  3.5910e-02,
        -3.9417e-03, -8.6156e-02,  5.6854e-02, -2.9188e-01,  2.2524e-01,
         2.1894e-02,  2.3718e-02,  8.2143e-03, -3.0098e-04,  5.9870e-02,
         6.0086e-02, -5.2447e-02,  1.0241e-03, -1.7487e-01, -9.1670e-02,
         2.0855e-01,  1.6946e-01,  2.3970e-01,  2.6465e-01,  3.9498e-02,
        -4.6418e-02, -8.3714e-02,  6.8795e-02,  3.4066e-02, -3.3607e-02,
        -2.4996e-02,  6.3738e-02, -5.4643e-02,  1.1619e-01,  2.9895e-02,
         1.3742e-01,  1.1049e-02,  1.5472e-01, -4.6465e-02,  1.0337e-01,
        -1.5915e-01, -6.7063e-03, -1.2842e-01, -6.1881e-02, -6.2523e-02,
        -9.9474e-02, -9.4578e-02, -2.4702e-01, -4.3068e-02,  3.4421e-02,
        -2.1124e-01, -7.0783e-02, -1.9899e-02,  1.3388e-01,  5.6165e-03,
         6.1075e-02,  9.1476e-02, -4.5349e-02, -1.7263e-01,  1.9484e-01,
         5.3868e-02,  3.1886e-02,  8.7658e-02, -7.2621e-02,  2.7197e-01,
         6.5463e-02, -2.0580e-01, -2.1885e-01,  1.8761e-01,  3.5011e-01,
         7.8589e-02,  7.0917e-02,  1.4324e-02, -3.0991e-01, -2.1720e-01,
        -1.0291e-01, -3.4604e-02, -2.6646e-01, -1.9697e-01, -1.5077e-01,
         1.2282e-01,  1.0292e-01,  8.9109e-02,  6.1043e-02,  7.1877e-02,
         3.0346e-02,  8.4473e-02,  1.4756e-01,  1.4474e-01,  1.2895e-01,
        -5.9344e-02,  9.3804e-03, -9.1633e-03, -4.9970e-02, -2.9700e-01,
        -2.4354e-02, -1.3106e-01, -1.2390e-01, -5.9186e-02, -3.9208e-03,
         1.5488e-03,  2.0990e-01,  5.1540e-02, -1.3729e-01, -9.8071e-02,
         2.0970e-02, -1.9359e-01, -2.4360e-01,  1.1235e-01, -1.0603e-01,
        -9.0062e-02, -3.6384e-02, -1.2083e-02,  9.1143e-02,  4.7130e-02,
         6.0333e-02, -1.3556e-03,  8.1600e-03,  8.0733e-02, -1.2516e-01,
        -2.4083e-01,  3.8862e-02,  5.3107e-02,  9.5203e-02, -2.0728e-01,
        -8.8920e-02,  2.0034e-01,  3.4568e-03, -2.9490e-01,  7.7522e-02,
        -9.7631e-02,  1.4702e-01,  1.7868e-01,  1.7191e-01,  1.2213e-01,
        -8.3747e-02, -1.2294e-01, -1.2607e-01, -2.2093e-01,  1.9614e-01,
         1.5410e-01, -2.0453e-01, -1.8280e-01,  6.2391e-03,  1.0263e-01,
        -1.4882e-01, -9.9284e-02,  9.4093e-04, -1.6266e-01, -3.0463e-01,
         1.1911e-01,  1.5220e-01,  2.2722e-01, -1.8098e-01, -2.0631e-01,
         1.4744e-02, -8.2195e-02,  1.2028e-01,  3.7020e-01,  6.3127e-02,
         1.1169e-01,  1.5448e-01,  4.8887e-02,  7.8455e-02,  1.3501e-01,
        -7.5803e-02,  2.2406e-02, -1.5751e-01, -4.6605e-02,  1.2325e-01,
        -4.0029e-02,  1.2143e-01,  1.3620e-01, -4.7455e-03, -3.3129e-02,
        -1.8860e-01,  3.9480e-01,  4.9912e-02,  9.1490e-03,  1.7894e-01,
        -1.8704e-01, -1.3542e-01,  3.0697e-03,  5.3634e-02,  2.6243e-01,
        -1.1380e-02,  2.2376e-01, -2.9218e-01, -7.9082e-02, -1.2248e-02,
         9.9634e-02, -2.5000e-02,  1.6048e-01,  4.1426e-02,  2.5987e-02,
        -2.1340e-01, -4.0298e-02, -2.6220e-02,  1.7621e-01,  2.6185e-01,
         1.1027e-01,  1.7500e-02,  6.8268e-02,  1.8378e-01, -3.3011e-01,
        -1.5542e-01, -2.5599e-01, -1.3341e-01, -2.0258e-01,  2.2564e-01,
        -1.3472e-01,  3.7761e-02,  3.6655e-02,  7.5629e-02,  1.0246e-01,
        -1.0177e-01, -6.4571e-02, -1.5015e-01,  3.8200e-03, -1.5490e-02,
        -1.5571e-01,  8.9356e-03, -6.0775e-02, -1.7780e-01, -1.2651e-01,
        -1.7461e-01,  1.2965e-01, -7.5977e-02,  1.3493e-01,  1.8063e-01,
        -7.5949e-02, -1.0598e-02,  2.4223e-01, -5.6316e-02,  1.3026e-01,
        -2.0480e-01,  1.4905e-02,  1.5085e-01,  1.5405e-02,  1.3051e-02,
         6.3953e-02, -1.6716e-02,  1.7797e-01, -6.6637e-02,  1.7301e-01,
        -6.4164e-02, -2.1577e-01,  1.2037e-01, -1.2639e-01, -2.8038e-01,
        -6.8198e-02, -2.4114e-01, -9.5860e-02,  1.1178e-01,  2.2461e-01,
         1.5825e-01, -1.4054e-01,  1.9468e-01, -9.1836e-03,  1.9157e-01,
         2.0017e-01, -1.1864e-02, -2.8219e-01,  1.5138e-01, -6.7829e-02,
        -6.4934e-02,  4.0055e-02, -5.0859e-02,  1.3145e-01,  6.9896e-02,
         1.3821e-01,  1.6557e-01,  4.5449e-01,  2.5689e-03,  1.9345e-01,
         7.9665e-02,  1.1735e-01, -4.5647e-02, -6.1549e-02, -7.3458e-02,
         1.1029e-02, -8.2906e-02,  7.0356e-02, -1.3390e-01, -1.2908e-02,
         1.5957e-01,  3.8056e-01,  5.9075e-02, -1.1987e-03,  9.1533e-02,
         1.1153e-01, -1.8652e-02, -1.2906e-01,  1.2624e-01, -2.1075e-01,
        -3.2252e-01,  1.1848e-01,  3.8668e-04, -2.9650e-01,  4.8925e-02,
        -1.2870e-02,  4.2964e-02,  1.3467e-01,  5.6260e-02, -8.5271e-02,
        -2.3309e-02,  1.6945e-01, -9.2490e-02, -5.2589e-02, -9.1831e-03,
         6.9878e-03,  1.6450e-01, -7.5774e-02,  3.1508e-03,  2.1683e-01,
         1.2214e-01,  4.7382e-01, -9.9964e-02, -4.7857e-02,  2.1545e-01,
         1.5531e-01, -1.0058e-01, -2.7667e-01,  1.0134e-01, -4.4625e-02,
         9.7356e-02,  5.7945e-02, -2.7370e-01,  3.0122e-02, -5.8835e-03,
         8.1799e-02, -1.4507e-01, -9.1005e-02,  6.9274e-02, -5.1755e-02,
        -5.6292e-02,  6.8155e-02, -1.9177e-02, -1.3183e-01,  9.4093e-02,
         1.9230e-02,  3.9346e-02,  6.9255e-02,  3.4681e-02,  2.0818e-02,
        -1.9504e-01,  9.2127e-02,  2.3041e-01,  1.0040e-01,  4.5649e-02,
         1.1597e-01, -5.9034e-02, -1.3721e-01,  2.4381e-01,  4.0898e-01,
        -1.9265e-01,  1.1437e-01, -3.0613e-01, -1.2485e-01,  1.1649e-01,
         3.7834e-02,  4.9181e-02,  5.1684e-02, -1.5508e-01, -1.8788e-01,
         1.0921e-01, -1.1774e-02, -2.0575e-02, -1.5302e-01,  1.8892e-01,
        -5.7836e-02,  1.4736e-01,  2.1780e-01,  1.6046e-01, -1.9181e-03,
        -2.5870e-02,  4.6923e-02,  1.1744e-01,  1.0342e-01, -3.0752e-02,
        -1.0622e-01, -1.0846e-01,  2.6482e-01, -1.2551e-01,  1.9661e-01,
        -3.3154e-02, -1.2351e-02, -1.8701e-01, -2.9192e-01,  5.0532e-02,
         8.6846e-02,  1.0639e-01, -8.5464e-02, -1.4202e-02, -2.2996e-01,
         3.8432e-02,  2.3030e-02, -1.7606e-01,  1.5660e-01,  8.3678e-02,
        -8.1883e-02, -1.8419e-02,  4.0662e-02,  7.5267e-02, -7.0003e-02,
         1.4612e-01, -2.5737e-01, -9.8118e-02,  6.9769e-02,  2.4592e-01,
         6.2562e-02,  1.4853e-01, -1.4999e-01,  2.0389e-01, -2.7684e-01,
        -1.2487e-01, -6.8469e-02,  1.0135e-01, -5.1454e-02,  1.4001e-01,
         2.7433e-01, -1.6028e-01,  2.7894e-02,  8.8843e-02,  4.1636e-02,
        -1.1878e-01, -5.3625e-02, -1.5687e-01, -9.8381e-02, -2.0323e-01,
        -2.5566e-01, -2.0570e-02, -1.8418e-01, -7.6142e-03, -6.3120e-02,
         3.3706e-01,  1.6975e-01, -6.6878e-02, -1.3130e-01,  3.2374e-02,
        -8.1465e-02, -3.7144e-02,  1.3793e-01, -2.9004e-02,  1.5601e-01,
         2.4551e-02,  5.6699e-02,  4.1813e-02, -1.4691e-01, -4.4159e-02,
        -2.3416e-01, -4.3536e-03,  5.2111e-02,  4.7838e-02,  1.3864e-01,
         4.6563e-02, -2.3080e-01, -9.7279e-02, -9.1622e-02, -3.1359e-03,
        -6.6427e-02,  1.5936e-01,  1.9557e-01,  1.3174e-01,  7.6669e-02,
         1.7215e-03,  1.1415e-01,  2.5091e-01, -7.4041e-02, -5.7345e-02,
         4.2821e-02,  3.6942e-02,  1.9231e-02, -1.4660e-01,  7.1702e-02,
        -7.5902e-02, -2.5113e-02, -1.0607e-01, -8.4713e-02, -2.1031e-01,
        -1.7293e-01,  7.7163e-02, -1.2576e-01,  2.6104e-01, -1.2562e-01,
        -2.7724e-01, -1.2538e-01, -7.6657e-02,  8.8664e-02, -2.5284e-01,
        -1.6252e-01,  3.8775e-02,  3.9907e-02, -1.9741e-02,  1.0563e-01,
         1.1271e-01,  8.6978e-02,  1.1259e-01,  9.7066e-03, -1.6419e-01,
         1.4227e-02,  9.9716e-02, -1.6769e-01, -1.6260e-01,  1.4248e-01,
        -7.5500e-02,  1.1518e-01, -5.4837e-02,  1.0956e-01, -2.7508e-01,
         2.0352e-01,  6.7775e-02,  5.8488e-02,  3.2493e-01, -9.9399e-02,
        -2.9976e-02,  1.2461e-01, -1.3528e-01,  1.0280e-01,  1.1318e-02,
         3.9539e-02, -5.3687e-02,  2.2015e-01, -2.0679e-02,  1.0811e-01,
         1.0993e-01,  1.3502e-03,  1.1362e-01, -1.2791e-01,  2.2994e-01,
         6.2931e-02, -7.8345e-02, -5.7997e-02,  1.8099e-01, -2.2018e-01,
        -1.4327e-02,  2.8777e-02, -9.5070e-02, -1.1013e-01,  1.8045e-02,
         1.1123e-01, -2.6708e-01, -5.4672e-02, -1.5491e-01, -7.7274e-02,
        -1.6136e-01, -4.4745e-02,  1.0205e-01, -1.3358e-01, -1.1287e-02,
         4.5260e-02, -1.2522e-01, -1.8165e-01,  5.0154e-02, -2.6171e-03,
         3.7949e-01,  1.1213e-01,  3.6203e-02, -4.7182e-02, -9.4211e-02,
        -1.1604e-01,  3.6563e-02, -2.4727e-02,  4.7772e-02, -8.5675e-02,
        -1.6967e-01, -5.8638e-02, -1.4009e-01,  1.1610e-01,  2.8562e-03,
        -9.3636e-02, -3.9161e-02,  1.5125e-01,  7.4825e-02, -6.8225e-02,
         1.5103e-01,  3.3081e-01, -2.1920e-01, -3.1229e-01, -4.7715e-02,
         5.1199e-02,  7.9663e-02,  1.8951e-01, -9.7365e-02, -3.2403e-01,
         2.8805e-02, -2.6628e-02, -1.3631e-01, -1.0991e-01,  4.3665e-02,
         2.0419e-02, -1.5933e-01, -2.5870e-01, -4.3043e-02, -2.0682e-01,
        -1.1994e-02,  6.4969e-02,  6.1605e-03, -1.1270e-02,  2.0855e-01,
         1.8658e-01, -1.0420e-01,  1.4682e-01, -9.9205e-02, -3.5093e-02,
         9.2007e-02,  3.1631e-01,  1.4881e-01,  4.2133e-01, -1.0462e-01,
        -8.4428e-03, -6.3761e-02, -1.3540e-02, -1.1085e-02,  1.4075e-01,
         8.5621e-02,  1.5731e-01, -9.9690e-02,  1.3546e-01,  1.2197e-02,
        -1.5778e-01,  4.0342e-02,  1.2232e-01, -3.0100e-02, -5.9391e-02,
         3.3655e-01,  3.6885e-01, -2.1382e-01, -9.2848e-02,  6.3848e-04,
        -1.2808e-01, -2.5621e-02,  4.0231e-02, -2.3377e-01, -1.8644e-02,
        -1.7079e-02, -4.1828e-02, -2.0315e-01, -1.4588e-01,  1.3271e-01,
        -6.5365e-02,  1.7809e-01, -5.9187e-02,  2.7536e-01,  4.6639e-02,
         4.1467e-03, -5.3287e-02,  6.4254e-02, -1.0991e-02,  1.2960e-01,
         1.4998e-02, -1.7829e-02,  3.2537e-02, -1.4453e-01,  3.1979e-01,
         2.0296e-01,  8.0112e-02,  3.6600e-01, -1.6045e-02, -1.7983e-01,
        -4.8352e-05,  2.8947e-02, -1.3556e-01,  1.4477e-02,  9.9369e-02,
         4.7612e-04,  1.1871e-01,  5.4118e-02,  4.1430e-01, -2.1997e-01],
       grad_fn=<UnbindBackward0>)))
[END images]'''.splitlines()[1:-1]
  assert_matches_stdout(actual, expected)


def check_digits(actual):
  # pylint: disable=line-too-long
  expected = '\n'.join(
      '''[START digits]
(1, PredictionResult(example=array([ 0.,  0.,  0., 12., 13.,  5.,  0.,  0.,  0.,  0.,  0., 11., 16.,
        9.,  0.,  0.,  0.,  0.,  3., 15., 16.,  6.,  0.,  0.,  0.,  7.,
       15., 16., 16.,  2.,  0.,  0.,  0.,  0.,  1., 16., 16.,  3.,  0.,
        0.,  0.,  0.,  1., 16., 16.,  6.,  0.,  0.,  0.,  0.,  1., 16.,
       16.,  6.,  0.,  0.,  0.,  0.,  0., 11., 16., 10.,  0.,  0.]), inference=1))
(2, PredictionResult(example=array([ 0.,  0.,  0.,  4., 15., 12.,  0.,  0.,  0.,  0.,  3., 16., 15.,
       14.,  0.,  0.,  0.,  0.,  8., 13.,  8., 16.,  0.,  0.,  0.,  0.,
        1.,  6., 15., 11.,  0.,  0.,  0.,  1.,  8., 13., 15.,  1.,  0.,
        0.,  0.,  9., 16., 16.,  5.,  0.,  0.,  0.,  0.,  3., 13., 16.,
       16., 11.,  5.,  0.,  0.,  0.,  0.,  3., 11., 16.,  9.,  0.]), inference=2))
[END digits]'''.splitlines()[1:-1])
  # pylint: enable=line-too-long
  assert_that(actual, equal_to([expected]))

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch(
    'apache_beam.examples.snippets.transforms.elementwise.runinference.print', str)
class RunInferenceTest(unittest.TestCase):
  def test_torch_unkeyed_model_handler(self):
    runinference.torch_unkeyed_model_handler(check_torch_unkeyed_model_handler)

  def test_torch_keyed_model_handler(self):
    runinference.torch_keyed_model_handler(check_torch_keyed_model_handler)

  def test_sklearn_unkeyed_model_handler(self):
    runinference.sklearn_unkeyed_model_handler(check_sklearn_unkeyed_model_handler)

  def test_sklearn_keyed_model_handler(self):
    runinference.sklearn_keyed_model_handler(check_sklearn_keyed_model_handler)

  def test_images(self):
    runinference.images(check_images)

  def test_digits(self):
    runinference.digits(check_digits)

if __name__ == '__main__':
  unittest.main()