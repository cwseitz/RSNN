import tensorflow as tf

tf.app.flags.DEFINE_integer('n_batch', 10, 'batch size of the testing set')

tf.app.flags.DEFINE_integer('n_out', 1, 'number of output neurons (number of target curves)')
tf.app.flags.DEFINE_integer('n_in', 10, 'number of input units')
tf.app.flags.DEFINE_integer('n_rec', 100, 'number of recurrent units')

tf.app.flags.DEFINE_integer('f0', 20, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 20, 'target rate for regularization')

tf.app.flags.DEFINE_integer('n_iter', 10, 'number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'number of time steps per sequence')
tf.app.flags.DEFINE_integer('print_every', 10, 'print statistics every K iterations')

tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'dampening factor to stabilize learning in RNNs')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('reg', 300., 'regularization coefficient')

tf.app.flags.DEFINE_float('p_e', 0.8, 'fraction of units that are excitatory')
tf.app.flags.DEFINE_float('p_ee', 0.16, 'excitatory-excitatory')
tf.app.flags.DEFINE_float('p_ei', 0.244, 'excitatory-inhibitory')
tf.app.flags.DEFINE_float('p_ii', 0.343, 'inhibitory-inhibitory')
tf.app.flags.DEFINE_float('p_ie', 0.318, 'inhibitory-excitatory')
tf.app.flags.DEFINE_float('mu', -0.64, 'mu for rec lognormal distribution')
tf.app.flags.DEFINE_float('sigma', 0.51, 'sigma for rec lognormal distribution')

tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', 0.03, 'threshold at which the LSNN neurons spike (in arbitrary units)')

tf.app.flags.DEFINE_bool('do_plot', True, 'interactive plots during training')
tf.app.flags.DEFINE_bool('random_feedback', True,
                         'use random feedback if true, otherwise take the symmetric of the readout weights')
tf.app.flags.DEFINE_bool('stop_z_gradients', False,
                         'stop gradients in the model dynamics to get mathematical equivalence between eprop and BPTT')
tf.app.flags.DEFINE_bool('gradient_check', True,
                         'verify that the gradients computed with e-prop match the gradients of BPTT')

tf.app.flags.DEFINE_string('eprop_or_bptt', 'bptt', 'choose the learing rule, it should be `eprop` of `bptt`')
