import numpy as np
import sys
import os
sys.path.append("../")
import tensorflow as tf
import tensorflow_probability as tfp
from decimal import Decimal
from metrics import ReplayBuffer,ExpectedLen,ExpectedReward

@tf.function
def gradient_add_scalar(self,grad_x,grad_y,s):
    n_train =  self.n_train
    res = [s*grad_x[i]+s*grad_y[i]  for i in range(n_train)]
    return res

@tf.function
def exploration_forcing(tot,delta,exploration):
    return  exploration

@tf.function
def no_reg(self,loss_gradients,reg_gradients):
    return loss_gradients

@tf.function
def next_action1(gathered,pos):
    return tf.einsum(
                        'bcij,bjc->bic',
                        gathered, pos,
                        name='ApplyActionEinsum'
                    )

@tf.function
def next_action1(gathered,pos):
    return tf.einsum(
                        'bcij,bjc->bic',
                        gathered, pos,
                        name='ApplyActionEinsum'
                    )


@tf.function
def next_action2(gathered,pos):
    pos = tf.experimental.numpy.swapaxes(pos,1,2)
    res =  tf.linalg.matvec(gathered, pos,name='ApplyAction')
    res =  tf.experimental.numpy.swapaxes(res,1,2)
    return res

class MultiGFlowCayleyLinear(tf.keras.Model):
    def __init__(self,
                 graph,
                 reward,
                 ncopy,
                 FlowEstimatorGen,
                 dd_v="float32",
                 name=None,
                 batch_size=64,
                 path_length=4,
                 initflow=1.0,
                 exploration_forcing=exploration_forcing,
                 reg_post=no_reg,
                 reg_fn=lambda x:tf.constant(0.),
                 grad_batch_size=10,
                 reward_rescale_estimator=None,
                 **kwargs
        ):
        if name is None:
            name = 'flow_on_'+graph.name
        super(MultiGFlowCayleyLinear, self).__init__(name=name, **kwargs)
        self.metric_list = [tf.keras.metrics.Mean(name="loss")]
        self.ncopy = ncopy
        self.graph = graph
        self.reward = reward # embedding -> RR_+
        self.embedding_dim = int(graph.embedding_dim)
        self.dd_v = dd_v
        self.reward_rescale_estimator = reward_rescale_estimator()
        FlowEstimator, FlowEstimator_options =  FlowEstimatorGen
        if FlowEstimator is None:
            self.FlowEstimator = FlowEstimator_options
        else:
            self.FlowEstimator = FlowEstimator(
                self.ncopy,
                self.graph.nactions,
                embedding_dim=self.embedding_dim,
                **FlowEstimator_options['options'],
                kernel_options=FlowEstimator_options['kernel_options']
            ) # embedding -> M(moves)
        self.path_length = path_length
        self.nactions = tf.constant(self.graph.nactions)
        self.exploration_forcing = exploration_forcing
        self.ref_initflow = tf.constant(initflow)
        self.initial_flow = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0.)(shape=(self.ncopy,), dtype=self.dd_v),
            trainable=True,
            constraint=tf.keras.constraints.non_neg(),
            name='init_flow'
        )
        self.grad_batch_size = grad_batch_size
        self.batch_size = batch_size
        self.reg_post=reg_post
        self.reg_fn = reg_fn

    def build(self,input_shape):
        # self.FlowEstimator.build((None,None,None,self.embedding_dim,self.ncopy))
        self.id_action = tf.constant(
            tf.concat([
                tf.reshape(tf.eye(self.graph.group_dim,dtype=self.graph.group_dtype),(1,self.graph.group_dim,self.graph.group_dim)),
                self.graph.actions,
            ],
            axis=0)
        )
        self.id_action_inverse = tf.constant(
            tf.concat([
                tf.reshape(tf.eye(self.graph.group_dim,dtype=self.graph.group_dtype),(1,self.graph.group_dim,self.graph.group_dim)),
                self.graph.reverse_actions,
            ],
            axis=0)
        )
        self.paths = tf.Variable(
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length, self.embedding_dim, self.ncopy), dtype=self.graph.representation_dtype),
            trainable=False)
        self.paths_true = tf.Variable(
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length, self.graph.group_dim,self.ncopy), dtype=self.graph.group_dtype),
            trainable=False)
        self.paths_actions = tf.Variable(
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length - 1,self.ncopy), dtype=self.graph.group_dtype), trainable=False)


        self.paths_reward = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.ncopy)),trainable=False)
        self.path_init_flow = tf.Variable(tf.ones((self.batch_size,self.path_length,self.ncopy)),trainable=False)
        self.forward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim,self.ncopy),
            dtype=self.graph.representation_dtype),
            trainable=False
        )
        self.backward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim,self.ncopy),
            dtype=self.graph.representation_dtype),
            trainable=False
        )
        self.reward_rescale = tf.Variable(tf.ones(()),trainable=False)


        self.action_step = tf.keras.layers.Dot(axes=(-1,1))
        self.encoding = tf.keras.layers.CategoryEncoding(num_tokens=self.nactions)

        self.embedding_layer = tf.keras.layers.Lambda(
            lambda x: self.permute(self.graph.embedding_fn(self.permute(x))),
            output_shape=(self.embedding_dim,self.ncopy),
            name='Embedding'

        )
        # tf.broadcast_to(self.graph.actions, (self.batch_size, *self.graph.actions.shape))

        self.Categorical = tf.keras.layers.Lambda(
            lambda F_r: tf.gather(
                self.graph.actions,
                tf.argmin(
                    tf.cumsum(tf.cast(F_r[:,:-1], 'float32'), axis=1) / tf.reduce_sum(F_r[:,:-1], axis=1, keepdims=True) < F_r[:,-1:],
                    axis=1
                )
            ),
            name='Categorical'
        )

        self.flow_base= tf.keras.Sequential([
            tf.keras.layers.Reshape((1, 1, self.embedding_dim, self.ncopy)),
            self.FlowEstimator,
            tf.keras.layers.Lambda(lambda x: x[:,0,0],name='Squeeze')
        ])
        self.permute = tf.keras.layers.Permute((2,1))


        self.apply_action = tf.keras.layers.Lambda(lambda gathered,pos: tf.linalg.matvec(gathered, pos,name='ApplyAction'))
        seeded_uniform_positions = tf.keras.layers.Input(shape=(self.path_length-1+self.graph.group_dim, self.ncopy))
        seeded_uniform, position = tf.split(seeded_uniform_positions,[self.path_length-1,self.graph.group_dim],axis=1)
        positions = [tf.cast(position,'int32')] #bjc
        embedded_positions = [self.embedding_layer(positions[-1])] #(bec)
        actions = []
        seeded_uniform_split = tf.split(seeded_uniform,self.path_length-1,axis=1)
        for i in range(self.path_length-1):
            F_r = tf.concat([self.flow_base(embedded_positions[-1]), seeded_uniform_split[i]], axis=1)
            gathered = self.Categorical(F_r) # bcij
            prev_pos = tf.transpose(positions[-1], perm=(0,2,1))
            positions.append(
                self.permute(tf.linalg.matvec(gathered, prev_pos, name='ApplyAction')) # bcij,bcj
            )
            embedded_positions.append(self.embedding_layer(positions[-1]))

        outputs = tf.stack(positions,axis=1), tf.stack(embedded_positions,axis=1)
        self.gen_path_model = tf.keras.Model(inputs=seeded_uniform_positions,outputs=outputs,name='Gen')

    def call(self,inputs):
        forward_edges = self.forward_edges
        backward_edges = self.backward_edges
        path_init_flow = self.path_init_flow
        paths_reward = self.paths_reward
        res = self.FlowCompute(forward_edges, backward_edges, path_init_flow, paths_reward)
        self.n_train = len(self.trainable_variables)
        return res
    def update_paths_embedding(self):
        self.paths.assign(self.graph.embedding(self.paths_true),axis=-2)
    def rotate_paths(self):
        self.paths_true.assign(tf.roll(self.paths_true,shift=1,axis=0))
        self.paths.assign(tf.roll(self.paths,shift=1,axis=0))
        self.paths_actions.assign(tf.roll(self.paths_actions,shift=1,axis=0))

    @tf.function
    def update_reward(self):
        self.paths_reward.assign(self.reward(self.paths_true[0], axis=-2))
    # @tf.function
    def FlowCompute(self,forward_edges,backward_edges, path_init_flow,paths_reward):
        f_out = tf.reduce_sum(self.FlowEstimator(forward_edges[:, :, 0:1])[:,:,0,:,:],axis=-2)
        R = paths_reward/self.reward_rescale
        f_init = path_init_flow * self.initflow_estimate() * self.reward_rescale


        f_in = tf.reduce_sum(
            tf.experimental.numpy.swapaxes(
                tf.linalg.diag_part(tf.experimental.numpy.swapaxes(
                    self.FlowEstimator(backward_edges[:, :, 1:]),
                    -1, -3)),
                -1, -2
            ),
            axis=-2
        )
        delta=1e-20
        # print('fout',f_out.shape)
        # print('R',R.shape)
        p = tf.math.cumsum(
            tf.math.log(delta + f_out) - tf.math.log(delta + f_out + R),
            exclusive=True,
            axis=1
        )
        return tf.stack([f_in, f_out, R, f_init, p, paths_reward], axis=-1)

    @tf.function
    def gen_path2_old(self, initial, delta=1e-20, exploration=0.):
        self.paths_true[:, :, 0].assign(initial) #grad_batch, batch, time, final
        self.paths[:, :, 0].assign(self.graph.embedding(self.paths_true[:, :, 0], axis=-2))
        for j in tf.range(self.grad_batch_size):
            for i in tf.range(self.path_length - 1):
                f_out = self.FlowEstimator(tf.expand_dims(self.paths[j, :, i:i+1],2))
                f_out_trans = tf.reshape(tf.experimental.numpy.swapaxes(f_out, -1, -2), (-1, self.graph.nactions))
                self.paths_actions[j, :, i].assign(
                    tf.reshape(
                        tf.random.categorical(
                            tf.math.log(delta + f_out_trans),
                            1,
                            dtype=self.paths_actions.dtype,
                            name='SampleNext'
                        )
                        ,
                        (-1,self.ncopy),
                        name='Categoricalreshape'
                    )
                )
                #
                # gathered = self.action_step([
                #     tf.cast(self.encoding(self.paths_actions[j, :, i]),'int32'),
                #     tf.broadcast_to(self.graph.actions,(self.batch_size,*self.graph.actions.shape))
                # ])
                # next_pos = next_action2(gathered, self.paths_true[j, :, i])

                gathered = tf.gather(self.graph.actions, self.paths_actions[j, :, i], name='GatherActions')
                pos = self.paths_true[j, :, i]
                next_pos = next_action2(gathered, pos)
                self.paths_true[j, :, i + 1].assign(next_pos)
                self.paths[j, :, i+1].assign(self.graph.embedding(self.paths_true[0, :, i+1],axis=-2))

    @tf.function
    def gen_path(self, initial, delta=1e-20, exploration=0.):
        for j in tf.range(self.grad_batch_size):
                seeded_uniform = tf.random.uniform(shape=(self.batch_size, self.path_length-1, self.ncopy))
                true_paths, embedded_paths = self.gen_path_model(tf.concat([tf.cast(initial[j],'float32'), seeded_uniform],axis=1))
                self.paths_true[j].assign(true_paths)
        # self.paths.assign(self.graph.embedding(self.paths_true,axis=-2))
        self.paths.assign(self.graph.embedding(self.paths_true,axis=-2))

    @tf.function
    def update_edges(self):
        self.forward_edges.assign(self.graph.embedding(tf.einsum('aij,btjc->btaic', self.id_action, self.paths_true[0]),axis=-2))
        self.backward_edges.assign(self.graph.embedding(tf.einsum('aij,btjc->btaic', self.id_action_inverse, self.paths_true[0]),axis=-2))
    @tf.function
    def update_init_flow(self):
        self.path_init_flow.assign(self.graph.density(self.paths_true[0], axis=-2))
    @tf.function
    def update_embedding(self):
        self.paths.assign(self.graph.embedding(self.paths_true, axis=-2))

    # @tf.function
    def update_training_distribution(self,delta=1e-20,exploration=0.,alpha=(0.9, 0.1)):
        initial = self.graph.sample(shape=(self.grad_batch_size, self.batch_size, self.ncopy), axis=-2)
        self.gen_path(initial, exploration=exploration)
        self.update_reward()
        self.update_edges()
        self.update_init_flow()
        self.initial_flow.assign(tf.math.log(
            (
                alpha[0]*self.initflow_estimate() +
                alpha[1]*self.aux_initflow_estimate1()
            ) / self.ref_initflow
        ))
        self.reward_rescale.assign(self.reward_rescale_estimator.fn_call())

    def evaluate(self, **kwargs):
       return {m.name : m.result() for m in self.metrics}

    @tf.function
    def initflow_estimate(self):
        return tf.math.exp(self.initial_flow)*self.ref_initflow

    @tf.function
    def aux_initflow_estimate1(self):
        return tf.reduce_mean(self.paths_reward[:,0])

    @property
    def metrics(self):
        return self.metric_list

    @tf.function
    def train_step(self, data):
        # for i in range(self.grad_batch_size):
            # self.rotate_paths()
            # self.update_edges()
            # self.update_reward()

        forward_edges = self.forward_edges
        backward_edges = self.backward_edges
        path_init_flow = self.path_init_flow
        paths_reward = self.paths_reward
        paths = forward_edges, backward_edges, path_init_flow, paths_reward
        with tf.GradientTape(persistent=True) as tape:
            Flownu = self.FlowCompute(*paths)
            reg = self.reg_fn(Flownu)
            loss = self.compiled_loss(Flownu,self.initflow_estimate())
        trainable_vars = self.trainable_variables
        # gradients = tape.jacobian(loss, trainable_vars)
        s = 1.
        # if i == self.grad_batch_size-1:
        #     s = 1/self.grad_batch_size
        # if i == 0:
        loss_gradients = tape.gradient(loss, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        reg_gradients = tape.gradient(reg, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        # else:
        #     loss_gradients = gradient_add_scalar(
        #         self,
        #         tape.gradient(loss, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO),
        #         loss_gradients, s
        #     )
        #     reg_gradients = gradient_add_scalar(
        #         self,
        #         tape.gradient(reg, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO),
        #         reg_gradients, s
        #     )
        for metric in self.metrics:
            # print(metric)
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "initflow":
                metric.update_state(self.initflow_estimate())
            else:
                metric.update_state(Flownu, reg_gradients)
        self.reward_rescale_estimator.update_state(Flownu)
        a = zip(self.reg_post(self, loss_gradients, reg_gradients), trainable_vars)
        self.optimizer.apply_gradients(a)
        return {
            m.name : m.result()[0]
            for m in self.metrics
            if m.name != 'loss'
            }
