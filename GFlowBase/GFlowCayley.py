import numpy as np
import sys
import os
sys.path.append("../")
import tensorflow as tf
from decimal import Decimal
from metrics import ReplayBuffer,ExpectedLen,ExpectedReward

@tf.function
def exploration_forcing(tot,delta,exploration):
    return  exploration

@tf.function
def no_reg(self,loss_gradients,reg_gradients):
    return loss_gradients


class GFlowCayleyLinear(tf.keras.Model):
    def __init__(self,
                 graph,
                 reward,
                 FlowEstimatorGen,
                 dd_v="float32",
                 name=None,
                 batch_size=64,
                 length_cutoff_factor=4,
                 initflow=1.0,
                 exploration_forcing = exploration_forcing,
                 neighborhood = 0,
                 improve_cycle=10,
                 reg_post=no_reg,
                 reg_fn=lambda x:tf.constant(0.),
                 **kwargs
        ):
        if name is None:
            name = 'flow_on_'+graph.name
        super(GFlowCayleyLinear, self).__init__(name=name, **kwargs)
        self.metric_list = [tf.keras.metrics.Mean(name="loss")]
        self.graph = graph
        self.reward = reward # embedding -> RR_+
        self.embedding_dim = int(graph.embedding_dim)
        self.dd_v = dd_v
        FlowEstimator, FlowEstimator_options =  FlowEstimatorGen
        self.FlowEstimator = FlowEstimator(
            self.graph.nactions,
            **FlowEstimator_options['options'],
            kernel_options=FlowEstimator_options['kernel_options']
        ) # embedding -> M(moves)
        self.improve_cycle = improve_cycle
        self.path_length = int(length_cutoff_factor * graph.diameter)
        self.nactions = tf.constant(self.graph.nactions)
        self.exploration_forcing = exploration_forcing
        self.ref_initflow = tf.constant(initflow)
        self.initial_flow = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0.)(shape=(1,), dtype=self.dd_v),
            trainable=True,
            constraint=tf.keras.constraints.non_neg(),
            name='init_flow'
        )
        self.neighborhood = self.graph.iter(neighborhood)
        self.pure_path_batch_size = batch_size
        self.batch_size = self.pure_path_batch_size * len(self.neighborhood)
        self.reg_post=reg_post
        self.reg_fn = reg_fn

    def build(self,input_shape):
        self.FlowEstimator.build((None,self.embedding_dim))

        self.paths = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.embedding_dim),dtype=self.graph.representation_dtype),trainable=False)
        self.paths_true = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.graph.group_dim),dtype=self.graph.group_dtype),trainable=False)
        self.paths_actions = tf.Variable(tf.zeros((self.batch_size,self.path_length-1,),dtype=self.graph.group_dtype),trainable=False)
        self.mean_MC = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0.)(shape=(), dtype=self.dd_v),
            trainable=False,
            name='mean_MC'
        )
        self.n_sample = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0)(shape=(2,), dtype='int64'),
            trainable=False,
            name='n_sample'
        )
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

        self.paths_reward = tf.Variable(tf.zeros((self.batch_size,self.path_length)),trainable=False)
        self.loss_val = tf.Variable(
            initial_value=tf.keras.initializers.Constant(1.)(shape=(), dtype=self.dd_v),
            trainable=False,
            name='init_flow'
        )
        self.path_init_flow = tf.Variable(tf.ones((self.batch_size,self.path_length)),trainable=False)
        self.path_density =  tf.Variable(tf.ones((self.batch_size,self.path_length)),trainable=False)

        self.FIOR = tf.Variable(tf.ones(
                (self.batch_size,self.path_length,4)
            ),
            trainable=False
        )

        self.forward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim),dtype=self.graph.representation_dtype),
            trainable=False
        )

        self.backward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim),dtype=self.graph.representation_dtype),
            trainable=False
        )

        self.state_batch_size=self.batch_size,self.path_length
        self.path_density_one = tf.constant(tf.ones_like(self.path_density))

    def call(self,inputs):
        res = self.FlowCompute()
        self.n_train = len(self.trainable_variables)
        return res

    @tf.function
    def update_reward(self):
        self.paths_reward.assign(
            tf.reshape(
                self.reward(
                    tf.reshape(self.paths_true,
                            shape=(-1,self.graph.group_dim)
                    )
                ),
                shape=self.paths_reward.shape
            )
        )

    @tf.function
    def FlowCompute(self):
        # TODO: Replace the reshape-apply by a linear time apply.
        nactions = self.nactions

        F = self.FlowEstimator(tf.reshape(self.forward_edges[:,:,0],shape=(-1,self.embedding_dim)))
        Fout = tf.reshape(
            tf.reduce_sum(F,axis=1),
            shape=(self.batch_size,self.path_length,1)
        )
        R = tf.reshape(self.paths_reward,shape=(self.batch_size,self.path_length,1))
        Finit = tf.reshape(self.path_init_flow,shape=(self.batch_size,self.path_length,1))*tf.math.exp(self.initial_flow)*self.ref_initflow
        Fin = tf.zeros_like(Fout)
        for i in tf.range(0, nactions, 1):
            Fin += tf.reshape(
                self.FlowEstimator(
                    tf.reshape(
                        self.backward_edges[:,:,i+1],
                        shape=(-1,self.embedding_dim)
                    )
                )[:,i],
                shape=(self.batch_size,self.path_length,1)
            )
        Flow = tf.concat([Fin, Fout, R,Finit],axis=-1)
        return Flow # (batch_size, path_length,4)


    @tf.function
    def gen_path(self,delta=1e-20,exploration=0.):
        batch_size = self.pure_path_batch_size
        self.paths_true[:batch_size,0].assign(tf.concat([self.graph.sample(batch_size)],axis=0))
        self.paths[:batch_size, 0].assign(self.graph.embedding(self.paths_true[:batch_size,0]))
        for i in tf.range(self.path_length - 1):
            Fout = self.FlowEstimator(self.paths[:batch_size, i])
            tot = tf.reduce_sum(Fout,axis=1,keepdims=True)/self.graph.nactions
            Fout = Fout + self.exploration_forcing(tot,delta,exploration)
            self.paths_actions[:batch_size,i].assign(
                tf.reshape(
                    tf.random.categorical(
                        tf.math.log(Fout),
                        1,
                        dtype=self.paths_actions.dtype
                    )
                    ,
                    (-1,)
                )
            )
            onehot = tf.one_hot(self.paths_actions[:batch_size,i],self.nactions,dtype=self.graph.group_dtype)
            self.paths_true[:batch_size, i + 1].assign(
                tf.einsum(
                    'ia,ajk,ik->ij',
                    onehot,
                    self.graph.actions,
                    self.paths_true[:batch_size, i]
                )
            )
            # print('s',batch_size)
            self.paths[:batch_size, i+1].assign(self.graph.embedding(self.paths_true[:,i+1])[:batch_size])
            # print(self.neighborhood.dtype)
        # if neighborhood:
        expanded_paths = tf.einsum('aij,btj->abti', self.neighborhood,self.paths_true[:self.pure_path_batch_size])
        expanded_paths = tf.reshape(expanded_paths, (self.batch_size,self.path_length,self.graph.group_dim))
        # assert(tf.reduce_all(expanded_paths[:self.pure_path_batch_size]==self.paths_true[:self.pure_path_batch_size]))
        self.paths_true.assign(expanded_paths)
        self.paths.assign(self.graph.embedding(self.paths_true))
        self.n_sample[1].assign(self.n_sample[1]+self.paths_true.shape[0]*self.paths_true.shape[1])


    @tf.function
    def gen_path_redraw(self,delta=1e-20,exploration=0.):
        batch_size = self.pure_path_batch_size//2
        # self.n_sample[1].assign(self.n_sample[1]+batch_size*self.paths_true.shape[1])
        self.paths_true[:batch_size,0].assign(tf.concat([self.graph.sample(batch_size)],axis=0))
        self.paths[:batch_size, 0].assign(self.graph.embedding(self.paths_true[:batch_size,0]))
        for i in tf.range(self.path_length - 1):
            Fout = self.FlowEstimator(self.paths[:batch_size, i])
            tot = tf.reduce_sum(Fout,axis=1,keepdims=True)/self.graph.nactions
            Fout = Fout + self.exploration_forcing(tot,delta,exploration)
            self.paths_actions[:batch_size,i].assign(
                tf.reshape(
                    tf.random.categorical(
                        tf.math.log(Fout),
                        1,
                        dtype=self.paths_actions.dtype
                    )
                    ,
                    (-1,)
                )
            )
            onehot = tf.one_hot(self.paths_actions[:batch_size,i],self.nactions,dtype=self.graph.group_dtype)
            self.paths_true[:batch_size, i + 1].assign(
                tf.einsum(
                    'ia,ajk,ik->ij',
                    onehot,
                    self.graph.actions,
                    self.paths_true[:batch_size, i]
                )
            )
            # print('s',batch_size)
            self.paths[:batch_size, i+1].assign(self.graph.embedding(self.paths_true[:,i+1])[:batch_size])
            # print(self.neighborhood.dtype)
        # if neighborhood:
        expanded_paths = tf.einsum('aij,btj->abti', self.neighborhood,self.paths_true[:self.pure_path_batch_size])
        expanded_paths = tf.reshape(expanded_paths, (self.batch_size,self.path_length,self.graph.group_dim))
        # assert(tf.reduce_all(expanded_paths[:self.pure_path_batch_size]==self.paths_true[:self.pure_path_batch_size]))
        self.paths_true.assign(expanded_paths)
        self.paths.assign(self.graph.embedding(self.paths_true))

    @tf.function
    def update_edges(self):
        self.forward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action,self.paths_true)))
        self.backward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action_inverse,self.paths_true)))
    @tf.function
    def update_init_flow(self):
        self.path_init_flow.assign(self.graph.density(self.paths_true))

    @tf.function
    def sort(self):
        # print(self.paths_reward)
        scores = tf.reduce_sum(self.paths_reward,axis=1)
        # print(scores)
        indices = tf.argsort(scores,axis=0)
        self.paths_true.assign(tf.gather(self.paths_true,indices,axis=0))
        self.paths.assign(tf.gather(self.paths,indices,axis=0))
        self.paths_actions.assign(tf.gather(self.paths_actions,indices,axis=0))
        self.paths_reward.assign(tf.gather(self.paths_reward,indices,axis=0))

    @tf.function
    def update_training_distribution(self,delta=1e-20,exploration=0.):
        self.gen_path(exploration=exploration)

        self.update_reward()
        self.update_ref_MC()
        for _ in range(self.improve_cycle):
            # print('redraw')
            self.sort()
            self.gen_path_redraw(exploration=exploration)
            self.update_reward()
        # self.sort()
        self.update_edges()
        self.update_init_flow()
        FinFoutRinit = self.FlowCompute()
        self.FIOR.assign(FinFoutRinit)
        path_density = self.path_density_foo(delta=delta)
        self.path_density.assign(tf.reshape(path_density,self.path_density.shape))
        alpha = [0.95,0.05,0.0,0.0,0.0]
        self.initial_flow.assign(tf.math.log(
            (
                alpha[0]*self.initflow_estimate()+
                alpha[1]*self.aux_initflow_estimate1()
            ) / self.ref_initflow
        ))

    @tf.function
    def update_ref_MC(self):
        self.mean_MC.assign(self.mean_MC + tf.reduce_mean(self.paths_reward[:,0]))
        # self.mean_MC.assign(tf.reduce_mean(self.paths_reward[:,0]))
        self.n_sample[0].assign(self.n_sample[0]+1)
        # self.n_sample[0].assign(1)

    @tf.function
    def path_density_foo(self,delta=0.,exploration=0.):
        FIOR = self.FlowCompute()
        p = FIOR[:,:, 1] / (
            delta+FIOR[:, :, 1] + FIOR[:, :, 2]
        )
        return tf.math.cumprod(p,exclusive=True,axis=1)


    @tf.function
    def logdensity_foo(self,delta=0.,exploration=0.):
        FIOR = self.FlowCompute()
        p = tf.math.log(delta+FIOR[:,:, 1]) - tf.math.log(delta+FIOR[:, :, 1] + FIOR[:, :, 2])
        return tf.math.cumsum(p,exclusive=True,axis=1)

    @tf.function
    def initflow_estimate(self):
        return tf.math.exp(self.initial_flow)*self.ref_initflow

    @tf.function
    def aux_initflow_estimate1(self):
        return tf.reduce_mean(self.paths_reward[:,0])

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.metric_list

    # @tf.function
    def train_step(self, data):
        with tf.GradientTape( persistent=True) as tape:
            Flownu = tf.concat(
                [
                    self.FlowCompute(),
                    tf.expand_dims(self.path_density,-1),
                    tf.expand_dims(self.logdensity_foo(),-1),
                    tf.expand_dims(self.path_density_foo(),-1),
                    tf.expand_dims(self.path_density_one,-1)
                ],
                axis=-1
            )
            reg = self.reg_fn(Flownu)
            loss = self.compiled_loss(Flownu,self.initflow_estimate())
        trainable_vars = self.trainable_variables
        # gradients = tape.jacobian(loss, trainable_vars)
        loss_gradients = tape.gradient(loss, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        reg_gradients = tape.gradient(reg, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        a = zip(self.reg_post(self,loss_gradients,reg_gradients), trainable_vars)
        self.optimizer.apply_gradients(a)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "initflow":
                metric.update_state(self.initflow_estimate())
            else:
                metric.update_state(Flownu, reg_gradients)

        print([x.name for x in self.metrics])
        return {
            m.name : m.result()
            for m in self.metrics
            }
