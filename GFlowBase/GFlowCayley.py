import numpy as np
import sys
import os
sys.path.append("../")
import tensorflow as tf
from decimal import Decimal

@tf.function
def exploration_forcing(tot,delta,exploration):
    return  ((tot+delta) * exploration)

@tf.function
def meanphi(self,phi,paths,proba):
    PHI = phi(tf.reshape(paths[:, :-1], (-1, paths.shape[-1])))
    P = (proba[:, :-1] - proba[:, 1:])
    return tf.reduce_mean(tf.reduce_sum(tf.reshape(PHI,paths[:,:-1].shape[:2]) * P,axis=1),axis=0)

@tf.function
def Rbar_Error(paths_reward,  density, delta=1e-8):
    RRbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density,axis=1),axis=0)
    RR = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.abs(RRbar-RR)/(delta+RR)

@tf.function
def Rhat_Error(FIOR,total_R,initial_flow,delta=1e-8):
    return tf.reduce_mean(
        tf.math.abs(initial_flow+FIOR[:, :, 0]-FIOR[:, :, 1]-FIOR[:, :, 2])/(delta+total_R)
    )

@tf.function
def expected_len(density):
    return tf.reduce_mean(tf.reduce_sum(density,axis=1))

@tf.function
def expected_reward(FoutStar,R,density,delta=1e-8):
    return tf.reduce_mean(tf.reduce_sum(density*R**2/(delta+FoutStar+R),axis=1),axis=0)


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
                 **kwargs
        ):
        if name is None:
            name = 'flow_on_'+graph.name
        super(GFlowCayleyLinear, self).__init__(name=name, **kwargs)
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
        self.path_length = int(length_cutoff_factor * graph.diameter)
        self.batch_size = batch_size
        self.nactions = tf.constant(self.graph.nactions)
        self.exploration_forcing = exploration_forcing

    def build(self,input_shape):
        self.FlowEstimator.build((None,self.embedding_dim))
        self.initial_flow = tf.Variable(
            initial_value=tf.keras.initializers.Constant(2.)(shape=(1,), dtype=self.dd_v),
            trainable=True,
            constraint=tf.keras.constraints.non_neg(),
            name='init_flow'
        )
        self.paths = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.embedding_dim),dtype=self.graph.representation_dtype),trainable=False)
        self.paths_true = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.graph.group_dim),dtype=self.graph.group_dtype),trainable=False)
        self.paths_actions = tf.Variable(tf.zeros((self.batch_size,self.path_length-1,),dtype=self.graph.group_dtype),trainable=False)

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
        self.density =  tf.Variable(tf.ones((self.batch_size,self.path_length)),trainable=False)

        self.FIOR = tf.Variable(tf.ones(
                (self.batch_size,self.path_length,3)
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
        self.density_one = tf.constant(tf.ones_like(self.density))

    def call(self,inputs):
        return self.FlowCompute()

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
        Finit = tf.ones_like(Fout)*self.initial_flow
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
    def gen_path_aux(self,delta=1e-8,exploration=0.):
        batch_size = self.batch_size
        self.paths_true[:,0].assign(self.graph.random(batch_size))
        self.paths[:, 0].assign(self.graph.embedding(self.paths_true[:,0]))
        for i in tf.range(self.path_length - 1):
            Fout = self.FlowEstimator(self.paths[:, i])
            tot = tf.reduce_sum(Fout,axis=1,keepdims=True)/self.graph.nactions
            Fout = Fout + self.exploration_forcing(tot,delta,exploration)
            Fcum = tf.math.cumsum(Fout, axis=1)
            Fcum = Fcum / tf.reshape(Fcum[:, -1], (-1, 1))

            self.paths_actions[:,i].assign(
                tf.reshape(
                    tf.random.categorical(
                        tf.math.log(Fcum),
                        1,
                        dtype=self.paths_actions.dtype
                    )
                    ,
                    (-1,)
                )
            )
            onehot = tf.one_hot(self.paths_actions[:,i],self.nactions,dtype=self.graph.group_dtype)
            self.paths_true[:, i + 1].assign(
                tf.einsum(
                    'ia,ajk,ik->ij',
                    onehot,
                    self.graph.actions,
                    self.paths_true[:, i]
                )
            )
            self.paths[:, i+1].assign(self.graph.embedding(self.paths_true[:,i+1]))
        self.forward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action,self.paths_true)))
        self.backward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action_inverse,self.paths_true)))

    def update_training_distribution(self,delta=1e-8,exploration=1e-1):
        self.gen_path_aux(exploration=exploration)
        self.update_reward()
        FinFoutRinit = self.FlowCompute()
        self.FIOR.assign(FinFoutRinit[:,:,:-1])
        self.density.assign(self.density_foo(delta=delta))

    @tf.function
    def density_foo(self,delta=1e-8,exploration=0.):
        FIOR = self.FlowCompute()
        p = FIOR[:,:, 1] / (
            delta+FIOR[:, :, 1] + FIOR[:, :, 2]
        )
        return tf.math.cumprod(p,exclusive=True,axis=1)

    @tf.function
    def logdensity_foo(self,delta=1e-8,exploration=0.):
        FIOR = self.FlowCompute()
        p = tf.math.log(delta+FIOR[:,:, 1])
        p -= tf.math.log(delta+FIOR[:, :, 1] + FIOR[:, :, 2])
        return tf.math.cumsum(p,axis=1,exclusive=True)


    # @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            Flownu = tf.concat(
                [
                    self.FlowCompute(),
                    tf.expand_dims(self.density,-1),
                    tf.expand_dims(self.logdensity_foo(),-1),
                    tf.expand_dims(self.density_one,-1)
                ],
                axis=-1
            )
            loss = self.compiled_loss(Flownu,Flownu)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        a = zip(gradients, trainable_vars)
        self.optimizer.apply_gradients(a)
        self.loss_val.assign(loss)
        return {
            'flow_init' : self.initial_flow,
            'loss':loss,
            'Elen':expected_len(Flownu[...,4]),
            'Eloglen':expected_len(tf.math.exp(Flownu[...,5])),
            'Erew':expected_reward(Flownu[...,1],Flownu[...,2],Flownu[...,4],),
            'Elogrew':expected_reward(Flownu[...,1],Flownu[...,2],tf.math.exp(Flownu[...,5]),),
            'densityError': tf.reduce_mean(tf.math.abs(Flownu[...,5]-tf.math.log(Flownu[...,4]))),
        }
