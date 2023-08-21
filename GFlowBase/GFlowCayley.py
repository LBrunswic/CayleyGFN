import numpy as np
import sys
import os
sys.path.append("../")
import tensorflow as tf
from decimal import Decimal



class GFlowCayleyLinear(tf.keras.Model):
    def __init__(self,
                 graph,
                 reward,
                 FlowEstimatorGen,
                 dd_v="float32",
                 name=None,
                 batch_size=64,
                 batch_memory=1,
                 length_cutoff_factor=4,
                 default_exploration=None,
                 initflow=1.0,
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
        self.batch_memory = batch_memory
        self.default_exploration = default_exploration
        self.nactions = tf.constant(self.graph.nactions)

    def build(self,input_shape): #(None,path_length,emb_dim)
        self.FlowEstimator.build((None,self.embedding_dim))
        self.initial_flow = tf.Variable(
            initial_value=tf.keras.initializers.Constant(2.)(shape=(1,), dtype=self.dd_v),
            trainable=True,
            constraint=tf.keras.constraints.non_neg(),
            name='init_flow'
        )
        self.paths = tf.Variable(tf.zeros((self.batch_size*self.batch_memory,self.path_length,self.embedding_dim)),trainable=False)
        self.paths_actions = tf.Variable(tf.zeros((self.batch_size,self.path_length,self.graph.nactions)),trainable=False)
        # self.Action = tf.Variable(tf.zeros((self.batch_size,self.graph.nactions)),trainable=False)
        self.id_action = tf.constant(
            tf.concat([
                tf.reshape(tf.eye(self.embedding_dim),(1,self.embedding_dim,self.embedding_dim)),
                self.graph.reverse_actions,
            ],
            axis=0)
        )
        self.id_action_inverse = tf.constant(
            tf.linalg.inv(self.id_action)
        )
        self.paths_reward = tf.Variable(tf.zeros((self.batch_size*self.batch_memory,self.path_length)),trainable=False)
        self.loss_val = tf.Variable(
            initial_value=tf.keras.initializers.Constant(1.)(shape=(), dtype=self.dd_v),
            trainable=False,
            name='init_flow'
        )
        self.path_init_flow = tf.Variable(tf.ones((self.batch_size*self.batch_memory,self.path_length)),trainable=False)
        self.density =  tf.Variable(tf.ones((self.batch_size*self.batch_memory,self.path_length)),trainable=False)

        self.FIOR = tf.Variable(tf.ones(
                (self.batch_size*self.batch_memory,self.path_length,3)
            ),
            trainable=False
        )
        print(self.FIOR.dtype)

        self.forward_edges = tf.Variable(
            tf.zeros((self.batch_size*self.batch_memory,self.path_length,1+self.nactions,self.embedding_dim)),
            trainable=False
        )
        print(self.forward_edges.dtype)
        self.backward_edges = tf.Variable(
            tf.zeros((self.batch_size*self.batch_memory,self.path_length,1+self.nactions,self.embedding_dim)),
            trainable=False
        )
        print(self.backward_edges.dtype)
        self.state_batch_size=self.batch_size*self.batch_memory,self.path_length
        self.density_one = tf.constant(tf.ones_like(self.density))
    @tf.function
    def update_reward(self):
        self.paths_reward.assign(
            tf.reshape(
                self.reward(
                    tf.reshape(self.paths,
                            shape=(-1,self.embedding_dim)
                    )
                ),
                shape=self.paths_reward.shape
            )
        )

    @tf.function
    def build_input(self):
         self.forward_edges.assign(tf.einsum('aij,btj->btai', self.id_action,self.paths))
         self.backward_edges.assign(tf.einsum('aij,btj->btai', self.id_action_inverse,self.paths))

    @tf.function
    def call(self,inputs): # input shape = (batch_size*path_length*(1+naction),emb_dim)
        nactions = self.nactions

        F = self.FlowEstimator(tf.reshape(self.forward_edges[:,:,0],shape=(-1,self.embedding_dim)))
        Fout = tf.reshape(
            tf.reduce_sum(F,axis=1),
            shape=(self.batch_size*self.batch_memory,self.path_length,1)
        ) + tf.reshape(self.paths_reward,shape=(self.batch_size*self.batch_memory,self.path_length,1))
        Fin = tf.ones_like(Fout)*self.initial_flow
        for i in tf.range(0, nactions, 1):
            Fin += tf.reshape(
                self.FlowEstimator(
                    tf.reshape(
                        self.backward_edges[:,:,i+1],
                        shape=(-1,self.embedding_dim)
                    )
                )[:,i],
                shape=(self.batch_size*self.batch_memory,self.path_length,1)
            )
        Flow = tf.concat([Fin,Fout],axis=-1)
        return Flow # (batch_size*path_length,2)

    @tf.function
    def gen_path_aux(self,delta=1e-8,exploration=1e-1,random_initial=True):
        batch_size = self.batch_size
        self.paths[batch_size:].assign( self.paths[:-batch_size])
        if random_initial:
            self.paths[:batch_size, 0].assign(self.graph.random(batch_size))
        else:
            self.paths[:batch_size, 0].assign(tf.stack([self.graph.initial] * batch_size))
        for i in range(self.path_length - 1):
            Fout = self.FlowEstimator(self.paths[:batch_size, i])
            tot = tf.reduce_sum(Fout,axis=1,keepdims=True)
            Fout = Fout+1/self.graph.nactions * (tot+delta) * exploration
            Fcum = tf.math.cumsum(Fout, axis=1)
            Fcum = Fcum / tf.reshape(Fcum[:, -1], (-1, 1))
            self.paths_actions[:,i+1].assign(tf.cast(tf.random.uniform((batch_size, 1)) <= Fcum, tf.float32))
            self.paths_actions[:,i+1].assign(tf.concat([self.paths_actions[:,i+1, :1], self.paths_actions[:,i+1, 1:] - self.paths_actions[:,i+1, :-1]], axis=1))
            self.paths[:batch_size, i + 1].assign(tf.einsum('ia,ajk,ik->ij', self.paths_actions[:,i+1], self.graph.actions, self.paths[:batch_size, i]))

    def update_training_distribution(self,delta=1e-8,exploration=1e-1):
        self.gen_path_aux(exploration=exploration)
        self.update_reward()
        self.FinOutReward()
        self.compute_density(delta=delta,exploration=0.)

    @tf.function
    def compute_density(self,delta=1e-8,exploration=1e-1):
        p = self.FIOR[:,:, 1] / (
            delta+self.FIOR[:, :, 1] + self.FIOR[:, :, 2]
        )
        self.density.assign(tf.math.cumprod(p,exclusive=True,axis=1))

    @tf.function
    def density_foo(self,delta=1e-8,exploration=1e-1):
        FIOR = self.FinOutReward_foo()
        p = FIOR[:,:, 1] / (
            delta+FIOR[:, :, 1] + FIOR[:, :, 2]
        )
        return tf.math.cumprod(p,exclusive=True,axis=1)

    @tf.function
    def FinOutReward_foo(self):
        batch_size,length = self.paths.shape[:2]
        FinFout = tf.reshape(
            self(tf.reshape(self.paths, shape=(-1,self.embedding_dim))),
            shape=(batch_size,length,2)
        )
        # R = tf.reshape(self.reward(tf.reshape(paths,shape=(-1,self.embedding_dim))),shape=(batch_size,length,1))
        R = tf.reshape(self.paths_reward,shape=(batch_size,length,1))
        return tf.concat([FinFout,R],axis=-1)

    @tf.function
    def FinOutReward(self):
        batch_size,length = self.paths.shape[:2]
        FinFout = tf.reshape(
            self(tf.reshape(self.paths, shape=(-1,self.embedding_dim))),
            shape=(batch_size,length,2)
        )
        # R = tf.reshape(self.reward(tf.reshape(paths,shape=(-1,self.embedding_dim))),shape=(batch_size,length,1))
        R = tf.reshape(self.paths_reward,shape=(batch_size,length,1))
        self.FIOR.assign(tf.concat([FinFout,R],axis=-1))


    @tf.function
    def train_step(self, data):
        # density = self.density
        # density = tf.ones_like(self.density)
        delta = 1.
        with tf.GradientTape() as tape:
            Flow = self(tf.reshape(self.paths,shape=(-1,self.embedding_dim)))
            Flow = tf.reshape(Flow,shape=(-1,2))
            nu = tf.reshape(self.density,shape=(-1,))
            loss = self.compiled_loss(Flow, nu, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        a = zip(gradients, trainable_vars)
        self.optimizer.apply_gradients(a)
        self.loss_val.assign(loss)
        return {'flow_init' : self.initial_flow,'loss':loss}
