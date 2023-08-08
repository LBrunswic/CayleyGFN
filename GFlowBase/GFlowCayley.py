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
        self.compile_build_dataset()
        self.default_exploration = default_exploration

    def build(self,input_shape): #(None,path_length,emb_dim)
        self.FlowEstimator.build((None,self.embedding_dim))
        self.initial_flow = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0.)(shape=(1,), dtype=self.dd_v),
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
        self.loss_val = tf.Variable(
            initial_value=tf.keras.initializers.Constant(1.)(shape=(), dtype=self.dd_v),
            trainable=False,
            name='init_flow'
        )
    @tf.function
    def call(self,inputs): # input shape = (batch_size*path_length*(1+naction),emb_dim)
        naction = self.graph.nactions
        F = self.FlowEstimator(inputs[:, 0])
        Fout = tf.reduce_sum(F,axis=1)
        Fin = 0
        for i in range(naction):
            Fin += self.FlowEstimator(inputs[:,i+1])[:,i]
        Flow = tf.transpose(tf.stack([Fin,Fout]))
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
            Fout = Fout+1/self.graph.nactions * tot * exploration
            Fcum = tf.math.cumsum(Fout, axis=1)
            Fcum = Fcum / tf.reshape(Fcum[:, -1], (-1, 1))
            self.paths_actions[:,i+1].assign(tf.cast(tf.random.uniform((batch_size, 1)) <= Fcum, tf.float32))
            self.paths_actions[:,i+1].assign(tf.concat([self.paths_actions[:,i+1, :1], self.paths_actions[:,i+1, 1:] - self.paths_actions[:,i+1, :-1]], axis=1))
            self.paths[:batch_size, i + 1].assign(tf.einsum('ia,ajk,ik->ij', self.paths_actions[:,i+1], self.graph.actions, self.paths[:batch_size, i]))
        if self.default_exploration is not None:
            pass
    def compile_build_dataset(self):
        @tf.function(input_signature=(tf.TensorSpec(shape=[None,self.embedding_dim], dtype=tf.float32),))
        def build_dataset(state_batch):
            return tf.einsum('aij,bj->bai', self.id_action,state_batch)
        self.build_dataset = build_dataset

    @tf.function
    def FinOutReward(self,paths):
        batch_size,length = paths.shape[:2]
        FinFout = tf.reshape(
            self(self.build_dataset(tf.reshape(paths, shape=(-1,self.embedding_dim)))),
            shape=(batch_size,length,2)
        )
        R = tf.reshape(self.reward(tf.reshape(paths,shape=(-1,self.embedding_dim))),shape=(batch_size,length,1))
        return tf.concat([FinFout,R],axis=-1)



    def show(self,folder='images',name='0'):

        F = self.FlowEstimator(self.tfnodes)
        def aux(x):
            return str(list(tf.cast(x, tf.int32).numpy()))
        # reward = self.reward(tfnodes)
        for node in range(self.edges.shape[0]):
            for action in range(self.edges.shape[1]):
                self.dot.edge(
                    self.nodes[node],
                    aux(self.edges[node,action]),
                    label='%.2E' % Decimal(float(F[node,action].numpy()))
                )
            # dot.node(nodes[node],nodes[node] +   )
        self.dot.node(aux(self.graph.initial),aux(self.graph.initial) + ' FI=%.2E' % Decimal(float(self.initial_flow.numpy())))
        self.dot.attr(label='%s' % self.loss_val.numpy())
        self.dot.render(os.path.join(folder,'%s' % name), view=False)

    @tf.function
    def train_step(self, data):
        states, mu_initial_reward = data
        Mu = mu_initial_reward[:,:1]
        initial = mu_initial_reward[:,1]
        reward = mu_initial_reward[:,2]
        with tf.GradientTape() as tape:
            Flow = self(states)
            FlowMu = tf.concat(
                [
                    Flow,
                    tf.math.log(1+Mu)
                ],
                axis=1
            )
            reward_initial = tf.stack(
                [
                    reward,
                    initial * self.initial_flow
                ]
            )
            loss = self.compiled_loss(FlowMu, reward_initial, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        a = zip(gradients, trainable_vars)
        self.optimizer.apply_gradients(a)
        self.loss_val.assign(loss)
        return {'flow_init' : self.initial_flow,'loss':loss}


    def gen_path_aux_2(self,initial_pos, delta=1e-8, exploration=1e-1):
        batch_size = int(initial_pos.shape[0])
        paths = np.zeros((batch_size,self.path_length,self.embedding_dim),dtype='float32')
        paths_actions = np.zeros((batch_size, self.path_length, self.graph.nactions),dtype='float32')
        paths[:, 0] = initial_pos
        for i in range(self.path_length - 1):
            Fout = self.FlowEstimator(paths[:batch_size, i])
            tot = tf.reduce_sum(Fout, axis=1, keepdims=True)
            Fout = Fout + 1 / self.graph.nactions * tot * exploration
            Fcum = tf.math.cumsum(Fout, axis=1)
            Fcum = Fcum / tf.reshape(Fcum[:, -1], (-1, 1))
            paths_actions[:, i + 1] = (tf.cast(tf.random.uniform((batch_size, 1)) <= Fcum, tf.float32))
            paths_actions[:, i + 1] = (tf.concat([paths_actions[:, i + 1, :1],
                                                           paths_actions[:, i + 1, 1:] - paths_actions[:,
                                                                                              i + 1, :-1]], axis=1))
            paths[:batch_size, i + 1] = (
                tf.einsum('ia,ajk,ik->ij', paths_actions[:, i + 1], self.graph.actions,
                          paths[:batch_size, i]))
        return paths

    def gen_true_path(self, delta=1e-8, initial='random'):
        if initial=='random':
            self.gen_path_aux(exploration=0)
            paths = self.paths
        else:
            paths = self.gen_path_aux_2(initial,exploration=0)
        batch_size, length = paths.shape[:2]
        FIOR = self.FinOutReward(paths).numpy()
        FIOR[:,2] = FIOR[:,0]-FIOR[:,1]
        proba = np.zeros((batch_size, length))
        proba[:, 0] = 1
        for i in range(1, length):
            proba[:, i] = proba[:, i - 1] * FIOR[:, i - 1, 1] / (
                    delta + FIOR[:, i - 1, 1] + FIOR[:, i - 1, 2]
            )
        out = []
        for i in range(batch_size):
            alpha = np.random.uniform(0.5, 1)
            for j in range(length):
                if alpha> proba[i,j]:
                    break
            out.append(paths[i,:j])
        print(proba)
        return out