import numpy as np
import sys
import os
sys.path.append("../")
import tensorflow as tf
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
                 grad_batch_size = 10,
                 reward_rescale_estimator = None,
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
        self.reward_rescale_estimator = reward_rescale_estimator()
        FlowEstimator, FlowEstimator_options =  FlowEstimatorGen
        if FlowEstimator is None:
            self.FlowEstimator = FlowEstimator_options
        else:
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
        self.grad_batch_size = grad_batch_size
        self.neighborhood = self.graph.iter(neighborhood)
        self.pure_path_batch_size = batch_size
        self.batch_size = self.pure_path_batch_size * len(self.neighborhood)
        self.reg_post=reg_post
        self.reg_fn = reg_fn
    def build(self,input_shape):
        self.FlowEstimator.build((None,self.embedding_dim))

        # self.mean_MC = tf.Variable(
        #     initial_value=tf.keras.initializers.Constant(0.)(shape=(), dtype=self.dd_v),
        #     trainable=False,
        #     name='mean_MC'
        # )
        # self.n_sample = tf.Variable(
        #     initial_value=tf.keras.initializers.Constant(0)(shape=(2,), dtype='int64'),
        #     trainable=False,
        #     name='n_sample'
        # )
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
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length, self.embedding_dim), dtype=self.graph.representation_dtype),
            trainable=False)
        self.paths_true = tf.Variable(
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length, self.graph.group_dim), dtype=self.graph.group_dtype),
            trainable=False)
        self.paths_actions = tf.Variable(
            tf.zeros((self.grad_batch_size, self.batch_size, self.path_length - 1,), dtype=self.graph.group_dtype), trainable=False)


        self.paths_reward = tf.Variable(tf.zeros((self.batch_size,self.path_length)),trainable=False)
        self.path_init_flow = tf.Variable(tf.ones((self.batch_size,self.path_length)),trainable=False)
        self.forward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim),
            dtype=self.graph.representation_dtype),
            trainable=False
        )
        self.backward_edges = tf.Variable(
            tf.zeros((self.batch_size,self.path_length,1+self.nactions,self.embedding_dim),
            dtype=self.graph.representation_dtype),
            trainable=False
        )
        self.reward_rescale = tf.Variable(tf.ones(()),trainable=False)

    def call(self,inputs):
        forward_edges = self.forward_edges
        backward_edges = self.backward_edges
        path_init_flow = self.path_init_flow
        paths_reward = self.paths_reward
        res = self.FlowCompute(forward_edges, backward_edges, path_init_flow, paths_reward)
        self.n_train = len(self.trainable_variables)
        return res
    def update_paths_embedding(self):
        self.paths.assign(self.graph.embedding(self.paths_true))
    def rotate_paths(self):
        self.paths_true.assign(tf.roll(self.paths_true,shift=1,axis=0))
        self.paths.assign(tf.roll(self.paths,shift=1,axis=0))
        self.paths_actions.assign(tf.roll(self.paths_actions,shift=1,axis=0))

    @tf.function
    def update_reward(self):
        self.paths_reward.assign(
            tf.reshape(
                self.reward(
                    tf.reshape(self.paths_true[0],
                            shape=(-1,self.graph.group_dim)
                    )
                ),
                shape=self.paths_reward.shape
            )
        )

    @tf.function
    def FlowCompute(self,forward_edges,backward_edges,path_init_flow,paths_reward):
        F = self.FlowEstimator(forward_edges[:, :, 0])
        Fout = tf.reduce_sum(F,axis=-1)
        R = paths_reward/self.reward_rescale
        Finit = path_init_flow * self.initflow_estimate() * self.reward_rescale
        Fin = tf.reduce_sum(
            tf.linalg.diag_part(self.FlowEstimator.call_pathaction_wise(backward_edges[:,:,1:])),
            axis=-1
        )
        delta=1e-20
        p = tf.math.cumsum(
            tf.math.log(delta + Fout) - tf.math.log(delta + Fout + R),
            exclusive=True,
            axis=1
        )

        return tf.stack([Fin, Fout, R, Finit,p,paths_reward], axis=-1)

    @tf.function
    def gen_path(self,initial,delta=1e-20,exploration=0.):
        batch_size = initial.shape[0]
        self.paths_true[:,:batch_size,0].assign(
            tf.reshape(
                initial,
                (self.grad_batch_size,batch_size,-1)
            )
        )
        self.paths[:,:batch_size, 0].assign(tf.reshape(
            self.graph.embedding(
                tf.reshape(self.paths_true[:,:batch_size,0],(self.grad_batch_size*batch_size,-1))
            ),
            (self.grad_batch_size,batch_size,-1)
        ))
        for i in tf.range(self.path_length - 1):
            for j in tf.range(self.grad_batch_size):
                Fout = self.FlowEstimator.call_statewise(self.paths[j,:batch_size, i])
                tot = tf.reduce_sum(Fout,axis=1,keepdims=True)/self.graph.nactions
                Fout = Fout + self.exploration_forcing(tot,delta,exploration)
                self.paths_actions[j,:batch_size,i].assign(
                    tf.reshape(
                        tf.random.categorical(
                            tf.math.log(delta+Fout),
                            1,
                            dtype=self.paths_actions.dtype
                        )
                        ,
                        (-1,)
                    )
                )
                onehot = tf.one_hot(self.paths_actions[j,:batch_size,i],self.nactions,dtype=self.graph.group_dtype)
                # print(onehot.shape)
                self.paths_true[j,:batch_size, i + 1].assign(
                    tf.einsum(
                        'ia,ajk,ik->ij',
                        onehot,
                        self.graph.actions,
                        self.paths_true[j,:batch_size, i]
                    )
                )
                self.paths[j,:batch_size, i+1].assign(self.graph.embedding(self.paths_true[0,:,i+1])[:batch_size])



    @tf.function
    def update_edges(self):
        self.forward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action,self.paths_true[0])))
        self.backward_edges.assign(self.graph.embedding(tf.einsum('aij,btj->btai', self.id_action_inverse,self.paths_true[0])))
    @tf.function
    def update_init_flow(self):
        self.path_init_flow.assign(self.graph.density(self.paths_true[0]))
    @tf.function
    def update_embedding(self):
        self.paths.assign(self.graph.embedding(self.paths_true))

    @tf.function
    def update_training_distribution(self,delta=1e-20,exploration=0.,alpha=(0.9,0.1)):
        initial = self.graph.sample(self.batch_size * self.grad_batch_size)
        self.gen_path(initial,exploration=exploration)
        self.update_reward()
        self.update_edges()
        self.update_init_flow()
        self.initial_flow.assign(tf.math.log(
            (
                alpha[0]*self.initflow_estimate()+
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
        for i in range(self.grad_batch_size):
            self.rotate_paths()
            self.update_edges()
            self.update_reward()

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
            s=1.
            if i==self.grad_batch_size-1:
                s = 1/self.grad_batch_size
            if i==0:
                loss_gradients = tape.gradient(loss, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
                reg_gradients = tape.gradient(reg, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            else:
                loss_gradients = gradient_add_scalar(self,
                    tape.gradient(loss, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO),
                    loss_gradients,s
                )
                reg_gradients = gradient_add_scalar(self,
                    tape.gradient(reg, trainable_vars,unconnected_gradients=tf.UnconnectedGradients.ZERO),
                    reg_gradients,s
                )
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                elif metric.name == "initflow":
                    metric.update_state(self.initflow_estimate())
                else:
                    metric.update_state(Flownu, reg_gradients)
            self.reward_rescale_estimator.update_state(Flownu)

        a = zip(self.reg_post(self,loss_gradients,reg_gradients), trainable_vars)
        self.optimizer.apply_gradients(a)



        print([x.name for x in self.metrics])
        return {
            m.name : m.result()
            for m in self.metrics
            }
