import numpy as np
import random
import tensorflow as tf
import maddpg_o.maddpg_local.common.tf_util as U
from maddpg_o.maddpg_local.common.distributions import make_pdtype
from maddpg_o.maddpg_local import AgentTrainer
from maddpg_o.maddpg_local.micro.replay_buffer_neighbor import ReplayBuffer


def p_train(make_obs_ph, act_space_n, p_index, p_func,  grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph = make_obs_ph
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        p_input = obs_ph
        p = p_func(p_input, int(act_pdtype_n[0].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution
        act_pd = act_pdtype_n[0].pdfromflat(p)
        act_sample = act_pd.sample()
        act = U.function(inputs=[obs_ph], outputs=act_sample)
        # target network
        target_p = p_func(p_input, int(act_pdtype_n[0].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        target_act_sample = act_pdtype_n[0].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph], outputs=target_act_sample)
        return act,  target_act

class PolicyTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, session, act_space_n, agent_index, args, local_q_func=False):
        self.session=session
        self.name = name
        self.n = len(obs_shape_n)
        self.num_agents = args.num_agents
        self.agent_index = agent_index
        self.args = args
        obs_ph = U.BatchInput(obs_shape_n[self.agent_index], name="observation"+str(self.agent_index)).get()


        self.act, self.target_act = p_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.get_p_variables()
        self.assign_weight()

    def action(self, obs):

        return self.act(obs[None])[0]

    def batch_action(self, obs):
        return self.act(obs)

    def batch_target_action(self, obs):
        return self.target_act(obs)

    def target_action(self, obs):
        return self.target_act(obs[None])[0]

    def get_p_variables(self,reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            self.p_variables=U.scope_vars(U.absolute_scope_name("p_func"))


    def get_weigths(self):
        weigths_dict=dict()
        weigths_dict['p_variables']=self.session.run(self.p_variables)
        return weigths_dict

    def assign_weight(self):
        self.assign_op = dict()
        self.assign_op['p_variables'] = []
        k1 = len(self.p_variables)
        self.x = []
        for i in range(k1):
            self.x.append(tf.placeholder(tf.float32, self.p_variables[i].get_shape()))
            self.assign_op['p_variables'].append(self.p_variables[i].assign(self.x[i]))

    def set_weigths(self, weight_dict):
        for i, weight in enumerate(weight_dict['p_variables']):
            self.session.run(self.assign_op['p_variables'][i], feed_dict = {self.x[i]: weight})

    def set_all_weights(self, weight_dict):
        for i, weight in enumerate(weight_dict['p_variables']):
            self.session.run(self.assign_op['p_variables'][i], feed_dict = {self.x[i]: weight})

class PolicyTargetPolicyTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, session, act_space_n, agent_index, args, local_q_func=False):
        self.session=session
        self.name = name
        self.n = len(obs_shape_n)
        self.num_agents = args.num_agents
        self.agent_index = agent_index
        self.args = args
        obs_ph = U.BatchInput(obs_shape_n[self.agent_index], name="observation"+str(self.agent_index)).get()

        self.act, self.target_act = p_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.get_p_variables()
        self.assign_weight()

    def action(self, obs):
        return self.act(obs[None])[0]

    def batch_action(self, obs):
        return self.act(obs)

    def batch_target_action(self, obs):
        return self.target_act(obs)

    def target_action(self, obs):
        return self.target_act(obs[None])[0]

    def get_p_variables(self,reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            self.p_variables=U.scope_vars(U.absolute_scope_name("p_func"))
            self.target_p_variables=U.scope_vars(U.absolute_scope_name("target_p_func"))

    def get_weigths(self):
        weigths_dict=dict()
        weigths_dict['p_variables']=self.session.run(self.p_variables)
        weigths_dict['target_p_variables']=self.session.run(self.target_p_variables)
        return weigths_dict

    def assign_weight(self):
        self.assign_op = dict()
        self.assign_op['p_variables'] = []
        self.assign_op['target_p_variables'] = []

        k1 = len(self.p_variables)

        self.x = []
        for i in range(k1):
            self.x.append(tf.placeholder(tf.float32, self.p_variables[i].get_shape()))
            self.assign_op['p_variables'].append(self.p_variables[i].assign(self.x[i]))

        self.y = []
        for i in range(k1):
            self.y.append(tf.placeholder(tf.float32, self.target_p_variables[i].get_shape()))
            self.assign_op['target_p_variables'].append(self.target_p_variables[i].assign(self.y[i]))


    def set_weigths(self, weight_dict):
        for i, weight in enumerate(weight_dict['p_variables']):
            self.session.run(self.assign_op['p_variables'][i], feed_dict = {self.x[i]: weight})

        for i, weight in enumerate(weight_dict['target_p_variables']):
            self.session.run(self.assign_op['target_p_variables'][i], feed_dict = {self.y[i]: weight})

    def set_all_weights(self, weight_dict):
        for i, weight in enumerate(weight_dict['p_variables']):
            self.session.run(self.assign_op['p_variables'][i], feed_dict = {self.x[i]: weight})

        for i, weight in enumerate(weight_dict['target_p_variables']):
            self.session.run(self.assign_op['target_p_variables'][i], feed_dict = {self.y[i]: weight})
