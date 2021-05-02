import numpy as np
import time

from es.filter import get_filter
from es.policyutils import tanh, relu, fclayer, ortho_init, lstmlayer

class Policy(object):

    def __init__(self, policy_params):

        self.numvars = policy_params['numvars']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        #if False:
        if True:
            self.observation_filter = get_filter(policy_params['ob_filter'], shape = (2,))
            self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def reset(self):
        pass

class MLPRowFeatureAttenttionPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.numvars = policy_params['numvars']
 
        hsize = policy_params['hsize']
        numlayers = policy_params['numlayers']
        embeddeddim = policy_params['embed']
        self.make_mlp_weights(policy_params['numvars']+1, embeddeddim, policy_params['hsize'], policy_params['numlayers'])

        # build up filter
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.numvars+1,))

        # for visualization
        """
        self.baseobsdict = []
        self.normalized_attentionmap = []
        self.cutsdict = []
        """
        self.t = 0 

    def make_mlp_weights(self, ob_dim, ac_dim, hsize, numlayers):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hsize = hsize
        self.numlayers = numlayers
        self.layers = []
        self.offsets = [0]
        for i in range(numlayers):
            if i == 0:
                layer = fclayer(nin=ob_dim, nout=hsize, act=tanh)
                self.offsets.append(ob_dim * hsize + hsize)
            else:
                layer = fclayer(nin=hsize, nout=hsize, act=tanh)
                self.offsets.append(hsize * hsize + hsize)
            self.layers.append(layer)
        finallayer = fclayer(nin=hsize, nout=ac_dim, act=lambda x:x, init_scale=0.01)
        self.layers.append(finallayer)
        self.offsets.append(hsize * ac_dim + ac_dim)
        self.offsets = np.cumsum(self.offsets)

    def act(self, ob, update=True, num=1):
        t1 = time.time()
        A,b,c0,cutsa,cutsb = ob
        baseob_original = np.column_stack((A, b))
        ob_original = np.column_stack((cutsa, cutsb))
        try:
            totalob_original = np.row_stack((baseob_original, ob_original))
        except:
            print(baseob_original.shape, ob_original.shape)
            print('no cut to add')
            return []

        totalob = self.observation_filter(totalob_original, update=update)

        baseob = totalob[:A.shape[0],:]
        ob = totalob[A.shape[0]:,:]

        # base values - embed
        x = baseob
        for layer in self.layers:
            x = layer(x)        
        baseembed = x

        # cut values - embed
        x = ob
        for layer in self.layers:
            x = layer(x)
        cutembed = x

        # generate scores for each cut
        attentionmap = cutembed.dot(baseembed.T)

        score = np.mean(attentionmap, axis=-1)
        assert score.size == ob.shape[0]

        score -= np.max(score)

        prob = np.exp(score) / np.sum(np.exp(score))

        if np.ndim(prob) == 2:
            assert np.shape(prob)[1] == 1
            prob = prob.flatten()

        if num == 1:
            one_hot = np.random.multinomial(1,pvals=prob)
            action = np.argmax(one_hot)
        elif num >= 1:
            # select argmax
            num = np.min([num, prob.size])
            threshold = np.sort(prob)[-num]
            indices = np.where(prob >= threshold)
            action = list(indices[0][:num])
            assert len(action) == num

        self.t += time.time() - t1
        return action

    def get_weights(self):
        w = []
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w)

    def update_weights(self, weights):
        idx = 0
        for i,j in zip(self.offsets[:-1],self.offsets[1:]):
            params = weights[i:j]
            self.layers[idx].update_weights(params)
            idx += 1

    def get_weights_plus_stats(self):     
        mu, std = self.observation_filter.get_stats()
        print('mu',mu,'std',std)
        w = []
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w), mu, std

class MLPRowFeatureLSTMEmbeddingPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.numvars = policy_params['numvars']
 
        hsize = policy_params['hsize']
        numlayers = policy_params['numlayers']
        rowembeddim = policy_params['rowembed'] # row embedding
        embeddeddim = policy_params['embed'] # attention embedding
        self.make_mlp_weights(rowembeddim, embeddeddim, policy_params['hsize'], policy_params['numlayers'])

        # embed
        self.embedlayer = lstmlayer(nin=1, nh=rowembeddim)
        self.rowembeddim = policy_params['rowembed']
        self.embedoffset = self.embedlayer.get_weights().size

        # build up filter
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.numvars+1,))

        # for visualization
        """
        self.baseobsdict = []
        self.normalized_attentionmap = []
        self.cutsdict = []
        """
        self.t = 0

    def make_mlp_weights(self, ob_dim, ac_dim, hsize, numlayers):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hsize = hsize
        self.numlayers = numlayers
        self.layers = []
        self.offsets = [0]
        for i in range(numlayers):
            if i == 0:
                layer = fclayer(nin=ob_dim, nout=hsize, act=tanh)
                self.offsets.append(ob_dim * hsize + hsize)
            else:
                layer = fclayer(nin=hsize, nout=hsize, act=tanh)
                self.offsets.append(hsize * hsize + hsize)
            self.layers.append(layer)
        finallayer = fclayer(nin=hsize, nout=ac_dim, act=lambda x:x, init_scale=0.01)
        self.layers.append(finallayer)
        self.offsets.append(hsize * ac_dim + ac_dim)
        self.offsets = np.cumsum(self.offsets)

    def rowembed(self, rows):
        s = np.zeros([rows.shape[0], self.rowembeddim * 2])
        for t in range(rows.shape[1]):
            newinput = np.expand_dims(rows[:,t], axis=1)
            h, snew = self.embedlayer(s, newinput)
            s = snew.copy()
        return h

    def act(self, observations, update=True, num=1):
        t1 = time.time()
        A,b,c0,cutsa,cutsb = observations
        baseob_original = np.column_stack((A, b))
        ob_original = np.column_stack((cutsa, cutsb))
        try:
            totalob_original = np.row_stack((baseob_original, ob_original))
        except:
            print('error')
            print(baseob_original.shape, ob_original.shape)
            return []

        # filter
        totalob = totalob_original
        
        # embed
        totalob_embed = self.rowembed(totalob)

        baseob_embed = totalob_embed[:A.shape[0],:]
        ob_embed = totalob_embed[A.shape[0]:,:]

        # base values - embed
        x = baseob_embed
        for layer in self.layers:
            x = layer(x)        
        baseembed = x

        # cut values - embed
        x = ob_embed
        for layer in self.layers:
            x = layer(x)
        cutembed = x

        # generate scores for each cut
        attentionmap = cutembed.dot(baseembed.T)

        score = np.mean(attentionmap, axis=-1)
        assert score.size == ob_embed.shape[0]

        score -= np.max(score)

        tem = 1.
        score /= tem
        prob = np.exp(score) / np.sum(np.exp(score))
        
        if np.ndim(prob) == 2:
            assert np.shape(prob)[1] == 1
            prob = prob.flatten()

        if num == 1:
            one_hot = np.random.multinomial(1,pvals=prob)
            action = np.argmax(one_hot)
        elif num >= 1:
            num = np.min([num, prob.size])
            threshold = np.sort(prob)[-num]
            indices = np.where(prob >= threshold)
            action = list(indices[0][:num])
            assert len(action) == num
        self.t = time.time() - t1
        return action

    def get_weights(self):
        w = []
        w.append(self.embedlayer.get_weights())
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w)

    def update_weights(self, weights):
        self.embedlayer.update_weights(weights[:self.embedoffset])
        weights = weights[self.embedoffset:]
        idx = 0
        for i,j in zip(self.offsets[:-1],self.offsets[1:]):
            params = weights[i:j]
            self.layers[idx].update_weights(params)
            idx += 1

    def get_weights_plus_stats(self):     
        mu, std = self.observation_filter.get_stats()
        print('mu',mu,'std',std)
        w = []
        w.append(self.embedlayer.get_weights())
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w), mu, std

