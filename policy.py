# define neural net \pi_\phi(s) as a class

class Policy(object):
    def __init__(self):
        self.model =

    def get_weights(self):
        return self.model.parameters

    def update_weights(self, new_weights):
        for v,v_ in zip(new_weights, self.model.parameters()):
            v_.data.copy_(v.data)
        return None

    def clone(self):
        actor_clone = Policy()
        for v,v_ in zip(self.model.parameters(), actor_clone.model.parameters()):
        v_.data.copy_(v.data)
        return actor_clone

    def compute_prob(self):
        """
        Compute probability for each action
        """
        pass

    def rollout(self, env):
        obs = env.reset()
        done = False
        rews =[]
        while not done:
            prob = self.compute_prob(np.expand_dims(obs,0))
            prob /= np.sum(prob)
            action = np.asscalar(np.random.choice(actsize, p=prob.flatten(), size=1)) #choose according distribution prob
            
            newobs, reward, done, info = env.step(list(action))
            rews.append(reward)
            obs = newobs
        return rews

class Policy(object):
    
    def __init__(self, policy_params):
        
        # TODO Define attention network 
        self.numvars = policy_params['numvars']


        self.model = torch.nn.Sequential(
                    torch.nn.Linear(obssize,16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 8),
                    torch.nn.ReLU(),
                    torch.nn.Linear(8, actsize)
                    #input layer of input size obssize
                    #intermediate layers
                    #output layer of output size actsize
                )
        
        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.actsize = actsize
        
        # TEST
        self.compute_prob(np.random.randn(obssize).reshape(1, -1))
    
    def compute_prob(self, states):
        """
        compute prob distribution over all actions given state: pi(s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        states = torch.FloatTensor(states)
        prob = torch.nn.functional.softmax(self.model(states), dim=-1)
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)
    
    def train(self, states, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        #pdb.set_trace()
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        Qs = torch.FloatTensor(Qs)
        
        # COMPUTE probability vector pi(s) for all s in states
        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # Compute probaility pi(s,a) for all s,a
        action_onehot = self._to_one_hot(actions, actsize)
        prob_selected = torch.sum(prob * action_onehot, axis=-1)
        
        # FOR ROBUSTNESS
        prob_selected += 1e-8

        # TODO define loss function as described in the text above
#         print(Qs.shape)
#         print(prob_selected.shape)
        loss = - torch.mean(Qs * torch.log(prob_selected))

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()
            
        return loss.detach().cpu().data.numpy()


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

        # TODO: why do we need to normalize? 
        # normalizatioon
        totalob_original = (totalob_original - totalob_original.min()) / (totalob_original.max() - totalob_original.min())
        totalob = self.observation_filter(totalob_original, update=update)

        baseob = totalob[:A.shape[0],:]
        ob = totalob[A.shape[0]:,:]

        # base values - embed
        x = baseob
        for layer in self.layers:
            #print(x.shape, layer.w.shape, layer.b.shape)
            x = layer(x)        
        baseembed = x

        # cut values - embed
        x = ob
        for layer in self.layers:
            #print(x.shape, layer.w.shape, layer.b.shape)
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
        #aux = np.asarray([self.weights, mu, std])
        #return aux
        w = []
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w), mu, std