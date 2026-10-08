"""One-step original Flax GRU/readout adapter (no weight conversion)."""
import jax
import jax.numpy as jnp
import flax.linen as nn

class PolicyAdapter:
    def __init__(self,trainer,network,config):
        # Audit guide:
        # Expose the frozen policy encoder, recurrent cell, and action/value readout
        # without converting its weights. Build JAX derivatives separately with respect
        # to memory and encoded input. Exact replay validation is required before
        # interpreting those derivatives.
        #
        self.trainer,self.network,self.config=trainer,network,config
        self.width=config['GRU_HIDDEN_DIM']
        self.full=jax.jit(self._full)
        self.update=jax.jit(self._update)
        self.readout=jax.jit(self._readout)
        self.jac_h=jax.jit(jax.jacfwd(lambda p,h,e:self._update(p,h[None],e[None])[0],argnums=1))
        self.jac_e=jax.jit(jax.jacfwd(lambda p,h,e:self._update(p,h[None],e[None])[0],argnums=2))
    def _full(self,params,h,obs,resets):
        # Audit guide:
        # Apply the original policy while capturing its normalized encoded input from
        # LayerNorm. Return the new carry, probabilities, value, encoded input, and
        # logits. The memory is after processing the supplied observation.
        #
        inputs=(jax.tree_util.tree_map(lambda x:x[None],obs),resets[None])
        ((new,pi,value),captured)=self.network.apply(params,h,inputs,capture_intermediates=lambda module,method:isinstance(module,nn.LayerNorm),mutable=['intermediates'])
        encoded=captured['intermediates']['LayerNorm_0']['__call__'][0][0]
        return new,pi.probs[0],value[0],encoded,pi.logits[0]
    def _update(self,params,h,e):
        # Audit guide:
        # Apply only the original GRU cell to previous memory and normalized encoded
        # input. Use the nested frozen checkpoint weights directly. This isolates
        # recurrence from the encoder and action heads.
        #
        return nn.GRUCell(features=self.width).apply({'params':next(value for name,value in params['params']['ScannedRNN_0'].items() if name.startswith('GRUCell_'))},h,e)[0]
    def _readout(self,params,h,obs):
        # Audit guide:
        # Reproduce actor/critic heads on supplied memory and apply the frozen protocol
        # action mask. A v2 initialization observation forces the action even if memory
        # perturbation changes unmasked logits. Value can still change at that forced
        # tick.
        #
        p=params['params']
        actor=nn.relu(nn.Dense(self.config['FC_DIM_SIZE']).apply({'params':p['Dense_0']},h))
        logits=nn.Dense(self.network.action_dim).apply({'params':p['Dense_1']},actor)
        if hasattr(self.trainer,'ego_action_mask'):
            legal=self.trainer.ego_action_mask(obs)
        else:
            legal=jnp.where((obs['is_t0']>.5)[...,None],self.trainer._ACTION_MASK_T0_J,self.trainer._ACTION_MASK_TGEQ1_J)
        logits=jnp.where(legal,logits,-jnp.inf)
        probabilities=jax.nn.softmax(logits,axis=-1)
        critic=nn.relu(nn.Dense(self.config['FC_DIM_SIZE']).apply({'params':p['Dense_2']},h))
        value=nn.Dense(1).apply({'params':p['Dense_3']},critic)[...,0]
        return probabilities,value,legal
