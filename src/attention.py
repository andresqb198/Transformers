import jax.numpy as jnp
from jaxtyping import Array
from jax.nn import softmax

class TransformedLayer:

    def __init__(self, X: Array):
        self._X = X

    def get_attentions_scores(self) -> Array:
        dimension_key = self._X.shape[-1]
        return (self._X @ self._X.T) / dimension_key
    
    def get_attention_weights(self):
        scores = self.get_attentions_scores()
        return softmax(scores, axis=-1)

        
