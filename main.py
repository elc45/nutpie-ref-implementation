import bridgestan as bs
import numpy as np
from ref_implementation import sample

#bs.set_bridgestan_path("/Users/eliotcarlson/Dropbox/nutpie-ref-implementation")
data = '{"D": 1}'

model = bs.StanModel("./normal.stan",data=data)
print("model compiled")

def U(q):
    return model.log_density(q)

def grad_U(q):
    return model.log_density_gradient(q)

trace = sample(U, grad_U, epsilon=0.01, current_q=np.array([0.1]), n_samples=100)

print(trace)