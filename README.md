Compressed Sensing with ISTA, FISTA and AFBN (an FISTA variant) optimization methods

The compressed sensing (CS) reconstruction model aim to obtain the sparse representation x in frequency domain from an sampled signal b represented in time domain. In math terms, that is represented by a linear inverse problem: 

 b =  PHI D x + w

where b is the sampled signal, PHI the measurement matrix, D the domain and x the signal to obtain. With x in hands, we can recontruct the orignal signal y by y = D x.
With some hypothesis under the orignal signal, the sparse solution from this problem can be obtained with hight problability solving the convex optimization problem:

minimeze  $$ || Ax - b||^2 + lambda  * ||x||, $$

where A = PHI D

That aproach bring to us the power for use all the tools from the convex optimization field. In that repository I will use the methos ISTA, FISTA and AFBN (an FISTA variant).
