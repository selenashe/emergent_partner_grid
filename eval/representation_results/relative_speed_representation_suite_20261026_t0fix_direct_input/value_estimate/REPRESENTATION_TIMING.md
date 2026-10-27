# Representation timing

Every curve is indexed by the current observation t. Events describe transitions before that observation; actual assignments describe transition t. Previous-memory inputs are h_(t-1) and therefore exclude observation t. The other features use observation t or the policy's updated h_t. Previous memory at the final valid observation L-1 is h_(L-2), which is held forward after termination.
