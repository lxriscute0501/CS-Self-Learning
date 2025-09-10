# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    n = len(z)
    a = np.zeros(n)
    ez_sum = 0.
    for i in range(n):
        ez_sum += np.exp(z[i])
    for i in range(n):
        a[i] = np.exp(z[i]) / ez_sum
    ### END CODE HERE ### 
    return a


# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='relu', name = "L1"),
        Dense(15, activation='relu',  name = "L2"),
        Dense(10, activation='linear', name = "L3"),
        ### END CODE HERE ### 
    ], name = "my_model" 
)
